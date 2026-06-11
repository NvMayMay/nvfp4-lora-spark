"""
Runtime monkeypatch: dynamic (request-time) attention-only LoRA on top of the
CUTLASS NVFP4 MoE backend, for Qwen3.5-122B-A10B compressed-tensors NVFP4 in
vLLM 0.22.1 on DGX Spark (GB10, sm_121).

All file:line anchors below refer to the qwen-serve venv install:
/home/veritan-spark-01/Veritan/.venvs/qwen-serve/lib/python3.12/site-packages/vllm

PROBLEM
=======
`--enable-lora` sets `FusedMoEConfig.is_lora_enabled=True` for EVERY FusedMoE
layer purely from `vllm_config.lora_config is not None`
(model_executor/layers/fused_moe/layer.py:334). The backend oracle then
rejects any experts kernel without LoRA support
(model_executor/layers/fused_moe/modular_kernel.py:580-581), and
`CutlassExpertsFp4` (experts/cutlass_moe.py:666, the only fast NVFP4 MoE
kernel on sm_121, capability families 100/110/120 at cutlass_moe.py:683-686)
inherits `supports_lora() -> False` (modular_kernel.py:747-753). With
`--moe-backend cutlass` forced, startup raises in `_return_or_raise`
(fused_moe/oracle/nvfp4.py:219-234, called from the explicit-backend path at
nvfp4.py:236-258 via CompressedTensorsW4A4Nvfp4MoEMethod.__init__ at
quantization/compressed_tensors/compressed_tensors_moe/
compressed_tensors_moe_w4a4_nvfp4.py:48-52). The LoRA-capable NVFP4 MoE
backends are unusable here: MARLIN repack OOMs 120B-class on this 131 GB box
and EMULATION lost LoRA support in 0.22.1 (no LoRAExpertsMixin).

Our adapter targets ONLY dense attention projections (q/k/v/o on the 12
full-attention layers), so the MoE layers genuinely have no LoRA. Dense
quantized linear + LoRA is fully supported independently of the MoE gate
(lora/layers/base_linear.py:198 applies quant_method then adds the punica
LoRA product), and punica kernels are proven on sm_121 on this box.

WHAT THIS PATCH DOES (4 pieces, all idempotent)
===============================================
A. FusedMoEConfig.is_lora_enabled becomes a property that always reads False
   (overrides the value written by fused_moe/layer.py:334; field declared at
   fused_moe/config.py:1274). Every consumer then behaves as if MoE LoRA is
   off:
     - modular_kernel.py:580-581 oracle gate -> CUTLASS passes,
     - oracle/unquantized.py:165-168 (forced-TRITON, not our path),
     - compressed_tensors_moe_wna16.py:222 (not our path).
   Dense-layer LoRA is untouched: it is driven by vllm_config.lora_config,
   not by FusedMoEConfig.

B. get_supported_lora_modules (lora/utils.py:208-229) no longer reports
   FusedMoE module suffixes (the `isinstance(module, FusedMoE)` branch at
   lora/utils.py:226-227). Effects:
     - LoRAModelManager._match_target_modules (lora/model_manager.py:685-691)
       stops matching `...mlp.experts`, so _create_lora_modules
       (model_manager.py:375-500) never wraps FusedMoE in FusedMoE3DWithLoRA /
       FusedMoEWithLoRA. Without this, the wrapper constructor would die on
       `assert moe_kernel.supports_lora()` (lora/layers/fused_moe.py:60-66)
       because piece A keeps the CUTLASS kernel.
     - WorkerLoRAManager._load_adapter (lora/worker_manager.py:99-148) builds
       expected_lora_modules WITHOUT "experts", so the stock
       check_unexpected_modules (lora/lora_model.py:212-242) already raises
       for any `*.experts*` adapter tensor.

C. Belt-and-braces fence: FusedMoEWithLoRA.can_replace_layer
   (lora/layers/fused_moe.py:396) and FusedMoE3DWithLoRA.can_replace_layer
   (lora/layers/fused_moe.py:561) return False, so no code path can construct
   the asserting wrapper while this patch is active.

D. Adapter-load guard + key remap, wrapping LoRAModel.from_lora_tensors
   (lora/lora_model.py:116-164, reached from from_local_checkpoint at
   lora_model.py:297-306):
     - RAISES a clear error if any adapter tensor targets an expert/MoE
       module (any `experts` path segment). Without the guard such tensors
       would at best hit the generic expected-modules ValueError and at worst
       be silently dropped: activate_adapter (lora/model_manager.py:309-316)
       just reset_lora()s modules with no matching weights, with only a
       debug log.
     - Remaps flat text-only PEFT keys
       `base_model.model.model.layers.N...` (produced by training against
       AutoModelForCausalLM, see scripts/train_qwen3_5_122b_rh_nvfp4_lora_ich.py)
       to `base_model.model.language_model.model.layers.N...` so they match
       the vLLM Qwen3_5MoeForConditionalGeneration module tree
       (models/qwen3_5.py:770-808 nests Qwen3_5MoeForCausalLM under
       `language_model`). The model's hf_to_vllm_mapper
       (models/qwen3_vl.py:1629-1635) only rewrites `model.language_model.*`
       and `model.visual.*` prefixes and would leave flat keys unmatched,
       which is exactly the silent-no-op case above.
     - Warns (does not raise) for `shared_expert` / `gate` segments: those
       are dense linears (qwen3_next.py:125-156) where LoRA does apply, but
       they are untested in this deployment.

WHY THIS IS SAFE FOR ATTENTION-ONLY ADAPTERS
============================================
The only thing `is_lora_enabled` does in this install is gate MoE kernel
selection (grep: layer.py:334 producer; modular_kernel.py:580,
oracle/unquantized.py:165, compressed_tensors_moe_wna16.py:222 consumers).
It feeds no buffer sizing and no runtime dispatch. The FusedMoE forward path
is byte-identical to the proven no-LoRA CUTLASS serve. Dense LoRA wrapping,
punica buffer allocation and qkv stacking (MergedQKVParallelLinearWithLoRA,
lora/layers/column_parallel_linear.py:431-489) key off
vllm_config.lora_config and packed_modules_mapping
(models/qwen3_5.py:440-450), which this patch does not touch.

ACTIVATION
==========
    from attention_only_lora_cutlass_moe import apply_patch
    apply_patch()
before the engine builds the model, in EVERY process (APIServer + spawned
EngineCore). Use serve/vllm_patches/sitecustomize.py with
PYTHONPATH=serve/vllm_patches and
VLLM_PATCH_ATTN_ONLY_LORA_CUTLASS_MOE=1 (see
serve/run_qwen35_122b_rh_ct_dynamic_lora.sh).

Applying it twice is harmless. Each piece stamps a sentinel attribute.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SENTINEL = "_attn_only_lora_cutlass_moe_patch"

# Path segments that indicate an adapter tensor targets the routed-experts
# FusedMoE module, which this deployment intentionally serves WITHOUT LoRA.
_BLOCKED_SEGMENTS = frozenset({"experts"})
# Dense MoE-adjacent modules where LoRA technically applies but is untested
# in this deployment; warn loudly instead of raising.
_WARN_SEGMENTS = frozenset({"shared_expert", "shared_expert_gate", "gate"})

_FLAT_PREFIX = "base_model.model.model.layers."
_FLAT_OLD = "base_model.model.model."
_FLAT_NEW = "base_model.model.language_model.model."


def _patch_a_fused_moe_config() -> None:
    """FusedMoEConfig.is_lora_enabled always reads False.

    Overrides the True written at fused_moe/layer.py:334; consumed by the
    oracle gate at modular_kernel.py:580-581.
    """
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig

    if getattr(FusedMoEConfig, _SENTINEL, False):
        return

    def _get(self) -> bool:
        return False

    def _set(self, value) -> None:
        # The dataclass __init__ (config.py:1274 default) and layer.py:334
        # both write here; record what was requested for debuggability but
        # never expose it.
        self.__dict__["_requested_is_lora_enabled"] = bool(value)

    FusedMoEConfig.is_lora_enabled = property(_get, _set)
    setattr(FusedMoEConfig, _SENTINEL, True)
    logger.info(
        "attention_only_lora_cutlass_moe: FusedMoEConfig.is_lora_enabled "
        "pinned to False (MoE backend oracle will keep CUTLASS)."
    )


def _patch_b_supported_lora_modules() -> None:
    """Drop FusedMoE suffixes from the supported-LoRA-modules discovery.

    Wraps lora/utils.py:208-229; rebinds the from-import in
    lora/model_manager.py:27 (called at model_manager.py:89).
    """
    import vllm.lora.model_manager as lora_model_manager
    import vllm.lora.utils as lora_utils

    original = lora_utils.get_supported_lora_modules
    if getattr(original, _SENTINEL, False):
        return

    def patched_get_supported_lora_modules(model):
        from vllm.model_executor.layers.fused_moe import FusedMoE
        from vllm.model_executor.layers.linear import LinearBase

        supported = original(model)

        # Suffixes contributed ONLY by FusedMoE modules (lora/utils.py:226-227
        # adds them). Keep a suffix if any LinearBase or embedding module also
        # uses it, so dense layers can never be knocked out by a name clash.
        moe_only: set[str] = set()
        for name, module in model.named_modules():
            if isinstance(module, FusedMoE):
                moe_only.add(name.split(".")[-1])
        for name, module in model.named_modules():
            if isinstance(module, LinearBase):
                moe_only.discard(name.split(".")[-1])
            embedding_modules = getattr(module, "embedding_modules", None)
            if embedding_modules is not None:
                for embedding_name in embedding_modules:
                    moe_only.discard(embedding_name)

        filtered = [s for s in supported if s not in moe_only]
        if moe_only:
            logger.info(
                "attention_only_lora_cutlass_moe: excluding FusedMoE module "
                "suffixes %s from LoRA wrapping (MoE stays on the CUTLASS "
                "kernel; adapters must not target them).",
                sorted(moe_only),
            )
        return filtered

    setattr(patched_get_supported_lora_modules, _SENTINEL, True)
    lora_utils.get_supported_lora_modules = patched_get_supported_lora_modules
    # model_manager imported the symbol by name; rebind it there too.
    lora_model_manager.get_supported_lora_modules = (
        patched_get_supported_lora_modules
    )


def _patch_c_wrapper_fence() -> None:
    """FusedMoE LoRA wrappers refuse to wrap anything.

    Fences lora/layers/fused_moe.py:396 (FusedMoEWithLoRA) and :561
    (FusedMoE3DWithLoRA) so from_layer (lora/utils.py:106-124) can never
    reach the `assert moe_kernel.supports_lora()` at
    lora/layers/fused_moe.py:60-66.
    """
    from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA

    if getattr(FusedMoEWithLoRA, _SENTINEL, False):
        return

    @classmethod
    def _never_replace(cls, *args, **kwargs) -> bool:
        return False

    FusedMoEWithLoRA.can_replace_layer = _never_replace
    FusedMoE3DWithLoRA.can_replace_layer = _never_replace
    setattr(FusedMoEWithLoRA, _SENTINEL, True)


def _remap_key(name: str) -> str:
    """Map flat text-only PEFT keys onto the ConditionalGeneration layout.

    base_model.model.model.layers.N...  (AutoModelForCausalLM training layout)
      -> base_model.model.language_model.model.layers.N...
         (vLLM Qwen3_5MoeForConditionalGeneration layout, qwen3_5.py:770-808)

    Keys already in `...model.language_model...` form are left alone; the
    model's hf_to_vllm_mapper (qwen3_vl.py:1629-1635) handles those inside
    parse_fine_tuned_lora_name (lora/utils.py:155-196).
    """
    if name.startswith(_FLAT_PREFIX):
        return _FLAT_NEW + name[len(_FLAT_OLD):]
    return name


def _check_and_remap_tensor_names(tensors: dict) -> dict:
    """Apply the MoE guard, then the flat-layout remap, to a tensors dict."""
    blocked: list[str] = []
    warned: list[str] = []
    for name in tensors:
        segments = name.split(".")
        if _BLOCKED_SEGMENTS.intersection(segments):
            blocked.append(name)
        elif _WARN_SEGMENTS.intersection(segments):
            warned.append(name)

    if blocked:
        raise ValueError(
            "attention_only_lora_cutlass_moe: this adapter targets MoE "
            f"expert modules ({blocked[:4]}{'...' if len(blocked) > 4 else ''}; "
            f"{len(blocked)} tensors total). This server is patched to keep "
            "the FusedMoE layers on the CUTLASS NVFP4 kernel, which has NO "
            "LoRA support, so these weights could never be applied. Refusing "
            "to load rather than silently serving the base experts. Use the "
            "merge-then-serve path instead "
            "(serve/run_qwen35_122b_rh_ct_lora.sh merge) or serve this "
            "adapter on a LoRA-capable MoE backend."
        )
    if warned:
        logger.warning(
            "attention_only_lora_cutlass_moe: adapter touches MoE-adjacent "
            "dense modules %s. LoRA does apply to these via punica, but this "
            "is untested in this deployment; validate outputs carefully.",
            warned[:8],
        )

    remapped = {}
    n_remapped = 0
    for name, tensor in tensors.items():
        new_name = _remap_key(name)
        if new_name != name:
            n_remapped += 1
        remapped[new_name] = tensor
    if n_remapped:
        logger.info(
            "attention_only_lora_cutlass_moe: remapped %d flat-layout adapter "
            "keys (model.layers.*) to the language_model.model.layers.* "
            "module tree.",
            n_remapped,
        )
    return remapped


def _patch_d_adapter_load_guard() -> None:
    """Guard + remap inside LoRAModel.from_lora_tensors.

    Wraps lora/lora_model.py:116-164. All adapter ingestion paths
    (from_local_checkpoint safetensors/bin/pt/tensorizer,
    lora_model.py:244-306) funnel through this classmethod.
    """
    from vllm.lora.lora_model import LoRAModel

    original = LoRAModel.from_lora_tensors.__func__
    if getattr(original, _SENTINEL, False):
        return

    def patched_from_lora_tensors(cls, lora_model_id, tensors, *args, **kwargs):
        tensors = _check_and_remap_tensor_names(dict(tensors))
        return original(cls, lora_model_id, tensors, *args, **kwargs)

    setattr(patched_from_lora_tensors, _SENTINEL, True)
    LoRAModel.from_lora_tensors = classmethod(patched_from_lora_tensors)


def apply_patch() -> None:
    """Apply all four pieces. Idempotent. Call before engine/model init."""
    _patch_a_fused_moe_config()
    _patch_b_supported_lora_modules()
    _patch_c_wrapper_fence()
    _patch_d_adapter_load_guard()
    logger.info(
        "attention_only_lora_cutlass_moe: patch active (CUTLASS MoE + "
        "dense-only dynamic LoRA, expert-targeting adapters rejected)."
    )
