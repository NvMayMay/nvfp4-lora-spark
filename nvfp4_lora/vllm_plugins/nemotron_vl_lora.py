"""Runtime-LoRA enablement for NemotronH_Nano_VL_V2 in vLLM.

vLLM's multimodal wrapper NemotronH_Nano_VL_V2 does not declare SupportsLoRA, so
`vllm serve --enable-lora` hard-errors at the startup gate
(v1/worker/lora_model_runner_mixin.py: "... does not support LoRA yet"). The inner
NemotronHForCausalLM DOES support LoRA, so this is a wrapper-only gap.

We fix it by MONKEYPATCHING the original class in place, not by registering a subclass:
vLLM resolves + caches the model class during config construction, BEFORE general_plugins
run, so a `ModelRegistry.register_model(<subclass>)` override loses the race (the ORIGINAL
class is still the one instantiated -- observed empirically). Adding the LoRA-contract
attributes to the original class object itself makes `supports_lora()` True regardless of when
the class was resolved (the SupportsLoRA check is a runtime_checkable Protocol over attributes).

Loaded as a `vllm.general_plugins` entry point so `register()` also runs inside the EngineCore
subprocess (where the model is actually built).
"""
from __future__ import annotations

from vllm.model_executor.models.interfaces import SupportsLoRA
from vllm.model_executor.models.nano_nemotron_vl import NemotronH_Nano_VL_V2
from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM
from vllm.model_executor.models.utils import WeightsMapper

# The inner LLM only declares q/k/v (packed as qkv_proj) as LoRA targets -- Mamba mixers and
# routed experts are NOT LoRA-served by this. An attention-target `both`/text LoRA is fine.
_EMBEDDING_MODULES = {
    "language_model.model.embed_tokens": "input_embeddings",
    "language_model.lm_head": "output_embeddings",
}
_LORA_SKIP_PREFIXES = ["mtp.", "language_model.mtp.", "language_model.model.mtp."]

# The wrapper's own base-weight mapper only maps `language_model.backbone -> language_model.model`.
# Extend it with the PEFT-LoRA key prefixes so an adapter's keys resolve to the SERVED module
# names (language_model.model.*), modeled on Mistral3ForConditionalGeneration.hf_to_vllm_mapper.
# The specific `language_model.backbone.` rule keeps base-weight loading identical (base keys are
# language_model.backbone.* / vision_model.* / mlp1.*; none start with bare `model.`/`backbone.`,
# so the broad LoRA prefixes below only ever match adapter keys).
_LORA_MAPPER = WeightsMapper(
    orig_to_new_prefix={
        "model.language_model.backbone.": "language_model.model.",
        "language_model.backbone.": "language_model.model.",
        "model.language_model.model.": "language_model.model.",
        "language_model.model.": "language_model.model.",
        "backbone.": "language_model.model.",
        "model.": "language_model.model.",
        "lm_head.": "language_model.lm_head.",
        "language_model.lm_head.": "language_model.lm_head.",
    },
    orig_to_new_substr={
        "A_log": "A",
        "embeddings": "embed_tokens",
    },
)


def register() -> None:
    """Add the LoRA contract to NemotronH_Nano_VL_V2 in place (idempotent)."""
    if getattr(NemotronH_Nano_VL_V2, "supports_lora", False):
        return
    NemotronH_Nano_VL_V2.supports_lora = True
    NemotronH_Nano_VL_V2.packed_modules_mapping = NemotronHForCausalLM.packed_modules_mapping
    NemotronH_Nano_VL_V2.embedding_modules = _EMBEDDING_MODULES
    NemotronH_Nano_VL_V2.lora_skip_prefixes = _LORA_SKIP_PREFIXES
    NemotronH_Nano_VL_V2.hf_to_vllm_mapper = _LORA_MAPPER
    # Nemotron-Omni is a (non-gated) MoE. vLLM's LoRA manager requires the MoE flags +
    # `get_expert_mapping` even for an attention-only adapter (it sets up the MoE-LoRA format for
    # the routed experts). Delegate to the inner LLM. (We do NOT LoRA the experts; this just lets
    # the LoRA manager build for a MoE model.)
    NemotronH_Nano_VL_V2.is_non_gated_moe = getattr(NemotronHForCausalLM, "is_non_gated_moe", False)
    NemotronH_Nano_VL_V2.is_3d_moe_weight = getattr(NemotronHForCausalLM, "is_3d_moe_weight", False)

    def _get_expert_mapping(self):
        # get_expert_mapping lives on the inner NemotronHModel (self.language_model.model), not
        # on NemotronHForCausalLM (self.language_model); resolve it wherever it is.
        lm = self.language_model
        fn = getattr(lm, "get_expert_mapping", None) \
            or getattr(getattr(lm, "model", None), "get_expert_mapping", None)
        if fn is None:
            raise AttributeError("get_expert_mapping not found on language_model or its .model")
        return fn()

    NemotronH_Nano_VL_V2.get_expert_mapping = _get_expert_mapping
    # The attribute patch already satisfies the runtime_checkable protocol; adding the base too
    # keeps `isinstance(model, SupportsLoRA)` correct for any direct check. Best-effort (MRO).
    if SupportsLoRA not in NemotronH_Nano_VL_V2.__mro__:
        try:
            NemotronH_Nano_VL_V2.__bases__ = NemotronH_Nano_VL_V2.__bases__ + (SupportsLoRA,)
        except TypeError:
            pass
