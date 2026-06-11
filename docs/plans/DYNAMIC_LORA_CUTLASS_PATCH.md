# Dynamic attention-only LoRA over CUTLASS NVFP4 MoE (Qwen3.5-122B-A10B, vLLM 0.22.1)

Status: patch written, NOT yet run (GPU owned by the 13h training run).
Artifacts:

- `serve/vllm_patches/attention_only_lora_cutlass_moe.py` (runtime monkeypatch)
- `serve/vllm_patches/sitecustomize.py` (env-gated application in every process)
- `serve/run_qwen35_122b_rh_ct_dynamic_lora.sh` (launcher, port 8000)

All file:line references below are against the installed tree
`/home/veritan-spark-01/Veritan/.venvs/qwen-serve/lib/python3.12/site-packages/vllm`
(version 0.22.1, commit g0decac0d9). Every line was re-verified by direct
source read on 2026-06-10; this is not inherited from the earlier audit.

## 1. Complete decision chain: `--enable-lora` -> MoE backend choice

Producer (the one and only write of the MoE-side flag):

1. `model_executor/layers/fused_moe/layer.py:334` - `FusedMoE.__init__`
   builds its `FusedMoEConfig` (ctor spans layer.py:318-341) with
   `is_lora_enabled=vllm_config.lora_config is not None`. Purely global:
   any `--enable-lora` makes every FusedMoE in the model report LoRA,
   regardless of what the adapter targets. The new `--lora-target-modules`
   filter does NOT feed this (it only restricts wrapping later, see
   `lora/model_manager.py:687-691` / `lora/utils.py:260-300`).
2. `model_executor/layers/fused_moe/config.py:1274` - field declaration on
   the plain `@dataclass FusedMoEConfig` (config.py:1246), default False.
   Comment at config.py:1276-1278 confirms its only purpose: filtering
   kernels via `is_supported_config`.

Consumers (every read of `is_lora_enabled` in the install, found by grep):

3. `model_executor/layers/fused_moe/modular_kernel.py:580-581` - the shared
   oracle gate `FusedMoEExperts.is_supported_config` (modular_kernel.py:536-582):
   `elif moe_config.is_lora_enabled and not cls.supports_lora(): reject("LoRA")`.
   This is the gate that kills CUTLASS.
4. `model_executor/layers/fused_moe/oracle/unquantized.py:165-168` - the
   unquantized oracle short-circuits to TRITON when LoRA is enabled. Not our
   path (our MoE is CT NVFP4) but it reads the same flag; under the patch it
   behaves as no-LoRA, which is consistent.
5. `model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py:222`
   - WNA16 `select_gemm_impl` swaps in `TritonWNA16Experts` when LoRA is
   enabled. Not our path (we are W4A4 NVFP4, not WNA16).

That is the exhaustive consumer list; nothing else in the package reads the
flag (the `is_lora_enabled` hits in `models/mamba*.py`, `jamba.py`,
`plamo2.py`, `zamba2.py`, `mamba_mixer.py:77/87/208/455` are a separate
model-level mechanism fed directly from `lora_config`, none of which is
instantiated by `qwen3_5.py`; grep of `models/qwen3_5.py`,
`models/qwen3_next.py` and `layers/mamba/gdn/qwen_gdn_linear_attn.py` shows
zero LoRA-conditional behavior).

Kernel-side support declarations consulted by gate (3):

6. `model_executor/layers/fused_moe/modular_kernel.py:747-753` -
   `FusedMoEExperts.supports_lora()` default False.
7. `model_executor/layers/fused_moe/experts/lora_experts_mixin.py:32-34` -
   `LoRAExpertsMixin` flips it True. Mixed into exactly three classes:
   `TritonExperts` (experts/triton_moe.py:54), `UnfusedOAITritonExperts`
   (experts/gpt_oss_triton_kernels_moe.py:744), `MarlinExperts`
   (experts/marlin_moe.py:693). The EMULATION kernel
   (`Nvfp4QuantizationEmulationTritonExperts`, experts/nvfp4_emulation_moe.py)
   does NOT mix it in: EMULATION+LoRA is gone in 0.22.1, confirming the
   audit.
8. `model_executor/layers/fused_moe/experts/cutlass_moe.py:666` -
   `CutlassExpertsFp4`, no mixin, so `supports_lora()` is False; device
   support at cutlass_moe.py:679-687 accepts capability families 100/110/120
   (GB10 sm_121 = family 120). (The `is_supported_config` override at
   cutlass_moe.py:1275 belongs to `CutlassExpertsW4A8Fp8`, not Fp4, and
   still delegates to the base gate at cutlass_moe.py:1288.)

Selection path for this deployment (CT NVFP4 + `--moe-backend cutlass`):

9. `model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py:48-52`
   - `CompressedTensorsW4A4Nvfp4MoEMethod.__init__` calls
   `select_nvfp4_moe_backend(config=self.moe, ...)` where `self.moe` is the
   per-layer `FusedMoEConfig` from (1).
10. `model_executor/layers/fused_moe/oracle/nvfp4.py:160-333` -
    `select_nvfp4_moe_backend`. Explicit backend path: nvfp4.py:236-258 maps
    `cutlass` -> `VLLM_CUTLASS` (nvfp4.py:141-157) -> `CutlassExpertsFp4`
    (nvfp4.py:118-123) -> `_return_or_raise` (nvfp4.py:219-234) calls
    `is_supported_config` per (3) and RAISES
    `"...does not support the deployment configuration since kernel does not
    support LoRA"` at nvfp4.py:234. Auto path (nvfp4.py:316-329) would walk
    FLASHINFER\* (unavailable here) -> VLLM_CUTLASS (rejected for LoRA) ->
    MARLIN (LoRA-capable, repack OOMs 120B on this box even with the chunked
    patch) -> EMULATION (rejected for LoRA per (7)).
11. Runtime LoRA wiring that would follow if a kernel HAD accepted LoRA:
    `lora/layers/fused_moe.py:27-70` `FusedMoEWithLoRA.__init__` asserts
    `moe_kernel.supports_lora()` (fused_moe.py:60-66, via
    `FusedMoEKernel.supports_lora` delegation at modular_kernel.py:1574-1575)
    and replaces the layer's quant method (fused_moe.py:68-70). This is the
    second kill site: even if the oracle were bypassed alone, the LoRA
    manager's wrap of `mlp.experts` would crash on this assert. Wrap origin:
    `lora/model_manager.py:133` -> `_create_lora_modules`
    (model_manager.py:375-500; FusedMoE special-cased at 434-440, wrapped via
    `from_layer` at 441-451 -> `lora/utils.py:106-124` -> `can_replace_layer`
    at lora/layers/fused_moe.py:396 (2D) / :561 (3D)). Module eligibility
    comes from `get_supported_lora_modules` (lora/utils.py:208-229; the
    FusedMoE branch at 226-227 adds the `experts` suffix) consumed at
    model_manager.py:89 and matched at model_manager.py:685-691.

So there are exactly two coupled gates: the oracle rejection (3 via 9/10)
and the wrapper assert (11). The patch addresses both, plus the silent-load
hazard described next.

## 2. What the patch overrides (and why it is safe here)

`attention_only_lora_cutlass_moe.apply_patch()` does four things:

- A. `FusedMoEConfig.is_lora_enabled` becomes a read-only-False property
  (setter swallows the write from layer.py:334 / the dataclass `__init__`).
  Consumers (3), (4), (5) all see False; oracle keeps `VLLM_CUTLASS` and the
  startup raise at nvfp4.py:234 disappears. Safe because the flag's ONLY
  role is kernel filtering (Section 1): it feeds no buffer sizing, no
  runtime dispatch, no weight handling. The MoE forward path is therefore
  byte-identical to the proven no-LoRA CUTLASS serve.
- B. `get_supported_lora_modules` (lora/utils.py:208) is wrapped to drop
  suffixes contributed only by FusedMoE modules (here: `experts`), rebinding
  the from-import in `lora/model_manager.py:27`. The LoRA manager then never
  attempts the FusedMoE wrap (no assert crash), and
  `WorkerLoRAManager._load_adapter` (lora/worker_manager.py:99-148) builds
  `expected_lora_modules` without `experts`, so the stock
  `check_unexpected_modules` (lora/lora_model.py:212-242) raises for any
  `*.experts*` tensor in a safetensors/bin/pt adapter.
- C. Fence: `FusedMoEWithLoRA.can_replace_layer` and
  `FusedMoE3DWithLoRA.can_replace_layer` return False, so no alternative
  path can construct the asserting wrapper.
- D. `LoRAModel.from_lora_tensors` (lora/lora_model.py:116-164) is wrapped
  to (i) RAISE a clear, patch-attributed error if any adapter tensor path
  contains an `experts` segment, (ii) WARN on `shared_expert`/`gate`
  segments (dense modules where LoRA does apply but is untested here), and
  (iii) remap flat text-only PEFT keys
  `base_model.model.model.layers.N.*` ->
  `base_model.model.language_model.model.layers.N.*`.

Why (iii) is required: the trainer
(`scripts/train_qwen3_5_122b_rh_nvfp4_lora_ich.py`) fine-tuned the
`AutoModelForCausalLM` text-only variant, and the saved adapter
(`checkpoint_step_50/adapter_model.safetensors`, header inspected
2026-06-10) has exactly 96 tensors named
`base_model.model.model.layers.{3,7,...,47}.self_attn.{q,k,v,o}_proj.lora_{A,B}.weight`.
vLLM serves the checkpoint as `Qwen3_5MoeForConditionalGeneration`
(registry.py:561-565), which nests the text model under `language_model`
(qwen3_5.py:770-808 -> 799-801). The model's `hf_to_vllm_mapper`
(qwen3_vl.py:1629-1635, applied during LoRA name parsing at
lora/utils.py:155-196 via worker_manager.py:127/141) only rewrites
`model.language_model.*` / `model.visual.*` prefixes, so flat keys would
parse to `model.layers.N.self_attn.q_proj`, match nothing, and
`activate_adapter` (lora/model_manager.py:285-324) would silently
`reset_lora()` every module (309-316, debug log only): a clean-looking
server that serves the base model. The remap plus the guard removes both
silent-failure modes.

### qkv fusion and o_proj mapping: confirmed workable

- `Qwen3NextAttention` (qwen3_next.py:207-285) builds
  `self_attn.qkv_proj = QKVParallelLinear(...)` (242-250) and
  `self_attn.o_proj = RowParallelLinear(...)` (252-258).
- `packed_modules_mapping` on the served class includes
  `"qkv_proj": ["q_proj","k_proj","v_proj"]` (qwen3_5.py:440-450, inherited
  into the CondGen mapping at qwen3_5.py:557-560), so
  `_register_packed_modules` (model_manager.py:710-721) +
  `_create_merged_loras_inplace` (model_manager.py:723-764,
  `PackedLoRALayerWeights.pack`) stack the three separate q/k/v LoRAs into
  the fused module, handled at runtime by `MergedQKVParallelLinearWithLoRA`
  (lora/layers/column_parallel_linear.py:431-489; selected because the
  packed list has 3 entries, can_replace_layer at 477-489).
- Shapes check out against the saved adapter: qkv_proj fuses
  `total_num_heads * (1 + attn_output_gate)` query heads (qwen3_next.py:245,
  output gate fused into q), so the wrapper's q slice is
  `num_heads*head_dim = 16384` (TP=1), matching `q_proj.lora_B [16384,16]`;
  k/v slices 512 match `[512,16]`; `o_proj` input 8192 matches
  `o_proj.lora_A [16,8192]` under `RowParallelLinearWithLoRA`. r=16 passes
  `PEFTHelper.validate_legal` (lora/peft_helper.py:114-128) against
  `--max-lora-rank 16`.

### GDN layers cannot accidentally receive these weights

`QwenGatedDeltaNetAttention` (layers/mamba/gdn/qwen_gdn_linear_attn.py:420+,
wired at qwen3_5.py:137-143 under `linear_attn`) contains only `conv1d`
(:467), `in_proj_qkvz` (:480, a single MergedColumnParallelLinear; the
"split when LoRA" comment at :478-479 is stale in this install, see
unconditional `create_qkvz_proj` at :566-590 and the CPU-only assert at
:1024), `in_proj_ba` (:491), `out_proj` (:545) plus non-linear params. No
module named `q_proj/k_proj/v_proj/o_proj` exists there, and the suffix
regex used for wrapping (lora/utils.py:250-257) cannot match `out_proj`
against `o_proj`. The 36 GDN layers' linears WILL be LoRA-wrapped (they are
LinearBase, with `in_proj_qkvz`/`in_proj_ba` packed per qwen3_5.py:448-449)
but receive no adapter weights and are reset per slot, i.e. exact zero
contribution.

### Loud-failure guarantee for expert-targeting adapters

Three independent layers must all be defeated for an expert tensor to load
silently:

1. patch D raises `ValueError` naming the patch and the merge-then-serve
   alternative for any `experts` path segment (covers safetensors, bin/pt,
   tensorizer, and direct `from_lora_tensors` callers);
2. with patch B, the stock expected-modules check
   (lora_model.py:227-242) no longer contains `experts`, so
   `*.experts.{0.gate_proj,gate_up_proj,...}` and bare `*.experts` forms
   raise the stock `"expected target modules in ... but received ..."`;
3. even if both were bypassed, no FusedMoE module is LoRA-wrapped, so
   `_create_merged_loras_inplace`/`activate_adapter` have no MoE
   destination; weights cannot be partially applied (they would be dropped,
   which is why guards 1-2 exist).

## 3. Risks

- Other readers of the global flag. Mitigated by exhaustive grep (Section 1
  list is complete for 0.22.1). Residual risk: a *future* vLLM upgrade adds
  a consumer; the patch pins behavior by class property so new readers also
  see False, which is correct for MoE-LoRA-off semantics but should be
  re-audited on any upgrade (same caveat as the peft 0.19.1 in-place patch).
- Punica buffer sizing for fused qkv. `MergedQKVParallelLinearWithLoRA`
  allocates 3 slices of `max_loras x max_lora_rank x dims` and the shrink/
  expand kernels run per slice. With `--max-loras 2 --max-lora-rank 16` the
  buffers are MBs. sm_121 punica is proven on this box (Nemotron-Nano-30B
  dynamic LoRA), but the 16384-wide q slice expand is a new shape; if the
  Triton autotune chokes, the failure is loud (kernel error), not silent.
- max-lora-rank interactions. Set exactly to 16 (adapter r). Loading any
  future r>16 adapter raises in `validate_legal`
  (lora/peft_helper.py:120-124); intended.
- Every dense linear gets wrapped, including GDN `conv1d` (a
  ColumnParallelLinear whose weight was reshaped at
  qwen_gdn_linear_attn.py:473) and the MoE router `gate`
  (qwen3_next.py:125). Wrapping is structural (forward still applies the
  base quant method then adds the LoRA product,
  lora/layers/base_linear.py:185-199); zero-weight slots contribute nothing.
  If GDN conv1d wrapping turns out to break the GDN forward (it reads
  weights directly, not via the module call), the visible symptom would be
  a startup/first-forward crash, not corruption; fallback is merge-then-serve.
- `--language-model-only` + LoRA: `_maybe_init_mm`
  (lora/model_manager.py:164-225) maps the punica wrapper onto the
  `language_model` prefix and tower LoRA stays disabled; `visual.*` modules
  get no wrapper and are skipped with a warning (model_manager.py:392-400).
  Expected log noise, not a problem.
- The key remap (patch D) is deployment-specific: it assumes flat-layout
  adapters belong to this model. Do not reuse this PYTHONPATH/env-var combo
  for serving other LoRA models; the env-var gate in sitecustomize.py exists
  precisely to keep the blast radius to this launcher.
- vLLM was upgraded 0.21 -> 0.22.1 on Jun 8 and the base CUTLASS serve has
  not been re-validated since; that is why validation step (a) exists.

## 4. Post-run validation sequence (after training completes, ~01:30)

All servers on port 8000 (Mistral path owns 8001). Run steps in order; stop
on first failure.

(a) Base serve, UNPATCHED sanity (bisects the Jun 8 vLLM upgrade from the
    patch):

    ./serve/run_qwen35_122b_nvfp4.sh   # or the rh_ct launcher pointed at BASE_DIR
    # expect: "Using 'VLLM_CUTLASS' NvFp4 MoE backend", a short completion
    # works, generation in the 11.8-14.5 tok/s band.

(b) Patched serve + adapter load:

    ./serve/run_qwen35_122b_rh_ct_dynamic_lora.sh
    # expect (per process, APIServer + EngineCore):
    #   [sitecustomize pid=...] applied attention_only_lora_cutlass_moe patch
    # expect: "Using 'VLLM_CUTLASS' NvFp4 MoE backend"
    # expect: log line "remapped 96 flat-layout adapter keys"
    # expect: GET /v1/models lists qwen3.5-122b-a10b-nvfp4 AND ich_v3_5
    # smoke both model names with a short completion.

(c) Distinguishing-prompt FT-vs-base check. `scripts/distinguish_ft.py`
    interface: `collect --url --model --output-jsonl` then
    `compare base.jsonl ft.jsonl` (100 fixed prompts, temperature 0; ICH
    domain prompts should differ, generic ones mostly not). With dynamic
    LoRA both variants are served simultaneously, no restart needed:

    python scripts/distinguish_ft.py collect --url http://localhost:8000 \
        --model qwen3.5-122b-a10b-nvfp4 --output-jsonl /tmp/qwen_base.jsonl
    python scripts/distinguish_ft.py collect --url http://localhost:8000 \
        --model ich_v3_5 --output-jsonl /tmp/qwen_ft.jsonl
    python scripts/distinguish_ft.py compare /tmp/qwen_base.jsonl /tmp/qwen_ft.jsonl
    # expect: ICH-domain prompts diverge, base-vs-base control run is identical.
    # If ZERO prompts differ, the adapter did not apply: check the remap log
    # line from (b) before suspecting the training run.

(d) Throughput vs merge-then-serve. Run the merge path
    (`run_qwen35_122b_rh_ct_lora.sh merge` then `serve`) and compare
    single-stream + 4-way concurrent tok/s against (b) on the same prompts
    (the distinguish_ft collect timings are a usable proxy; the May 31
    baseline is 11.8-14.5 tok/s). Expected: base-model requests identical to
    (a); adapter requests a few percent slower (punica adds a shrink/expand
    per wrapped linear). If dynamic LoRA costs more than ~15 percent vs the
    merged serve, prefer merged for the demo and keep dynamic for iteration.

(e) Adapter hot-swap. With VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 (set by the
    launcher):

    curl -s -X POST http://localhost:8000/v1/load_lora_adapter \
      -H 'Content-Type: application/json' \
      -d '{"lora_name": "ich_v3_5_step50", "lora_path": "/home/veritan-spark-01/Veritan/Sandbox/adapters/qwen3_5_122b_a10b_rh_nvfp4_lora_ich_v3_5/checkpoint_step_50"}'
    # expect: 200, model listed, completions differ from both base and best.
    # also negative-test the guard with a deliberately wrong adapter
    # (any MoE-targeting LoRA): expect the patch's ValueError, not a silent load.
    curl -s -X POST http://localhost:8000/v1/unload_lora_adapter \
      -H 'Content-Type: application/json' -d '{"lora_name": "ich_v3_5_step50"}'
