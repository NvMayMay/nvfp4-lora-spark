# Fork Plan: Qwen3.5-122B and Mistral-Small-4-119B NVFP4 LoRA

**Status:** **APPROVED v4** — locked after 3 rounds of iterative review with persistent Sonnet + codex gpt-5.5. Both reviewers approved v4. Round-by-round audit trail at `/tmp/fork_plan_v{2,3,4}_{sonnet,codex}_round{1,2,3}.{md,txt}`.
**Hardware:** DGX Spark, single GB10, 130 GB UMA
**Synthesized from:** 3 Sonnet research reports (Qwen3.5+compressed-tensors,
Mistral-Small-4, trainer-refactor) and the existing Mistral quantization
plan.

## What the research turned up that changes the picture

1. **Qwen3.5-122B is hybrid Mamba+attention** (like Nemotron-H), not pure
   attention. 36 of 48 layers are `linear_attention` (Mamba2/GatedDeltaNet
   SSM, not quantized); only 12 are `full_attention`. The cached-prefix /
   Mamba-multitoken work from the Nemotron line is at least partially
   relevant later.
2. **Qwen3.5 expert weights are stored as fused 3D parameters**
   (`gate_up_proj: (256, 2*1024, 3072)`). `named_modules()` walks for
   `nn.Linear` silently skip every routed-expert weight. This is the P0
   blocker for expert-LoRA on Qwen3.5 — unavoidable for the obvious
   targeting.
3. **compressed-tensors NVFP4 key naming confirmed exactly**:
   `weight_packed` (uint8), `weight_scale` (fp8_e4m3fn), `weight_global_scale`
   (fp32 per-tensor), plus a new `input_global_scale` (fp32) with no
   ModelOpt equivalent. Low-nibble packing matches ModelOpt; `_unpack_nibbles`
   works unchanged. Detection via key sniff: `weight_packed` → CT,
   `weight` → ModelOpt.
4. **Mistral-Small-4-119B is sparse MoE (128 experts, top-4) AND multimodal**
   (Pixtral vision encoder embedded). Active compute ~6.5B-equivalent. Can be
   loaded text-only by never passing pixel_values; vision weights stay
   resident unless explicitly dropped.
5. **Mistral-Small-4 quant path B (in-house from BF16) is byte-for-byte
   schema-compatible with Qwen3.5's NVFP4 format** — one loader path
   serves both. Option A (consolidated loader) is isolated work; Option
   B is the unifying choice.
6. **Most of the existing trainer core is portable** (loss, dequant,
   linear, ~70% of the trainer machinery). The Nemotron-specific
   patches are isolated to a few named helpers and can be guarded.
   Pre-M1a and Pre-M1b are fully portable; Pre-M1c is Nemotron-only.

## Naming conventions used throughout this plan (codex round-2 note)

Mistral terminology spans three layers that this plan uses interchangeably:

| Layer | Token | Notes |
|---|---|---|
| Public model name | **Mistral-Small-4-119B-2603** | The actual model we want to fine-tune. |
| HF `config.json` `model_type` | **`mistral3`** | The HF Transformers model-type identifier the axolotl BF16 config declares. Misleading: it's used for both Mistral 3 family wrappers AND Mistral-Small-4 (which dispatches to `Mistral4*` modeling code internally). |
| HF modeling class prefix | **`Mistral4`** (e.g. `Mistral4ForConditionalGeneration`, `Mistral4Attention`, `Mistral4NaiveMoe`, `Mistral4DecoderLayer`) | The actual Python class names that get instantiated when you load Mistral-Small-4. |

When this plan says `model_family="mistral3"`, that's the HF model_type string (matches the axolotl config). When it cites classes like `Mistral4Attention` or `Mistral4DecoderLayer`, those are the actual modeling classes that get instantiated. **Not a typo — this is the upstream inconsistency.**

Qwen3.5 is consistent: model_type `qwen3_5_moe`, classes `Qwen3_5Moe*`.

## Strategic decisions baked into this plan (v2 — post four-reviewer consensus)

| Decision | Choice | Why |
|---|---|---|
| Mistral quant path | **Option B** (in-house quant from axolotl BF16) | Output is **weight-storage-compatible** with Qwen3.5 (per gpt-5.4 walk-back — not byte-for-byte schema-identical). One loader path; Option A is 2-4 weeks of isolated work. |
| Quant mode for Mistral | **NVFP4A16 data-free** | Runs in 30-60 min on Spark; full W4A4 would need cloud calibration. For a LoRA-FT base, A16 is fine. Companion quant plan Stage 4 recipe uses `scheme="NVFP4A16"`, not `"NVFP4"`. |
| Trainer shape | **Per-model scripts + shared `nvfp4_lora/training_utils.py`** | Per trainer-refactor research. Lower coupling than a unified dispatcher. |
| Both-models expert-LoRA scope (v1) | **Strictly attention-only LoRA + tightened acceptance gates** | Both models have the same fused-3D blocker. v1 ships an integration baseline, not guaranteed adaptation. Shared-expert LoRA is explicitly a v1.1/v2 decision with separate target counts and acceptance gates. A5/B4 must verify with on/off deltas, target-count assertions, frozen-base control. |
| Nemotron-Super Pre-M1c/d | **Preserve for future Qwen3.5 long-context** (not "shelve indefinitely") | Qwen3.5 has 36 GatedDeltaNet SSM layers — different kernel from Nemotron Mamba2, same descriptor-pressure class. Lessons may recur. |
| Order of execution | **Phase 0 shared infra first; then by readiness or parallel** | Both models share the same fused-3D + loader blockers. Pre-fork hardening is the actual bottleneck. Qwen3.5 is already quantized; Mistral needs download+quant. With two engineers, parallel tracks AFTER Phase 0 lands. |

Each of these is challengeable — flag any you disagree with.

## Phase 0: Pre-fork hardening (~2-3 weeks)

The work that has to land before either model-specific track can start.
All of it is independently useful for the Nemotron line too — no
regression risk to existing v1.1.

Phase 0 grew from "~1 week" to "~2-3 weeks" after the four-reviewer
consensus surfaced a previously-missed BLOCKER (Phase 0.6) — both
Mistral and Qwen3.5 routed-expert base weights need a fused-3D
loader + forward path before even attention-only LoRA v1 is unblocked.

### 0.1 Compressed-tensors NVFP4 decode

**Files**: [`nvfp4_lora/dequant.py`](../../nvfp4_lora/dequant.py) (107 lines total),
[`nvfp4_lora/loader.py`](../../nvfp4_lora/loader.py)

- Extend [`dequantize_nvfp4_weight()` at dequant.py:40](../../nvfp4_lora/dequant.py#L40)
  to accept both key conventions. Add a new entry point — `dequantize_nvfp4_weight_ct(weight_packed, weight_scale, weight_global_scale, input_global_scale=None, …)` — that wraps the existing fp32 path. Or a `format: Literal["modelopt","compressed_tensors"]` arg on the existing function. Either way: the inner math (`_unpack_nibbles` + LUT + per-group + per-tensor scale) is unchanged per Agent A's finding that the packing order matches.
- In [`loader.py`](../../nvfp4_lora/loader.py): add key-sniff in
  [`list_quantized_modules()` at loader.py:38](../../nvfp4_lora/loader.py#L38)
  and the per-module record builder
  [`_collect_quantized_linear_records()` at loader.py:185](../../nvfp4_lora/loader.py#L185):
  if `{prefix}.weight_packed` is present in the safetensors index, this
  is compressed-tensors; if `{prefix}.weight` (uint8) is present, this
  is ModelOpt. The non-NVFP4 weight path
  [`load_non_nvfp4_weights()` at loader.py:594](../../nvfp4_lora/loader.py#L594)
  currently skips only `("weight", "weight_scale", "weight_scale_2", "input_scale", "bias")`.
  **Expand the CT skip set to the full artifact list**:
  `weight_packed`, `weight_scale`, `weight_global_scale`,
  `input_global_scale`. Reviewers verified KV-cache scales
  (`*.k_scale`, `*.v_scale`) are NOT a concern per
  `Qwen3.5 config.json` `"kv_cache_scheme": null` and Mistral
  `params.json` `"kv_cache_scheme": null`.

### 0.2 Loader generalization (rewritten post-consensus)

**File**: [`nvfp4_lora/loader.py`](../../nvfp4_lora/loader.py)

- Rename [`load_nemotron_with_nvfp4_lora()` at loader.py:696](../../nvfp4_lora/loader.py#L696)
  → `load_model_with_nvfp4_lora(model_family: Literal["nemotron_h","mistral3","qwen3_5_moe"], …)`. The
  existing call sites in `train/train_super_nvfp4.py` and
  `train/train_nano_nvfp4.py` get explicit `model_family="nemotron_h"`.
- **Replace [`make_key_translator()` at loader.py:56](../../nvfp4_lora/loader.py#L56)
  with an explicit per-family prefix-map architecture.** The prior
  `skip_prefixes`-tweak proposal was wrong (codex gpt-5.4 + Sonnet pass 2
  consensus): the existing single-level `named_children()` search at
  loader.py:88 fails for BOTH Qwen3.5
  (`Qwen3_5MoeForConditionalGeneration` owns `model`, which owns
  `language_model`, which owns `.layers`) AND Mistral3
  (`Mistral3ForConditionalGeneration` owns `model`, which owns
  `language_model`). The model-prefix-not-found branch raises
  `RuntimeError` before any silent corruption — but the failure happens
  100% of the time on either model.

  Replacement design:
  - Per-family prefix map: `{"nemotron_h": ("backbone.", "model."),
    "mistral3": ("model.language_model.",),
    "qwen3_5_moe": ("model.language_model.",)}` (verify exact prefixes
    via init_empty_weights() walk in 0.5).
  - Backup: if the per-family map doesn't match, fall through to a
    recursive child search that returns the full dotted path with
    `.layers`.
  - `skip_prefixes` retains its narrower meaning: keys to drop entirely
    (e.g. Nemotron's `mtp.`, multi-token predictor).
- Per-family `target_lora_suffixes` defaults:
  - **Nemotron-H**: `("q_proj","v_proj")` unchanged.
  - **Mistral4**: `("q_b_proj","kv_b_proj","o_proj")` —
    **Multi-head Latent Attention (MLA) names** (Sonnet round-1 caught
    this). `Mistral4Attention.__init__` uses `q_a_proj`/`q_b_proj`
    (low-rank-decomposed Q) and `kv_a_proj_with_mqa`/`kv_b_proj` (KV
    compression) because `q_lora_rank=1024` and `kv_lora_rank=256` in
    `params.json`. There is NO `self.q_proj` or `self.v_proj` —
    `("q_proj","v_proj","o_proj")` would silently match only `o_proj`
    on 36 layers (36 modules instead of intended 108). The chosen
    minimal v1 set `("q_b_proj","kv_b_proj","o_proj")` = 36 × 3 = 108
    modules, all MLA-significant. `q_a_proj` and `kv_a_proj_with_mqa`
    are low-rank down-projections; not LoRA-targeted in v1 (revisit in
    v1.1 if attention adaptation undersupplied).
  - **Qwen3.5**: `("q_proj","v_proj","o_proj")` — Qwen3.5 uses standard
    multi-head attention on its 12 full_attention layers (not MLA);
    standard names hold. 12 × 3 = 36 modules.

### 0.3 Guard Nemotron-specific patches (corrected)

**File**: [`train/train_super_nvfp4.py`](../../train/train_super_nvfp4.py)

The fix is an early `model_family != "nemotron_h"` guard at the top of
each. Functions to guard (state-corruption labels corrected per Sonnet
pass 2 + codex consensus — only `enable_sdpa_causal_no_mask` actually
mutates state before crash):

| Function | Line | Notes |
|---|---|---|
| `enable_sdpa_causal_no_mask()` | [550](../../train/train_super_nvfp4.py#L550) | **STATE-CORRUPTING** — overwrites `model.model._update_causal_mask` at L582 before the L660 attention-module-count crash. Highest-priority guard. |
| `enable_moe_sparse_no_one_hot()` | [707](../../train/train_super_nvfp4.py#L707) | Class-name guarded; safe crash |
| `enable_mamba_cached_multitoken()` | [722](../../train/train_super_nvfp4.py#L722) | `NemotronHMamba2Mixer` lookup; safe crash |
| `set_mamba_chunk_size()` | [840](../../train/train_super_nvfp4.py#L840) | Same class dep; safe crash |
| `enable_grouped_layer_checkpointing()` | [864](../../train/train_super_nvfp4.py#L864) | **NOT state-corrupting** (Sonnet pass 2 correction): the `AttributeError` on `remote_mod.NemotronHOutput` at L872 fires before the `base_model.forward` replacement at L963. Still needs a guard for clarity, lower urgency than `enable_sdpa_causal_no_mask`. |
| `run_routing_census()` | [967](../../train/train_super_nvfp4.py#L967) | `NemotronHTopkRouter` hooks; safe crash |
| `make_hybrid_cache()` | [1029](../../train/train_super_nvfp4.py#L1029) | Instantiates `NemotronHHybridDynamicCache` |
| `make_static_prefix_attention_cache()` | [1036](../../train/train_super_nvfp4.py#L1036) | Mamba state assumption |
| `make_cache_readonly_for_suffix()` | [1072](../../train/train_super_nvfp4.py#L1072) | Mamba state assumption |
| `prefill_prefix_cache()` | [1094](../../train/train_super_nvfp4.py#L1094) | cached_prefix mode driver |
| `cached_prefix_suffix_loss()` | [1135](../../train/train_super_nvfp4.py#L1135) | cached_prefix mode driver |
| `run_cached_prefix_compare()` | [1172](../../train/train_super_nvfp4.py#L1172) | cached_prefix eval helper |

`enable_sdpa_causal_no_mask` gets a hard early raise (state-corruption
risk). The rest get defensive class-name asserts with helpful error
messages.

### 0.4 Extract `nvfp4_lora/training_utils.py`

**New file** + extractions from
[`train/train_super_nvfp4.py`](../../train/train_super_nvfp4.py).

Move into the shared module (the existing implementations stay
unchanged; we just re-home them):

| Function | Current loc | Notes |
|---|---|---|
| `_CURRENT_PHASE` + `set_current_phase()` | [train_super_nvfp4.py:90-100](../../train/train_super_nvfp4.py#L90-L100) | Watchdog phase tagging — agnostic |
| `save_adapter()` | [train_super_nvfp4.py:220](../../train/train_super_nvfp4.py#L220) | PEFT adapter save — agnostic |
| `load_adapter_weights()` | [train_super_nvfp4.py:293](../../train/train_super_nvfp4.py#L293) | PEFT adapter load — agnostic |
| `mask_prompt_labels()` | [train_super_nvfp4.py:474](../../train/train_super_nvfp4.py#L474) | Label masking — agnostic, parameterize tokenizer |
| `build_optimizer()` | [train_super_nvfp4.py:521](../../train/train_super_nvfp4.py#L521) | Optimizer dispatch — agnostic post-extraction. **Note (Sonnet pass-1 carry-over)**: requires promoting `LR` module-level constant at [train_super_nvfp4.py:523](../../train/train_super_nvfp4.py#L523) to an explicit `lr: float` function parameter. Per-trainer scripts then pass their own LR. |
| `_lm_head_ce()` dispatcher | [train_super_nvfp4.py:39-58](../../train/train_super_nvfp4.py#L39-L58) | Per Pre-M1a — agnostic post-refactor |

The big training loop in `main()` at
[train_super_nvfp4.py:1332](../../train/train_super_nvfp4.py#L1332) — and
specifically the per-step phase tagging at L1724/1734/1742/1748/1757 —
stays Nemotron-specific (it includes the cached-prefix branches). A
slim model-agnostic equivalent will be re-implemented in each new trainer
script.

The MoE sparse-no-one-hot implementation at
[`_moe_sparse_no_one_hot_impl()` line 674](../../train/train_super_nvfp4.py#L674)
is generic (per Agent C) and worth lifting to `training_utils.py` for
reuse by Mistral and Qwen3.5; only its bind-helper
[`enable_moe_sparse_no_one_hot()` line 707](../../train/train_super_nvfp4.py#L707)
is Nemotron-class-name-guarded and can stay.

### 0.6 NEW — Fused-CT-expert loader + frozen forward path (~5-7 days)

**The four-reviewer consensus surfaced this as a BLOCKER that both
Sonnet passes missed.** Both Mistral4 (`Mistral4NaiveMoe`) and Qwen3.5
(`Qwen3_5MoeSparseMoeBlock`) declare the routed-expert weights as
**fused 3D `nn.Parameter`** (e.g.
`gate_up_proj: (num_experts, 2*intermediate, hidden)`), while
on-disk safetensors expose them as **per-expert compressed-tensors
keys** (e.g. `model.layers.N.mlp.experts.E.gate_proj.weight_packed`).

The current loader walks `nn.Linear` instances at
[loader.py:185](../../nvfp4_lora/loader.py#L185)
(`_collect_quantized_linear_records()` filters
`if not isinstance(module, nn.Linear)`) and at
[loader.py:329](../../nvfp4_lora/loader.py#L329)
(`replace_nvfp4_modules()` same filter). It does NOT have a path to:
1. Discover the fused 3D parameters (they're `nn.Parameter`, not
   `nn.Linear`)
2. Assemble the fused 3D in-memory tensor from per-expert CT safetensors
   keys
3. Forward the routed batched matmul over the (potentially
   quantized) fused storage

**This blocks even attention-only LoRA v1** because the base MoE forward
will hit meta tensors and crash (or worse, silently misbehave).
Attention-only LoRA does NOT solve the loading-side problem.

Two implementation options:

**Option P-A — Custom `NVFP4Experts3D` quantized container**:
- New `nn.Module` storing
  `weight_packed: (num_experts, out, in//2)` uint8,
  `weight_scale: (num_experts, out, in//group_size)` fp8_e4m3fn,
  `weight_global_scale: (num_experts,)` fp32.
- Override the model-family MoE block's forward to perform
  per-active-expert dequant + matmul.
- Most invasive — needs per-model-family forward patches.
- **Workspace shape (codex round-1 correction)**: a full
  `(num_experts, out, in)` dequant workspace would be multi-GB per
  layer (Qwen3.5 single layer fused `gate_up_proj` scratch at bf16:
  256 × 2048 × 3072 × 2 bytes ≈ 3 GB; Mistral4: 128 × 4096 × 4096 × 2
  bytes ≈ 4 GB) — unsafe and recreates the bf16 fallback we're
  avoiding. **Bounded design**: per-active-expert scratch
  `(max_active_experts_per_batch, out, in)` where
  `max_active_experts_per_batch ≤ k × batch_size × seq_len /
  num_experts × safety_factor`. For Qwen3.5 (k=8) at batch=1 seq=2048
  this caps the scratch at ~64-128 experts worth. Alternative: tiled
  matmul chunks that bound the dequant footprint independent of
  `num_experts`. Final choice deferred to 0.6.b sub-task.
- Memory at full retention: experts stay NVFP4 (~0.5 bytes/param),
  matches existing scale.

**Gate/up/down assembly semantics (codex round-1 finding)**:
- Both Mistral4 (`Mistral4NaiveMoe.gate_up_proj: (E, 2*I, H)`) and
  Qwen3.5 (`Qwen3_5MoeSparseMoeBlock.gate_up_proj: (E, 2*I, H)`) fuse
  `gate_proj` and `up_proj` along the output dimension. **Order
  matters — a swap produces non-NaN-but-wrong outputs that pass the
  forward acceptance gate.** Each model's forward dictates which half
  is `gate` and which is `up`; assembly from per-expert
  `experts.E.gate_proj.weight_packed` + `experts.E.up_proj.weight_packed`
  safetensors keys must match this ordering.
- 0.6.b' (NEW sub-stage): verify per-family fused tensor ordering from
  modeling source. Add a small parity test: for one expert in one
  layer, materialize the fused 3D tensor from per-expert CT keys,
  dequant to bf16, compare against (a) the published HF model's
  loaded value if available via compressed-tensors at the Linear
  level, or (b) per-expert dequant of the same source compared to
  a hand-rolled fused-tensor dequant under both possible orderings.
  Acceptance: bit-identical match.
- `down_proj: (E, H, I)` — single-fused-axis tensor, no ordering
  ambiguity.

**Option P-B — `compressed-tensors` runtime dequant integration**:
- Check whether `compressed-tensors` library provides a fused-3D
  expert runtime module (`Linear`-equivalent for fused MoE storage).
- If yes: integrate, lower implementation cost.
- If no: fall through to P-A.
- Adds runtime dependency on `compressed-tensors` lib (currently NOT
  in `qwen-peft` venv per Agent A research).

**Phase 0.6 sub-stages**:
- 0.6.a — ~~investigate `compressed-tensors` v0.X API surface for
  fused-3D-quantized container support~~ **COMPLETED** (execution log):
  `compressed_tensors 0.15.0.1` in qwen-serve venv. `CompressedLinear.from_linear()`
  raises `"CompressedLinear is no longer supported"`. No MoE/fused-3D
  container exists. `NVFP4PackedCompressor.can_compress()` requires
  `module_type == torch.nn.Linear`. **Option P-B is unavailable**; Option
  P-A is the committed path. Reusable primitives confirmed:
  `unpack_fp4_from_uint8` (compressed_tensors path, high-nibble-first
  ordering — different from our existing `_unpack_nibbles`), `dequantize`
  (full dequant math). vLLM's `flashinfer_fp4_moe.py` confirms the
  fused-3D pattern (`w13_weight: (num_experts, 2*I, H)`, `w2_weight:
  (num_experts, H, I)`) and the gate/up packing-order question is real
  (`reorder_w1w3_to_w3w1` helper exists).
- 0.6.b — build `NVFP4Experts3D` (Option P-A). Use our existing
  `dequantize_nvfp4_weight` (extended for 3D batched) as the dequant
  primitive — same code path as the rest of the loader.
- 0.6.c — patch loader: new
  `_collect_fused_expert_records()` walks model for fused-3D MoE
  blocks (per-family class registry), assembles the quantized 3D
  storage from per-expert CT safetensors keys.
- 0.6.d — verify with 1-token forward through one
  Mistral/Qwen3.5 MoE block. Non-NaN output mandatory.

**Risk**: if BOTH Option P-A and Option P-B are too invasive, we may
need to fall back to a fully-dequanted bf16 storage path. **Corrected
memory math (Sonnet round-1)** — prior v2 counted only
`gate_up_proj` and stated 154 GB; the full per-model number
including `down_proj` is **~232 GB each**:

- Qwen3.5 (256 experts, hidden=3072, moe_intermediate=1024, 48 expert
  layers):
  - `gate_up_proj`: 256 × (2×1024) × 3072 × 2 bytes × 48 layers
    = ~154.6 GB
  - `down_proj`: 256 × 3072 × 1024 × 2 bytes × 48 layers = ~77.3 GB
  - **Total: ~232 GB**
- Mistral4 (128 experts, hidden=4096, expert_hidden=2048, 36 layers):
  - `gate_up_proj`: 128 × (2×2048) × 4096 × 2 bytes × 36 layers
    = ~154.6 GB
  - `down_proj`: 128 × 4096 × 2048 × 2 bytes × 36 layers = ~77.3 GB
  - **Total: ~232 GB**

**Doesn't fit 130 GB UMA** — the gap is even larger than the v2 math
suggested. Quantized storage is mandatory; the work in 0.6 has no easy
escape hatch.

**Acceptance gates (codex round-1 corrected — split into required and
advisory)**:

REQUIRED (failure blocks Phase 0):
- No meta tensors remain on any routed-expert parameter after Phase 0.6
  loader runs (verified via `[p for n,p in model.named_parameters() if p.is_meta]`).
- 1-token forward through one fully-loaded MoE block for each model
  family produces non-NaN, non-Inf logits.
- Correct target counts per Phase 0.5 gate 3 (this is the gate that
  catches "loaded but wrong" cases).
- Peak memory during the 1-token forward stays below 130 GB UMA with
  a 10 GB headroom.
- 0.6.b' parity test (bit-identical fused-tensor reconstruction
  against per-expert dequant ground truth) passes.

ADVISORY (recorded but does not block Phase 0):
- Tokens/sec on 1-token forward, compared against an explicitly-defined
  baseline (top-k bf16 synthetic with dequanted weights, single MoE
  block). Initial target: within 3× of baseline; tightened in
  follow-on work.

### 0.5 Smoke tests + acceptance gates (strengthened post-consensus)

**Run**: existing CPU smoke tests in [`smoke_tests/`](../../smoke_tests/) —
specifically `test_loss_parity_pre_m1a.py` and
`test_dequant_workspace_pre_m1b.py` (both shipping in Pre-M1, total 22
tests).

Plus **new per-family integration smoke** that exercises
`load_model_with_nvfp4_lora(model_family="...", …)` against each model
directory and asserts (codex consensus — prior "loader parses configs
without error" was too weak):

1. **Module tree builds via `init_empty_weights()`** for each family.
2. **Prefix translator resolves** — no `RuntimeError` from the
   replacement per-family-prefix-map logic.
3. **Expected quantized-module count matches per family** (verified
   against the safetensors index NVFP4-key set):
   - Nemotron-Nano: existing count unchanged
   - **Mistral4**: attention modules use MLA names — count
     `36 × len(target_lora_suffixes)` for attention-only v1 (108 with
     `("q_b_proj","kv_b_proj","o_proj")`). Shared expert is
     `Mistral4MLP.{gate_proj,up_proj,down_proj}` = 36 × 3 = 108 modules
     (not targeted in v1; counted separately). Routed-expert fused-3D
     count tracked separately (handled by Phase 0.6).
   - **Qwen3.5**: 12 full_attention layers × 3 standard attention proj
     = 36 attention NVFP4 records targeted in v1; routed-expert
     fused-3D count tracked separately (handled by Phase 0.6).
4. **Expected LoRA-target count matches** the resolved
   `target_lora_suffixes` × layers — fails fast if the suffix list hits
   only shared expert on Mistral (the false-positive trap).
5. **No meta tensors remain** after `load_model_with_nvfp4_lora()`
   completes (catches Phase 0.6 fused-expert loader regressions).
6. **1-token forward** per family produces non-NaN logits. CPU is
   sufficient for the module-tree + prefix tests; GPU needed for the
   1-token forward.

**Phase 0 acceptance**: all 6 gates pass for all three families
(nemotron_h, mistral3, qwen3_5_moe). Failing any gate is a Phase 0
blocker.

## Phase 1A: Mistral track (~1-2 weeks after Phase 0)

### A1: Download axolotl BF16 (~2-4 hours wall clock, background)

```bash
huggingface-cli download axolotl-ai-co/Mistral-Small-4-119B-2603-BF16 \
  --local-dir /home/veritan-spark-01/Veritan/Models/Mistral-Small-4-119B-2603-BF16-HF
```

Disk check first: needs ~240 GB free at `/home/veritan-spark-01/Veritan/Models/`.

### A2: Quantize BF16 → NVFP4A16 in-house (~30-60 min)

Use the recipe in
[`docs/plans/mistral_bf16_to_nvfp4_quant.md`](mistral_bf16_to_nvfp4_quant.md)
(I wrote it in parallel with this plan). **One revision from Agent B's
findings**: switch from generic `scheme="NVFP4"` to **NVFP4A16**
(data-free, weight-only). Sequential pipeline still required; working
set ~3-5 GB.

Output: HF-format NVFP4 compressed-tensors checkpoint at
`/home/veritan-spark-01/Veritan/Models/Mistral-Small-4-119B-2603-NVFP4-HF/`.
**Uses the same compressed-tensors NVFP4 storage convention as Qwen3.5-122B-A10B-NVFP4** (per gpt-5.4 walk-back from "byte-for-byte schema-identical" — see strategic table). Specifically:
same `quant_method: compressed-tensors`, `format: nvfp4-pack-quantized`,
`scale_dtype: torch.float8_e4m3fn`, `group_size: 16`.

### A3: Sanity-load + serve (~30 min)

Modify [`serve/run_mistral_small4_nvfp4.sh`](../../serve/run_mistral_small4_nvfp4.sh)
to point at the new HF-format quant dir. Drop `--config-format mistral`,
`--tokenizer-mode mistral`, `--load-format mistral` (the existing flags
that consume Mistral consolidated format). The HF-format quant uses
standard vLLM defaults. Keep `--enforce-eager` and `--moe-backend cutlass`.

Generate from a fixed prompt, compare qualitatively to BF16 baseline.

### A4: Write `train/train_mistral_nvfp4.py` (~3-5 days)

**New file**, modeled on
[`train/train_nano_nvfp4.py`](../../train/train_nano_nvfp4.py) (existing
NVFP4 LoRA pattern) rather than the much larger
[`train/train_super_nvfp4.py`](../../train/train_super_nvfp4.py) which
has all the Mamba/cached-prefix machinery. Uses the Phase 0 shared
`nvfp4_lora/training_utils.py`.

- **LoRA targets: attention-only MLA projections for v1** —
  `("q_b_proj","kv_b_proj","o_proj")` on all 36 layers. **NOT**
  `("q_proj","v_proj","o_proj")` — Mistral4 uses Multi-head Latent
  Attention (Sonnet round-1 caught this). Verified: `Mistral4Attention`
  declares `q_a_proj`/`q_b_proj` (Q low-rank decomposition with
  `q_lora_rank=1024`), `kv_a_proj_with_mqa`/`kv_b_proj` (KV
  compression with `kv_lora_rank=256`), `o_proj`. Standard `q_proj`
  and `v_proj` do not exist. Targeting them would silently match only
  `o_proj` (36 modules) — would FAIL the A5 target-count gate but only
  because the count is wrong; if anyone updated the count to 36 the
  smoke would pass on a tiny adaptation surface.

  The prior "up_proj+down_proj on routed experts" recommendation
  remains retracted: Mistral4 routed experts are fused-3D
  `nn.Parameter`, and `("up_proj","down_proj")` would silently match
  only the shared `Mistral4MLP` expert while missing all 128 routed
  experts — the **shared-expert false-positive trap**. Real
  routed-expert LoRA requires Phase 0.6 + a v2 fused-3D LoRA wrapper.

  v1.1 / v2 backlog: optionally extend MLA targets to include
  `q_a_proj` and `kv_a_proj_with_mqa` (the low-rank down-projections);
  then layer-1 routed-expert LoRA via fused-3D `NVFP4LoRAExperts3D`.
- Skip vision encoder entirely: never pass `pixel_values`; freeze the
  vision branch at load. **Correct attribute paths** (per Sonnet pass 2
  + codex gpt-5.4 verification): `model.model.vision_tower` and
  `model.model.multi_modal_projector` (Mistral3 wraps as
  `model.model.vision_tower`, NOT `model.vision_tower`). Specifically:
  ```python
  for p in model.model.vision_tower.parameters():
      p.requires_grad = False
  for p in model.model.multi_modal_projector.parameters():
      p.requires_grad = False
  ```
- **No Mamba patches, no cached-prefix-suffix mode** (Mistral is
  pure-attention; the Phase 0 guards on those Nemotron-only helpers
  prevent accidental calls).
- Standard `loss.backward()` loop using the
  [`_lm_head_ce()` dispatcher (Pre-M1a)](../../nvfp4_lora/loss.py) and
  the dequant workspace pool (Pre-M1b).
- max_seq_len defaults to 4096 (the same as the current Mistral serve
  script's setting).
- Adapter saved via the shared `save_adapter()` to
  `/home/veritan-spark-01/Veritan/Sandbox/adapters/mistral_small4_<dataset>_<config>`.

### A5: First LoRA smoke (~1 day, **tightened acceptance**)

5-step LoRA training on a tiny ICH subset. The consensus reviewers
flagged that the prior "loss goes down" gate is **too weak** — it would
false-pass on the shared-expert trap because shared-expert LoRA can
produce a real loss decrease while routed-expert capacity is
untouched. Tightened gates:

- **Target-count assertion**: exactly the intended MLA attention
  modules are LoRA-wrapped. Count = 36 layers ×
  `len(("q_b_proj","kv_b_proj","o_proj"))` = 36 × 3 = 108 LoRA
  modules. Verified at run time against `init_empty_weights()` module
  inventory before training starts. NOT 108 + N_shared_expert_proj.
  Failing this count means the suffix list either has wrong MLA names
  or silently captured shared experts (the `Mistral4MLP.up_proj`
  collision risk if anyone reverts to `up_proj`).
- **Loss decrease vs frozen-base control**: run an identical 5-step
  loop with all LoRA params frozen at init. Adapter-on loss curve must
  diverge from frozen-control by a margin > step-noise. Distinguishes
  real adaptation from initialization noise.
- **LoRA-A/B param movement** (codex round-1 correction — replaces
  prior overly-strict "monotonic norm growth" gate that can fail
  valid 5-step training):
  - REQUIRED: nonzero gradients on LoRA params after step 1 (catches
    "optimizer not seeing the modules" — the actual failure we care
    about).
  - REQUIRED: at least one optimizer step changes adapter tensors
    above a small tolerance (e.g. `||p - p_init||_F > 1e-5`).
  - REQUIRED: aggregate `sum_M ||W_M_after - W_M_init||_F` over all
    LoRA modules increases by step 5 vs step 1.
  - ADVISORY/diagnostic: per-module `||A||_F` and `||B||_F` traces —
    no monotonicity gate; logged for debugging only.
- **Adapter-on/off eval delta** on a tiny eval set: compute log-likelihood
  under adapter-on vs adapter-off (set `lora_scale=0` for off). Delta
  must be measurable; sign and magnitude matter less for the smoke than
  the fact that the adapter produces a different output distribution.
- **Vision encoder zero-grad check**: assert
  `all(p.grad is None for p in model.model.vision_tower.parameters())`
  after step 1.
- **Routed-expert no-touch check**: assert routed-expert parameters
  (the fused-3D ones from Phase 0.6) have `.grad is None`. Catches
  any accidental gradient flow.
- No CUDA OOM at s=4096 batch=1.
- Adapter saves via `save_adapter()` and reloads via
  `load_adapter_weights()` with bit-identical LoRA params.

## Phase 1B: Qwen3.5 track (~2-3 weeks after Phase 0)

Can run in parallel with Phase 1A if you have the bandwidth — but the
Qwen3.5-specific 3D expert decision is the gating one.

Model already downloaded at
`/home/veritan-spark-01/Veritan/Models/Qwen3.5-122B-A10B-NVFP4/` —
2 HF safetensors shards + `model.safetensors.index.json` (confirmed via
the earlier inspection). Quant config: `quant_method: compressed-tensors`,
`model_type: qwen3_5_moe`, architectures `['Qwen3_5MoeForConditionalGeneration']`.

### B0: Decide on expert-LoRA scope (1-hour decision, but needed before B3)

Agent A confirmed the routed-expert weights are stored as a single fused
3D parameter `gate_up_proj: (256, 2*1024, 3072)` on
`Qwen3_5MoeSparseMoeBlock`. `named_modules()` walks for `nn.Linear` —
which is what
[`_collect_quantized_linear_records()` at loader.py:185](../../nvfp4_lora/loader.py#L185)
does today — silently skips every routed-expert weight. The shared
expert (1 per block) IS a proper `nn.Linear` and would be picked up.

| Option | Trainable params (r=8) | Effort | Files touched |
|---|---|---|---|
| **B0.a** Attention-only (`q_proj`, `v_proj`, `o_proj` on 12 full_attention layers; **shared_expert.* explicitly NOT included in v1**) | ~5M | **Minimal AFTER Phase 0.6 fused-CT-expert base-loader lands** (codex round-1 correction: existing loader CAN'T load routed-expert base weights, so even attention-only LoRA needs 0.6 first). LoRA wrapping path itself is the existing `NVFP4LoRALinear`. | New `train/train_qwen35_nvfp4.py` only |
| **B0.b** Custom 3D-aware NVFP4LoRA wrapper for fused expert weights | ~200M+ | New `NVFP4LoRAExperts3D` class; 1-2 weeks careful work | New module in `nvfp4_lora/`; loader extension; integration tests |
| **B0.c** Unfuse `gate_up_proj` 3D parameter into 256 distinct nn.Linear modules at load time | ~200M+ | Loader hook to rebuild the MoE block; doubles allocation count | `nvfp4_lora/loader.py` (+~150 lines); risk of memory regression |

**Decision baked into v3: B0.a.** Strictly attention-only LoRA for v1,
explicitly excluding shared-expert wrapping. The B0 decision is no
longer a v1 fork point — it's been folded into the strategic table.
B0.c (unfuse on load) remains as a v2 backlog item but is not pursued
in this scope.

Note: B0.a depends on Phase 0.6 (the fused-CT-expert base loader)
landing first — the existing `nn.Linear`-only loader walk cannot
materialize the routed-expert base weights, so even attention-only
LoRA's base forward will hit meta tensors without 0.6.

### B1: Verify Qwen3.5 load through HF transformers (~half day)

Confirm `AutoModelForCausalLM.from_pretrained` works through the Phase 0
generalized loader. Test on CPU first via `init_empty_weights()` to
surface module-tree issues before paying GPU init cost. Specifically:

- Module class names: confirm `Qwen3_5MoeAttention` and
  `Qwen3_5MoeSparseMoeBlock` are present at the expected layer indices.
- `q_proj`, `k_proj`, `v_proj`, `o_proj` exist as `nn.Linear` only on
  the 12 `full_attention` layers (layers 3, 7, 11, 15, ..., 47 per
  Agent A); the other 36 are `linear_attention` with non-quantized SSM
  params.
- Safetensors keys: spot-check
  `model.layers.3.self_attn.q_proj.weight_packed` exists; spot-check
  `model.layers.0.self_attn` is the linear_attention block (no q/k/v
  Linears).
- `_collect_quantized_linear_records()` returns the expected set (12
  layers × 4 attention projections = 48 NVFP4 Linears) when only
  attention LoRA is targeted. Add 48 shared-expert weights if shared
  experts are also targeted.

### B2: Decide on hybrid arch training mode (~1 day analysis)

The 36 linear_attention layers contain SSM state (GatedDeltaNet, per
Agent A — NOT vanilla Mamba2). Two options:

- **B2.a** Standard full-sequence training: HF's standard
  `model(input_ids).loss`. Simple, no Nemotron patches, GatedDeltaNet
  state evolves layer-by-layer per token like any RNN. Suffices for
  ICH inputs at ≤4k tokens.
- **B2.b** Adapt the Nemotron cached_prefix_suffix machinery (lives in
  [`train_super_nvfp4.py:1094-1230` region](../../train/train_super_nvfp4.py#L1094))
  to Qwen3.5's GatedDeltaNet. Different state shape, different
  chunk-scan kernel than the Mamba2 work. ~3-4 weeks if pursued.

**Recommend B2.a for v1.** Revisit B2.b only if Qwen3.5 ICH training
needs long context AND hits a descriptor-cliff analogous to the
Nemotron-Super one at s>2048. Pre-M1c work would partially carry over
in that case (workspace-arg patching pattern), though the specific
kernel is different.

### B3: Write `train/train_qwen35_nvfp4.py` (~5-7 days)

**New file**, modeled on
[`train/train_nano_nvfp4.py`](../../train/train_nano_nvfp4.py) similar to
the Mistral track. Standard HF trainer with the Phase 0 shared utils.
LoRA targeting per B0 decision; no Mamba/cached-prefix patches (per B2.a).

Key wiring:
- `load_model_with_nvfp4_lora(model_family="qwen3_5_moe", target_lora_suffixes=("q_proj","v_proj","o_proj"), …)` — note: `o_proj` included to match Phase 0.2 declaration; missing it would produce 24 LoRA modules instead of the declared 36 and fail B4 gate (codex/Sonnet round-2 catch)
- Standard `model(input_ids).loss` (no cached-prefix machinery)
- Watchdog phases from shared `set_current_phase()`
- Chunked CE loss from `nvfp4_lora.loss._lm_head_ce`
- max_seq_len 4096 to match the existing Qwen3.5 serve config

### B4: First LoRA smoke (~1 day, **tightened acceptance**)

Same tightened criteria as A5 (target-count assertion, frozen-base
control, LoRA-A/B norm growth, on/off eval delta, routed-expert
no-touch check, OOM, adapter round-trip). Qwen3.5-specific assertions
on top:

- **Full_attention-layers-only check**: assert LoRA-wrapped modules
  are exclusively on layers 3, 7, 11, ..., 47 (verified via
  `config.text_config.layer_types` + `full_attention_interval: 4` —
  codex gpt-5.4 corrected this attribute path; the prior plan claimed
  a non-existent `attention_layer_indices` field).
- **GatedDeltaNet SSM params frozen**: assert
  `[p for p in model.named_parameters() if "linear_attn" in name]`
  all have `requires_grad=False`. Catches accidental SSM training.
- **Routed-expert no-touch** (same as A5): the 256 routed experts'
  fused-3D params have `.grad is None`.

## Phase 2: Run ICH fine-tunes (~ongoing, runs per model)

Per the user's existing ICH workflow: full ICH dataset, target adapter
sizing per model, periodic checkpointing, eval against held-out set.
Same machinery for all three model families now.

## Honest risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| HF transformers version doesn't support `mistral3` AND `qwen3_5_moe` AND `nemotron_h` simultaneously | Med | High — breaks Nemotron regression | Pin during Phase 0; if conflict surfaces, sequence by venv (mistral-quant venv separate from training venv) |
| compressed-tensors `input_global_scale` key surfaces as warnings or breaks the non-NVFP4 weight load path | Med | Med | Agent A flagged this; Phase 0.1 includes explicit handling in non-NVFP4 weight loader |
| Mistral data-free quantization quality is poor enough to undermine LoRA-FT quality | Low-Med | Med | Stage A3 sanity inference catches catastrophic loss; if marginal, fall back to GPTQ (~6-12h on Spark) |
| Qwen3.5 attention-only LoRA capacity is insufficient for ICH adaptation | Med | Low | B0.a is "ship something"; can escalate to B0.c if eval results disappoint |
| Qwen3.5 Mamba layers create descriptor-cliff at training time | Med | Med | Confirmed concern; if hit, Pre-M1c work becomes useful again. Don't preemptively port. |
| Disk space exhaustion (238 GB BF16 + 70 GB NVFP4 + existing models) | Med | Med | Phase 0 check; consider deleting consolidated NVFP4 Mistral (~60 GB recoverable) once HF quant verifies |
| Mistral vision encoder eats GPU memory at LoRA training time even when unused | Med | Med | A4 freezes ViT (`requires_grad=False`) but the weights stay resident; the quant plan also excludes vision from quantization so the vision branch is BF16-resident. Implementation watchpoint (codex round-3): if memory pressure surfaces during training, surgically remove (`del model.model.vision_tower; del model.model.multi_modal_projector`) or offload to CPU at load time before training starts. |

## Decisions for you to make right now

These each block specific phases. Flagging together so you can answer in
one pass:

1. **Order: Mistral first, parallel, or Qwen3.5 first?**
   I recommend Mistral first. Cleaner path to a working LoRA fine-tune
   in ~2 weeks; Qwen3.5 has the expert-LoRA decision blocking serious work.

2. **Qwen3.5 expert-LoRA: B0.a (attention-only, ship fast) or B0.c
   (unfuse + full expert LoRA, slower)?**
   I recommend B0.a for v1, B0.c as a follow-up if the attention-only
   capacity is inadequate.

3. **Mistral quant: NVFP4A16 (data-free, fast, runs on Spark) or NVFP4 W4A4
   (calibrated, better quality, needs cloud OR 6-12h Spark GPTQ)?**
   I recommend NVFP4A16. We can re-quantize later if quality is bad.

4. **Nemotron-Super Pre-M1c/d: shelve or cancel?**
   I recommend shelve. Mamba kernel patches may be useful for Qwen3.5
   long-context work later; the analysis effort is already done so
   shelving is cheap.

5. **Mistral training: text-only (skip pixel_values, freeze ViT) or
   multimodal (include image inputs in the dataset)?**
   I recommend text-only for v1. The ICH dataset is text. Vision support
   is a separate feature.

6. **Trainer venv strategy: one shared venv for all three models, or
   per-model venvs?**
   I recommend trying one shared venv first; only split if transformers
   version pinning forces it.

## Time and effort summary (v2 — post-consensus)

| Phase | Duration | Blocks |
|---|---|---|
| Phase 0 (now includes 0.6 fused-CT-expert loader) | **~2-3 weeks** | Everything below |
| Phase 1A (Mistral) | 1-2 weeks | Phase 0 |
| Phase 1B (Qwen3.5) | 1-2 weeks (no B0 decision needed; attention-only v1 baked in) | Phase 0 |
| Phase 2 (ICH FTs) | per-model ongoing | Phase 1A or 1B |
| **Total to first working FT** | **4-5 weeks** | |
| **Total to both working** | **5-7 weeks** | |

Phase 0 grew from "~1 week" to "~2-3 weeks" because the new 0.6 fused-CT
work (5-7 days) is a hard blocker. Phase 1B shrank slightly because the
B0 decision (attention-only vs fused-3D-LoRA) is now decided in advance
(attention-only v1 + post-v1 fused-3D LoRA backlog item).

Excludes wall-clock waiting (BF16 download ~3 hours, quantization ~1 hour,
training runs hours-to-days per epoch).

## Out of scope (explicitly)

- Multimodal/vision LoRA for Mistral
- Long-context training (>4k) for Mistral or Qwen3.5
- Full W4A4 quantization for Mistral (cloud-only)
- DoRA on Qwen3.5 (in-proj-qkv NaN risk; user memory flags this)
- Production serving optimization (separate from training work)
- Multi-Spark distributed training
- BF16 master-weights training (the third option from the original
  Nemotron-Super descriptor-cliff plan)
