# Serve path for the two ICH v3.5 LoRA targets (Qwen3.5-122B RH-NVFP4, Mistral-Small-4-119B RH-NVFP4-HF)

Status: feasibility memo, written 2026-06-10 while the Qwen training run still owns
the GPU. Everything below is from source reading and existing logs only; nothing
was executed against the GPU. Scripts referenced here must not run until the
training run completes (expected around 01:30).

vLLM under investigation: `vllm 0.22.1` in `/home/veritan-spark-01/Veritan/.venvs/qwen-serve`
(dist-info dated Jun 8; the May 31 serve logs in `logs/` were produced by 0.21.0,
so 0.22.1 has not yet served anything on this box). All `vllm/...` citations below
are relative to
`/home/veritan-spark-01/Veritan/.venvs/qwen-serve/lib/python3.12/site-packages/vllm/`.

## The three decisive facts

1. **Qwen3_5 MoE is natively registered in vLLM 0.22.1.**
   `Qwen3_5MoeForConditionalGeneration` maps to `qwen3_5.py`
   (`model_executor/models/registry.py:562-564`). The model class chain declares
   `SupportsLoRA` (`model_executor/models/qwen3_5.py:433-438`) and the
   GatedDeltaNet hybrid layers are implemented
   (`model_executor/models/qwen3_5.py:43-44` imports
   `QwenGatedDeltaNetAttention` from `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`;
   the decoder layer wires `linear_attn` at `qwen3_5.py:138`).

2. **compressed-tensors NVFP4 loads and runs on sm_121, but only without LoRA on the MoE.**
   The dense W4A4 scheme is `CompressedTensorsW4A4Fp4`
   (`model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py:25`),
   which picks a GEMM from the NVFP4 dense kernel list
   (`model_executor/kernels/linear/__init__.py:378-392`: FlashInfer-CUTLASS,
   CUTLASS, Marlin, ... Emulation). The MoE side routes through the NVFP4 MoE
   oracle (`model_executor/layers/fused_moe/oracle/nvfp4.py`), same backend set
   the Nemotron investigation mapped out (`serve/README.md`,
   `docs/TROUBLESHOOTING.md`). `CutlassExpertsFp4` accepts device capability
   family 120, which sm_121 satisfies
   (`model_executor/layers/fused_moe/experts/cutlass_moe.py:681-686`).
   Proof it works end to end on this box: `logs/qwen35_122b_nvfp4_serve.log`
   (May 31, vLLM 0.21) shows the identical checkpoint layout (the
   `Qwen3.5-122B-A10B-NVFP4` dir is byte-for-byte the same index/total_size as
   `RedHatAI-Qwen3.5-122B-A10B-NVFP4`: 76,419,752,928 bytes, 149,100 tensors)
   serving via `VLLM_CUTLASS` at 11.8-14.5 tok/s generation.

3. **Runtime LoRA over quantized attention linears is supported in principle, but the MoE oracle kills it for this model, and MLA absorption kills it for Mistral.**
   - Dense linears: the LoRA wrapper calls the base layer's quant method and adds
     the LoRA product on the side
     (`lora/layers/base_linear.py:198`:
     `output = self.base_layer.quant_method.apply(self.base_layer, x, bias)` then
     `_apply_lora_to_output`). So LoRA over compressed-tensors NVFP4 qkv/o
     projections is architecturally fine; no dequant of the base is needed.
   - MoE layers: with `--enable-lora`, every FusedMoE layer gets
     `is_lora_enabled=True` purely from `vllm_config.lora_config is not None`
     (`model_executor/layers/fused_moe/layer.py:334`). The kernel oracle then
     rejects any expert kernel without LoRA support
     (`model_executor/layers/fused_moe/modular_kernel.py:580-581`). Default
     `supports_lora()` is False (`modular_kernel.py:746-752`); the only NVFP4
     expert kernels that mix in `LoRAExpertsMixin` are `TritonExperts`,
     `UnfusedOAITritonExperts`, and `MarlinExperts`
     (`model_executor/layers/fused_moe/experts/marlin_moe.py:693` etc.).
     `CutlassExpertsFp4` does not, and notably the EMULATION kernel
     (`experts/nvfp4_emulation_moe.py`) does not either, so the 0.21 era
     "EMULATION+LoRA crashes in the Triton kernel" failure mode has become a
     clean oracle rejection in 0.22.1. Forcing `--moe-backend cutlass` with LoRA
     raises `ValueError` at startup (`oracle/nvfp4.py:219-234` via the explicit
     backend branch at `oracle/nvfp4.py:256`).
     The new deployment-time filter `--lora-target-modules`
     (`config/lora.py:48-51`, CLI at `engine/arg_utils.py:1294`) restricts which
     modules get *wrapped* (`lora/model_manager.py:389` via
     `is_in_target_modules`, `lora/utils.py:260`), but it does NOT refine the
     global `is_lora_enabled` flag the MoE oracle sees, so it cannot rescue the
     CUTLASS MoE path.
     Net for Qwen: `--enable-lora` forces the MoE onto MARLIN, the one backend
     the Nemotron-Super investigation showed cannot fit a 120B-class repack on
     this box (`serve/README.md:22-24`, `serve/README.md:40-47`,
     `docs/TROUBLESHOOTING.md`). Path A is blocked for the 122B.
   - MLA (Mistral): vLLM 0.22.1's LoRA manager does handle the aliasing of MLA
     projection modules (the same module object is registered both on
     `self_attn` and inside the MLA wrapper); it wraps once and rewires every
     alias to the wrapper (`lora/model_manager.py:385-431`,
     `named_modules(remove_duplicate=False)` plus `wrapped_by_id`). So LoRA on
     `q_b_proj` and `o_proj` would actually be in the forward path
     (`model_executor/layers/mla.py:126-180`). But `kv_b_proj` is weight-absorbed
     at load time into BF16 `W_UK_T`/`W_UV` copies
     (`model_executor/layers/attention/mla_attention.py:835-863`), and the decode
     path multiplies against those absorbed copies
     (`mla_attention.py:742-751`), while prefill paths call the `kv_b_proj`
     module (`mla_attention.py:639`). A `kv_b_proj` LoRA therefore applies during
     prefill and silently does not apply during decode: inconsistent K/V between
     the two phases. Our Mistral adapter targets `kv_b_proj`, so request-time
     LoRA in vLLM is unsound for it regardless of anything else.

## An extra, Mistral-specific blocker: the RH HF checkpoint cannot load in vLLM 0.22.1 at all

- The checkpoint declares top-level `Mistral3ForConditionalGeneration`
  (registered, `registry.py:491-493`) but its `text_config` is `model_type:
  "mistral4"` (MLA + 128-expert MoE). vLLM has **no** mistral4 text backbone:
  `grep -r mistral4` over the vLLM tree returns nothing. The mistral-format
  Large-3 class exists (`MistralLarge3ForCausalLM`, `registry.py:172`,
  subclassing `DeepseekV3ForCausalLM`), but it only remaps mistral-native
  consolidated names (`model_executor/models/mistral_large_3.py:12-37`), not HF
  names.
- Worse, the checkpoint's `text_config` carries a stale
  `architectures: ["Mistral3ForConditionalGeneration"]` (verified by parsing the
  local config with transformers 5.8.1). `mistral3.py:439-444` calls
  `init_vllm_registered_model(hf_config=config.text_config)`, and
  `with_hf_config` only synthesizes architectures when the field is None
  (`config/vllm.py:633-644`). Resolution therefore loops back into
  `Mistral3ForConditionalGeneration` with a `Mistral4Config`, whose `__init__`
  immediately reads `config.projector_hidden_act` / `config.vision_config`
  (`mistral3.py:413-415`) and will die with `AttributeError`.
- The May 31 Mistral serve success (`logs/mistral_small4_nvfp4_serve.log`,
  resolved architecture `PixtralForConditionalGeneration`, TRITON_MLA) used the
  mistral-native consolidated checkpoint
  `Models/Mistral-Small-4-119B-2603-NVFP4`, which has since been deleted from
  disk. Only the RH HF checkpoint remains.
- What CAN load the RH HF checkpoint is the transformers + repo-loader stack:
  the v3.5 adapter currently on disk was trained through exactly that path
  (`scripts/train_mistral_rh_nvfp4_lora_ich_smoke.py:136-189`:
  `AutoModelForImageTextToText.from_config` on meta, `NVFP4Experts3D` MoE,
  `load_non_nvfp4_weights` for the BF16 attention, in the qwen-serve venv).

## Checkpoint and adapter facts that drive the merge design

- Qwen base (`RedHatAI-Qwen3.5-122B-A10B-NVFP4`): full-attention `self_attn.{q,k,v,o}_proj`
  ARE quantized (the quant-config ignore list covers only `linear_attn.*`, MoE
  gates, lm_head, and the vision tower). CT on-disk trio per linear:
  `weight_packed` (uint8), `weight_scale` (fp8_e4m3), `weight_global_scale`
  (fp32 divisor), plus `input_global_scale` (fp32, activation side, untouched by
  a weight merge). Only 2 shards of ~36 GB each.
- Per layer, q/k/v share `weight_global_scale` AND `input_global_scale`
  (verified layer 3: 15377.1611 / 40.7273 across q,k,v; o_proj has its own).
  vLLM fuses q/k/v into one `qkv_proj` and warns + max-fuses if the global
  scales differ (`compressed_tensors_w4a4_nvfp4.py:117-127`). A requantization
  after merging MUST keep the q/k/v trio on one shared per-tensor scale.
- Qwen adapter (training tonight): 96 tensors,
  `base_model.model.model.layers.{3,7,...,47}.self_attn.{q,k,v,o}_proj.lora_{A,B}.weight`
  (12 full-attention layers x 4 projections), r16 alpha32 (inspected
  `checkpoint_step_50/`). The train script writes the final adapter to the
  adapter root and the best-val copy to `best/`
  (`scripts/train_qwen3_5_122b_rh_nvfp4_lora_ich.py:616-619`).
  Note the adapter tail is `model.layers.N...` while the base checkpoint keys
  are `model.language_model.layers.N...`: the merge must insert
  `language_model.`.
- The LoRA was trained against the *dequantized* base
  (`nvfp4_lora/linear.py`: forward is `x @ dequant(W).T + scale * x @ A.T @ B.T`),
  so "dequant -> add delta -> requantize" reproduces the training-time function
  up to one requantization error, which the merge script logs per tensor.
- Mistral base (`RedHatAI-Mistral-Small-4-119B-2603-NVFP4-HF`): attention is
  NOT quantized (`ignore` includes `re:.*self_attn.*`), so the adapter targets
  (`q_b_proj`, `kv_b_proj`, `o_proj`, regex
  `^model\.language_model\..*\.(q_b_proj|kv_b_proj|o_proj)$`, r16 alpha32) are
  plain BF16 linears. Merging is the exact BF16 update `W += (alpha/r) B @ A`,
  no requantization involved.

## Decision matrix: Qwen3.5-122B-A10B (RH NVFP4, attention quantized)

| Path | Verdict | Blockers (source) | Confidence | Perf class |
|---|---|---|---|---|
| A. vLLM base + `--enable-lora --lora-modules` | **Blocked** | Global `is_lora_enabled` (`fused_moe/layer.py:334`) + oracle rejection (`modular_kernel.py:580`) forces MARLIN MoE; MARLIN repack OOMs 120B-class on Spark (`serve/README.md:22-24`); EMULATION no longer supports LoRA in 0.22.1 (no `LoRAExpertsMixin`); forced `--moe-backend cutlass` raises at startup (`oracle/nvfp4.py:256`). `--lora-target-modules` does not bypass the global flag. Additionally the adapter's `model.layers.*` naming would need remapping to `language_model.*`. | High (read 0.22.1 source end to end) | n/a |
| B. Merge-then-serve (new `scripts/merge_lora_into_ct_nvfp4.py`, then CUTLASS recipe) | **Recommended** | Requant noise on 48 attention tensors (logged per tensor, expected cosine > 0.999 since delta << base); vLLM 0.21 -> 0.22.1 upgrade since the last successful base serve is unvalidated (re-smoke the base path first if anything looks off). | Medium-high | ~12-15 tok/s single stream, ~40 tok/s aggregate at low concurrency (matches base CUTLASS numbers from `logs/qwen35_122b_nvfp4_serve.log` and `serve/README.md:13`) |
| C. Transformers OpenAI server on the training stack (NVFP4LoRALinear loader) | Works by construction (it is literally the training forward path) but no server exists for the Qwen NVFP4 loader yet; would need a sibling of the Mistral server below. | None functional; effort + speed only. | High (training already exercised this forward) | Low single-digit tok/s (per-layer NVFP4 dequant each forward, single stream) |

## Decision matrix: Mistral-Small-4-119B (RH NVFP4-HF, BF16 MLA attention)

| Path | Verdict | Blockers (source) | Confidence | Perf class |
|---|---|---|---|---|
| A. vLLM base + `--enable-lora` | **Blocked twice** | (1) Checkpoint cannot load: no mistral4 text backbone in vLLM (grep) and stale `text_config.architectures` recurses `mistral3.py:439` into a text config lacking `vision_config` (`mistral3.py:413-415`, `config/vllm.py:633-644`). (2) Even if loadable, `kv_b_proj` LoRA is inconsistent under MLA weight absorption (`mla_attention.py:835-863` vs `:742-751`). | High | n/a |
| B. Merge-then-serve via vLLM | Merge itself is trivial and exact (BF16 targets), but there is no vLLM-loadable artifact to merge into: the RH HF dir will not load (above) and the mistral-format dir was deleted. Future option: re-download the mistral-format checkpoint and port the BF16 deltas onto `layers.N.attention.{wq_b,wkv_b,wo}.weight` (name map in `mistral_large_3.py:12-37`), then serve with the proven `run_mistral_small4_nvfp4.sh` recipe. Not tonight. | Medium (porting step unvalidated) | Would be vLLM-class (TRITON_MLA + CUTLASS) |
| C. Transformers OpenAI server reusing the training loader + PEFT adapter, merged in memory (`merge_and_unload` on BF16 linears is exact) | **Recommended** | peft 0.19.1 in qwen-serve needs the in-place `WeightConverter` kwarg-filter patch (already applied; re-apply if peft is ever reinstalled, see memory note). Vision tower stays on meta in this load path, so the server pins generation device explicitly. | High (training ran this exact load path on this box) | Single-stream transformers MoE with Triton NVFP4 dequant; expect low single-digit tok/s (Qwen3.6-35B-A3B reference was ~10 tok/s; 119B A6B will be slower) |

## What was written where

- `scripts/merge_lora_into_ct_nvfp4.py` (new, py_compile checked only): the
  compressed-tensors counterpart of `scripts/merge_lora_into_nvfp4.py`.
  Reuses the repo's own primitives so merge numerics match training numerics:
  dequant via `nvfp4_lora.dequant.dequantize_nvfp4_weight(format="compressed_tensors")`,
  requant via `quantize_to_nvfp4_2d` from `scripts/quantize_mistral_to_nvfp4.py`
  (CT convention, `per_tensor_max_override` used to keep q/k/v on one shared
  global scale). Includes `--self-test` (CPU round-trip on random tensors) and
  `--dry-run` (adapter/base coverage check, no writes).
- `serve/run_qwen35_122b_rh_ct_lora.sh`: merge + serve wrapper for the Qwen
  target. `merge` subcommand produces
  `Models/RedHatAI-Qwen3.5-122B-A10B-NVFP4-ich-v3.5`, `serve` subcommand is the
  proven CUTLASS recipe (clone of `run_qwen35_122b_nvfp4.sh` flags) pointed at
  the merged dir. DO NOT RUN until training completes.
- `serve/serve_mistral_rh_nvfp4_lora_openai.py` +
  `serve/run_mistral_small4_rh_lora.sh`: OpenAI-compatible transformers server
  for the Mistral target. Loads the base exactly like
  `train_mistral_rh_nvfp4_lora_ich_smoke.py` (meta init, `NVFP4Experts3D`,
  `load_non_nvfp4_weights`, dequant workspaces), attaches the PEFT adapter, and
  by default merges it into the BF16 attention in memory. Endpoints are reused
  from `Sandbox/serve_qwen3_6_35b_a3b_openai_transformers.py` by module import,
  so behavior matches the existing house server. DO NOT RUN until training
  completes.

## Why a sibling merge script instead of editing `merge_lora_into_nvfp4.py`

The required edits are not cosmetic; the original would need all of the
following, which together touch most of its core paths (this doubles as the
requested TODO list if someone prefers in-place edits later):

1. `adapter_key_to_base_key` hardcodes the NemotronH `backbone.` prefix
   (`merge_lora_into_nvfp4.py:56-97`). Needs the Qwen mapping
   `base_model.model.model.layers.N...` -> `model.language_model.layers.N...`
   (and pass-through for keys already carrying `model.language_model.`).
2. Weight/scale key layout: modelopt `.weight` / `.weight_scale` /
   `.weight_scale_2` (`:300-314`) vs CT `.weight_packed` / `.weight_scale` /
   `.weight_global_scale`, plus CT's extra `.input_global_scale` which must be
   passed through untouched.
3. Dequant backend: `modelopt...NVFP4QTensor.dequantize` (`:195-203`) assumes
   the per-tensor scale is a multiplier; CT stores the reciprocal (divisor), see
   `nvfp4_lora/dequant.py:113-119`. Swap in
   `dequantize_nvfp4_weight(..., format="compressed_tensors")`.
4. Requant backend: `NVFP4QTensor.quantize` (`:218-230`) emits modelopt-layout
   scales; CT needs `quantize_to_nvfp4_2d` semantics
   (`quantize_mistral_to_nvfp4.py:61-133`), and crucially a shared
   `per_tensor_max_override` across each layer's q/k/v so the fused-qkv global
   scales stay equal (vLLM max-fuses and warns otherwise,
   `compressed_tensors_w4a4_nvfp4.py:117-127`). The original has no concept of
   scale groups.
5. Shard strategy: the original streams shard-by-shard, which is fine for
   Nemotron's many small shards but the Qwen base is 2 x ~36 GB; the new script
   computes the 48 merged tensors first (via the index), then rewrites each
   shard once. Expect ~40 GB peak host RAM per shard during the rewrite, fine
   once training has released the UMA.

## Post-run validation sequence (cheapest first, run after ~01:30)

1. **Training postmortem (no GPU):** confirm the trainer exited cleanly; check
   `adapters/qwen3_5_122b_a10b_rh_nvfp4_lora_ich_v3_5/adapter_model.safetensors`
   exists at the root with 96 tensors, and compare `metrics.jsonl` tail vs
   `best/`. Decide root (final) vs `best/` for the merge input.
2. **Merge self-test (CPU, seconds):**
   `python scripts/merge_lora_into_ct_nvfp4.py --self-test`.
3. **Merge dry run (file reads only, seconds):**
   `... --dry-run` with the real base + adapter dirs; verify it reports 48
   matched targets, 12 q/k/v scale trios, and 0 unmatched adapter keys.
4. **Run the merge (minutes, GPU or CPU):**
   `serve/run_qwen35_122b_rh_ct_lora.sh merge`. Then read
   `merge_stats.jsonl`: expect merge_cosine >= 0.999 and small relative error;
   flag anything below 0.995. Verify output dir size ~72 GB and the index plus
   tokenizer/config files copied.
5. **Serve the merged Qwen (10-15 min load):**
   `serve/run_qwen35_122b_rh_ct_lora.sh serve`. Watch for the
   `Using 'VLLM_CUTLASS' NvFp4 MoE backend` line; smoke a short completion;
   sanity-check throughput vs the 11.8-14.5 tok/s base numbers. If 0.22.1
   misbehaves where 0.21 worked, re-serve the unmerged base via
   `serve/run_qwen35_122b_nvfp4.sh` to bisect vLLM-upgrade vs merge.
6. **FT-vs-base behavioral check:** one ICH v3.5 held-out prompt against the
   merged server; optionally `scripts/distinguish_ft.py` or an A/B against the
   base recipe.
7. **Mistral server (longest single step, ~10-20 min load):**
   `serve/run_mistral_small4_rh_lora.sh`, then `/health`, then the same ICH
   smoke prompt. If adapter attach fails with
   `TypeError: WeightConverter.__init__`, the peft in-place patch was lost;
   re-apply per the memory note before retrying.
8. **Optional research, strictly after the above:** vLLM-loading experiments for
   the RH HF Mistral (`--hf-overrides` on `text_config.architectures`; expect
   either a transformers-backend fallback of unknown MLA quality or config-field
   mismatches), and the mistral-format re-download + delta-port plan from the
   decision matrix.
