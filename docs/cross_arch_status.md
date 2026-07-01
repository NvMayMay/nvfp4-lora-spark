# Cross-architecture expert-LoRA status

Mechanism-specific functional status (NOT a quality claim; the value question is a documented NO-GO on
EM for typical format tasks -- experts buy calibration, not exact-match). See
docs/plans/cross_arch_expert_lora_testing.md for the plan and the pass bar.

Legend: TRAIN = base loads + experts attach + loss drops + expert lora_B moves from zero-init + clean
save/exit (hang-fixed). SERVE = runtime-LoRA can actually apply the expert delta at request time.

## Fused-3D expert LoRA (NVFP4Experts3D; expert_lora_r path)
Serve path = vLLM EMULATION backend (the LoRA-capable NVFP4 MoE backend on sm_121; CUTLASS/flashinfer
report supports_lora=False). Slow single-stream but correct.

| family | model | TRAIN | SERVE (runtime-LoRA) | notes |
|--------|-------|-------|----------------------|-------|
| glm4_moe | GLM-4.5-Air-106B-A12B | PASS | PASS (emulation) | proven end-to-end (train->rekey->serve->learned behavior); 90/90 expert lora_B nonzero |
| qwen3_5_moe | Qwen3.5-122B-A10B | PASS | PASS (emulation) | proven end-to-end. T2: train PASS (96/96 expert lora_B). Root cause of the earlier no-op was a REKEY bug, not a vLLM gap: the expert/attn rekey kept the flat `model.layers.N` path, but the vision-wrapped Qwen3.5 base serves its decoder under `language_model.model.layers.N`, so vLLM's 3D->2D MoE LoRA converter found neither expert key and left the stacked buffers zero. Fixed by re-keying to the wrapped path (logprob delta -5.47 sum / 2.71 max per token). See findings #1-#3 |
| mistral3/mistral4 | Mistral-Small-4-119B | PASS | pending (emulation expected) | T3: 72/72 expert lora_B nonzero, expert-only; loss 13.66->9.02; force-native-expert-only path (BF16 MLA attn); seq2048; clean save+exit |

## Per-module expert LoRA (per-expert NVFP4 nn.Linear; ordinary target-modules)
| family | model | TRAIN | SERVE (runtime-LoRA) | notes |
|--------|-------|-------|----------------------|-------|
| nemotron_h | Nemotron-3-Nano-30B-A3B | PASS | **BLOCKED-ROUTED** | trains fine (5920/5934 expert lora_B nonzero, expert-only), but serve is blocked -- see finding |

### T1 binding verdict (2026-06-30): BLOCKED-ROUTED -- but that is CONSERVATIVE, not a merge-only truth
Binding-contract verdict on the trained adapter:
- 5934/5934 targets resolve to base modules (identity rekey, keys map cleanly);
- only 46/5934 LoRA-live at serve (shared-expert MLPs); 5888/5934 marked BLOCKED-ROUTED.

CORRECTION (codex/gpt-5.5 review, grounded in vLLM source): the BLOCKED-ROUTED verdict is a BUG in our
binding check, not a hard fact. `nybbloris/plan.py` marks EVERY routed-expert target blocked
UNCONDITIONALLY, without checking the selected serve backend -- so it would wrongly block GLM too, yet we
SERVE GLM routed-expert LoRA at runtime via the emulation backend. Runtime-LoRA for Nemotron routed
experts is feasible IN PRINCIPLE:
- Nemotron-H is NOT a bespoke no-LoRA module at serve: `NemotronHMoE` builds a standard vLLM `FusedMoE`,
  and `NemotronHForCausalLM` declares `SupportsLoRA, MixtureOfExperts, is_non_gated_moe=True`, using the
  same `select_nvfp4_moe_backend` family as GLM. So forcing a LoRA-wired backend (emulation/marlin) should
  apply the delta.
- The per-module-trained adapter can be REKEYED into vLLM's stacked FusedMoE LoRA layout (per-module vs
  fused-3D is a train-representation issue, not a serve-kernel one).
- ONE Nemotron-specific catch: it is NON-GATED MoE -- `w13` is ONLY `up_proj` (no gate / no `w3`),
  `w2 = down_proj`. A GLM-style `w13=[gate,up]` rekey would be WRONG; if vLLM's LoRA context assumes two
  `w13` shards, that is the likely small patch.

REVISED serve status for per-module routed experts: **LIKELY runtime-LoRA-servable via the emulation
backend (same path as GLM), PENDING three things**: (1) fix the binding check to be backend-aware
("routed expert => live iff selected FusedMoE backend applies LoRA"); (2) a NON-GATED-aware rekey
(up_proj->w1, down_proj->w2, no fake gate); (3) a GPU logits-move test forced to moe_backend=emulation.
NOT proven yet, but NOT merge-only either.

IMPLICATION for the capability claim: "expert-LoRA across architectures" -- fused-3D families serve at
runtime via emulation (GLM proven; Qwen3.5/Mistral-4 under test); per-module routed experts (Nemotron)
TRAIN and are LIKELY runtime-servable via emulation pending the rekey + GPU confirmation above. Do not
state "merge-for-serve only" -- that was the conservative binding-check artifact.

> **SUPERSEDED by FINDING #3 (2026-07-01).** Findings #1 and #2 below are the diagnostic trail, not the
> current status. Qwen3.5-122B expert-LoRA now SERVES (logprob delta confirmed). Root cause was a
> wrapped-model rekey bug, not a vLLM apply gap. Read #3 for the resolution; #1/#2 retained for history.

### FINDING #1 (T2 serve, 2026-06-30): GLM-shaped expert rekey is NOT directly Qwen3.5-servable
Brought up Qwen3.5-122B on box B with `--moe-backend emulation --enable-lora` (the LoRA-capable path,
vLLM 0.22.1) + the rekeyed T2 adapter. vLLM rejected the adapter at `add_lora`:
`expected target modules in {'linear_fc2','down_proj','qkv','experts','conv1d','o_proj',
'shared_expert_gate','v_proj','gate','gate_proj','in_proj_a','in_proj_z','in_proj_qkv','proj','q_proj',
'linear_fc1','in_proj_b','k_proj','up_proj','out_proj'}`.
- The adapter_config target_modules (down_proj/gate_proj/q/k/v/o/up_proj) are ALL in that set, so the
  CONFIG is fine. The mismatch is the TENSOR-KEY LAYOUT: vLLM's Qwen3.5 exposes routed experts as a FUSED
  `experts` LoRA module (note 'experts' singular in the set), but rekey_expert_lora_for_vllm.py EXPANDS to
  PER-EXPERT keys (`...mlp.experts.0.gate_proj.lora_A`, `.1.`, ...) -- the layout that DID serve on GLM.
- So the rekey is family-specific at SERVE (exactly codex's rekey-shape caution). GLM's vLLM model took
  per-expert keys; Qwen3.5's vLLM model (Qwen3_5MoeForConditionalGeneration) wants the fused `experts`
  layout. FOLLOW-UP: a Qwen3.5-specific rekey (keep the stacked/fused `experts` layout, or map to whatever
  Qwen3_5Moe's LoRA loader consumes) + re-test the emulation serve. NOT a fundamental block -- a rekey-
  mapping fix. Train side is fully PASS; only the serve rekey format needs per-family work.

### FINDING #2 (T2 serve, 2026-06-30): v2 adapter LOADS on Qwen3.5 but emulation does NOT APPLY it (no-op)
After codex's fused-3d rekey, vLLM 0.22.1 (--moe-backend emulation --enable-lora --language-model-only)
ACCEPTED the v2 adapter: "Loaded new LoRA adapter: t2 ... vllm_rekey_v2", emulation backend engaged, serve
READY. BUT a rigorous apply-check shows it is a SILENT NO-OP:
- greedy generation (no-think): base == adapter on 2 Spider prompts (ambiguous -- base already saturates
  these trivial prompts with correct SQL);
- DECISIVE -- prompt-echo LOGPROBS (/v1/completions echo): base sum=-53.8779 vs t2 sum=-53.8779, max
  per-token |delta|=0.0000. Identical forward pass => the LoRA delta is NOT applied.
ROOT CAUSE (hypothesis): GLM uses vLLM's 2D per-expert MoE LoRA path, which the emulation backend's apply()
IS wired for (inherits TritonExperts.apply + MoELoRAContext). Qwen3.5 is is_3d_moe_weight=True -> the 3D
FUSED MoE LoRA path (FusedMoE3DWithLoRA), which the emulation backend LOADS but does NOT execute in apply().
So the serve gap moved from "rekey won't load" (fixed) to "3D LoRA not applied by emulation".
STATUS: Qwen3.5 fused-3d expert-LoRA -- TRAIN PASS, rekey/LOAD PASS, APPLY = NO-OP on emulation. Needs the
3D-fused-MoE LoRA wired into the emulation apply() (or a backend that executes 3D LoRA: marlin?, GPU-gated).
CAVEAT: logprob test assumes vLLM applies LoRA to echo logprobs (it should); corroborated by the greedy
null. CREDIBILITY WIN: a greedy-only test would have passed this falsely (saturation); the logprob check
caught the no-op -- this is why output-delta alone is not a valid serve-apply proof.

### FINDING #3 (T2 serve, RESOLVED 2026-07-01): the no-op was a wrapped-model REKEY bug, not a vLLM gap
Findings #1/#2 are now SUPERSEDED. Instrumenting vLLM's MoE LoRA path (probe on FusedMoEWithLoRA.set_mapping,
Nvfp4QuantizationEmulationTritonExperts.apply, LoRAExpertsMixin.apply_w13/w2_lora, and
LoRAModelManager._convert_3d_to_2d_moe_lora) showed the research's leading "context-identity" hypothesis was
WRONG: the MoELoRAContext lands on the exact emulation experts object that executes (fe_id matches), and the
LoRA kernels DO fire. The real cause: the 3D->2D converter logged `CONVERT_3D_2D module=
language_model.model.layers.0.mlp.experts had_mod=False had_base=False` -- it looked up the adapter's expert
weights under vLLM's WRAPPED runtime path and found neither key, so the stacked buffers stayed zero
(adapter_enabled=[0,0], all maxabs=0.0) and the forward was identical.
ROOT CAUSE: Qwen3.5-122B is a vision-wrapped `...ForConditionalGeneration` base; vLLM serves its decoder under
`language_model.model.layers.N...`. The expert rekey (and the attention passthrough) emitted the flat
`base_model.model.model.layers.N...` path, which on a wrapped base resolves to a module vLLM never builds.
GLM-4.5-Air never hit this because it is a plain text MoE (`model.layers.N`, no wrapper).
FIX: re-key all decoder keys `base_model.model.model.layers.` -> `base_model.model.language_model.model.layers.`
(the same `language_model` swap scripts/rekey_lora_for_vllm.py already applies for attention). Verified on GPU:
prompt-echo logprobs base=-53.8779 vs adapter=-59.3489 (sum delta -5.47, max per-token |delta|=2.71) -> APPLIES.
Proven adapter: Sandbox/adapters/t2_qwen35_122b_vllm_rekey_v3. Durable fix: rekey_expert_lora_for_vllm.py now
auto-detects a wrapped base (architectures `...ForConditionalGeneration` or vision_config/text_config) and
applies the wrapped re-key (`--wrapped {auto,yes,no}`); v4 from the fixed script byte-matches the proven v3.
LESSON: the binding contract's static key-presence check (QC A7/B4) could not catch this -- the adapter
LOADED and bound; only the runtime logprob-delta check exposed the zero buffers. Backs the QC's call to make
the contract wrapped/format-aware and to add a runtime logprob-delta regression.
