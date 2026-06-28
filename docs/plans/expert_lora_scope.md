# Expert-LoRA scope (LoRA on routed MoE experts) — feasibility + plan

Status: SCOPED, greenlit as a long-road Phase-3 R&D item (not v1). Assessed 2026-06-27.
Reviewed by gpt-5.5 (xhigh); static serving de-risk run (no GPU). Raw review: scratchpad/codex_expert_lora_review.txt.

## Goal
Make the routed MoE experts a LoRA-trainable surface (today they are frozen NVFP4).
Must preserve the core contract: a small bf16 delta applied at SERVE time over a
frozen NVFP4 base — NO merge / requantize.

## Why this matters
In MoE models the routed experts are ~99% of params. Today nybbloris only adapts
attention + (dense/shared) FFN; the experts (the bulk of the model's knowledge) are
untouched. Expert-LoRA is the difference between "we adapt the attention of an MoE"
and "we actually fine-tune the experts."

## Serving feasibility (the gating question) — DE-RISKED STATICALLY
vLLM 0.22.1 fused-MoE LoRA is gated per expert-kernel: `FusedMoEWithLoRA.__init__`
asserts `moe_kernel.supports_lora()`, which defaults False and is flipped True only by
mixing in `LoRAExpertsMixin` (provides apply_w13_lora/apply_w2_lora -> punica triton).

NVFP4 (compressed-tensors `nvfp4-pack-quantized`, e.g. GLM-4.5-Air) routes through
`compressed_tensors_moe_w4a4_nvfp4.py` -> `select_nvfp4_moe_backend()` (oracle/nvfp4.py).
Backend -> experts class -> LoRA support:

| backend (forceable via moe_backend=...) | experts class | LoRA? |
|---|---|---|
| flashinfer_trtllm/cutlass/cutedsl/b12x | FlashInfer*Experts | NO |
| cutlass (VLLM_CUTLASS) | CutlassExpertsFp4 | NO |
| **marlin** | **MarlinExperts(LoRAExpertsMixin, MarlinExpertsBase)** -- apply() actually CALLS apply_w13_lora L823 / apply_w2_lora L848 | **YES, genuinely applies** |
| ~~emulation~~ | Nvfp4QuantizationEmulationTritonExperts(TritonExperts) -- inherits supports_lora()==True BUT overrides apply() (L83), and that apply() has ZERO lora refs | **NO -- silent no-op trap** |

>>> SUPERSEDED (see "## Verdict" and the 2026-06-28 parity update below): the claim that
>>> emulation is a silent no-op is WRONG. nvfp4_emulation_moe.py:apply() ends in
>>> `super().apply()` = TritonExperts.apply (fully LoRA-wired) and DOES receive the
>>> MoELoRAContext via the supports_internal_mk reuse branch. The "0 lora refs" was a grep
>>> artifact (wiring is inherited). EMULATION IS THE VALIDATED ONE-BOX PATH. The text below
>>> is kept for the investigation record only.

KEY RESULT (LATER PROVEN WRONG -- see banner above): "MARLIN is the only
genuine stock path; the emulation backend is a TRAP: it inherits `supports_lora()==True`
from TritonExperts so it PASSES the `FusedMoEWithLoRA` assert, but its OVERRIDING `apply()`
never calls apply_w13_lora/apply_w2_lora (grep: 0 lora refs in nvfp4_emulation_moe.py) -- so
the adapter is SILENTLY IGNORED." (Refuted: the grep missed the inherited super().apply().)

=> Feasibility hinges ENTIRELY on **marlin running on sm_121** (unverified; oracle/nvfp4.py
L171-173 already disables FLASHINFER_B12X on SM121 pending an upstream CUTLASS guard, so sm_121
backend support IS fiddly). IF marlin does not run on sm_121, an upstream vLLM patch IS
required -- but a SMALL one: wire the existing mixin hooks into emulation's apply() (mirror
TritonExperts.apply) for a guaranteed-correct (slow) reference path. Fast flashinfer/cutlass
backends would need the mixin for production throughput regardless.

ALWAYS validate any serve-with-LoRA path with gpt-5.5's logits-move test (huge delta on one
expert; compare logits LoRA-on vs off). The emulation no-op proves a backend can pass the
capability assert yet apply nothing.

## Train side (tractable; the hard part is parity, not autograd)
Extend `nvfp4_lora/experts.py` (`NVFP4Experts3D` + `_GroupedDequantExpertLinear`) to
carry trainable per-expert bf16 A/B and emit their grads (the grouped bmm already
gathers per-expert token groups via expert_idx; add x@Ae^T@Be^T + accumulate adapter
grads; base stays frozen NVFP4). Add expert targets to family/target resolution.

The HARD part (per gpt-5.5) is train<->serve numerical PARITY, NOT the autograd:
- w13 gate/up stacking order must match vLLM's stacked layout exactly
- w13 LoRA added pre-activation; w2 LoRA at vLLM's add point
- padded/duplicate expert_idx token groups -> correct masked scatter-add of grads
- our NVFP4 dequant math must match the served backend's dequant (marlin vs emulation differ!)
- LoRA scaling, rank<=128 (kernel assert), dtype, accumulation order
- local/global expert ids under EP/TP sharding vs the stacked adapter layout
MUST build a tiny deterministic MoE parity test (train-forward == serve-kernel) BEFORE real training.

## Plan (ordered; Phase 0 is the gate)
- Phase 0a (DONE, no GPU): static check -- only MARLIN genuinely applies LoRA for NVFP4; emulation
  is a silent no-op (passes assert, apply() ignores the delta); fast defaults lack the mixin.
- Phase 0b (GPU, ~0.5d): serve GLM-4.5-Air-NVFP4 with `moe_backend="marlin"` + a hand-built tiny
  per-expert stacked adapter; run gpt-5.5's logits-move test (huge delta on one expert -> logits
  must move vs LoRA-off). This simultaneously proves (i) marlin runs on sm_121 and (ii) LoRA is
  actually applied. GATE: if marlin won't run on sm_121, fall back to the SMALL patch (wire mixin
  hooks into emulation.apply()) for a correct-but-slow reference path -- do NOT rely on stock
  emulation (it silently ignores the adapter).
- Phase 1 (TRAIN-SIDE DONE 2026-06-27, CPU-complete; reviewed by 4 agents: 2 Opus + 2 codex/gpt-5.5
  on max -- unanimous SHIP-WITH-FIXES, all consensus fixes applied; 18 expert-LoRA tests, 159 suite total).
  Implemented: module per-expert LoRA + CLI --expert-lora-r/alpha/dropout + loader wiring + adapter
  save/load round-trip. Review fixes applied: (a) module defaults alpha=2*r when 0 (was a silent dead
  adapter via scale=0); (b) per-expert Kaiming init (3D kaiming_uniform_ had fan_in=r*in, wrong by
  ~sqrt(r)); (c) zero padded grouped lanes before LoRA work (avoid 0*NaN in bwd); (d) resume reads
  expert_lora from adapter_config + fails loud if expert keys present but model has no expert LoRA, or
  on partial/shape-mismatched blocks (was: silently dropped the trained delta); (e) loud warn when
  --expert-lora-r set but mode!=native; (f) meta-tensor guard in save; (g) corrected adapter_config
  tensor_shapes + "experimental" stamp; (h) est-optimizer-state-GB log at load. Tests added for
  split_gate_up_scales=True (was untested), dropout train/eval, alpha-default, init-scale, resume guards.
  REMAINING Phase 1 detail (was, now mostly done): expert target resolution is implicit (all routed
  experts adapted; a per-projection / down-only knob is a future nicety).
- Phase 1 (original plan note): train-side per-expert A/B in experts.py + grouped autograd; expert target resolution.
  (NVFP4Experts3D gained opt-in lora_r/lora_alpha/lora_dropout; per-expert bf16 A/B for gate_up (E,r,h)/
  (E,2i,r) and down (E,r,i)/(E,h,r); delta added in BOTH the grouped and per-expert forward paths; A/B
  shapes mirror vLLM stacked w13/w2). lora_r=0 is an exact no-op. CPU-tested: tests/test_expert_lora.py
  (8 tests: wiring, zero-init no-op, trainable-only-adapter, grouped==per-expert parity, float64
  gradcheck) -- full suite 149 passed. REMAINING Phase 1: plumb a --expert-lora-r CLI flag through the
  loader so the trainer instantiates experts with lora_r>0 (optimizer already auto-collects requires_grad
  params), and add expert target resolution.
- Phase 2 (1-2d): save adapter in vLLM stacked w13/w2 per-expert format; binding-contract/rekey
  recognizes expert modules; tiny-MoE train-vs-serve PARITY test; PASS verdict.
- Phase 3 (GPU): train->bind->serve->measure on GLM-4.5-Air; confirm delta applies + moves a metric.

## Caveats (gpt-5.5 confirmed mine + added)
- Adapter size grows a lot (256 experts x 2 proj): even small r is ~1-2GB bf16 vs tens of MB
  attn-only. Mitigate small r and/or w2(down)-only.
- Sparse expert grads (~64 tok/expert at seq2048/top8/256experts) -> noisy; needs more data/epochs.
- ROUTER stays frozen -> expert-LoRA cannot fix bad routing (a real benefit ceiling).
- Throughput: emulation backend is slow; LoRA adds bf16 low-rank work on every routed expert.
- Optimizer-state memory >> adapter size during training.
- EP/TP expert sharding must match the stacked adapter layout.
- Marlin is NOT a contract-preserving workaround if it implies requantizing away NVFP4 (here it
  serves the NVFP4 weights directly, so it IS contract-preserving IF it runs on sm_121).

## Verdict: PROVEN feasible on ONE box (2026-06-28)

Expert-LoRA serving works on a single GB10/sm_121, end to end, with stock vLLM 0.22.1
and NO vLLM source patch. Validated on GLM-4.5-Air-NVFP4:

  train (--expert-lora-r, native stacked save) -> rekey (scripts/rekey_expert_lora_for_vllm.py)
  -> serve (serve/run_glm45_air_nvfp4_expert_lora.sh: --moe-backend emulation --enable-lora,
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True) -> the expert delta IS applied.

Logits-move proof (same prompt, temperature 0):
  base: " Paris. It is one of the most beautiful and famous cities in the world"
  myft (large synthetic per-expert delta): ".weixin track不完smith追entr asymmetric ... 红辣椒"
The adapter loaded ("Loaded new LoRA adapter: name 'myft'", "Using fused MoE LoRA
implementation") and visibly scrambled the output -> the per-expert LoRA is genuinely
applied to the experts, not a silent no-op.

### Corrections to the earlier (wrong) verdict, from the 4-agent panel + source reads:
1. EMULATION is NOT a silent no-op. Its apply() ends in `super().apply()` =
   TritonExperts.apply, which is fully LoRA-wired (calls apply_w13_lora/apply_w2_lora) and
   receives the MoELoRAContext via the supports_internal_mk reuse branch. My earlier
   "0 lora refs -> no-op" was a grep artifact (the wiring is inherited). NO PATCH NEEDED.
2. MARLIN's load repack is PER-LAYER (~1 GB transient, 4-bit->4-bit), not a whole-model 2x.
3. The GLM marlin OOMs were compounded by a MISSING flag: PYTORCH_CUDA_ALLOC_CONF=
   expandable_segments:True (the load-bearing GB10 UMA flag; the working cutlass launcher
   uses it at util 0.80). With it, GLM-Air serves fine on one box.

### Backend choice for expert-LoRA serving on one box
- EMULATION: loads cheap (no repack), applies expert LoRA, JITs on sm_121. Correct but
  slow (dequantizes experts per forward) -> serving-for-iteration. This is the validated path.
- MARLIN: faster native expert-LoRA GEMM; one-box-viable with expandable_segments + sane
  util + low rank (repack is only ~1 GB/layer). Not yet validated end-to-end here; emulation
  was sufficient to prove feasibility.
- cutlass/flashinfer: experts NOT LoRA-capable -> attention-only LoRA only (the existing
  run_glm45_air_nvfp4_dynamic_lora.sh path).

### Remaining (not blocking the feasibility claim)
- Train<->serve numerical parity (a real trained adapter, not the synthetic large delta used
  for the logits-move proof) -> validate a GLM expert-LoRA actually IMPROVES a task.
- Marlin one-box validation for the faster serving path.
- Process lesson: never trust supports_lora()==True alone; confirm apply() applies AND that
  logits actually move; and always set expandable_segments on GB10.

## Update 2026-06-28: real-adapter train<->serve parity CONFIRMED
Beyond the synthetic-delta logits-move proof, a REAL GLM-4.5-Air expert-LoRA was trained on
GPU (100 steps on Spider, expert_lora_r=4 + attn r=8, loss 13.1 -> ~0.1; 45 MoE blocks
adapted), rekeyed (34,928 vLLM tensors), and served on ONE box (emulation). Same Spider prompt:
  gold: SELECT count(*) FROM singer
  base: triple-backtick sql / SELECT COUNT(*) FROM singer; (markdown-fenced, uppercase)
  myft: SELECT count(*) FROM singer;                       (clean lowercase -- the trained Spider format)
The trained expert-LoRA reproduces its learned behavior at serve time (strips markdown,
lowercases) -- coherent, distinct from base, training-consistent. Full pipeline validated:
train (real fused-MoE GPU) -> rekey -> serve (emulation, one box) -> learned behavior applied.
Not measured: a Spider EM lift (GLM-Air likely saturates the strict set-match like Qwen3-32B;
parity + coherence was the objective). Faster marlin serving remains 2-box only.
