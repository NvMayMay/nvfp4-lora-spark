# Bring expert-LoRA online -- prioritized roadmap (DRAFT for review)

State today (2026-06-28): the full loop is PROVEN end-to-end on ONE GB10/sm_121 but is
EXPERIMENTAL (on branch feat/expert-lora-trainside): train (--expert-lora-r, real GLM-4.5-Air
fused-MoE on GPU, dry-run + 100-step run) -> rekey (scripts/rekey_expert_lora_for_vllm.py,
native stacked -> vLLM per-expert) -> serve (--moe-backend emulation + --enable-lora,
expandable_segments) -> the trained delta APPLIES (behavioral parity: the adapter reproduces
its learned Spider output format vs base). 18 train-side CPU tests. 4-agent reviewed.

Known constraints carried into this roadmap:
- The ROUTER stays frozen -> expert-LoRA adapts what routing already selects; it cannot re-route.
- Adapter is large (~1-2 GB bf16 for GLM-Air at small r) vs tens of MB attention-only.
- Backend matrix (NVFP4 MoE, sm_121): emulation = LoRA-capable but slow (per-forward dequant);
  marlin = LoRA-capable + faster BUT OOMs one box (needs 2-box) and vLLM treats sm_121 as
  non-FP4-native (weight-only fallback); cutlass/flashinfer = fast but NOT LoRA-capable.
- Not yet done: rigorous numerical parity; a measured task-LIFT (GLM saturates like Qwen-32B).

## Priority order

### P0 -- Serving-path decision (gates everything else)
Emulation is correct-but-slow. Pick the production serving path:
  A. Ship EMULATION as iteration-grade (one-box, slow but proven). Lowest effort; unblocks P1-P5.
  B. 2-box TP MARLIN (faster; one box OOMs). Medium-high effort (multi-node serve infra); the
     185 Gb/s link is validated, GLM is on both boxes.
  C. Upstream vLLM fix for a FAST one-box NVFP4-MoE-LoRA on sm_121 (add LoRAExpertsMixin to the
     cutlass/flashinfer experts, or fix marlin's sm_121 FP4 path + load memory). High effort,
     upstream dependency, best long-term.
Recommendation: start with A to unblock the pipeline, scope B in parallel; treat C as upstream.

### P1 -- Rigorous train<->serve numerical parity
Tiny deterministic MoE: assert train-time expert-LoRA forward == serve-time kernel output within
tolerance, for the CHOSEN backend (dequant math differs marlin vs emulation). The scope's stated
hard part; required for trust before any quality claim.

### P2 -- Serve-flow integration (productize)
Wire the rekey into the serve path (auto-rekey an expert adapter on serve); the binding contract
(`nybbloris inspect`) classifies expert adapters (PASS / NEEDS-REKEY / EMPTY); a first-class
`nybbloris serve` recipe for expert-LoRA (today it is a hand-run vllm command + a separate rekey).

### P3 -- Real task-LIFT demonstration
Train an expert-LoRA on a model/task WITH headroom (NOT GLM-Air, which saturates) and show a
MEASURED served lift (EM or domain metric), base vs adapter. Proves value, not just mechanism.

### P4 -- Tests + CI
rekey round-trip test; serve-format/key-mapping test; wire the P1 parity test as a CI gate.
(18 train-side CPU tests already exist.)

### P5 -- Merge + status flip + docs
Merge off the experimental branch to main; flip "experimental" -> "supported"; document the
router-frozen ceiling, adapter size, training optimizer-state memory, and the backend matrix.

## Rough effort
- Iteration-grade online (emulation, one box): ~days (P0=A + P2 + P3 + P4 + P5).
- Production-fast online: ~weeks (P0=B 2-box infra or C upstream + the rigorous P1 parity).

---

## Reviewer input (Sonnet max + gpt-5.5 max, 2026-06-28) and REVISED order

The two reviewers DISAGREED on the top gate, productively:
- Sonnet: front-load the VALUE question. We already have BEHAVIORAL parity, so prove the
  capability is worth anything (does expert-LoRA beat ATTENTION-ONLY LoRA on a task with
  headroom?) before building infra or rigorous parity. Rigorous numerical parity is
  trust-building, not value-unlocking; push it later.
- gpt-5.5: front-load CORRECTNESS. A "no lift" result is uninterpretable without numerical
  parity (could be saturation / frozen router / miskeyed adapter / backend mismatch /
  training). Lock the adapter ABI + deterministic parity first.

Reconciliation (the synthesis we will work from):
- P0  CHEAP correctness floor (NOT the full harness): rekey ROUND-TRIP test + a
      CATASTROPHIC-mismatch parity check (>1% rel error = bug; <0.1% = normal bf16 rounding).
      We already have behavioral parity, so this is cheap and removes gpt-5.5's
      "uninterpretable" objection without doing the full fp32 harness yet. (~hours)
- P1  THE VALUE EXPERIMENT (Sonnet's highest-leverage, now interpretable thanks to P0):
      3 arms -- base vs expert-LoRA vs ATTENTION-ONLY LoRA at comparable rank -- on a model/
      task WITH headroom (e.g. Qwen3-32B on a structured task where base EM < 50%; reuse
      prep_spider/eval_retention). Measure BOTH the task metric AND emulation tokens/sec.
      GO/NO-GO: if expert-LoRA does not beat attention-only enough to justify ~1-2 GB adapter
      + emulation slowdown, shelve/narrow. (~1-2 weeks incl. training)
- P2  Productize the emulation serve flow: auto-rekey on serve; `nybbloris inspect` classifies
      expert adapters; LOAD-TIME FAIL-CLOSED validation (base/adapter/backend mismatch ->
      hard error, never silent partial serve); serve recipe. (~days)
- P3  RIGOROUS numerical parity harness as a CI gate (gpt-5.5's P0, promoted here once value
      is shown): tiny deterministic MoE, frozen known routing, compare expert-output + final
      logits + routed-token subset, per chosen backend. (~days, underpriced -- expert
      ordering/dtype/scaling/packed-layout debugging)
- P4  Performance decision, MEASURED not guessed: emulation accept criteria (min tok/s, p95,
      concurrency, ctx, memory headroom); scope 2-box marlin vs upstream fast backend against
      those numbers. (B ~weeks; C = R&D track, off critical path)
- P5  Merge + status: "preview/experimental-supported" until correctness + integration + CI +
      one measured served lift; only then "supported". Docs: router-frozen ceiling, adapter
      ABI/versioning, training optimizer-state memory, backend matrix, multi-adapter +
      fallback policy.

New cross-cutting items both reviewers surfaced (fold into the above):
- Adapter ABI / versioning (expert-id order, layer naming, rank/alpha/dtype, MoE layout,
  base-model + quant + tokenizer hashes) + load-time fail-closed validation. [gpt-5.5]
- Throughput is a SPEC not "slow": measure emulation tok/s vs cutlass (use eager_lora_serve_probe
  / serve_parity_vllm). [both]
- The attention-only LoRA BASELINE is mandatory in P1 -- it is the real competitor (100s of MB,
  fast cutlass path). If expert-LoRA does not beat it per-GB and per-tok/s, the feature is a
  curiosity. [Sonnet]
- Frozen-router eval: include task cases where frozen routing SHOULD help and where it should
  NOT, so the capability is not overgeneralized; quantify what fraction of task tokens route to
  the experts the adapter benefits. [both]
- Multi-adapter / fallback semantics: is expert-LoRA single-adapter-only, composable with
  attention LoRA, hot-swappable; what happens if a fast backend is requested with an expert
  adapter (hard error vs downgrade-to-emulation). [gpt-5.5]

HIGHEST-LEVERAGE NEXT ACTION (both converge, modulo order): the cheap correctness floor (rekey
round-trip + catastrophic-mismatch check) THEN the 3-arm value experiment (base/expert/attention)
with throughput -- this answers "is it worth building" before any heavy infra.
