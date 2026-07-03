# Path 1: Cheap-Alternatives Spike — descriptor cliff stop-loss

**Status:** DRAFT — awaiting adversarial audit
**Hard timeline cap:** 1 week (5 working days) from start. If at end of week the spike has not produced a decisive answer, escalate.
**Goal:** Decide whether the descriptor-pool cliff at ~s2048-3000 can be moved by **cheap, off-the-shelf interventions** before committing 14-22 weeks to Path 2 (full static graph engine).

## Why this spike exists

Both adversarial audits of the Path 2 plan (`static_cuda_graph_engine.md`)
converged on REVISE with the same secondary recommendation: try the cheap
alternatives first. They are unrepresented in our experiment journal
(`docs/LONG_CONTEXT_EXPERIMENTS.md`) and each takes ~1 day to test. If any
one of them shifts the cliff materially, Path 2 is unnecessary.

The "wins" are stratified by what they would mean for the project:

| Outcome | Implication |
|---|---|
| Cliff shifts to s8k+ via one cheap intervention | Path 2 cancelled, ship the cheap fix |
| Cliff shifts to s4k via one cheap intervention | Useful but Path 2 still needed for s16k+ target |
| No intervention shifts the cliff | Path 2 is justified, but with the audit findings baked in |
| One intervention reduces descriptor growth rate (slope <0.4 events/tok) but not the absolute cliff | Stack interventions before deciding |

## Pre-work (day 0, ~3 hours, independent merit)

These two changes are worth shipping regardless of how the spike goes. They
unblock several downstream experiments.

### PW1: Eliminate `.item()` from `nvfp4_lora/loss.py`

`loss.py:23` (chunked CE) and `loss.py:114` (Liger FLCE wrapper) both call
`(shift_labels != -100).sum().item()` unconditionally. Sonnet's audit
flagged this as a hard blocker for any CUDA graph capture; it's also a
host-sync that costs us ~50-200 µs per forward (minor but free to fix).

- **Change**: precompute `valid_count` as a 0-dim tensor; pass into the
  function; use `torch.clamp(valid_count, min=1).to(loss.dtype)` as divisor.
- **Verify**: forward parity with current implementation on a 100-step run
  (loss curves overlay).
- **Risk**: the `valid_count == 0` fast path uses a Python `if` on the
  scalar value. Keeping this branch in eager mode is fine for the spike;
  for Path 2 we'd refactor to a `torch.where`.

### PW2: Diagnostic instrumentation

Add per-region descriptor-delta logging via `torch.cuda.memory_stats()`
deltas around: prefix prefill, forward (excl. prefix), forward backward,
optimizer step, loss compute. Tag with the existing phase-tagged watchdog
mechanism. Output to `logs/spike_descriptor_attribution.jsonl`.

- **Why**: every subsequent experiment in the spike measures
  `num_device_alloc` per region. Without this we can't tell which
  intervention helped which region.
- **Risk**: instrumentation overhead. Keep it to ~20 measurement points per
  step, not per-layer.

---

## Experiment design principle

Each experiment has:
- **Question** it answers
- **Setup** (one-line CLI delta from current baseline)
- **Measure** (the killer metric — what we look at)
- **Decision criteria** (concrete thresholds for SUCCESS / PARTIAL / FAIL)
- **Time budget** (if exceeded, log result-so-far and move on)

Baseline for all experiments: `--training-mode cached_prefix_suffix
--train-suffix-len 2048 --prefix-chunk-len 8192 --max-len 262144 --stop-at-step
5` (5 steps gives us enough samples to bound run-to-run variance; we measure
mean num_device_alloc over steps 2-5, dropping step 1 as warmup).

**All experiments must compare against the same baseline run, captured fresh
at spike-start with PW1+PW2 applied.** Run the baseline twice to bound
eager-vs-eager noise.

---

## Experiment 1 — Allocator config sweep (day 1, ~6 hours)

### Question
Does changing the CUDA caching allocator's behavior, with no other changes,
shift the descriptor cliff?

### Setup
4 variants, each a 5-step run at s2048:
- E1a: `PYTORCH_CUDA_ALLOC_CONF=` (default, baseline)
- E1b: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- E1c: `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`
- E1d: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512`

### Measure
- `num_device_alloc` mean over steps 2-5
- `num_alloc_retries` (allocator pressure proxy)
- Peak CUDA reserved
- Throughput (tokens/sec)

### Decision criteria
- **SUCCESS**: any variant reduces `num_device_alloc` ≥40% AND lets us complete
  a 3-step run at s4096 without NVRM trip. → re-run the certification ladder
  at the winning variant.
- **PARTIAL**: ≥20% reduction in `num_device_alloc` but s4096 still trips. →
  log winner, carry forward as baseline for E2/E3.
- **FAIL**: <20% reduction across all variants. → fall through to E2.

### Time budget
6 hours total (4 runs × ~1h each + 30 min analysis).

---

## Experiment 2 — Allocator pre-warm and reuse (day 2, ~6 hours)

### Question
If we pre-warm the allocator with a "phantom forward" at the eventual target
suffix length before training starts, do subsequent forward+backward steps
reuse those allocations and avoid descriptor accounting growth?

### Setup
- E2a: At training start, run one forward+backward at `STATIC_SUFFIX_LEN=s2048`
  inside `torch.no_grad()` then a second forward+backward with grads,
  discarding both. Then start real training.
- E2b: As E2a but at `STATIC_SUFFIX_LEN=s4096`.
- E2c: As E2a but at `STATIC_SUFFIX_LEN=s8192`.

Apply on top of the winner from E1 (or E1a baseline if E1 failed).

### Measure
Same as E1, plus: descriptor count of the warm-up forward+backward (so we
know what we "paid" upfront).

### Decision criteria
- **SUCCESS**: real-training-step `num_device_alloc` drops ≥30% relative to E1
  baseline at s2048; s4096 certification ladder runs clean.
- **PARTIAL**: ≥10% reduction OR allocator stats show fewer retries even if
  num_device_alloc unchanged.
- **FAIL**: no reduction in num_device_alloc OR retries.

### Time budget
6 hours. If E2c fits in memory at all on GB10 (~ 130 GB UMA at warm time), it's
informative; if not, log and skip.

---

## Experiment 3 — `torch.compile(mode="reduce-overhead")` partial (day 3, ~8 hours)

### Question
Does `torch.compile`'s built-in CUDA graph mode (`reduce-overhead`) capture
clean enough of the trainable-suffix path to materially reduce descriptors,
without us doing manual capture?

### Setup
Three variants, each a 5-step run, on top of the best result from E1/E2:

- E3a: Compile only the trainable-suffix `forward` of one Mamba block
  (isolated, not full model). Sanity check — does compile even succeed on a
  patched Nemotron-H block?
- E3b: Compile the LoRA `nvfp4_lora.linear.NVFP4LoRALinear.forward`. This
  module is small, called many times, and is a known descriptor source via
  `_DequantLinear.backward`.
- E3c: Compile the full trainable-suffix forward+backward as a single
  `torch.compile(mode="reduce-overhead", fullgraph=False)` wrapper.

For each, log compile errors / graph breaks / recompilations.
`TORCH_LOGS=recompiles,graph_breaks` env var on.

### Measure
Same as E1, plus:
- Number of graph breaks reported by torch.compile
- Number of recompilations (a sign of dynamic shape pollution)
- Compile time (one-time cost)

### Decision criteria
- **SUCCESS** of E3c: ≥50% reduction in `num_device_alloc` AND ≤3 graph breaks
  AND throughput within 1.5x of eager. → certification ladder at s4k, s8k.
- **PARTIAL**: E3a OR E3b works isolated but E3c breaks. → suggests the
  compile path is theoretically viable but needs Path-2-style cleanup of
  custom autograd Functions before it lands. Logs feed into Path 2.
- **FAIL**: graph breaks in the dozens or recompilations on every step. →
  compile path is dead; only manual capture (Path 2) remains.

### Time budget
8 hours. `torch.compile` warmup alone can take 5-15 minutes per variant on a
model this size; budget for it.

---

## Experiment 4 — Isolated capture sanity check (day 4, ~6 hours)

### Question
With PW1 applied and using manual `torch.cuda.graph(...)` (NOT
`make_graphed_callables`) on a single Mamba block forward, can we get a
flat-descriptor replay? This is the absolute floor of feasibility for Path 2.

### Setup
- E4a: One Mamba block forward (no backward) at static shape s=512. Manual
  capture, 100 replays, measure `num_device_alloc` per replay.
- E4b: One Mamba block forward+backward at s=512. Same measurement.
- E4c: Same as E4b at s=2048.

For each, ensure: Triton kernels warmed (5 eager calls), no `.item()` in path,
no `nn.Module` hooks attached, no `torch.utils.checkpoint`. Pre-allocate
inputs/outputs/grad-outputs.

### Measure
- `num_device_alloc` per replay (the killer — if this isn't ~0, capture is
  buying us nothing)
- Numerical parity with eager (atol/rtol on output tensor)

### Decision criteria
- **SUCCESS**: replay descriptor delta is 0-5 events. → Path 2 is technically
  feasible at the single-block level; the work in Path 2 is "make this work
  for the full network", which is hard but bounded.
- **PARTIAL**: replay delta 5-50 events. → capture is reducing descriptor
  count but not eliminating it; investigate which kernel still allocates.
- **FAIL**: replay delta ≥50 events OR capture raises
  `cudaErrorStreamCaptureUnsupported`. → Path 2 is in much deeper trouble
  than the audit suggested; escalate.

### Time budget
6 hours. This experiment is also a "free" diagnostic for Path 2 regardless of
outcome.

---

## Experiment 5 — Stacked best-of (day 5, ~6 hours)

### Question
If E1+E2 each delivered PARTIAL gains, does stacking them produce SUCCESS?

### Setup
Run the certification ladder (s2k, s4k, s8k) with the best E1 config + the
best E2 warmup applied + (optionally) E3b LoRA module compiled. Real training
loop, watchdog armed, 3 clean steps required per rung.

### Measure
- Certified ladder rung reached
- num_device_alloc at each rung
- Step-time at each rung

### Decision criteria
- **SUCCESS**: s8k certifies clean. → spike concluded successfully, ship
  the stacked config as v1.2.
- **PARTIAL**: s4k certifies but s8k trips. → ship at s4k as v1.2, Path 2
  remains justified for s16k+.
- **FAIL**: no improvement over s2k baseline. → Path 2 with audit-revised
  plan is the only option.

### Time budget
6 hours.

---

## Decision tree

```
              ┌──────────────────────────────┐
              │ PW1+PW2 applied (always)     │
              └──────────────┬───────────────┘
                             │
                             ▼
                     ┌──────────────┐
                     │ E1: allocator│
                     │ sweep        │
                     └───┬──────┬───┘
                         │      │
                  SUCCESS│      │FAIL/PARTIAL
                         │      │
                         ▼      ▼
                ┌────────────┐  ┌──────────────┐
                │ Ship E1    │  │ E2: pre-warm │
                │ winner as  │  └──┬─────────┬─┘
                │ v1.2       │     │         │
                │ SPIKE DONE │SUCC.│         │FAIL/PARTIAL
                └────────────┘     ▼         ▼
                              ┌─────────┐   ┌──────────────────┐
                              │ Ship E2 │   │ E3: torch.compile│
                              │ winner  │   │ partial/full     │
                              │ as v1.2 │   └─┬─────────┬───┬──┘
                              │ SPIKE   │     │         │   │
                              │ DONE    │SUCC.│ PARTIAL │  FAIL
                              └─────────┘     ▼         ▼   ▼
                                   ┌──────────┐  ┌────────────────┐
                                   │ Ship E3c │  │ E4: isolated    │
                                   │ as v1.2  │  │ capture sanity │
                                   │ SPIKE    │  │ check           │
                                   │ DONE     │  └──┬──────────┬──┘
                                   └──────────┘     │          │
                                              SUCC/PARTIAL    FAIL
                                                    │          │
                                                    ▼          ▼
                                             ┌──────────┐  ┌─────────────┐
                                             │ E5: stack│  │ ESCALATE:   │
                                             │ best-of  │  │ Path 2 in   │
                                             │ + ladder │  │ deeper      │
                                             └──┬────┬──┘  │ trouble than│
                                                │    │     │ audit said. │
                                            SUCC/   FAIL   │ Stop spike. │
                                            PARTIAL  │     └─────────────┘
                                                │    │
                                                ▼    ▼
                                       ┌─────────┐ ┌─────────────────┐
                                       │ Ship    │ │ Path 2 with     │
                                       │ ladder  │ │ audit revisions │
                                       │ winner  │ │ is the only     │
                                       │ as v1.2 │ │ option. Commit  │
                                       │ SPIKE   │ │ 14-22 weeks.    │
                                       │ DONE    │ └─────────────────┘
                                       └─────────┘
```

## What "SUCCESS" means quantitatively

The spike is a success if at end of week we have **any one** of:

1. A shippable v1.2 config that certifies cleanly at s4k or higher with a
   watchdog-armed 3-step run. (Direct training-budget win.)
2. Decisive evidence that no off-the-shelf intervention shifts the cliff,
   accompanied by a measured descriptor attribution that points Path 2 at
   the right rewrites. (Path 2 derisk win.)

The spike is a **failure** if at end of week we have neither — i.e. we ran
out of time before any decisive conclusion. In that case the next step is
NOT "extend the spike" but "step back and reconsider the project framing"
with the user.

## What this spike will NOT tell us

- It will not validate Path 2's full architecture. E4 only tests one block.
- It will not surface the MoE static-buffer infeasibility (already known
  from Sonnet's audit).
- It will not improve numerical quality of training. Only descriptor
  pressure.
- It will not test multi-Spark or BF16-master alternatives (those are
  Option B / Option C, separate efforts).

## Risks during the spike

| Risk | Likelihood | Mitigation |
|---|---|---|
| `torch.compile` warmup takes >2h on full model | Med | Time-box E3c at 2h; if not done, log graph-break count and move on. |
| Allocator change introduces silent NaN / training instability | Low | Each variant run gets a loss-curve check; abort if loss diverges from baseline. |
| `expandable_segments` reduces num_device_alloc but the cliff was actually NVRM-level not torch-allocator-level | High | E1 measure includes journalctl NVRM scan; if num_device_alloc drops but NVRM still fires, we learn something different but still useful. |
| E4 isolated capture succeeds but real model fails for orthogonal reasons | Med | E4 is feasibility floor, not predictor. Document explicitly. |
| Hardware unavailable for full week (CUDA recovery needed, etc.) | Low | Spike has slack built in (6h budgets where ~3h is real work). |

## Deliverables

By end of day 5:

- `docs/spikes/path1_results.md` — one-pager with: per-experiment outcome,
  final decision, recommended next action.
- `docs/LONG_CONTEXT_EXPERIMENTS.md` — entries SUPER-LC-100 through ~LC-115
  for each variant tested.
- `logs/spike_descriptor_attribution.jsonl` — raw measurement data.
- One of: (a) merged v1.2 PR with shipping config OR (b) green-light brief
  for Path 2 with audit-revised plan and concrete descriptor attribution
  feeding milestone definitions.

## Out-of-scope guards

- No model changes (no LoRA rank/target changes, no loss reweighting).
- No HF transformers version bump.
- No new dependency installs except: `triton` upgrade if E4 demands it.
- No multi-Spark experimentation.
- No checkpoint format changes.
