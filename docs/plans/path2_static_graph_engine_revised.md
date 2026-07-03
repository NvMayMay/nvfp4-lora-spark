# Path 2 (REVISED): Static CUDA Graph Training Engine — audit-incorporated

**Status:** STANDBY — only proceed if Path 1 (`path1_alternatives_spike.md`)
concludes that no cheap intervention shifts the cliff.
**Timeline (revised after audit):** 14-22 weeks single-engineer focused work.
**Target:** s32k certified, **s40-48k honest stretch** (not s64k — that claim
was wrong, see audit).

## Why this plan was rewritten

Both adversarial audits of the prior `static_cuda_graph_engine.md` returned
REVISE with five convergent blockers. This revision bakes them in as
mandatory pre-M1 work, not as risks-to-manage.

## What changed vs prior plan

| Prior plan | Revised plan |
|---|---|
| `make_graphed_callables` as primary path | **Manual `torch.cuda.graph(...)` with explicit input/output/grad buffers** as primary path. `make_graphed_callables` is unusable due to reentrant checkpoint + `torch.autograd.grad` incompatibility. |
| MoE per-expert static buffers | **Per-expert sub-graph dispatch**: one captured graph per expert, replayed only for active experts. ~880 descriptors worst case (22 experts × 40 layers). |
| s64k stretch | **Dropped. s40-48k is the honest byte-bound ceiling.** |
| 6-10 week timeline | **14-22 weeks**, with most of the new budget in pre-M1 refactors and the MoE redesign. |
| `.item()` / `torch.empty_like` ignored | **Pre-M1 mandatory refactor of three files** before any capture work. |

## Pre-M1: mandatory refactors (Weeks 1-4)

These are not part of M1's GO/NO-GO; they are **gating prerequisites** that
must merge cleanly before any capture experiments begin.

### Pre-M1a: Loss function refactor (~1 week)

**Files**: `nvfp4_lora/loss.py`

**Changes**:
- Eliminate `.item()` from `_ChunkedFrozenLMHeadCE.forward` (line 23) and
  `liger_fused_lm_head_ce` (line 114). Pass `valid_count` as a 0-dim tensor
  precomputed in eager before the captured region.
- Replace the `valid_count == 0` Python branch with a
  `torch.where(valid_count == 0, zero_loss_path, normal_loss_path)`
  formulation.
- Replace `(flat_hidden.float() * 0.0).sum()` autograd-preserving zero with
  an explicit `torch.zeros((), requires_grad=True)` that has a backward via
  `flat_hidden.sum() * 0.0`.

**Verify**: numerical parity vs current loss on a 100-step run, loss curves
overlay within bf16 noise.

### Pre-M1b: NVFP4 dequant workspace argument (~1 week)

**Files**: `nvfp4_lora/dequant.py`, `nvfp4_lora/linear.py`

**Changes**:
- Add optional `out: torch.Tensor | None = None` parameter to
  `dequantize_nvfp4_weight`. If provided, write into it; if None, allocate
  as today.
- Modify `_DequantLinear.backward` (linear.py:60-66) to accept a workspace
  buffer for `W_bf16` recomputation. For the captured path, this buffer is
  pre-allocated once per unique weight shape.
- Add a `dequant_workspace_pool` to `loader.py` that allocates one buffer
  per unique `(out_features, in_features)` shape across the model.

**Verify**: parity test that `dequantize_nvfp4_weight(W, S, S2)` and
`dequantize_nvfp4_weight(W, S, S2, out=preallocated)` produce bit-identical
output.

### Pre-M1c: Mamba2 SSD kernel workspace patches (~1-2 weeks)

**Files**: `mamba_ssm/ops/triton/ssd_combined.py` (vendored fork)

**Changes**:
- Fork the upstream `mamba_ssm` package into `vendored/mamba_ssm/` (already
  partially done? check repo state).
- Modify `_mamba_chunk_scan_combined_bwd` (lines 422-435) to accept
  pre-allocated `dx_workspace`, `ddt_workspace`, `dB_workspace`,
  `dC_workspace` parameters. Today these are `torch.empty_like(...)`
  allocations inside backward.
- Modify `alloc_tile_workspace` (determinism.py:85-88) to accept a
  pre-allocated buffer parameter. The determinism flag still gates whether
  contents are zeroed (`buffer.zero_()`) or untouched.
- Pre-warm `triton.autotune` on `_chunk_scan_bwd_ddAcs_stable` and other
  autotuned kernels with the exact shapes/dtypes/strides used in capture.
  Compile-and-discard a dummy call before the capture window.

**Verify**: per-step backward descriptor count drops to <50 events for one
Mamba block at s=2048. (Today: empirically dominant contributor to the 0.4
events/token slope.)

### Pre-M1d: Hook-free training infrastructure (~3 days)

**Files**: `train/train_super_nvfp4.py`

**Changes**:
- Remove or guard the routing census hooks (line 962) so they're never
  attached during capture-enabled steps.
- Add a `capture_safe_mode` flag that enforces: no forward/backward hooks
  registered, autocast cache disabled.

**Verify**: `nn.Module._forward_hooks` is empty for every module in the
captured path.

### Pre-M1 GO/NO-GO gate

All four refactors must merge with parity tests passing. If any of pre-M1a/b/c
fails (e.g. Mamba kernel patches break the existing eager path), escalate
before continuing.

---

## M1: Descriptor attribution + alternatives reconfirm (Week 5)

This milestone is mostly redundant if Path 1 was completed; in that case M1
collapses to ~2 days of confirming the Path 1 measurements still hold after
the pre-M1 refactors.

**Goal**: precise per-region descriptor budget for the post-refactor
baseline. The refactors themselves may have moved descriptors around (e.g.
Pre-M1b should drop the ~40k dequant temp count substantially).

**Tasks**:
- Re-run the descriptor attribution from Path 1 PW2 on the refactored code.
- Confirm Mamba backward descriptor count dropped per Pre-M1c.
- Update attribution report.

**GO gate**: ≥60% of remaining per-step descriptor budget is in regions
where Path 1 E4 demonstrated flat-replay capture (or, if Path 1 was skipped,
demonstrate on a 2-layer toy here).

---

## M2: Per-expert MoE sub-graph capture (Weeks 6-9)

**Goal**: prove that the MoE FFN can be captured as N=512 small graphs and
dispatched per-active-expert, with the dispatch happening in eager between
replays.

**Approach**:
- For each expert, capture one static-shape graph with input/output buffers
  sized to `MAX_TOKENS_PER_EXPERT_PER_LAYER` (configurable, default 512).
  At static shape, with pre-warmed Triton kernels, the capture should be
  one-time and the replay descriptor-flat.
- In the eager router, after top-k routing yields per-token expert
  assignments, group tokens by expert. For each expert with active tokens,
  copy tokens into its input buffer (with zero-padding to capture cap),
  replay its graph, read its output buffer.
- Per-step descriptor cost: ~22 active experts × ~40 MoE layers = ~880
  descriptors for MoE total (vs prior plan's infeasible 388 GB pool).

**Tasks**:
- Implement per-expert capture helper.
- Implement eager dispatch loop.
- Measure: descriptor count, throughput, parity vs eager.

**GO gate**: per-expert replay descriptor delta <5 events, end-to-end MoE
descriptor count <1000, throughput within 1.5x of eager.

**Risk**: if active-expert count per token is much higher than 22 effective
(e.g. tail tokens routing to obscure experts), descriptor count creeps up.
Mitigate by aggregating low-traffic experts into a "leftover" graph.

---

## M3: Mamba block capture + full forward (Weeks 10-12)

**Goal**: capture the full trainable-suffix forward (89 layers: Mamba +
MoE FFN + attention) as a single graph.

**Tasks**:
- Capture each Mamba block forward+backward as its own static-shape graph.
- Capture attention layers (with `is_causal=True`, no materialized mask).
- Compose with MoE per-expert dispatch from M2.
- Compose with the lm_head + chunked CE (post-Pre-M1a refactor).
- Verify forward loss parity with eager: bf16 absolute tolerance based on
  measured eager-vs-eager variance (audit point: don't pick arbitrary atol;
  measure baseline noise first).

**GO gate**: forward loss within `max(5x eager-eager-p99, 1e-3)` at s=8192;
descriptor count per step <500.

---

## M4: Backward capture as its own engineering project (Weeks 13-16)

**Goal**: capture the full backward such that LoRA gradients are produced
inside the graph.

**Tasks**:
- Manual capture of backward with explicit grad-output input buffers and
  grad-input output buffers. No `make_graphed_callables`.
- Pre-allocate workspaces for all the pre-M1c-patched Mamba backward calls,
  Pre-M1b-patched dequant calls, and any other custom autograd Functions.
- Resolve the grouped-checkpointing question: either rewrite the
  checkpointing path to be capture-clean, or disable checkpointing within
  capture and verify byte budget at the chosen STATIC_SUFFIX_LEN.
- Gradient lifecycle: `param.grad` allocated once before capture, replay
  writes in place; zero via `grad.zero_()` outside the captured region.
- Parity test: 2-microstep gradient accumulation comparison against eager.

**GO gate**: LoRA `.grad` parity within `max(5x eager-eager-p99, 1e-2)` for
each tensor (per-tensor max/mean/p99 reported); total per-step descriptor
count <600.

**Risk**: this is the highest-risk milestone. The audit suggests this should
be its own multi-week project. If the gate fails, fall back to forward-only
capture (per-token descriptor savings ≈ forward-share of the 0.4 events/tok
slope, likely 20-30%).

---

## M5: Integration + watchdog + certification (Weeks 17-20)

**Goal**: wire the graphed engine into the main training loop, certify the
ladder.

**Tasks**:
- CLI flag `--graphed-suffix-engine` defaulted OFF, eager path preserved.
- Watchdog integration: segment captures into ~10-layer groups with eager
  sync points between them. Watchdog can poll between segments.
- Run certification ladder at s4k, s8k, s16k, s32k.
- Update README's certified-configurations table.
- Long-context experiment journal entries (SUPER-LC-200+).

**GO gate**: s32k certified (three watchdog-clean steps, journalctl-clean
for NVRM at each rung); per-step throughput within 2x of eager s2048.

---

## M6: Stretch and slip buffer (Weeks 21-22)

Buffer for milestones sliding right. If M5 lands clean, attempt s40k
certification.

---

## Failure modes (post-audit)

| Risk | Likelihood | Fallback |
|---|---|---|
| Pre-M1c Mamba kernel patches don't merge cleanly (upstream rejects, kernel internals more complex than expected) | High | Vendor a permanent fork; accept maintenance burden |
| Per-expert dispatch in M2 has too-high overhead in the eager router | Med | Batch graph replays via `cudaGraphInstantiate` reuse; coalesce experts into super-experts |
| M4 backward capture fundamentally infeasible due to autograd graph dynamism | Med | Ship forward-only capture; document descriptor reduction is ~20-30% not 80% |
| Grouped checkpointing rewrite leaves bytes-at-s32k untenable | Med-High | Cap target at s16k certified |
| `cudaErrorStreamCaptureUnsupported` mid-capture from an unidentified Python-side allocator call | Med | Bisect by capturing progressively smaller regions until failure point identified; patch |
| Watchdog blind to NVRM mid-segment | Low (segmented capture mitigates) | Tighter segments + post-segment health check |

## Effort estimate (revised)

- **Engineering**: 14-22 weeks single-engineer focused.
- **Compute**: $0 marginal (single Spark).
- **Risk-adjusted ceiling**: s16k certified is the realistic floor of
  success; s32k is the target; s40-48k is honest stretch (byte-bound).

## Decision points for user

- **Slack on Mamba upstream**: do we vendor a fork (1 week) or attempt
  upstream patches (4-6 weeks calendar including review)?
- **MoE dispatch overhead**: if M2 dispatch eats throughput, do we accept
  the regression or pivot to coalesced super-experts?
- **Pause point**: end of Pre-M1 (week 4) is the cheapest abort. ~4 weeks
  of code refactors that are independently useful for non-graph work.
- **Hard pause**: end of M2 (week 9). If MoE per-expert dispatch fails, the
  entire MoE strategy needs rethinking and that probably means abandoning.

## Relationship to Path 1

If Path 1 succeeded (cliff shifted via cheap intervention): this plan does
not execute. Ship Path 1 results as v1.2.

If Path 1 produced PARTIAL gains: this plan executes with the Path 1
winner baked into the baseline. Pre-M1 still runs because the refactors
are mandatory regardless.

If Path 1 produced no gain: this plan executes as written, with the
descriptor attribution from Path 1 PW2 feeding M1.
