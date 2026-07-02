# Plan: Static CUDA Graph Training Engine for NVFP4 LoRA on GB10

**Status:** DRAFT — awaiting adversarial audit
**Author:** v1.1 ship + post-mortem (May 2026)
**Goal:** Break the CUDA descriptor-pool ceiling that caps trainable-suffix length
at ~s2048-s3072 on Nemotron-3 Super 120B NVFP4. Target post-engine ceiling:
**s32k+ certified, s64k+ fit-tested**, on the same single-Spark hardware.

---

## 1. Problem statement

`v1.1` ships at certified s2048 because of a **CUDA descriptor-pool ceiling**, not a
byte ceiling. Empirically (`docs/LONG_CONTEXT_EXPERIMENTS.md`):

- Backward descriptor count grows ~0.4 events per suffix token (LC-061 onward).
- At s2048 the per-step `num_device_alloc` baseline is ~800 + ~820 per-token = ~1620.
- Above ~3000 suffix tokens the NVRM `NV_ERR_NO_MEMORY` cliff fires at
  `mem_desc.c:1359` even with abundant unified memory.
- Single-component swaps (save_on_cpu, Mamba chunk_size, Liger FLCE) move bytes
  but not descriptors.

This means the descriptor cliff is the *only* lever that matters for breaking
through long context on this hardware. Bytes are already comfortable: at s2048,
peak CUDA reserved is ~95-105 GB out of 130 GB UMA.

## 2. Why static CUDA graphs are the right fix

A `torch.cuda.CUDAGraph` replay issues **one** descriptor for the entire
captured region regardless of how many kernel launches it contains. Pre-allocated
workspace tensors inside the captured region are reused on replay; no new
descriptor accounting fires.

Concretely: if we capture the trainable-suffix forward + backward as a single
static-shape graph, the per-step descriptor cost drops from ~1620 to a constant
that depends only on the optimizer and any out-of-graph code (dataloader,
loss reduction, watchdog tags). Suffix length becomes bounded by available
memory bytes, not by descriptor accounting.

This is the technique NVIDIA's own training stack (Megatron-LM / NeMo) uses
internally for NVFP4-quantized model training. We're re-implementing in
HF-Trainer territory what they implemented in Megatron.

## 3. Scope and non-goals

**In scope:**
- Static CUDA graph capture of the trainable-suffix forward pass.
- Static CUDA graph capture of the corresponding backward pass.
- Pre-allocation of every workspace tensor used during the step.
- Static padding of all dynamic shapes (suffix length, MoE dispatch, Mamba
  chunk grid, attention masks).
- Integration with the existing `cached_prefix_suffix` training mode and the
  existing watchdog/profile infrastructure.
- Numerical parity verification against eager mode.
- Long-context certification ladder up to at least s32k.

**Explicitly out of scope:**
- Multi-Spark distributed training (option B in the brief — separate effort).
- Replacing LoRA with full fine-tuning (option C in the brief).
- Capturing the cached-prefix prefill (one-shot, variable length, runs once,
  not worth capturing).
- Capturing the optimizer step (Adafactor row/col stats are tiny; eager step
  outside the graph is fine and simpler).
- New base-model support beyond Nemotron-3 Super 120B (Nano/Qwen3.5/Mistral
  ports come later if this works).

## 4. Architecture sketch

```
┌────────────────────────────────────────────────────────────────────────┐
│ Training step (one optimizer update)                                     │
│                                                                          │
│  Eager (out-of-graph):                                                   │
│    ├── Dataloader, tokenizer, batch shaping                              │
│    ├── set_current_phase("step:fwd")                                     │
│    └── Pad suffix to STATIC_SUFFIX_LEN (e.g. 4096, 8192, 16384, 32768)   │
│                                                                          │
│  Captured graph (one descriptor per replay):                             │
│    ├── Forward: Mamba/MoE/attention stack with pre-allocated buffers    │
│    │    over [prefix K/V + Mamba SSM cache] + [static-padded suffix]    │
│    ├── Chunked frozen CE loss on lm_head                                 │
│    └── Backward: gradient flow through LoRA params only                  │
│                                                                          │
│  Eager (out-of-graph):                                                   │
│    ├── Loss scalar reduction + logging                                   │
│    ├── Adafactor optimizer step on LoRA params                           │
│    ├── set_current_phase("step:idle")                                    │
│    └── Watchdog tick                                                     │
└────────────────────────────────────────────────────────────────────────┘
```

Inputs to the captured region: pre-existing prefix K/V slab + Mamba state
(allocated and populated once during prefill), static-padded suffix tokens,
static-padded suffix mask, and LoRA params (whose `.data` is updated by the
eager optimizer between replays).

Outputs from the captured region: loss scalar + LoRA `.grad` tensors (in
place on the param objects).

## 5. Technical challenges and proposed handling

### 5.1 Variable suffix length

Suffix length varies per document (1..STATIC_SUFFIX_LEN). Graph capture
requires a single static shape.

**Approach**: pad to `STATIC_SUFFIX_LEN` with a mask. Loss reduction uses
`valid_count = mask.sum()`; the `valid_count == 0` fast path in
`nvfp4_lora/loss.py` (round-2 fix) already preserves the autograd graph.

**Padding cost**: compute is fully padded, so per-step work scales with
`STATIC_SUFFIX_LEN` not actual suffix length. For docs much shorter than the
static cap this wastes compute. Mitigation: train at multiple static caps
(e.g. capture three graphs: s2048, s8192, s32768) and dispatch per-doc to
the smallest one that fits the doc's suffix.

### 5.2 MoE top-k routing

Nemotron-3 Super uses k=22 of 512 experts per token. The dispatch pattern
(which token → which expert) is data-dependent. Both the existing sparse
dispatch path and a "dense fake-routing" path are candidates.

**Approach A (default)**: keep the existing sparse-no-one-hot dispatch but
allocate the routing index tensor and per-expert work-tensors at
`MAX_TOKENS_PER_EXPERT = STATIC_SUFFIX_LEN * k / num_experts * SAFETY_FACTOR`
size, pre-allocate before capture, and use `torch.cuda.synchronize` + replay
to verify the static buffer is large enough. Mask out unused slots.

**Approach B (fallback)**: dense-route — all experts fire for all tokens,
weighted by a soft router output. This eliminates the data-dependent
dispatch entirely but multiplies expert FFN compute by `num_experts/k =
23x`. Probably unaffordable.

**Risk**: if the per-expert workload is wildly uneven across steps (some
experts get 0 tokens, others get many), capture-time buffer sizing has to
target the worst case. Need to measure the actual distribution first.

### 5.3 Mamba2 SSD chunk-scan

`mamba_chunk_scan_combined` iterates over `chunk_size`-token chunks. Internal
workspace allocations happen per chunk. **However**, if we hold
`chunk_size` and `seq_len` static, the chunk loop iterates a fixed number of
times, and each iteration's internal allocations are deterministic in shape
— graph capture should record them as one-time allocations and replay them
without re-firing descriptor accounting.

**Risk**: the SSD kernel may use `torch.empty(...)` with shape derived from
runtime tensors. Need to inspect the Triton or CUDA source. If yes, must
either patch the kernel to take a workspace tensor argument or pre-call it
once before capture so the cache holds the allocation.

### 5.4 Attention causal mask

Mask shape depends on `prefix_len + suffix_len`. With static suffix length
and static prefix length (per cached-prefix slab) this is already static.
The existing `--sdpa-causal-no-mask` path uses `is_causal=True` instead of
materializing the mask, which we should keep.

### 5.5 LoRA delta computation

`(lora_B @ lora_A @ x) * (alpha/r)` per target module. With pooled loader
buffers (round-2 fix) the `lora_A` and `lora_B` views are already
pre-allocated. The matmul intermediate `(lora_A @ x)` is a new allocation
per layer per step. Pre-allocate per-layer scratch buffers before capture.

### 5.6 Chunked frozen CE loss

Already chunked over `loss_chunk_tokens` (default 512). Each chunk
allocates a small intermediate. With static suffix length the chunk count
is fixed; the per-chunk allocations are static-shape; capture replays them
in place.

### 5.7 Backward graph capture

`torch.cuda.graphs.make_graphed_callables` handles the forward+backward
capture as one unit if used correctly. But:
- Grouped checkpointing (`enable_grouped_layer_checkpointing` in
  `train_super_nvfp4.py`) inserts recompute logic that may not capture
  cleanly. Need to verify or replace with a static recompute path inside
  the captured region.
- Mixed precision (bf16 forward, fp32 master in optimizer) is fine —
  capture happens at bf16 within the step.

**Approach**: capture the per-layer forward + backward as nested graphs,
then assemble. Fall back to capturing forward-only (and running backward in
eager) if the unified capture fails — partial gain but not zero gain.

### 5.8 Cached prefix interaction

Prefix prefill is one-shot, variable length, runs at training start (and
never during a step). It stays in eager mode. The prefix K/V slab and
Mamba SSM cache are inputs to the captured graph, not internal allocations.

### 5.9 Watchdog and profiling

Phase-tagged watchdog has been validated through v1.1. The
`set_current_phase()` calls happen in eager between captures; this is
fine. `torch.cuda.memory_stats()` polling needs to be done outside the
captured region — inside the graph the stats would only reflect capture-
time allocations.

## 6. Milestones and derisking gates

Each milestone has a **GO/NO-GO gate**. NO-GO at any gate means re-plan,
not push through.

### M1 — Diagnostic groundwork (Week 1-2)

**Goal**: precisely attribute the per-step descriptor budget to specific
code regions.

**Tasks**:
- Add fine-grained `num_device_alloc` deltas around: prefix prefill,
  per-layer forward, per-chunk Mamba SSD scan, per-expert MoE FFN, lm_head
  CE, backward recompute, optimizer step.
- Build a one-shot allocation-attribution report at s2048.
- Run `make_graphed_callables` on a single dummy 2-layer Nemotron block
  with static-shape inputs as a sanity check (does HF transformers + our
  patches even capture cleanly?).
- Verify Mamba SSD chunk-scan internal allocation behavior (source-dive or
  empirical: capture, then check descriptor count on replay vs first call).

**GO gate**: at least 60% of the per-step descriptor budget must be
attributable to code regions that are theoretically captureable (i.e. NOT
in eager dataloader / optimizer / out-of-step code). If <40%, the whole
plan has limited ceiling — escalate to user before continuing.

### M2 — Static-shape pad of one layer (Week 3-4)

**Goal**: prove a single Mamba block + a single MoE block can be captured
with static shapes.

**Tasks**:
- Implement static-shape pad helpers for: suffix length, MoE dispatch,
  Mamba chunk grid.
- Pre-allocate workspace pool for one Mamba block and one MoE block.
- Capture each as a `make_graphed_callables` graph.
- Bench `num_device_alloc` per replay; bench wall-clock vs eager.

**GO gate**: a single captured Mamba + MoE block round-trip must show
≥10x reduction in descriptor count on replay versus first call, with
numerical parity (tensor `.allclose()` with atol=1e-3, rtol=1e-3 in bf16).

### M3 — Full forward graph (Week 5-6)

**Goal**: capture the full trainable-suffix forward (89 layers) for the
chosen STATIC_SUFFIX_LEN.

**Tasks**:
- Extend the pre-allocated pool to all layers.
- Capture forward only (no backward yet).
- Verify forward loss matches eager forward loss within fp tolerance.
- Run at multiple STATIC_SUFFIX_LEN values (2048, 8192, 16384) and confirm
  descriptor count stays roughly constant per replay.

**GO gate**: forward loss parity to within 5e-3 absolute at s8192;
descriptor count per replay <500 (vs. ~1600 baseline at s2048).

### M4 — Backward capture (Week 7-8)

**Goal**: capture full forward + backward, end-to-end.

**Tasks**:
- Either capture fwd+bwd as one unit via `make_graphed_callables` or use
  manual capture with `torch.cuda.graph(...)` and explicit `loss.backward()`
  inside.
- Resolve grouped-checkpointing interaction (likely: disable checkpointing
  inside the captured region and rely on byte budget being adequate at the
  cap).
- Verify LoRA `.grad` tensors match eager `.grad` to within fp tolerance.
- Run one full step at s8192 and measure descriptor count.

**GO gate**: per-step descriptor count <600 (vs. ~1620 at s2048 baseline,
which is now released by static graph capture). LoRA gradient parity
within atol=1e-2.

### M5 — Integration + long-context certification ladder (Week 9-10)

**Goal**: integrate the graphed engine into the training loop, run the
watchdog-certified ladder up to s32k.

**Tasks**:
- Wire graphed step into the main training loop, gated by a CLI flag
  (`--graphed-suffix-engine`).
- Eager fallback preserved (CLI default OFF until verified).
- Run certification at s4k, s8k, s16k, s32k — three watchdog-clean steps
  at each.
- Add to `docs/LONG_CONTEXT_EXPERIMENTS.md`.
- Update README's certified-configurations table.

**GO gate**: s32k certified (three clean steps with watchdog armed,
journalctl-clean for NVRM errors); throughput within 2x of eager at s2048
(graphed engine should be FASTER per token at s32k due to amortization).

### M6 — Stretch: cached-prefix swap and multi-cap dispatch (Week 11-12)

**Goal**: support multiple graph caps with per-doc dispatch and validate
cached-prefix invalidation strategy.

This is buffer for the milestones above sliding right.

## 7. Failure modes and fallbacks

| Risk | Likelihood | Fallback |
| --- | --- | --- |
| Mamba SSD kernel internally allocates per chunk regardless of capture | Med | Patch kernel to take workspace tensor; if not feasible, capture forward only and run Mamba backward eagerly |
| MoE dispatch can't be sized with static buffer (per-expert load too variable) | Med | Capture per-expert sub-graphs, route in eager between graphs (partial gain) |
| Backward fails to capture due to autograd recompute interactions | Med-High | Capture forward only, run backward in eager. Half the gain but still real. |
| Grouped checkpointing forces dynamic allocation inside capture | High | Disable checkpointing inside capture; rely on bytes (we have headroom at s32k) |
| HF transformers patches don't compose with `make_graphed_callables` | Low-Med | Rewrite the trainable-suffix forward as a self-contained module that doesn't go through HF's wrapper |
| Descriptor count drops but throughput regresses (capture overhead per step) | Low | Capture once, reuse across thousands of steps; capture cost amortizes |
| s32k still hits a NEW cliff (allocator workspace, not descriptors) | Med | Investigate; bytes at s32k should be ~120GB, within UMA budget. Static graphs avoid descriptor accounting but not byte budget. |

## 8. Test methodology

Three levels of test, run at every milestone:

1. **Smoke**: one captured step produces a finite loss without crashing.
2. **Parity**: captured-step loss and gradients within fp tolerance of
   eager equivalent on identical inputs. Tolerance: bf16 means atol=1e-3,
   rtol=1e-3 on loss; atol=1e-2 on LoRA `.grad` (looser due to backward
   compounded rounding).
3. **Throughput + descriptor budget**: 50-step run, measure mean
   `num_device_alloc` per step (the killer metric — if it doesn't drop by
   ≥3x, the work is wasted) and wall-clock per token.

Long-context certification at M5 follows the existing v1.1 protocol:
three clean watchdog-armed steps + journalctl scan for NVRM at each rung.

## 9. Concrete deliverables

By milestone:
- M1: `scripts/profile_descriptor_attribution.py` + report; sanity-check
  notebook proving graph capture works on a toy.
- M2: `nvfp4_lora/static_capture/` module skeleton; per-block capture
  helpers.
- M3: `--graphed-forward-only` CLI flag in `train_super_nvfp4.py`;
  forward parity test in `smoke_tests/`.
- M4: `--graphed-step` CLI flag (full fwd+bwd); gradient parity test.
- M5: `--graphed-suffix-engine` certified at s32k; new README section;
  new `LONG_CONTEXT_EXPERIMENTS.md` entries SUPER-LC-100+ for the graphed
  ladder.

## 10. Effort estimate

- **Engineering**: ~6-10 weeks single-engineer focused work.
- **Compute**: ~$0 marginal (single Spark, already owned).
- **Risk-adjusted ceiling**: s16k certified is the realistic floor of
  success; s32k is the target; s64k is stretch. If all milestones land
  cleanly, s64k+ is plausible because bytes are the only remaining
  constraint.

## 11. Decision points for user

- **Cap target**: do we commit upfront to s8192 / s16384 / s32768 as the
  static capture cap, or build the multi-cap dispatcher from M3 onwards?
  (Multi-cap adds ~1 week.)
- **Backward capture**: if M4 backward capture fails, do we ship
  forward-only graphed (real but partial gain) or push another sprint
  trying to make backward work?
- **Pause point**: M1 GO gate is the cheapest place to abort if the
  descriptor attribution comes out unfavorable. ~2 weeks of engineering
  spent before commit-or-walk decision.
