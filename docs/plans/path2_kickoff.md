# Path 2 Kickoff — Pre-M1 schedule + M0 gate

**Status:** ACTIVE
**Decision:** β (skip the standalone spike, start Pre-M1 refactors that are mandatory regardless). Half-day `cudaMallocAsync` + `--prefix-chunk-len` probe runs as a side-channel during week 1.
**Real go/no-go:** the M0 gate at the end of Pre-M1c (Mamba kernel patches), not anything earlier.

## Why this kickoff exists

Two adversarial audit rounds (Path 2 base plan + Path 1 spike plan, 5 reviewers
total) converged on:

1. Single-component swaps don't move the descriptor cliff (LC-062 through LC-069
   already proved this).
2. The cheap-alternatives surface area is much smaller than the Path 1 spike
   assumed: `expandable_segments:True` is already the baseline
   ([train_super_nvfp4.py:28](../../train/train_super_nvfp4.py#L28)), and
   `torch.compile` graph-breaks on the
   [monkey-patched Mamba forward](../../train/train_super_nvfp4.py#L819).
3. The Pre-M1 refactors in `path2_static_graph_engine_revised.md` are
   independently valuable (eliminate `.item()` syncs, pre-allocate dequant
   workspace) regardless of whether Path 2 ever ships.
4. The real "does Path 2 work" question is decided by Pre-M1c (Mamba
   kernel workspace patches), not by anything cheaper.

The kickoff therefore: starts Pre-M1 immediately, runs a half-day side-channel
probe of two genuinely-untested allocator levers, and treats the end of
Pre-M1c as a hard go/no-go gate.

## Schedule overview

```
Week 1                Week 2                Week 3                Week 4
├── Pre-M1a (loss)    ├── Pre-M1b (dequant) ├── Pre-M1c (Mamba)   ├── Pre-M1c (cont.)
│   ~1 week           │   ~1 week           │   ~1-2 weeks        │   + Pre-M1d (hooks)
│                     │                     │
└── α-lite probe      └── (slack)           └── (slack)           └── M0 GATE
    ~half a day                                                       decision point
```

If Pre-M1c finishes early (single week), Pre-M1d slides into week 3 and
week 4 is M0 evaluation + writeup.

## Pre-M1a — Loss function refactor (Week 1, ~3-5 days)

**Files:** `nvfp4_lora/loss.py`

### Concrete changes

1. **`_ChunkedFrozenLMHeadCE.forward` ([loss.py:23](../../nvfp4_lora/loss.py#L23)):**
   - Drop `valid_count = int((shift_labels != -100).sum().item())`.
   - Add a `valid_count: torch.Tensor` parameter to `forward`. Caller must
     pass it as a 0-dim tensor precomputed in eager.
   - Save `valid_count` (the tensor, NOT a Python int) into `ctx`. Sonnet's
     audit caught that `ctx.denom = denom` at
     [loss.py:42](../../nvfp4_lora/loss.py#L42) is currently a Python int;
     this must change to `ctx.save_for_backward(valid_count_tensor)` or
     equivalent.

2. **Denominator clamping in `_ChunkedFrozenLMHeadCE.forward` ([loss.py:24](../../nvfp4_lora/loss.py#L24)):**
   - Today `denom = max(1, valid_count)` is a Python-scalar guard against
     div-by-zero in the chunked loop. `_ChunkedFrozenLMHeadCE.forward`
     does NOT have an explicit `if valid_count == 0: return ...` fast path
     (Sonnet's audit confirmed this — that branch only exists in
     `liger_fused_lm_head_ce`).
   - Replace with `denom_tensor = torch.clamp(valid_count_tensor, min=1).to(loss_dtype)`,
     compute the chunked loop unconditionally, and divide by `denom_tensor`
     at the end. No graph-structure-changing branch needed for this
     function.

3. **Backward ([loss.py:47-81](../../nvfp4_lora/loss.py#L47-81)):**
   - Retrieve `valid_count` from saved tensors; use
     `divisor = torch.clamp(valid_count, min=1).to(grad_output.dtype)` as the
     denominator in place of the current Python-int `ctx.denom`.

4. **`liger_fused_lm_head_ce` ([loss.py:114](../../nvfp4_lora/loss.py#L114)):**
   - Same `.item()` removal.
   - The actual `if valid_count == 0: return (flat_hidden.float() * 0.0).sum()`
     fast path at [loss.py:115-120](../../nvfp4_lora/loss.py#L115-120) needs
     a `torch.where` treatment: compute the normal path unconditionally,
     also compute `(flat_hidden.float() * 0.0).sum()` unconditionally, and
     select via `torch.where(valid_count_tensor == 0, zero_path, normal_path)`.

5. **Call sites in `train_super_nvfp4.py`:**
   - `_lm_head_ce` dispatcher (the entry point all callers use) is the
     right place to compute `valid_count_tensor = (labels != -100).sum()`
     once and pass it through to the autograd Function.
   - Verified call sites that flow into `_lm_head_ce` (Sonnet round-2
     audit added the line-1832 site that the earlier enumeration
     missed):
     - `cached_prefix_suffix_loss` ([train_super_nvfp4.py:1150](../../train/train_super_nvfp4.py#L1150))
     - `run_cached_prefix_compare` ([train_super_nvfp4.py:1176](../../train/train_super_nvfp4.py#L1176)
       and [:1204](../../train/train_super_nvfp4.py#L1204)) — two calls,
       one for each of the two loss paths it compares.
     - Full-sequence training path early-return variant ([train_super_nvfp4.py:1722](../../train/train_super_nvfp4.py#L1722))
     - Full-sequence training path inside the epoch loop ([train_super_nvfp4.py:1832](../../train/train_super_nvfp4.py#L1832))
   - **Before merge**: grep for all `_lm_head_ce(` invocations and verify
     the count is 5 (any additional caller introduced since this audit
     must be reviewed for the new API).
   - If `_lm_head_ce` is the integration point, none of the call sites need
     to change for API reasons — but the smoke-test coverage at the
     Pre-M1a gate must exercise all 5 (chunked CE mode, Liger FLCE
     mode, compare-loss mode = 3 modes × call sites that exercise
     each mode).

### Verification

The parity gate is a deterministic gradient micro-test, NOT the 100-step
training run (Codex audit: a refactor that replaces Python branching with
tensor control flow can preserve forward losses while changing gradients).
The 100-step run is a smoke test on top.

- **Parity gate — deterministic gradient micro-tests** (fast, must pass first):
  - Fixed `(logits, targets, valid_mask)` triples with `valid_count` ∈
    {0, 1, mid, full}. Compute forward loss and `torch.autograd.grad`
    against logits for each case under both implementations.
  - **Each case run with `grad_output ∈ {1.0, 0.5}`**: the existing
    backward has a noted fp32-promotion quirk
    ([loss.py:76](../../nvfp4_lora/loss.py#L76) comment) that's
    "currently safe because callers pass grad_output=1.0" — but
    `effective_grad_accum=2` and other scaled-loss configurations pass
    non-unit `grad_output` through `loss.backward()`. The refactor
    changes the divisor dtype path; the non-unit case must be tested
    explicitly to catch any scale × divisor regression. (Sonnet round-2
    audit.)
  - Compare element-wise: `torch.allclose(grad_old, grad_new, atol=1e-5, rtol=1e-5)`
    must hold for every (case × grad_output) pair.
  - Specifically exercise the zero-count case for both `_ChunkedFrozenLMHeadCE`
    (which has no explicit branch today and must produce a meaningful
    autograd-preserving zero after the refactor) and `liger_fused_lm_head_ce`
    (which has the explicit branch becoming a `torch.where`).
  - NaN stress: chunk with all `-100` labels alongside a normal chunk;
    verify no NaN propagation through `torch.where`.
- **Smoke test — 100-step training overlay**: identical seed, identical
  data; loss curves must overlay within bf16 noise
  (`max(|loss_old - loss_new|) < 5e-4` over 100 steps). This is a sanity
  check, not the parity gate.
- **Mode coverage**: Both gate and smoke run in chunked CE mode, Liger
  FLCE mode, AND compare-loss mode.
- **Wall-time budget acknowledgment**: a 100-step run at s=2048 takes
  ~5.3 hours (per `super_lc_062`). Three modes × 100-step smoke = ~16
  hours wall time. Gradient micro-tests run in seconds. Budget Pre-M1a
  verification as a 24-hour wall clock window, not a 1-hour task.

### Pre-M1a GO criteria

Gradient micro-tests pass element-wise + smoke test loss overlay clean
across all three modes + diff is clean. Merge only after all three pass.

## α-lite probe — Half-day side-channel (Week 1, ~4 hours)

Runs in parallel with Pre-M1a (which is mostly editor work). Two
genuinely-untested levers, each one or two short runs.

### Probe 1: `backend:cudaMallocAsync`

```bash
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
python train/train_super_nvfp4.py \
  --model-dir ... --train-file ... --adapter-dir ... \
  --batch 1 --grad-accum 1 --max-len 262144 \
  --training-mode cached_prefix_suffix --train-suffix-len 2048 \
  --prefix-chunk-len 8192 --loss-mode chunked_frozen_ce --loss-chunk-tokens 512 \
  --optimizer adafactor \
  --sdpa-causal-no-mask --pooled-loader-buffers --moe-sparse-no-one-hot \
  --mamba-cached-multitoken \
  --watchdog-min-available-gb 2 --watchdog-nvrm-errors --profile-memory-phases \
  --stop-at-step 5 --no-save-final-adapter --no-save-optimizer-state
```

Two runs back-to-back with `release_cuda.sh` between them per
[TROUBLESHOOTING.md:170](../TROUBLESHOOTING.md#L170).

**Decision:**
- If s2k completes with same loss curve AND `dev_alloc` differs by >50% from
  the `expandable_segments:True` baseline → try s4k certification. If s4k
  passes 3 clean steps → write up, this is potentially v1.2.
- If s2k completes with similar `dev_alloc` → cudaMallocAsync is not the lever;
  log as known-negative, continue Pre-M1.
- If s2k fails or destabilizes → log, revert, continue Pre-M1.

### Probe 2: `--prefix-chunk-len 2048`

Same baseline command as above with `--prefix-chunk-len 2048` instead of `8192`.
Codex #2 flagged that long prefix chunks might leak descriptors into the
trainable region. Lowering the chunk size produces shorter no_grad forward
chains during prefill; the hypothesis is that this reduces the descriptor pool
state at the moment the trainable suffix forward begins.

**Decision:** same structure as probe 1. If `dev_alloc` during the
trainable-suffix forward (not the prefill) drops meaningfully, this is a real
lever and we re-test at s4k.

### Probe budget

4 hours total including 2 runs each at s2k (~30 min each), GPU recovery
between runs, write-up. Both probes run during Pre-M1a editor work, not
serially. If either probe needs more than 4 hours, stop and log; don't let
it inflate.

### Probe negative outcome

Both probes coming up neutral is the **expected** outcome given the audit
findings. Log under `LONG_CONTEXT_EXPERIMENTS.md` as SUPER-LC-070 (cudaMallocAsync)
and SUPER-LC-071 (prefix_chunk_len). Continue Pre-M1 unchanged.

### Probe positive outcome — interrupt policy

If `dev_alloc` drops materially AND s4k certifies in either probe, pause
Path 2 and re-run the audit cycle on whatever the new ceiling is. We may
not need Path 2 anymore.

Concrete interrupt rules (Codex audit: this needs to be specified, not
hand-waved):

- **If probe is positive BEFORE Pre-M1a is merged**: stop Pre-M1a at
  PR-ready state, do NOT merge until re-audit completes. Pre-M1b and
  Pre-M1c do not start.
- **If probe is positive AFTER Pre-M1a is merged**: Pre-M1a stays in main
  (it has standalone correctness value — eliminating a host sync and
  clarifying the autograd graph). Pre-M1b and Pre-M1c do not start until
  re-audit completes.
- **Re-audit scope**: same multi-reviewer pattern as the Path 2 base-plan
  and kickoff audits. Question becomes: "given the new ceiling, is Path 2
  still the right next step or should we ship at the new ceiling and
  pivot effort?"

This is the asymmetric-upside case — low probability, high payoff,
half-day cost.

## Pre-M1b — Dequant workspace argument (Week 2, ~3-5 days)

**Files:** `nvfp4_lora/dequant.py`, `nvfp4_lora/linear.py`, `nvfp4_lora/loader.py`

### Concrete changes

1. **`dequantize_nvfp4_weight` ([dequant.py:40](../../nvfp4_lora/dequant.py#L40)):**
   - Add optional `out: torch.Tensor | None = None` parameter.
   - If provided, the final `.reshape(out_feat, in_feat).to(out_dtype)`
     materializes into `out` rather than allocating a new tensor. Use
     `torch.Tensor.copy_` or write the reshape directly into the provided
     buffer.
   - Verify dtype/shape of `out` if provided; raise if mismatched.

2. **`_DequantLinear` — thread workspace through `apply()`, store as
   non-differentiable `ctx` attribute (NOT `save_for_backward`)** (Codex
   round-2 audit caught two real bugs in the earlier "save_for_backward"
   design):

   *Why not `save_for_backward` for the workspace*: tensors saved via
   `save_for_backward` participate in autograd's version-counter check.
   If two `NVFP4LoRALinear` modules share a workspace (which the pool's
   by-shape sharing requires) and both forwards run before either
   backward, the first backward's `out=workspace` write bumps the
   workspace version. The second backward's `ctx.saved_tensors` unpack
   then raises `RuntimeError: one of the variables needed for gradient
   computation has been modified by an inplace operation`.

   The correct pattern is to attach the scratch buffer as a plain `ctx`
   attribute and return `None` for its gradient slot:

   - Modify `_DequantLinear.apply()` signature ([linear.py:33](../../nvfp4_lora/linear.py#L33))
     to accept a 6th positional argument: `w_bf16_workspace: torch.Tensor`.
   - In `forward`, store via `ctx.w_bf16_workspace = w_bf16_workspace.detach()`.
     Do NOT use `ctx.save_for_backward` for the workspace. Only the
     existing forward inputs needed for gradient correctness go through
     `save_for_backward`.
   - In `backward` ([linear.py:61-64](../../nvfp4_lora/linear.py#L61-64)),
     retrieve via `ctx.w_bf16_workspace` and pass as `out=ctx.w_bf16_workspace`
     to `dequantize_nvfp4_weight(...)`.
   - **The `backward()` return signature changes from 5 slots to 6**. The
     6th slot (for the workspace) must return `None` since the workspace
     is a non-differentiable buffer. Forgetting this raises "incorrect
     number of gradients" at runtime.
   - **Required regression test**: two same-shape `NVFP4LoRALinear`
     modules share a single workspace; run both forwards (no
     intermediate backward), then run a single combined backward with
     `torch.autograd.set_detect_anomaly(True)`. Must complete without
     version-counter errors.

3. **`NVFP4LoRALinear.forward` — own the workspace reference**
   ([linear.py:163-165](../../nvfp4_lora/linear.py#L163-165) call site):
   - Each `NVFP4LoRALinear` module holds a reference to its assigned
     workspace tensor (assigned at model load by the pool — see loader.py
     changes below). The forward passes this workspace through
     `_DequantLinear.apply(x, self.weight_uint8, ..., self.w_bf16_workspace)`.

4. **`loader.py` — workspace pool, allocated AFTER pooled_loader_buffers**:
   - At model load time, after `_collect_quantized_linear_records` has
     enumerated all NVFP4 modules and after the existing pooled
     loader buffers are zeroed
     ([loader.py:460](../../nvfp4_lora/loader.py#L460)), build a workspace
     pool: process-global `dict[tuple[int, int, torch.dtype], torch.Tensor]`
     keyed by `(out_features, in_features, dtype)`. The `dtype` key
     element is required because
     [linear.py:48](../../nvfp4_lora/linear.py#L48) calls
     `dequantize_nvfp4_weight(..., out_dtype=x.dtype)` — i.e. the
     workspace dtype tracks the activation dtype, not a hardcoded bf16.
     Today all activations are bf16 so the pool has a single dtype, but
     keying without dtype is a latent defect that would silently select
     a wrong-dtype workspace under any future fp16/fp32 activation path
     (Sonnet round-3 audit).
   - Allocate one buffer per unique `(out_features, in_features, dtype)`,
     **with `requires_grad=False` explicitly set and verified**. Modules
     with identical shapes AND dtypes share a workspace; safe because
     backward is sequential per CUDA stream AND because the workspace
     is held as a non-differentiable `ctx` attribute (see #2 above) so
     version counters do not gate backward.
   - Assign `module.w_bf16_workspace = pool[(out_features, in_features, module_dtype)]`
     for every `NVFP4LoRALinear` instance, where `module_dtype` is the
     dtype the module's `_DequantLinear.forward` will produce (matched
     to expected activation dtype at load time).
   - **Load-time invariant check**: after assignment, iterate every
     module and assert
     `module.w_bf16_workspace.requires_grad is False`. Fail-fast at
     load if violated.
   - **Explicitly NOT views into `lora_b_pool` or any other existing
     buffer.** Workspaces are not zero-initialized (overwritten on every
     use) and must not participate in the `lora_b_pool.zero_()` round-2
     fix invariant.

### Verification

- **Bit parity:** `dequantize_nvfp4_weight(W, S, S2)` and
  `dequantize_nvfp4_weight(W, S, S2, out=preallocated)` produce
  `torch.allclose(..., atol=0, rtol=0)` output. (Both run in fp32 internally
  then cast to bf16; should be bit-identical.)
- **Backward parity:** 100-step LoRA training run, refactored backward vs
  current backward, identical seed and data. Loss curves within bf16 noise.
- **Memory:** confirm peak CUDA reserved drops by approximately
  `num_unique_shapes * largest_shape_bytes` worth of transient allocations
  per backward step.
- **Descriptor count:** the spike of ~40k dequant calls per backward step
  ([linear.py:60-66](../../nvfp4_lora/linear.py#L60-66) × 512 experts × 40
  layers × up+down) should produce a measurable drop in `num_device_alloc`.
  Quantify this — it's the first data point for whether Pre-M1 alone moves
  the cliff.

### Pre-M1b GO criteria

Bit-parity + 100-step parity + measurable `dev_alloc` reduction during
backward.

## Pre-M1c — Mamba kernel workspace patches (Weeks 3-4, ~1-2 weeks)

**This is the M0 gate.** The reason: the audit identified the Mamba SSD
backward's `torch.empty_like` allocations as the dominant remaining
descriptor source. If we can't patch this kernel cleanly to accept
pre-allocated workspaces, Path 2 fundamentally doesn't work, because no
amount of graph capture will help if the captured region itself allocates
per-call.

### Version pin (read this first)

- **Installed `mamba_ssm` version: 2.3.2.post1** (confirmed via
  `pip show mamba_ssm`).
- In `mamba_ssm 2.3.2.post1`:
  - `_mamba_chunk_scan_combined_bwd` is defined at `ssd_combined.py:396`
    (function start). The `torch.empty_like(B|C|dt)` allocations are at
    lines ~422, 427, 435 within the body.
  - `alloc_tile_workspace` is at `mamba_ssm/utils/determinism.py:80` with
    signature `(base_shape, tile_dim, dtype, device, deterministic, *, zero_init=True)`.
- **Vendor the exact commit/tag corresponding to 2.3.2.post1**, not
  upstream HEAD. Record the upstream commit hash in
  `vendored/mamba_ssm/UPSTREAM_COMMIT` so future Triton/torch upgrades
  can diff cleanly.

### Internal milestones inside Pre-M1c

The audit (Codex finding 3) flagged that compressing five distinct
workstreams into a single 2-week window with the entire Path 2 decision
attached is unwise. Pre-M1c is therefore decomposed into four internal
gates **C0/C1/C2/C3**. Hit them in order; if any slips, escalate at that
gate rather than at M0.

| Gate | Target | Content | Pass criteria |
|---|---|---|---|
| **C0** | End W3 D1 | Vendored fork imports cleanly; sys.path shim prefers vendored over installed; existing training reproduces eager loss on a 10-step run AND vendored-vs-installed `dev_alloc` matches within ±1% at B2v measurement (see below) | loss overlay within bf16 noise over 10 steps (NOT bit-identical — Triton autotune non-determinism on first import is acceptable); `dev_alloc_at_B2v` within ±1% of `dev_alloc_at_B2`; no import errors |
| **C1** | End W3 | Allocation source attribution: per-region descriptor breakdown of the **unpatched** vendored bwd path (state = B2v), identifying which specific `torch.empty_like` / `alloc_tile_workspace` / autotune-internal allocations dominate. **Numerically establishes the Mamba share of total backward `dev_alloc` against the B2v denominator — this is the input to the M0 threshold** | written attribution report with %share table; `M_share` is the Mamba bwd `dev_alloc` divided by `dev_alloc_at_B2v` |
| **C2** | Mid W4 | Workspace-arg patch compiles + passes eager parity micro-tests (single Mamba block forward+backward, fixed inputs, gradient atol=1e-5) **AND gradient-parity under grouped checkpointing** with the shape-shared workspace bundle — see checkpoint-safety test in Verification below | per-parameter gradient `allclose(atol=1e-5)` for (a) patched vs unpatched-vendored, single Mamba block; (b) checkpointed vs non-checkpointed under patched kernel on a single Mamba block (eager + recompute sharing the same bundle); (c) two Mamba blocks in different checkpoint groups, both sharing the same pool-allocated bundle (cross-block sharing under checkpointing) — all three pass gradient allclose |
| **C3** | End W4 | Allocation-count measurement at B3 baseline (see M0 GATE below); autotune pre-warm verified (50-step run with no Triton recompile after step 0) | quantitative `dev_alloc_at_B3` number in hand |

**Escalation rule**: if C0 slips to end of W3, OR C1 slips to mid W4, M0
is "not decidable on schedule" and we either extend Pre-M1c by one
fixed-budget week (decided at the C1 slip) or replan.

### Concrete changes

1. **Vendor `mamba_ssm 2.3.2.post1`** into `vendored/mamba_ssm/`. Record
   commit hash. Add sys.path shim in `train_super_nvfp4.py` near the
   existing `os.environ.setdefault` block at line 28.

2. **Patch `_mamba_chunk_scan_combined_bwd`** (line 396 in installed
   version) to accept pre-allocated `dx_workspace`, `ddt_workspace`,
   `dB_workspace`, `dC_workspace` parameters. The function already accepts
   `dx`, `ddt`, `dB`, `dC` optionally; extend the same pattern to internal
   workspace tensors and remove `torch.empty_like(...)` calls in the path.

3. **Patch `alloc_tile_workspace`** (`determinism.py:80`) to accept a
   pre-allocated buffer parameter. Existing `deterministic` flag still
   gates whether contents are zeroed.

4. **Triton autotune pre-warm**:
   - Identify every `@triton.autotune` decorated kernel in the SSD
     backward path (`_chunk_scan_bwd_ddAcs_stable` and others).
   - Add a pre-warm step at model-load time that calls each autotuned
     kernel with the exact `(batch, seqlen, nheads, dstate, chunk_size,
     dtype, stride)` used in real training. Discard outputs.
   - Verify subsequent training calls hit the autotune cache.

5. **Workspace pool integration — shape-shared (single bundle in
   Nemotron-Super case)**:
   - At model-load time, build a Mamba workspace pool keyed by
     `(out_features, in_features, chunk_size, dtype)`. Allocate one
     dx/ddt/dB/dC bundle per unique key.
   - In Nemotron-3 Super, **all 40 Mamba2 blocks have identical
     `(hidden, intermediate, chunk_size, dtype)`** (confirmed via
     `super_lc_038*.log:23` "Mamba cached multi-token patch: enabled for
     40 modules"; the Super config has 88 total layers, 40 of them
     Mamba2). So the pool collapses to a single bundle in production,
     shared across all 40 blocks. The shape-shared design is
     intentional — Codex round-3 audit caught an earlier internal
     contradiction between "per-block bundles" and "shape-shared
     bundles." This section now picks shape-shared and the C2 gate
     verifies it.
   - **Safety argument under sequential bwd**: at any wall-clock instant
     only one Mamba block's backward is executing on the active CUDA
     stream. Within that backward, the bundle is allocated, written,
     read, and effectively released before the next block's backward
     starts. The bundle is therefore safe to share across blocks within
     a single sequential bwd pass.
   - **Safety argument under grouped checkpointing**: when
     [`enable_grouped_layer_checkpointing`](../../train/train_super_nvfp4.py#L851)
     fires the forward twice (once eager, once on recompute), both
     passes write the same shared bundle. Eager fwd writes; recompute
     fwd writes (overwriting eager); recompute bwd reads (correct
     value from recompute fwd); eager bwd does not exist within the
     checkpoint context (its grad flows through the recompute path).
     The hazard would be if eager bwd were to consume the workspace
     value the eager fwd wrote — but the checkpoint contract ensures
     it does not.
   - **C2 must verify both safety arguments empirically**, not just
     accept them on paper. See C2 sub-tests below.
   - **Bundle sizing**: workspaces are sized for the largest configured
     suffix length (allocated once at load; resized only if the user
     re-launches with a different `--train-suffix-len`).
   - **`requires_grad=False` invariant**: same load-time assertion as
     the dequant pool.

### Verification

This is where the gate happens. We need to demonstrate:

- **Eager parity (block-level)**: Pre-M1c-patched Mamba block forward+backward
  produces bit-identical output to the unpatched vendored version on 100
  randomized fixed-shape inputs. Element-wise `torch.allclose(atol=1e-5,
  rtol=1e-5)`. Tested at C2.
- **Eager parity (model-level)**: end-to-end 100-step training run with
  patched vs unpatched Mamba; loss overlay within bf16 noise. Smoke
  test, not parity gate.
- **Backward `dev_alloc` measurement**: see M0 GATE below for the
  empirically-grounded threshold methodology.
- **No autotune re-tune in production**: stress-test by running 50 steps
  at s=2048 after the autotune pre-warm. No Triton recompilation should
  occur after step 0.
- **Checkpoint-safety test (per the C2 gate)**: loss-equality is
  insufficient because both eager and recompute can write the same
  shared bundle and silently produce matching-but-wrong gradients
  (Codex round-2 audit). Real test is **gradient parity** across three
  sub-tests at fixed seed/inputs, designed against the shape-shared
  workspace model (Codex round-3 audit resolved the per-block vs
  shape-shared ambiguity in favor of shape-shared):
  1. Patched vs unpatched-vendored, single Mamba block: per-parameter
     `torch.allclose(grad, ..., atol=1e-5, rtol=1e-5)`.
  2. Checkpointed vs non-checkpointed under the patched kernel on a
     **single Mamba block** sharing one bundle between eager fwd and
     recompute fwd: `enable_grouped_layer_checkpointing` on vs off,
     gradient allclose required. This verifies the eager-fwd writes
     are correctly overwritten by recompute-fwd before recompute-bwd
     consumes them.
  3. **Two distinct Mamba blocks** in different checkpoint groups,
     both sharing the same pool-allocated bundle (cross-block sharing
     under sequential bwd + checkpointing): gradient allclose required.
     This verifies the cross-block sharing safety argument from the
     Pre-M1c workspace pool spec.
  All three must pass to clear C2.

### M0 GATE — measurement baselines

The threshold for GREEN is **derived from the C1 attribution measurement,
not picked a priori** (Codex audit: a fixed 60% threshold has no empirical
basis until we know what fraction of backward `dev_alloc` Mamba actually
owns today).

**Named measurement points** (Codex round-2 audit: the baseline used for
`M_share` must be allocation-equivalent to whatever we evaluate B3
against, so we add `B2v` to bridge the vendoring transition):

| Label | Branch state | What it measures |
|---|---|---|
| **B0** | `main` before Pre-M1a | current production behavior; ground truth |
| **B1** | post-Pre-M1a merge | loss `.item()` syncs removed; tensor-divisor in chunked CE |
| **B2** | post-Pre-M1b merge | dequant workspace pool active; ~40k temp allocs eliminated per backward |
| **B2v** | post-Pre-M1b + **unpatched vendored Mamba** (state after C0) | identical to B2 except `mamba_ssm` served from the vendored fork. Verifies vendoring did not perturb allocator behavior. |
| **B3** | post-Pre-M1c patches (branch, not merged yet) | Mamba SSD bwd pre-allocated workspaces |

Per-step backward `dev_alloc` is measured at each of B0/B1/B2/B2v/B3 at
identical inputs (fixed seed, fixed data, s=2048, three runs averaged).

### M0 GATE — threshold derivation

At C1 (end of week 3), we have a Mamba-share % from the unpatched
vendored bwd path **measured against `dev_alloc_at_B2v`** (NOT B2 — the
vendoring transition might shift allocator behavior, and we need the
denominator to match the baseline B3 is compared against). Call it
`M_share` ∈ [0, 1].

The **isolated Pre-M1c effect** we expect is at most `M_share` of the
B2v backward `dev_alloc` budget. The GREEN threshold is set to capture
**at least 75% of the expected effect**:

- `GREEN_THRESHOLD = 0.75 × M_share × dev_alloc_at_B2v`
- A successful Pre-M1c drops `dev_alloc` from B2v to B3 by at least
  `GREEN_THRESHOLD` events. Formally:
  `drop = dev_alloc_at_B2v − dev_alloc_at_B3` (a positive number when
  the patch works; GREEN requires `drop ≥ GREEN_THRESHOLD`).

If C0 detects `dev_alloc_at_B2v` differs from `dev_alloc_at_B2` by more
than ±1%, the vendoring itself has shifted allocator behavior and we
investigate before continuing — that's a C0 fail, not a free pass.

Two cases worth flagging upfront:

1. **If M_share at C1 turns out small (<30%)** — Mamba is not the
   dominant source. Pre-M1c can't move the cliff much regardless of
   patch quality. **This is C1 telling us to abort early** — Pre-M1c
   work doesn't even need to complete. The decision is "the audit was
   wrong about where descriptors live; we need to find the real source
   before committing more weeks to Path 2." Document and pivot.
2. **If M_share is large (>70%)** — proceed with Pre-M1c with confidence
   that the patch can move the cliff. Standard M0 evaluation applies.

### M0 GATE — decision table

All "drop" values below are `dev_alloc_at_B2v − dev_alloc_at_B3` (positive
when the patch reduces descriptor count).

| Outcome | Decision |
|---|---|
| C1 reveals M_share <30% | **RED-EARLY.** Pivot before completing Pre-M1c. Pre-M1a/b still ship. |
| Patch lands + `drop ≥ GREEN_THRESHOLD` + eager parity holds | **GREEN.** Proceed to M1/M2. Estimated remaining timeline: 14-18 weeks. |
| Patch lands + `0.5 × GREEN_THRESHOLD ≤ drop < GREEN_THRESHOLD` | **AMBER.** See AMBER protocol below. |
| Patch lands + `drop < 0.5 × GREEN_THRESHOLD` | **AMBER → likely RED.** Run AMBER protocol; expect RED. (Intervals are half-open per Codex round-3 audit: at exact `drop == GREEN_THRESHOLD`, GREEN wins; at exact `drop == 0.5 × GREEN_THRESHOLD`, AMBER wins.) |
| C0 fails (B2v differs from B2 by >±1%) | **RED.** Vendoring perturbed allocator behavior; investigate root cause before proceeding. |
| C2 slips past mid-W4 (patch doesn't compile or fails parity) | **RED.** Path 2 can't address the dominant source on a tractable timeline. Pivot to Option B or C. |
| Patch lands but breaks eager parity at C2 | **RED.** Can't merge incorrect kernels. Stop. |
| Checkpoint-safety test fails at C2 (gradient parity check across the three sub-tests) | **RED.** Workspace sharing under recompute corrupts gradients. Stop. |

### AMBER protocol (Codex audit: AMBER must be timeboxed)

If M0 is AMBER, we have **1 calendar week** of investigation. Required
deliverables at end of AMBER week:

- **Residual attribution table**: where are the descriptors not killed by
  Pre-M1c coming from? (loss path? dequant path? attention? MoE?)
- **Patch candidate**: a concrete proposal — file, function, estimated
  effort in days — that would close the gap to GREEN_THRESHOLD.

Conversion rules at end of AMBER week:

- AMBER → GREEN: a bounded patch with estimated effort ≤1 week is
  identified AND expected (justified, not hand-waved) to cross
  GREEN_THRESHOLD when applied. Schedule that patch as M0.5 before
  proceeding to M1/M2.
- AMBER → RED: residual is unattributable, OR every candidate fix is
  >1 week of work, OR the gap to GREEN_THRESHOLD is more than the
  remaining engineering budget can plausibly close. Pivot to Option
  B or C.

AMBER cannot extend more than 1 week. There is no AMBER → AMBER state.

### Why the M0 gate is the cheapest abort

By end of week 4, we've invested 4 weeks of refactor work that is
independently valuable:
- Pre-M1a removes a host sync from the loss path and cleans up the
  autograd graph for any future graph-capture work.
- Pre-M1b removes ~40k transient allocations per backward step,
  measurably reducing allocator pressure (B2 vs B0 will quantify this).

If M0 RED forces a pivot, Pre-M1a and Pre-M1b still ship as v1.2-cleanup
PRs IF AND ONLY IF B2's `dev_alloc` drop relative to B0 is non-trivial
(>5% of total backward `dev_alloc`). If the drop is negligible, they
ship as "internal correctness PRs" not "v1.2" — see "v1.2 framing"
below.

## Pre-M1d — Hook-free training infrastructure (Week 4, ~2-3 days)

Only proceeds if M0 is GREEN or AMBER. Concrete changes per the audit:

1. Remove or guard routing census hooks
   ([train_super_nvfp4.py:962](../../train/train_super_nvfp4.py#L962)) so they
   are never attached during capture-enabled steps.
2. Add `--capture-safe-mode` flag that enforces: no forward/backward hooks
   registered, `torch.set_autocast_cache_enabled(False)`, and a startup audit
   that enumerates all hooks on the module being graphed.

Verification: `nn.Module._forward_hooks` is empty for every module in the
captured path under `--capture-safe-mode`.

## Risk register (re-stated, post-audit)

| Risk | Likelihood | Trigger | Mitigation |
|---|---|---|---|
| Pre-M1c patch is rejected upstream (irrelevant short-term, important if we want to maintain) | High | Path 2 ships with a vendored fork | Plan upstream contribution post-ship; accept fork burden until then |
| Triton autotune pre-warm fails to suppress retunes | Med | Step 1+ Triton recompilation observed | Force autotune disable via env var; accept fixed config penalty |
| **Triton/PyTorch/CUDA version drift** invalidates Pre-M1c patches | Med | Triton signature change between installed-at-kickoff and installed-at-merge | Pin Triton + PyTorch + CUDA versions before C0; any upgrade during weeks 1-4 requires re-audit before continuing |
| **Grouped checkpointing × workspace pool**: recompute mid-eager-bwd corrupts workspace contents | Med-High | Loss diverges under `enable_grouped_layer_checkpointing` after Pre-M1c | C2 includes a checkpoint-safety test; if it fails, either disable checkpointing inside capture-readiness paths or allocate per-recompute workspace bundles |
| `NVFP4LoRALinear.w_bf16_workspace` attribute initialization slows model startup | Low | Slow load time | Lazy-allocate at first forward rather than at model load; measure load time at B2 vs B0 |
| α-lite probe positive: cudaMallocAsync moves the cliff | Low | s4k certifies clean under cudaMallocAsync | Apply interrupt policy above; pause Path 2, re-audit |
| Pre-M1a refactor introduces gradient regression masked by loss-overlay smoke test | Med | Gradient micro-tests fail at parity gate | Bisect within `loss.py`; revert the specific change; consider keeping `.item()` for now and deferring to Path 2 manual capture |
| **C1 reveals Mamba share <30%** of backward `dev_alloc` | Med-Low | Attribution report at end of W3 | RED-EARLY: pivot before completing Pre-M1c; route attribution data into Option B/C planning |
| `_lm_head_ce` change of API affects callers outside the three identified ones | Low | Compile error in some training mode | grep for all `_lm_head_ce(` invocations before merge; add CI smoke for each loss mode |
| M0 RED: no path forward on single Spark | Med | Mamba patch doesn't work or share is small | Pivot to Option B or C; Pre-M1a and Pre-M1b ship as v1.2 only if B2 vs B0 drop is meaningful |

## Deliverables and intermediate decision points

Each week ends with both deliverables and an **explicit go/no-go decision**
that gates the next week (Codex audit: weekly deliverables without
decision points hide drift; M0 should not be the only decision in 4
weeks).

**Week 1**:
- *Deliverables*: Merged PR for Pre-M1a (loss refactor); SUPER-LC-070
  (cudaMallocAsync) + SUPER-LC-071 (prefix_chunk_len 2048) logged;
  `dev_alloc` measurement at B0 (current main) and B1 (post-Pre-M1a).
- *End-of-week decision*: merge Pre-M1a only if gradient micro-tests
  pass AND smoke loss overlay clean. If parity fails, bisect + revert;
  Week 2 cannot start with broken loss.

**Week 2**:
- *Deliverables*: Merged PR for Pre-M1b (dequant workspace pool);
  `dev_alloc` measurement at B2; isolated Pre-M1b effect documented.
- *End-of-week decision*: record `B2 - B0` and `B2 - B1` deltas. If
  Pre-M1b moves `dev_alloc` substantially relative to B1, that tells us
  the dequant temps were a larger share than expected and Pre-M1c's
  threshold (calculated at C1) will be lower. Update Pre-M1c expectations
  before starting it.

**Week 3**:
- *Deliverables*: vendored `mamba_ssm 2.3.2.post1` (C0); attribution
  report on Mamba's share of B2 backward `dev_alloc` (C1).
- *End-of-week decision*: **this is the early-abort point.** If C1
  reveals M_share <30%, escalate to RED-EARLY immediately. Do not start
  the patching workstream. Pre-M1a/b already merged; pivot decision
  begins now.

**Week 4**:
- *Deliverables*: patched Mamba kernels (C2); B3 measurement (C3);
  M0 evaluation: GREEN/AMBER/RED call with the empirically-derived
  threshold.
- *End-of-week decision*: per the M0 GATE table above. If GREEN, merge
  Pre-M1c + Pre-M1d, brief for M1. If AMBER, kick off AMBER protocol
  (1-week timebox). If RED, write the pivot brief.

## "v1.2 ship" framing — when this is honest

(Codex audit: claiming Pre-M1a/b ship as v1.2 only holds if they produce
measurable user-visible improvement; otherwise it's overclaim.)

At end of Week 2, `B2 vs B0` quantifies whether Pre-M1a + Pre-M1b have
moved a user-visible metric:

- **Ship as v1.2** if `B2 vs B0` shows ≥10% reduction in per-step
  backward `dev_alloc` OR ≥5% step-time improvement OR enables a
  certified s2k+ rung that B0 couldn't.
- **Ship as "internal cleanup PRs"** if the improvement is real but
  small (<10% dev_alloc, no certification change). No user-facing
  v1.2 release; merge to main as plain refactors.
- **Don't ship** if either refactor produces a regression in any
  metric. Hold on a feature branch for Path 2 internal use.

This is decided at end of Week 2 with concrete numbers in hand.

## What success looks like at end of week 4

In the GREEN case:
- Three merged PRs (loss refactor, dequant workspace, Mamba patches + hook
  guards) that independently improve eager training cleanliness.
- A precise per-region descriptor budget showing the residual after Pre-M1
  and pointing M1 at the next sources.
- A green-light decision to commit to weeks 5-22 of Path 2 with concrete
  understanding of what's left.

In the RED case:
- Three merged PRs of independent value (still worth shipping as v1.2).
- A precise diagnosis of why Path 2 doesn't work on this hardware.
- A reasoned recommendation for Option B (multi-Spark) or Option C (BF16
  master on Nano-30B).

Either outcome is informative. Neither outcome wastes the four weeks.

## Pinned environment (record before Week 1)

Capture and freeze for the duration of Pre-M1:

- PyTorch version + CUDA build
- Triton version
- `mamba_ssm` version (target: 2.3.2.post1; record exact upstream commit
  hash for the vendor copy)
- HF transformers version (currently 5.8.1)
- Driver/NVRM version

A change to any of these during weeks 1-4 forces re-audit before
continuing — they affect kernel signatures, autotune behavior, and the
Mamba patch baseline.

## Out-of-scope guards

- No model changes (LoRA rank, targets, loss weighting).
- No HF transformers version bump (would invalidate the Mamba patch
  baseline; transformers updates to `NemotronHMamba2Mixer` modeling code
  could break the existing monkey-patch).
- No multi-Spark experimentation.
- No new dependencies except: `mamba_ssm` vendored fork.
- **No Triton, PyTorch, or CUDA version changes**. If the autotune pre-warm
  work requires a Triton upgrade, that triggers re-audit per the pinned
  environment rule above.

## Sign-off — assumptions (must hold or revise this doc first)

(Codex audit: assumptions must be empirical claims or be converted to
acceptance criteria.)

1. **Mamba SSD backward kernel patches are the highest-risk single piece
   of Path 2.** This is supported by the prior audit findings but
   *quantitatively confirmed at C1* via the Mamba-share attribution
   report. If C1 contradicts this assumption (M_share <30%), the kickoff
   stops; we don't push through.

2. **Pre-M1a and Pre-M1b have measurable independent value.** This is an
   *acceptance criterion*, not a default-true assumption — tested at end
   of Week 2 via the `B2 vs B0` measurement. If the measurement shows
   negligible improvement, they ship as cleanup PRs not v1.2, per the
   "v1.2 ship framing" section.

3. **The α-lite probe is allowed to short-circuit Path 2** without
   further audit IF it produces a clean s4k certification. If the probe
   produces only proxy improvement without certification, no
   short-circuit; continue Path 2.

If any of these assumptions is invalidated during execution, stop and
revise this document before starting next-week work.
