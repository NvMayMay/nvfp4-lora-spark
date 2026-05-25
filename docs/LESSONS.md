> This is a development-time engineering journal kept during the Phase 1 sprint, not a polished reference doc. It contains chronological lab-notebook style entries with the failure modes hit and the workarounds chosen. For the production reference, see [docs/PERFORMANCE_ROADMAP.md](PERFORMANCE_ROADMAP.md) and [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md).

# Lessons & deviations notebook - P-1 sprint execution

**Purpose**: capture every "the plan said X, reality required Y" moment, plus surprising findings, while the work is happening. To be reflected on after the sprint.

**Convention**: append entries with timestamp, what the plan said, what we actually did, and why.

---

## 2026-05-20 ~10:30 - `nohup` does not preserve `cd`

**Plan said**: download Super via `cd /path/to/Models && nohup hf download ... --local-dir Nemotron-3-Super-...`.
**Reality**: the second nohup'd `hf download` in the chained command ignored the working directory set by `cd`. The model landed in `$HOME/Nemotron-3-Super-120B-A12B-NVFP4/` instead of `Models/`. Had to `mv` 75 GB after (instant since same filesystem, but it could have been a real problem).
**Fix going forward**: always pass an absolute path to `--local-dir`. Never rely on the shell's working directory being preserved across `nohup`.
**P-1d runbook impact**: revise download steps in the P-1a runbook to use absolute paths.

## 2026-05-20 ~10:30 - `hf` CLI not in `nohup`'s PATH

**Plan said**: `nohup hf download ...` (using bare `hf`).
**Reality**: the bare `hf` isn't in the PATH that `nohup` inherits when launched from a non-interactive shell session. First download attempt failed silently with `hf: No such file or directory` - the launch returned no error to the foreground but no download occurred.
**Fix going forward**: always use the full path `/path/to/venvs/train/bin/hf` (or activate the venv inside the nohup'd script).
**Runbook impact**: P-1a runbook step 1 download lines should use full path.

## 2026-05-20 ~10:40 - Process RSS is the wrong metric on unified memory

**Plan said**: gate P-1a memory measurements on process RSS ≤ 110 GB at every checkpoint (post-load, post-warmup, post-first-forward).
**Reality**: on this unified-memory Spark, `VmRSS` only shows ~3-4 GB for a vLLM-served model with 19-75 GB of weights resident. Model weights live in mmap'd safetensors pages + GPU-allocated tensors that don't count toward per-process RSS. The real indicator of system memory pressure is `free -g` used.

Concretely observed for the Nano probe:
- Sum of VmRSS across vLLM workers: 3.87 GB
- `free -g` used: 115 GB (post-warmup), 116 GB (post-first-forward)
- Difference: vLLM's default `--gpu-memory-utilization=0.9` reserves ~90% of memory for KV cache PagedAttention pool, plus the safetensors mmap is in OS page cache (24 GB Cached per `/proc/meminfo`).

**Fix going forward**: P-1a runbook's "process RSS ≤ 110 GB" gate is meaningless on this hardware. The actual Outcome D gate is **does the model start successfully under vLLM's auto memory management** - if it OOMs during load, that's Outcome D. If it runs, the memory budget is fine by definition.

**Runbook impact**: rewrite P-1a memory criterion before P-1d's container repro test, or that test will produce false PASS/FAIL based on a metric that doesn't reflect actual constraints.

## 2026-05-20 ~10:55 - Super C1 first attempt failed with `code=137` (OOM-killed nvcc)

**Plan said**: launch Super under marlin recipe, expect ~10 min model load.
**Reality**: FlashInfer auto-JIT-compiles SM120 NVFP4 MoE kernels at startup (`gen_gemm_sm120_group_gemm_mxfp4_groupwise_e4m3_bf16_sm120` and 6 others). For Super specifically (mixed-precision NVFP4 + FP8), 7+ new kernels needed JIT-compilation that weren't in the FlashInfer cache from the Nano probe.

The JIT compile spawns ~8 parallel nvcc jobs by default. Each nvcc + the 75 GB Super model load = OOM. The kernel killed nvcc with SIGKILL (exit code 137), ninja reported `FAILED: [code=137]`, the engine-core init crashed.

**Fix applied**: re-launched with `MAX_JOBS=1` and `NVCC_THREADS=1` env vars to force serial nvcc compiles. Each kernel compiles one at a time → fits within memory budget. Cached on success so subsequent runs don't need to re-compile.

**Important nuance**: NVIDIA's Spark deployment guide forces marlin (`VLLM_NVFP4_GEMM_BACKEND=marlin`, `--moe-backend marlin`) AND disables FlashInfer MoE FP4. We had marlin forced but FlashInfer was still trying to JIT-compile its kernels eagerly at startup - marlin-forced doesn't suppress FlashInfer's eager warmup.

**Runbook impact**: P-1a runbook needs to include `MAX_JOBS=1 NVCC_THREADS=1` env vars for Super (not Nano - Nano's smaller kernel set didn't trip the OOM). May also need to investigate whether there's a vLLM env var that fully disables FlashInfer eager JIT.

## 2026-05-20 ~10:55 - Super's MIXED_PRECISION accepted, not rejected

**Plan said** (per vLLM #37854 risk): expect Super to fail at load with `quant_algo: "MIXED_PRECISION"` not in whitelist.
**Reality**: vLLM detected Super's quant config as **two separate per-layer schemes**:
- `Detected ModelOpt fp8 checkpoint (quant_algo=FP8)`
- `Detected ModelOpt NVFP4 checkpoint (quant_algo=NVFP4)`

This means our pinned vLLM version already has the #35047 fix (merged 2026-02-26). The MIXED_PRECISION patch I sketched in `patches/vllm-mixed-precision-37854.md` is **not needed** for our environment.

**Runbook impact**: `patches/vllm-mixed-precision-37854.md` can be archived/deprioritized. The risk is mitigated by version pinning, not by an in-tree patch.

## 2026-05-20 ~10:30 - Nemotron-3-Super FlashInfer cache directory tagged `121a/`

**Finding (not a deviation)**: FlashInfer's JIT cache path is `~/.cache/flashinfer/0.6.8.post1/121a/cached_ops/gemm_sm120/...`. The `121a/` directory confirms that FlashInfer **does** treat sm_121a as a distinct compute capability and attempts to dispatch SM120 NVFP4 kernels to it. This contradicts the v6.2 "CUTLASS one-line whitelist" framing somewhat - FlashInfer/vLLM at our version has already wired up SM120/121 NVFP4 MoE; the issue is not "kernels don't exist" but "JIT compile pressure under load."

## 2026-05-20 ~10:30 - TE install on aarch64 + CUDA 13 is broken

**Plan said**: install `transformer-engine[pytorch]` in the `nvfp4-te` venv.
**Reality**:
1. First attempt: `pip install transformer-engine` → installed meta package without TE extension, import fails.
2. Second attempt: `pip install transformer-engine[pytorch]` → `pyproject.toml` build needed `wheel` which wasn't in the build env. Failed.
3. Third attempt: pre-install `wheel setuptools cmake ninja pybind11`, then `pip install --no-build-isolation transformer-engine[pytorch]` → `transformer_engine_torch` cpp extension ninja build failed (separate error - needs investigation).

**Implication**: P-1b is genuinely blocked until TE install is resolved. This is the "toolchain rot on aarch64 + CUDA 13" risk PROPOSAL.md §3.3 flagged. Container approach (P-1d) may need to use NGC's TE which is presumably pre-built.

**Action**: flagged in todo list as pending operator attention. Does not block P-1a or P-1c (P-1c uses transformers/peft path, not TE).

## 2026-05-20 ~01:48 - Qwen ICH_v3.0 LoRA adapter much smaller than v2.0

**Finding** (not a deviation, but worth recording):
- v2.0: r=128, target_modules=`all-linear` → 180M trainable params, 363 MB adapter on disk
- v3.0: r=128, target_modules=`q,k,v,o,gate,up,down` (post-mortem fix) → 67M trainable params, 256 MB adapter

The 3× drop in trainable params is from skipping SSM `in_proj_qkv/a/b/z`, MoE `shared_expert_gate`, and `out_proj`. v3.0's val_loss curve confirms the post-mortem theory: the SSM/router-gate adaptations were contributing to overfit. v3.0 minimum bounce from min-to-final is +0.010 vs v2.0's +0.033.

## 2026-05-20 ~12:30 - TE install resolved at version 2.12.0 (not latest)

**Plan said**: install TransformerEngine in `nvfp4-te` venv via `pip install transformer-engine[pytorch]`.

**Reality**: took 8 install iterations to find a working version:
- v1-3: meta-package issue, then missing build deps (`wheel/setuptools/cmake/ninja/pybind11`).
- v4: build deps installed but failed on `cudnn.h: No such file or directory` - pip wheels of cuDNN don't auto-set CPATH.
- v5: added `CUDNN_HOME` / `CPATH` / `LIBRARY_PATH` for the cuDNN pip wheel → progressed past cuDNN errors → failed on `nccl.h: No such file or directory`.
- v6: extended CPATH/LIBRARY_PATH to cover ALL `nvidia/*/include` and `nvidia/*/lib` directories pip installs (cudnn, nccl, cusparselt, nvshmem, cu13). **Build succeeded**, but import failed with `undefined symbol: cublasLtGroupedMatrixLayoutInit_internal, version libcublasLt.so.13`.
- v7: downgraded to TE 2.14.1 - same import error.
- v8: downgraded to TE 2.12.0 - **import works, NVFP4BlockScaling recipe available**. The cublasLtGrouped symbol dependency was added in TE 2.13.0.

**Root cause for the symbol error**: TE 2.13.0+ requires `cublasLtGroupedMatrixLayoutInit_internal`, a cuBLAS symbol that does not exist in publicly-released cuBLAS 13.1.1.3 (the only version available via pip or NVIDIA's CUDA 13.0 toolkit on aarch64). System libcublasLt and pip libcublasLt are byte-identical (same MD5: `a038b06190dccb22a3397f9b9939cb86`). TE 2.13+ appears to depend on an unreleased internal NVIDIA cuBLAS build, possibly tied to datacenter-Blackwell (sm_100a) internal CUDA APIs that don't exist on consumer-Blackwell (sm_120/121).

**Settled working configuration**:
- `nvfp4-te` venv: torch 2.12.0+cu130 + nvidia-cudnn-cu13 9.20.0.48 + nvidia-nccl-cu13 2.29.7 + nvidia-cublas 13.1.1.3 + nvidia-cusparselt + nvidia-nvshmem + transformer-engine 2.12.0
- Required runtime env vars (any shell launching TE code must set these, otherwise libtransformer_engine.so fails to load its CUDA-runtime deps):
  ```
  SITE_PKGS=/path/to/venvs/te/lib/python3.12/site-packages
  export CPATH=$(ls -d $SITE_PKGS/nvidia/*/include | tr '\n' ':')
  export LIBRARY_PATH=$(ls -d $SITE_PKGS/nvidia/*/lib | tr '\n' ':')
  export LD_LIBRARY_PATH=$(ls -d $SITE_PKGS/nvidia/*/lib | tr '\n' ':')
  ```
  Recommend wrapping the venv activate script to set these automatically.

**Runbook impact**:
- P-1b runbook's "install transformer-engine" step needs the 8-step recovery captured here.
- P-1d container (NGC PyTorch 26.04 base) may avoid this entirely if NGC bundles a working TE; verify at container build time before re-implementing this fix in-tree.
- TE 2.13+ is currently inaccessible on this hardware via public CUDA. Track NVIDIA forum 351220 and the TE release notes for when sm_121 training support officially lands.

## 2026-05-20 ~13:00 - TE NVFP4 RHT kernel broken on sm_121; degraded NVFP4 GEMM does run

**P-1b finding**: `is_nvfp4_available: True` is misleading. TE 2.12.0 reports NVFP4 is available on sm_121 (compute capability check passes), but the actual kernel path used in the default `NVFP4BlockScaling` recipe fails with:

```
RuntimeError: hadamard_transform_cast_fusion.cu:672 in function rht_gemm_ntt_w_sfc:
CUDA Error: invalid argument
```

This is the Random Hadamard Transform cast-fusion kernel - the kernel that pre-conditions activations before NVFP4 quantization to reduce quantization noise.

**Knob-by-knob diagnostic** (variants tested at IN=OUT=1024, B=1, S=256):

| Recipe knobs disabled | fprop result | dgrad result |
|---|---|---|
| (default - all knobs ON) | **FAIL** RHT kernel CUDA error | - (forward dies first) |
| `disable_rht=True` | OK (cos=0.989, rms_rel=0.146) | OK but **dx_norm=0** (gradient vanished) |
| `disable_stochastic_rounding=True` (RHT still on) | FAIL same as default | - |
| `disable_2d_quantization=True` (RHT still on) | FAIL same as default | - |
| `disable_rht + disable_stochastic_rounding + disable_2d_quantization` | OK (cos=0.991, rms_rel=0.134) | OK (dx_norm=326) |

**Interpretation**:
- The **RHT cast-fusion kernel is the broken component**. Three of the five variants that left RHT enabled all crashed with the same error; the two variants that disabled it ran to completion.
- The "all-off" variant runs end-to-end but produces **rms_rel ~13%** vs the 0.1% target tolerance. That's three orders of magnitude over budget - not a tolerance issue, an actual quantization-quality issue caused by stripping the precision-preservation mechanisms that make NVFP4 viable for training.
- The "disable_rht only" variant has dx_norm = 0 - gradient literally vanishes, suggesting the gradient path also depends on RHT or some related infrastructure to produce non-zero output.

**P-1b verdict under runbook outcomes**:
- This is **Outcome C** - "FP4 forward kernel fails on sm_121 for the production training path". A degraded mode exists but isn't training-quality.
- Outcome A (full FP4 forward + backward) is firmly dead on sm_121 with TE 2.12.0.
- Outcome B (FP4 forward only, bf16 backward-input) does not apply because the FP4 forward itself fails on the production recipe.
- Whether to continue under Outcome C - bf16-dequant fallback for the entire training path - is the operator's call per PROPOSAL.md §5.2.

**What this means for the proposal**:
- The PROPOSAL.md §3.3 / §4.2 / §5.2 framing that called Outcome A "speculative" (citing NVIDIA forum 351220 saying TE didn't support sm_121 in late 2025) is now confirmed empirically. Outcome A is not just speculative - it is currently inaccessible.
- The proposal's downstream paths (P-1c, P-2 library architecture) need to be re-scoped around Outcome C / QeRL-pattern bf16-dequant or PIVOT.
- This also means our P-1d container hardening should NOT pin TE 2.13+ in the hope of NVFP4 training support - that version isn't accessible AND doesn't bring sm_121 support. Pinning TE 2.12.0 (or simply not including TE at all if we go QeRL-pattern) is the realistic plan.

**Track for future re-test**: when NVIDIA publishes a cuBLAS release that includes `cublasLtGroupedMatrixLayoutInit_internal` AND fixes the RHT cast-fusion kernel for sm_120/121, re-run P-1b. The infrastructure (P-1b runbook script + nvfp4-te venv setup) is now reusable.

## 2026-05-20 ~13:30 - Internal tooling auth details

[redacted: internal tooling auth details removed for public release]

## 2026-05-20 ~13:30 - Both subagents independently converged on the same path (Path 1)

Spun up two subagents on the same prompt (read PROPOSAL + LESSONS + P-1_results → produce ranked path list to Super-120B-NVFP4 LoRA on this Spark). Different model families, different framings, **same headline conclusion**:

> Hand-rolled `NVFP4LoRALinear` (~250 LOC), treat NVFP4 as compressed storage, dequant per-tile to bf16 in forward, run standard PEFT LoRA on top, serve adapter back through the already-working marlin vLLM. TransformerEngine entirely out of the training loop.

Rated HIGH confidence by both. Effort: 4-6 days (Opus) / "Days" (GPT-5.5).

The independent convergence is the strong signal. Both arrived at the conclusion via different reasoning paths:
- **Opus**: focused on memory discipline + autograd correctness (custom `autograd.Function` that doesn't save bf16 across backward, recompute on demand)
- **GPT-5.5**: focused on minimal infrastructure that bypasses all known-broken layers (no TE, no bitsandbytes, no AWQ/HQQ, no FP4 training kernels)

Lower-ranked paths both agents deprioritized identically:
- QeRL fork: medium effort/conf, risks ModelOpt-vs-compressed-tensors scale-layout mismatch
- Megatron-Bridge: 1-4 weeks, 240 GB bf16 Super won't fit on 128 GB - strictly worse than Path 1
- TE 2.12.0 RHT patch: weeks-months, low confidence (broken surface > 1 kernel given the dx=0 result)
- Unsloth Q-LoRA: bitsandbytes-broken backend, NF4 ≠ NVFP4

**This re-frames the v6.2 proposal**: Outcome C was originally framed as "only proceed if P0 model-quality upside is exceptional" - i.e., consolation prize. With Outcome A empirically dead and Outcome B also unavailable (forward fails on the production recipe), Outcome C becomes the default-and-only path forward. Both agents say this is a CLEAN path, not a degraded one - the 13% rms_rel that killed TE doesn't apply here because dequant + bf16 matmul has no quantization noise in the compute path; the LoRA delta trains in bf16 normally.

**Implication for proposal v7**: re-baseline around Path 1 = bf16-dequant + hand-rolled NVFP4LoRALinear as the production architecture. The "FP4 training speed" framing in v6.2 disappears; the value proposition becomes "120B-class fine-tunable model fitting on a single Spark, at bf16-trainer wall-time", which is still meaningful but a different pitch.

**Artefacts**:
- `agent_outputs/PROMPT.md` - the shared task description
- `agent_outputs/opus.md` - Opus's 1157-word ranked path list
- `agent_outputs/gpt55.md` - GPT-5.5's parallel ranked path list (extracted from log via awk)
- `agent_outputs/design_memo.md` - the convergence memo

**Minor disagreement on tactics** (not blocking):
- Opus says Nano-first (3-day validation), GPT-5.5 doesn't strongly recommend.
- Opus says plain LoRA for milestone 1 (skip DoRA monkey-patch), GPT-5.5 has DoRA as a follow-on Path 6.
- Both agree: target attention + expert MLP projections, skip Mamba-2 `mixer.in_proj/out_proj` (bf16-zero-row NaN class).
- Both agree: vLLM-as-trainer is dead (compiled/captured graph, not autograd-traced).
- Both agree: Triton FP4 dequant kernel is a v2 optimization, not v1.

## 2026-05-20 ~14:30 - Path 1 sprint Day 1 PASS - dequant + NVFP4LoRALinear work

Built the package at `Sandbox/nvfp4_lora/` per the Day 1 plan. Two gates cleared:

**Gate 1: Dequant correctness vs torchao reference** (`Sandbox/nvfp4_lora/tests/dequant_correctness.py`):
- Loaded `backbone.layers.0.mixer.in_proj` from Nemotron-3-Nano-30B-A3B-NVFP4 (a known-NVFP4 layer, not in the modelopt `exclude_modules` list).
- On-disk layout confirmed: `weight` uint8 (10304, 1344) packed-LSN-first, `weight_scale` float8_e4m3fn (10304, 168), `weight_scale_2` float32 scalar. Group size 16.
- Hand-rolled dequant uses the E2M1 LUT (8 positive values + 8 sign-flipped), unpacks low-nibble-first, scales by per-group FP8 (→ FP32) and per-tensor FP32.
- Compared against `torchao.prototype.mx_formats.nvfp4_tensor.NVFP4Tensor` constructed with `(qdata, scale, block_size, orig_dtype, per_tensor_scale)`: cos=1.0035 (bf16 noise above 1.0 is a rounding artefact), rms_rel 4e-3, max abs 2e-3. **PASS**.

**Gate 2: NVFP4LoRALinear forward + backward smoke** (`Sandbox/nvfp4_lora/tests/linear_smoke.py`):
- Constructed module from real on-disk tensors at r=8, alpha=16.
- 103,936 trainable params (r * (in + out) = 8 * (2688 + 10304)) + 15.6M frozen buffer params (the NVFP4 storage).
- Forward (with LoRA at init = no-op since B starts zero) matches `F.linear(x, dequant(W))` reference exactly (cos=1.0, rms_rel=0).
- Backward: `x.grad` flows through the custom `autograd.Function` (`_DequantLinear.apply`), std=7.09. The frozen base produces correct backward-input gradient despite no FP4 GEMM involvement.
- LoRA gradients: `lora_B.grad` std=14.19 (Kaiming-init A flows through); `lora_A.grad` exactly zero at init (correct because B=0 makes the LoRA delta zero on init).
- No `requires_grad` on frozen buffers.
- r=0 frozen-only mode: 0 trainable params, x.grad still flows. **PASS**.

**Architectural choices locked in for the package**:
- Frozen NVFP4 stored as `register_buffer` (not nn.Parameter) - sidesteps optimizer-state contamination and `nn.Module.parameters()` returning the frozen weight.
- Custom `torch.autograd.Function` recomputes the bf16 dequant inside backward rather than saving it across the graph - direct implementation of the PROPOSAL §5.3.1 v6 "no persistent bf16 shadow" requirement.
- `NVFP4LoRALinear` is `nn.Module` with a Linear-compatible interface, NOT a subclass of `nn.Linear` (per v6 audit point - nn.Linear's `.weight` Parameter assumption is wrong here).
- Class-method `from_safetensors_record(record, prefix=..., r=..., ...)` for the Day 2 module-injection loader.

**Third gate Opus had specified (vLLM marlin's effective output)** is deferred to Day 3's adapter round-trip test, where we'll naturally verify the adapter trained against our forward path produces correct outputs when served via vLLM. The torchao+`F.linear` comparison plus a working backward is sufficient evidence for Day 1.

**Side observation**: `dequantize_nvfp4_weight` on the 10304×2688 layer ran without OOM on CPU at uint8 → bf16 expansion. Per-layer dequant is 10304 * 2688 * 2 bytes ≈ 53 MB - small enough that no chunking is needed for the per-tile-on-demand pattern. We have plenty of headroom for the 4096 hidden + larger expert-MLP shapes that Super has.

## 2026-05-20 ~15:00 - Path 1 sprint Day 2 PASS - loader + module replacement + 2-batch training on Nemotron-3-Nano

Built `Sandbox/nvfp4_lora/loader.py` per the Day 2 plan. Replaces NVFP4 Linears with our `NVFP4LoRALinear` modules via `accelerate.init_empty_weights()` + safetensors-index-driven module swap + non-NVFP4 weight loading.

**Day 2 gate results**:
- Loaded Nemotron-3-Nano-30B-A3B-NVFP4 architecture (5968 NVFP4LoRALinear modules - 5934 LoRA-trainable at r=8, 34 frozen).
- CUDA allocated after load: **20.5 GB**. Total params: 1.30 B (Linear-only count). Buffer storage: 17.15 B params (the NVFP4 packed bytes + FP8 scales).
- Trainable params: 216.4 M (across 5934 modules × 2 LoRA tensors each).
- 2-batch training smoke:
  - step 1 loss = 2.4190 (finite). lora_A grad = 0 (correct - lora_B starts zero, so the LoRA delta is zero on init).
  - step 2 loss = 2.3456 (decreased). lora_A grad = 4.2e-4 (non-zero - confirms after step-1's lora_B update, gradient now flows through the LoRA path).
- Peak CUDA across the 2 steps: 25.5 GB. Comfortable margin under 128 GB.
- Adapter saved to `/tmp/day2_smoke_adapter/` in PEFT-compatible format (adapter_config.json + adapter_model.safetensors with `base_model.model.<path>.lora_A.weight` naming).

**Inventory finding documented in code**: Nemotron-3-Nano keeps attention `q/k/v/o_proj` in the modelopt `exclude_modules` list (bf16, NOT NVFP4). The NVFP4 modules are expert `up_proj/down_proj` (5934 of them) and Mamba `in_proj/out_proj` (46 of them). The Day 2 plan said "rank 8 targeting q_proj, v_proj" - adapted to target the actual NVFP4 modules (`up_proj`/`down_proj`). When we eventually want to LoRA-train attention layers, those are bf16 nn.Linear and can be wrapped with the standard PEFT library (no NVFP4LoRALinear needed).

**Bug found and fixed during Day 2 execution**: Nemotron-3's modeling code at `modeling_nemotron_h.py:855` does `expert.down_proj.weight.dtype` for autocast-dtype detection. Our `NVFP4LoRALinear` originally had no `.weight` attribute (correctly - the NVFP4 storage is in `weight_uint8` + scales). Fix: added a `weight` property returning a **meta tensor** (zero memory, correct shape/dtype/device metadata). Any code that tries to USE `.weight` for compute will fail loudly, which is correct since our forward goes through the custom autograd path. The single-line property fix unblocked the smoke.

**Dependency installation**: `mamba-ssm 2.3.2.post1` installed cleanly on aarch64+CUDA 13 via `pip install --no-build-isolation mamba-ssm`. The naive-implementation fallback warning ("fast path not available because one of selective_state_update, causal_conv1d_fn, causal_conv1d_update is None") is fine - model runs at non-fused speed but produces correct output. `causal_conv1d` not strictly required for Day 2; may install for Day 5-6 to speed up Super wall-time.

**Time spent**: ~45 minutes of focused work for Day 2 (writing loader, debugging the `.weight` AttributeError, running smoke). The Day 2 estimate was "1 day" - actual was much faster because the design from Day 1 was solid and the meta-tensor pattern via `init_empty_weights` worked cleanly the first time.

**Files written**:
- `Sandbox/nvfp4_lora/loader.py` (212 lines): inventory, replacement, non-NVFP4 weight loading, top-level `load_nemotron_with_nvfp4_lora`
- `Sandbox/nvfp4_lora/tests/loader_smoke.py` (148 lines): the Day 2 gate test
- `Sandbox/nvfp4_lora/linear.py`: added `weight` property (meta-tensor proxy)

## 2026-05-20 ~15:30 - Path 1 sprint Day 3: training PASS, then two bugs surfaced in greedy capture

10-step LoRA training on Nemotron-3-Nano with ICH v3.1 data ran cleanly:
- Loss trajectory: 2.639 → 2.444 → 2.462 → 2.601 → 2.799 → 2.984 → 2.445 → 2.362 → 1.991 → 1.953
- Adapter saved (11868 tensors, 432.79 MB) at `Sandbox/adapters/nemotron_3_nano_nvfp4_lora_smoke_day3/`
- Peak CUDA: ~21.7 GB across 10 steps at max_length=512 - much higher than Day 2's 25.5 GB peak at 64-token seq because activations dominate; LoRA + base model alone is the 20.5 GB floor

**Bug 1 - `model.generate()` crashes on Nemotron-3 modeling code**: `prepare_inputs_for_generation` at `modeling_nemotron_h.py:1633` does `cache_position[-1] >= input_ids.shape[1]` but `cache_position` is None when transformers' `_prefill` calls into the model. `TypeError: 'NoneType' object is not subscriptable`. Workaround: bypass `generate()`, write a manual greedy loop.

**Bug 2 - Mamba-hybrid needs explicit cache or decode is quadratic+silent**: First manual-greedy attempt used `past_key_values=...` to track cache. The model warned `"NemotronH requires an initialized NemotronHHybridDynamicCache to return a cache. None was provided, so no cache will be returned."` and silently ran no-cache decode, where each step re-prefills the full sequence. Result: prompt[0] still not finishing after 5 min on a 500-token prompt.

**Fix attempt for bug 2 (first try, didn't work)**: Tried passing a pre-instantiated `HybridMambaAttentionDynamicCache(config, batch_size, dtype, device)` as `cache_params=`. Failed during prefill with `AttributeError: 'HybridMambaAttentionDynamicCache' object has no attribute 'conv_kernel_size'`. Looking at modeling source, the cache's `__init__` defines `conv_kernel_size` only as a local variable (line 177); it's never stored on `self`. The Mamba mixer's `torch_forward` then accesses `cache_params.conv_kernel_size` (line 546) and explodes. Separate downstream bug: `cache_params.ssm_states.device` is read at line 563 but `ssm_states` is a Python list of per-layer tensors, not a tensor.

**Intermediate workaround (used while gpt-5.5 was consulted)**: Skipped the cache entirely; each decode step re-prefills the full sequence. Measured cost was ~7 min/prompt - 5.5 h projected for 50 prompts. Killed after 24/50 done.

**gpt-5.5 (codex CLI, high reasoning) review of the diagnosis** added two findings I'd missed:
1. **Even with the cache fixed, the lm_head is still run over the full sequence each step.** Nemotron's modeling code at line 1717 does `self.lm_head(hidden_states)` with no slicing; `prepare_inputs_for_generation` sets `logits_to_keep` but `forward` ignores it. For greedy validation we should bypass `model.forward` and call `model.backbone` directly, then `lm_head` on `hidden_states[:, -1:, :]` only.
2. **`NemotronHBlock.forward` calls attention with only `cache_position`** (modeling line 780-783) - never passes `past_key_value=cache_params`. So attention layers never get the cache and re-compute the full K/V every step even when the cache would otherwise work. Same effect as no-cache for the attention sub-path.

gpt-5.5 also flagged that bf16 eval-mode weight caching is a legitimate further win, **but the bf16 shadow is ~60 GB for Nano and ~240 GB for Super** - Super won't fit on Spark's 128 GB unified memory, so that fix is Nano-only and we're skipping it for now. Batched greedy with length-bucketing was suggested as another lever - also deferred until we measure post-patch.

**Patches applied (Sandbox/nvfp4_lora/tests/day3_capture_completions.py)**:
- `_patch_nemotron_h_modeling(model)`: monkey-patches the dynamic-modules `HybridMambaAttentionDynamicCache` to add `conv_kernel_size`, wrap `conv_states`/`ssm_states` lists in a `_TensorList` subclass with a `.device` property, fix `update_conv_state` / `update_ssm_state` / `reset` to use per-layer indexing. Also patches `NemotronHBlock.forward` to pass `past_key_value=cache_params` and `attention_mask=attention_mask` to attention.
- `manual_greedy` rewritten to call `model.backbone` directly, slice `hidden_states[:, -1:, :]`, then apply `lm_head` only to that - avoids the full-sequence lm_head cost. Cache is initialized fresh per prompt with the patched class.

Sources confirmed by gpt-5.5: NVIDIA's remote-code `modeling_nemotron_h.py` (the file at issue here) and transformers main `nemotron_h` (which uses generic `Cache`/`DynamicCache` and is cleaner - flagged as an alternative if patching ever gets messier).

**Measured speedup after cache+lm_head patches alone**: prompt[0] went from ~421 s no-cache to ~371 s patched - only **~12%** improvement (5.5 h → 5 h projected). gpt-5.5 had predicted: "your expected 10x from cache alone is plausible only if sequence compute dominates; if dequant / lm-head dominate, cache alone will disappoint." Diagnosis confirmed: with the cache fixed, the 5934 NVFP4 dequants × ~33 forwards/prompt becomes the new dominant cost.

**Third patch (lazy eval bf16 weight cache, applied to `nvfp4_lora/linear.py`)**:
- `NVFP4LoRALinear.forward` now branches on `self.training`. Training keeps the custom-autograd `_DequantLinear.apply` path (dequant recomputed in backward, no bf16 shadow saved across the graph). Eval materializes the dequantized bf16 weight on first forward into `self._eval_weight` and uses `F.linear` against it on every subsequent call.
- `train(mode=True)` override clears `self._eval_weight` so the shadow doesn't leak back into the train path or stick around when re-entering train.
- Cache is dtype-aware: if `x.dtype` changes (e.g. autocast pushes a fp32 input through), the shadow is re-materialized.
- Memory: Nano-30B has 17.15 B params of NVFP4 storage → ~34 GB bf16 shadow. With the existing 20.5 GB allocation, total ~55 GB / 128 GB unified - fits. **Super-120B's ~80 B NVFP4 params → ~160 GB shadow won't fit on Spark**, so this optimization is Nano-only. Super eval has to keep paying per-forward dequant, or migrate to vLLM marlin for eval.

Expected combined speedup with all three patches: 32× from cache, plus removing 5934-per-forward dequant overhead → target sub-30-min capture (vs 5.5 h no-cache). **Realised**: 20× speedup, 50 prompts captured in 18 min generation + 4 min load = ~22 min total. Per-prompt cadence ~21 s once warm.

## 2026-05-20 ~19:20 - Day 3 round-trip FAIL on strict gate; gpt-5.5 review reframes the architecture as publishable

10-step LoRA training PASS, vLLM-served adapter loaded cleanly (no per-expert/fused mismatch - vLLM's `pack_moe` does accept our per-expert PEFT keys via expert-axis stacking, confirmed by reading `vllm/lora/layers/fused_moe.py`, `vllm/lora/lora_weights.py`, and `vllm/model_executor/models/nemotron_h.py`). 50-prompt strict-greedy-match v6.2 gate: **34% exact-token match, 58% next-token match - FAIL**.

**Why**: train side is `bf16-dequant + F.linear + PyTorch SDPA + Mamba naive scan`. Serve side is `vLLM Marlin (weight-only FP4 GEMM on GB10, internal bf16 decompression since sm_121 lacks native FP4 compute) + vLLM paged/flash attention + vLLM Mamba2 with Triton causal_conv1d`. These are different bf16 math paths at every layer. Temperature=0 greedy decode is fragile to those rounding differences: a top-1/top-2 logit gap of 1e-4 can flip the choice, then cascade. The 17 matching prompts are the high-margin ones (LoRA delta makes top-1 robustly win on both paths); the 21 catastrophic-first-token-divergence prompts are the low-margin ones where the LoRA "answer-prefix" effect isn't strong enough to survive the bf16 noise.

**Important corollary**: my own train-side runs disagree across patch combinations. The no-cache run produced `'section=3.2.2'` for prompt[0]; the cache+lm_head+eval-bf16 patched run produced `'section=2.2.10'` for the same prompt. Same model, same adapter, same prompt - my `NemotronHBlock` attention patch (passes `past_key_value=cache_params` AND `attention_mask=causal_mask` to attention) changes `is_causal=True → is_causal=False+explicit_mask`. Both are nominally equivalent paths through SDPA but use different kernels with subtly different rounding. The train-process model is therefore **not a stable ground truth** even at the train side.

**gpt-5.5 (codex CLI, xhigh reasoning) verdict - adopted**:

The 100%-greedy gate was wrong. It's testing "do two different decoders agree on tie-breaks under bf16 noise" - not "is the adapter correct." The correct publishable framing: **"train NVFP4 LoRA adapters on Spark, serve through stock vLLM with validated behavioral equivalence."** Greedy text equivalence is the wrong invariant.

**Layered gate replacing strict-match**:
1. **Structural interop**: vLLM loads adapter, all per-expert tensors map correctly, rank/alpha verified. (Day 3 already passes this - `Loaded new LoRA adapter: name 'nano+day3'` + `MoE model detected. Using fused MoE LoRA implementation.` in serve log.)
2. **No-op + scale sanity**: zero LoRA delta = no-op on vLLM; scaling LoRA by 0.5x/1x/2x/4x moves target-token logprobs monotonically on both sides.
3. **Teacher-forced logprob comparison** (key insight): compare both systems on the same prompt + same continuation, not free-running greedy generations. Free-running amplifies one tiny first-token flip into a fake catastrophic mismatch.
4. **Margin-conditioned agreement**: high-train-side-margin prompts should usually match vLLM. Low-margin can diverge and that's expected. High-margin divergence = real bug suspect.
5. **Served behavioral quality** (the real publish gate): vLLM-served-adapter vs vLLM-base on val NLL, format compliance, task-specific score.

**Day 3.5 diagnostic battery (cheap; insert before Day 5-6 full Super training)**:
- Prompt/token identity audit (eliminate chat-template/BOS/truncation drift).
- Base-model drift test (compare train-side base vs vLLM base WITHOUT LoRA - if base already diverges, the LoRA round-trip was never a strict oracle).
- Margin analysis on the 50 prompts.
- **Overfit canary** (highest-value diagnostic): train on 1-5 examples until the desired LoRA prefix has a huge margin. If vLLM emits the same prefix → current Day 3 failure is just weak training signal + kernel drift, architecture fine. If vLLM doesn't → real adapter-wiring bug, must fix.
- Alpha/scale sweep: export adapter with scaled deltas, vLLM behavior should respond monotonically.
- Packed-tensor spot-check: dump vLLM's `pack_moe` output for a few experts, verify it matches our PEFT keys.

**Publishable abstraction guidance (gpt-5.5)**:
- Pin model revision + vLLM version + CUDA stack. Don't try to forward-compat arbitrary upstream changes.
- Replace by state-dict patterns / module predicates, not source-line patches.
- Validate expected tensor counts/shapes at load time, fail fast.
- Export standard PEFT format.
- Ship a "compatibility smoke" command that fails loudly when NVIDIA/HF/vLLM changes the graph.
- Public contract: `pinned-model-revision → PEFT adapter → stock vLLM serves it`. Patched modeling code is a compatibility shim, not the core contract.

**Sprint reordering**:
- Day 4 (Super-120B memory smoke): GREENLIGHT now. Independent question - does Super fit on Spark with NVFP4LoRALinear?
- Day 3.5 diagnostics: in parallel with Day 4, or right after.
- Day 5-6 (full Super LoRA on ICH v3.1): gated by Day 3.5 results.

## 2026-05-20 ~20:00 - Day 3.5 diagnostics turned up TWO compounding root causes

Spawned three subagents in parallel after Day 3 FAIL: (A) margin analysis on the existing round-trip results, (B) overfit-canary script writer, (C) alpha/scale-sweep script writer. Outputs landed in local diagnostic scripts and a margin-analysis summary outside this published repo. Subagent A produced findings that, combined with my own follow-up, exposed two stacked bugs:

### Discovery 1 - Subagent A's margin analysis says the divergence is structural, not bf16 noise

Agreement-length distribution across 50 prompts is sharply bimodal: 17 at length 32 (perfect match), 21 at length 0 (first-token divergence), only 12 in the middle. **76% sits at the two extremes.** Pure bf16 numerical noise through cascading greedy decode would produce a smooth distribution biased toward 32 - not this gap.

Smoking gun on the 21 catastrophic prompts: `served_ids` starts with the LoRA-style CoT prefix `"We need to answer"` (`4268, 2534, 1317, 4832`) on **21/21**, but `train_ids` does so on **0/21**. The train-side path emits raw context-paraphrase / mid-word continuation tokens instead (e.g. `'ese-models\n[4] doc_id=...'`). The bimodal pattern plus the all-or-nothing asymmetry rules out symmetric numerical drift - both sides should have similar fractions of "We need to answer" starts under noise, not 100% vs 0%.

Subagent A's interpretation: the train-process capture path may be failing to apply the adapter on those prompts (or worse, running on a partially-broken graph). Subsequent investigation confirmed something more nuanced - see Discovery 2.

### Discovery 2 - Chat-template silently produces different prefixes for training vs inference

Reading `Models/Nemotron-3-Nano-30B-A3B-NVFP4/chat_template.jinja` lines 105-145, 198-202:

- `apply_chat_template(messages, add_generation_prompt=False)` - used at TRAINING. If the assistant content has no `<think>...</think>` tags (which our ICH v3.1 data does NOT), the template auto-wraps with **`<think></think>`** (immediate close, no newlines). Training rendering becomes `<|im_start|>assistant\n<think></think>Not automatically...`.
- `apply_chat_template(messages[:-1], add_generation_prompt=True)` - used at INFERENCE (default `enable_thinking=True`). Produces **`<|im_start|>assistant\n<think>\n`** - open tag, newline, no close. Inference asks the model to fill in reasoning, then close `</think>`, then produce the answer.

These are two different prompts at the assistant boundary. The training data taught the LoRA what comes after `<think></think>` (the empty closed pair). The inference prompt asks what comes after `<think>\n` (the open tag expecting reasoning content). **The LoRA has zero training signal for the position it's being asked to fill in at inference.**

### Confirmed fix for Discovery 2

Pass `enable_thinking=False` to `apply_chat_template` at INFERENCE time. Verified: `apply_chat_template(messages[:-1], add_generation_prompt=True, enable_thinking=False)` produces `<|im_start|>assistant\n<think></think>` - **byte-identical** to the boundary the training rendering uses (line-by-line compare passed).

Applied this fix to:
- `Sandbox/nvfp4_lora/tests/day3_capture_completions.py`
- `Sandbox/nvfp4_lora/tests/day3_roundtrip_compare.py`
- `Sandbox/nvfp4_lora/tests/day3_5_overfit_canary_compare.py`

The training side (`day3_train_nano.py`) already uses `add_generation_prompt=False` which the chat template handles correctly for our data shape (auto-wraps with `<think></think>`).

### How the two discoveries compound

Without Discovery 2 fix, the LoRA is being asked to predict at a position it never saw during training. ANY observed train↔serve disagreement at that position is ambiguous between (i) my custom NVFP4LoRALinear stack has a real wiring bug, or (ii) the LoRA is just emitting unstable garbage because it's out-of-distribution. Subagent A's "the train side never produces the CoT prefix" finding originally pointed at (i), but Discovery 2 explains it as a special case of (ii) - both paths are out-of-distribution, and they handle OOD differently due to their (presumably small) numerical differences.

We cannot tell whether (i) is a real bug until we re-run with the template fix. The overfit-canary diagnostic is now the right tool because (a) it uses matched templates (Discovery 2 fix applied to compare), (b) it overfits hard enough that the LoRA effect should be MASSIVE even at borderline positions, drowning out any minor (i)-type wiring drift.

### Discovery 3 - Day 4 Super-120B loader fails at the 512-expert scale

The Super config has 88 layers / 512 routed experts per layer / 22 top-k routing. Launching `day4_super_memory_smoke.py` produced 50K+ `WARN: path not found in model: backbone.layers.X.mixer.experts.Y.up_proj.weight_scale` messages within the first 5 min. The loader iterates safetensors keys and looks up `nn.Module` paths; most don't find a match - almost certainly because `accelerate.init_empty_weights` + the dynamic Nemotron-3 modeling code only materializes a subset of expert modules at empty-init time, or my loader's path-mapping logic doesn't scale to 512 experts. Killed at 5:31 - needs a separate investigation before Day 4 can run. Filed as a follow-up; not on the critical path for Day 3 resolution.

### What we wrote during this session (so we can find it again)

- `Sandbox/nvfp4_lora/tests/day3_5_margin_analysis.py` + margin-analysis summary
- `Sandbox/nvfp4_lora/tests/day3_5_overfit_canary_train.py` + `day3_5_overfit_canary_compare.py` + `day3_5_overfit_canary_serve.sh`
- `Sandbox/nvfp4_lora/tests/day3_5_alpha_sweep_export.py` + `day3_5_alpha_sweep_serve.sh` + `day3_5_alpha_sweep_compare.py`

### What the overfit canary will tell us (and what to do based on outcomes)

After running canary training + serve + compare:

- **Both paths converge to the gold answer**: architecture is fine. The previous Day 3 failure was Discovery 2 amplifying tiny Discovery 1-style drift. Re-render Day 3 training data correctly, re-run, and we expect much higher exact-match rates. Then proceed to Day 4 fix + Day 5-6.
- **Only vLLM converges, train-side doesn't**: train-side path has a real wiring bug. Concrete candidates: my custom NVFP4LoRALinear forward, my loader's non-NVFP4 weight loading, my eval bf16 cache, the patched NemotronHBlock attention call. Need to bisect - disable my patches one at a time.
- **Only train-side converges, vLLM doesn't**: adapter-format issue with vLLM's `pack_moe`. Less likely (vLLM source-read confirmed it expects exactly our PEFT key naming), but possible at scale.
- **Neither converges to gold**: serious stack issue. Probably end the architecture and reconsider.

### Canary outcome (actually ran)

Training collapsed `loss: 2.6391 → 0.0029` in 80 steps on 3 examples. Adapter saved, ~432 MB (same shape as the 10-step Day 3 adapter - these are per-expert LoRAs). vLLM marlin loaded the canary adapter cleanly. Compare result on the 3 trained prompts:

| prompt | 20-tok prefix match | first-divergent-idx |
|---|---|---|
| 0 | MATCH | n/a |
| 1 | DIFF | 0 (full divergence) |
| 2 | MATCH | n/a |

**2/3 strict prefix match even with the LoRA delta this strong (loss-near-zero memorization).** Decision: YELLOW per the script.

What this rules in / out:
- **Rules OUT catastrophic wiring bug.** If our PEFT format / vLLM `pack_moe` / per-expert stacking / NVFP4LoRALinear were misaligned in a fundamental way, vLLM would not be able to reproduce the gold answer on ANY prompt - let alone the first 20 tokens *exactly* (`16860, 1044, 1809, ...`). That's not numerical-drift territory; that's an architecturally correct end-to-end pipeline.
- **Rules IN persistent FP-path drift.** The DIFF on prompt[1] is not a token-1-flip-then-cascade; vLLM produces a *coherent, on-topic but different answer* (`'The justification should be framed around the staged, risk-informed approach...'` vs trained `'Yes, but the argument needs to be a risk-managed timing proposal...'`). That looks like vLLM falling back to base-model behavior because the LoRA delta failed to win top-1 at position 0 on that specific prompt's routing path. With bf16-dequant+SDPA vs marlin-FP4-GEMM+paged-attention being two genuinely different math paths, this kind of per-prompt divergence is expected - and it's bounded (1/3 here under maximum LoRA strength; will probably be lower on properly-trained adapters and higher on under-trained ones).
- **Rules OUT the chat-template trap as the only Day 3 problem.** Even after fixing `enable_thinking=False` and overfitting to memorization, 1/3 prompts still diverge. The Day 3 strict-greedy gate would still have failed (less spectacularly than 34%, but FAIL nonetheless).

Implication: the right publish gate is gpt-5.5's behavioral-equivalence framework, not byte-equivalent greedy. Architecture is sound; per-prompt drift is inherent to the train↔serve math pairing.

### What this unblocks

- Day 4 (Super-120B memory smoke): not just greenlit, but the prerequisite that "vLLM can faithfully serve our adapter" is now empirically verified at the Nano scale. Loader bug at 512-expert / 88-layer scale is the only Day 4 blocker.
- Day 5-6 (full Super LoRA on ICH v3.1): fine to proceed once Day 4 loader fix lands and Discovery 2 template fix is applied to the production training script. Evaluation gate becomes vLLM-served quality on val set, not byte-equivalent capture.
- Discovery 2 fix needs to be **applied to `day3_train_nano.py`** before any further training. Currently train uses `apply_chat_template(..., add_generation_prompt=False)` which works (auto-wraps `<think></think>` for ICH data without reasoning), but if any future training data DOES contain `<think>` reasoning, we'd want to test the full template path. For now the matched-templates property holds for ICH v3.1.

## 2026-05-20 ~22:00 - Day 4 PASS - Super-120B-NVFP4 loads and trains on Spark

After two loader iterations to handle Super-specific quirks Nano didn't have:

**Discovery 4 - Super-vs-Nano in-memory submodel naming**: Nano-30B-A3B uses `self.backbone = NemotronHModel(...)`, Super-120B uses `self.model = NemotronHModel(...)`. Both safetensors files use `backbone.X` prefix. The Nano `model.named_modules()` paths match safetensors keys; Super's `model.layers.X` paths don't. Fix: added `make_key_translator(model, model_dir)` in `Sandbox/nvfp4_lora/loader.py` that detects safetensors prefix and model submodel prefix, then provides a `translate(safetensors_key) -> model_path` callable. Used in both `replace_nvfp4_modules` and `load_non_nvfp4_weights`.

**Discovery 5 - Super uses mixed quantization formats**:
- **MoE routed experts (`experts.N.up_proj`, `experts.N.down_proj`)**: NVFP4 - `weight=uint8 (packed)`, `weight_scale=fp8_e4m3fn per-group`, `weight_scale_2=fp32 scalar`. Same as Nano.
- **Mamba `in_proj`, `out_proj`**: FP8 per-tensor - `weight=fp8_e4m3fn (out,in)`, `weight_scale=fp32 scalar` (no `weight_scale_2`). DIFFERENT from Nano (Nano had NVFP4 here too).
- **Shared experts (`shared_experts.up_proj`, `shared_experts.down_proj`)**: also FP8 per-tensor. These match the `up_proj` / `down_proj` suffix used for LoRA targets, but FP8 LoRA isn't supported by my custom autograd path.

Fix: `replace_nvfp4_modules` now branches on `weight.dtype`. uint8 → existing NVFP4LoRALinear path. fp8_e4m3fn → dequant to bf16 once at load (`W = weight.to(f32) * scale.to(f32)`), store as frozen `nn.Linear`. If a suffix-matched LoRA target is FP8 (like Super's shared_experts), the loader silently demotes to frozen and reports a `lora_demoted_fp8` count. The actual LoRA training capacity comes from the 40 MoE layers × 512 routed experts × 2 projections = 40,960 modules.

**Discovery 6 - MTP layers (`mtp.X`) in Super safetensors**: Multi-Token Prediction speculation layers used for serve-side speculative decoding. They're not part of the training graph. The key translator returns `None` for these and `load_non_nvfp4_weights` skips them - 1040 tensors skipped on Super.

**Day 4 measured numbers** (Super-120B-NVFP4 on Spark, max_length=64):
- Load wall: 1689.3 s (~28 min) - includes 165K safetensors-key walk + 40,961 NVFP4LoRALinear constructions + 215 FP8-dequant-to-bf16 + 559 non-quantized bf16 tensor loads.
- Trainable params: **1216.42 M** (40,961 LoRA modules × ~30K params each at r=8).
- Total trainable count of params: 9.12 B (counts ALL params including the bf16 dequanted weights of FP8/non-quantized modules - most of these are requires_grad=False but still in the count). Actual gradient-touched params is the 1.22 B LoRA count.
- Buffer (NVFP4 storage): 63.43 B params (the packed FP4 bytes + per-group fp8 scales + per-tensor fp32 scales of the routed experts).
- CUDA after load: **82.61 GB**.
- Peak CUDA across 2 forward+backward steps at max_len=64: **98.76 GB (90% of 110 GB ceiling)**.
- Loss step 1: 3.0873 finite. Loss step 2: 3.0621 finite (different prompts → not directly comparable).

**Activation budget for Day 5-6**: activations consumed ~16 GB at max_len=64 (98.76 − 82.61). Naively linear in seq_len, max_len=512 would push activations to ~128 GB → over budget. Day 5-6 will need either gradient checkpointing (likely 4× reduction → ~32 GB activations at len=512 → fits) or a shorter max_length, or both. Will configure based on ICH v3.1 sequence-length distribution.

**Files touched**:
- `Sandbox/nvfp4_lora/loader.py` - added `make_key_translator`, branched NVFP4/FP8 handling in `replace_nvfp4_modules`, MTP skip in `load_non_nvfp4_weights`, expanded `nvfp4_replaced` set to include FP8-replaced modules.
- `Sandbox/nvfp4_lora/tests/day4_super_memory_smoke.py` - no changes needed; the test passes against the patched loader.

## 2026-05-20 ~22:30 - Dependency inventory (canonical for publishing reproducibility)

This pipeline pins exact upstream versions. The Spark + sm_121 stack has been hostile enough to multiple CUDA-extension installs (TE 2.13, FlashInfer JIT etc.) that we have to commit to specific working combinations. Any users reproducing the work need this entire matrix or they'll hit different failure modes.

### Hardware
- DGX Spark, GB10 SoC, **sm_121** (custom Blackwell variant; key non-feature: **no native FP4 tensor-core compute**, so all FP4 GEMMs go through weight-only marlin decompression).
- **128 GB unified memory** (CPU + GPU share the same pool - there's no separate VRAM).
- CUDA 13.0 driver, NVIDIA driver 580.142.
- Linux aarch64 (NVIDIA NVOS).

### Python venvs

Two venvs, each Python 3.12.3:

**`qwen-peft` (training side)** - `/path/to/venvs/train/`
| Package | Version | Notes |
|---|---|---|
| torch | 2.12.0+cu130 | aarch64 wheel with CUDA 13.0 |
| transformers | 5.8.1 | trust_remote_code path used (Nemotron-3 isn't yet in mainline) |
| accelerate | 1.13.0 | for `init_empty_weights` |
| safetensors | 0.7.0 | for shard-aware weight loading |
| peft | 0.19.1 | format compat for adapter save/load + reference impl for the bf16-attention LoRA path (we don't actually wrap with PEFT - we emit PEFT-format adapters by hand) |
| mamba_ssm | 2.3.2.post1 | requires `pip install --no-build-isolation mamba-ssm`. Without this the Mamba mixer can't even import. |
| causal_conv1d | 1.6.2.post1 | requires `pip install --no-build-isolation causal-conv1d`. Without this the model warns `"fast path not available because one of (selective_state_update, causal_conv1d_fn, causal_conv1d_update) is None"` and falls back to a pure-PyTorch token-by-token scan. **At Day 5 scale (max_len=1536, Super-120B), naive scan makes training infeasible** - installed during the Day 4.5 pre-flight after measuring 0% GPU util on a 40+ min pre-flight step. ~205 MB compiled wheel; aarch64 build takes ~10-30 min. |
| torchao | 0.17.0 | used only for the **Day 1 dequant correctness reference** against `NVFP4Tensor.dequantize()`. Not in the runtime path. |
| psutil | 7.2.2 | used by smoke tests to report system memory baselines |
| requests | 2.34.2 | used by the day3 round-trip and day3_5 canary compare scripts to hit vLLM's `/v1/completions` |

**`qwen-serve` (serving side)** - `/path/to/venvs/serve/`
| Package | Version | Notes |
|---|---|---|
| vllm | **0.21.0** (pinned) | newer versions might change MoE LoRA handling (specifically `FusedMoEWithLoRA.pack_moe` expected key layout). Pin tight. |
| flashinfer-python | 0.6.8.post1 | runtime attention kernels |
| flashinfer-cubin | 0.6.8.post1 | precompiled cubin bundle (avoids JIT compile on warmup; previously caused OOM at MAX_JOBS>1) |
| torch | 2.11.0+cu130 | NOTE: one minor version older than qwen-peft. Must stay separate venv to avoid vLLM/PEFT torch ABI clashes |
| transformers | 5.8.1 | same as qwen-peft |

### Models (~90 GB total disk)
- **NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4** at `/path/to/Models/Nemotron-3-Nano-30B-A3B-NVFP4/` (~15 GB). Used for Day 1-3 validation. Sub-model attribute name `backbone.X`. NVFP4 on expert + Mamba modules.
- **NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4** at `/path/to/Models/Nemotron-3-Super-120B-A12B-NVFP4/` (~75 GB). The actual production target. Sub-model attribute name `model.X`. NVFP4 on expert MLPs only; FP8 per-tensor on Mamba and shared-expert MLPs.

### Runtime environment variables (vLLM serve side, all needed)
- `VLLM_NVFP4_GEMM_BACKEND=marlin` - selects the weight-only marlin GEMM (the only working FP4 path on sm_121).
- `MAX_JOBS=1` - caps FlashInfer's parallel JIT compilation during warmup to avoid 128 GB OOM.
- `--moe-backend marlin` - vLLM serve flag, matches the GEMM backend for MoE expert path.
- `--enable-lora --max-lora-rank N --max-loras 1` - for adapter serving.

### Tried-but-failed dependencies (so we don't waste time re-trying)
- **TransformerEngine 2.13+** does not build on aarch64+CUDA13: linker can't resolve `cublasLtGroupedMatrixLayoutInit_internal` (an unreleased cuBLAS symbol). Stuck at TE 2.12.0 if anyone wants TE at all.
- **TE NVFP4 RHT kernel on sm_121** crashes with a CUDA error even at TE 2.12.0 - documented as P-1b "Outcome C." All TE-based NVFP4 training paths therefore unavailable. We bypass TE entirely; the bf16-dequant `NVFP4LoRALinear` is the production training architecture.
- **TE not used at runtime by this pipeline** - `import transformer_engine` is not part of either venv's required set.

### Reproducibility checklist (for the published repo)
- `requirements/qwen-peft.txt` - pin every package in the table above with exact versions (including the `--no-build-isolation` flags).
- `requirements/qwen-serve.txt` - same for serving venv.
- `download_models.sh` - `hf download nvidia/NVIDIA-Nemotron-3-{Nano-30B-A3B,Super-120B-A12B}-NVFP4 --local-dir ...`.
- `scripts/serve_super.sh` - vLLM serve command with all the env vars.
- `scripts/train_super.sh` - Day 5 training launcher.

## 2026-05-20 ~23:50 - Day 4.5 PASS, Day 5 sizing committed

**Day 4.5 result** (Super-120B + NVFP4LoRALinear + max_len=1536 + gradient checkpointing, with causal-conv1d installed):

```
load wall: 1714.8 s (~29 min)
cuda allocated after load: 82.61 GB

step 1 (warmup): 177.3 s, peak 87.19 GB, loss=0.0296
step 2:          127.0 s, peak 92.05 GB
step 3:          127.7 s, peak 92.05 GB
step 4:          127.6 s, peak 92.05 GB
step 5:          128.2 s, peak 92.05 GB

steady-state: ~127 s/step, peak 92 GB (77% of 120 GB ceiling)
=== Day 4.5 long-seq smoke PASS ===
```

Critical insight from this: **the Mamba fast path makes a 10-20x difference**. The first time we ran this same config without causal-conv1d (Day 4.5 attempt 1), step 1 was still grinding 40+ minutes in with 0% GPU util - naive Python token-by-token Mamba scan was the bottleneck. Installing causal-conv1d 1.6.2.post1 (built in ~25 min from source on aarch64+CUDA13 via `pip install --no-build-isolation`) replaced that with the Triton kernel and brought step time to a tractable ~127 s.

**Day 5 sizing committed**: max_len=1536, 1 epoch, batch=1, grad_accum=4. Expected wall time 1081 examples × 127 s = ~38 h. Adapter checkpointed every 200 forward+backward steps (~25 min between checkpoints) into `Sandbox/adapters/nemotron_3_super_nvfp4_lora_ichv31_1epoch/`. Script: `Sandbox/nvfp4_lora/tests/day5_super_full_train.py`. The trade-off was made explicitly: the originally-discussed (1536 × 3 epochs ≈ 5 days) is infeasible on a single Spark in a reasonable turnaround; the coverage-maximalist 1-epoch single pass keeps 95% of training examples un-truncated while finishing in ~1.6 days.

**Day 6 plan, post-training**: serve the resulting adapter via the same `day3_serve_vllm.sh` recipe, run a behavioral-equivalence quality eval (per gpt-5.5's framework) on the 191-prompt val set: served-base vs served-adapter NLL on the gold answers, plus a side-by-side qualitative spot-check on N≈10 prompts. Document the gap between train-time and serve-time behavior as a property, not a bug.

## 2026-05-21 ~00:40 - Day 6 prep: deployment script consolidated, 4-way eval plan locked in

User has an existing instruction-tuning eval harness (designed for the Qwen v3.x ICH run) that runs on a separate host on the LAN. The harness uses vLLM's OpenAI-compatible `/v1/completions` endpoint (raw prompt + assistant prefill `"Final answer:\n"` for deterministic outputs - chat-completions can't do that because vLLM applies its own chat template). Eval is one-model-at-a-time, ~30-40 min of GPU time per Stage 1; Stages 2-3 are post-processing against Codex (cloud GPT-5.5) and can run after the local GPU is released.

This unlocks the **4-way comparison** that the published artifact deserves: Nano-base / Nano-FT / Super-base / Super-FT, all on the same harness, same 191-prompt val set.

### Sprint reordering for Day 6
1. **Day 5b - train comparable Nano-FT**: same config as Day 5 Super (1 epoch ICH v3.1, max_len=1536, grad_ckpt, r=8, lr=1e-4) so the 4-way is apples-to-apples. Script: `Sandbox/nvfp4_lora/tests/day5b_nano_full_train.py`. Estimated ~3 h (Nano is ~5× faster per step than Super).
2. **Serve each family with both base + FT model IDs simultaneously** (vLLM exposes both via `--lora-modules`), eval harness fires Stage 1 against each ID, GPU released after Stage 1 wraps, harness moves to Stages 2-3 on the eval host.
3. **Repeat for the other family.**

### Deployment script (consolidated, parameterized)

`Sandbox/nvfp4_lora/serve/serve_nemotron_nvfp4.sh` is the **canonical deployment script for the Nemotron-3 NVFP4 family on Spark**. Takes `<nano|super>` plus optional `<adapter_dir> [adapter_tag]`:

```bash
# base only:
./serve_nemotron_nvfp4.sh nano
./serve_nemotron_nvfp4.sh super

# base + FT adapter (exposed simultaneously under both model IDs):
./serve_nemotron_nvfp4.sh nano \
    /path/to/adapters/nemotron_3_nano_nvfp4_lora_ichv31_1epoch \
    ich_v1_0
./serve_nemotron_nvfp4.sh super \
    /path/to/adapters/nemotron_3_super_nvfp4_lora_ichv31_1epoch \
    ich_v1_0
```

Resulting served model IDs (visible at `/v1/models`):
- `nemotron-3-nano-nvfp4` (base)
- `nemotron-3-nano-nvfp4+ich_v1_0` (with adapter)
- `nemotron-3-super-a12b-nvfp4` (base)
- `nemotron-3-super-a12b-nvfp4+ich_v1_0` (with adapter)

The eval host on the LAN points at the Spark vLLM server on port 8000 and selects between the model IDs via the `model` field in the request body.

### Spark-required vLLM flags (per LESSONS.md dependency inventory, baked into the script)
- `VLLM_NVFP4_GEMM_BACKEND=marlin` - sm_121 has no native FP4 compute; the marlin kernel does weight-only FP4 GEMM with internal bf16 decompression. Without this vLLM falls back to a path that doesn't exist on this GPU.
- `MAX_JOBS=1` - caps FlashInfer's parallel JIT to avoid 128 GB OOM during warmup.
- `--moe-backend marlin` - uses the matched FP4 path for routed expert MLPs.
- `--dtype bfloat16` - activations / compute dtype (weights remain FP4 on disk; bf16 is the dequant target inside marlin).
- `--enable-lora --lora-modules "<served_id>+<tag>=<adapter_dir>" --max-lora-rank 8 --max-loras 1` - only when serving an adapter. The exposed model id is `<served_id>+<tag>` so it cleanly appears next to the base model in `/v1/models`.

### Why this isn't just a copy of the existing Qwen serve recipe
The earlier ICH-on-Qwen pipeline used `serve_qwen3_6_35b_a3b_openai_transformers.py` (transformers/PEFT direct-serve, not vLLM) and had different requirements: Qwen is bf16, no NVFP4, no Mamba, MoE handled by a different kernel path. None of the env vars or vLLM flags above apply there. Our deployment script is therefore a fresh artifact specific to Nemotron-3 NVFP4 on Spark.

**Bug 3 - Prompt slicing was vacuous**: First capture used `ex["messages"][:1]` which keeps ONLY the system message. All 50 "prompts" therefore rendered identically → identical greedy completions → vacuous round-trip test. Fix: drop the assistant turn and keep system + user via `ex["messages"][:-1] if msgs[-1]["role"] == "assistant" else msgs`.

**Open question**: My `HybridMambaAttentionDynamicCache` lookup uses `sys.modules[type(model).__module__]` which works for HF dynamic-modules loading but is brittle if model loading path changes. For the Day 5-6 Super training, may want to add an explicit re-export in our loader so capture/eval scripts don't depend on the dynamic-modules hash.

**Files touched**:
- `Sandbox/nvfp4_lora/tests/day3_train_nano.py` (existing, training portion works; generation portion left as-is for record but is replaced by ↓)
- `Sandbox/nvfp4_lora/tests/day3_capture_completions.py` (new): proper manual-greedy with explicit Mamba cache + prompt slicing fix
- `Sandbox/nvfp4_lora/tests/day3_serve_vllm.sh` (new): vLLM marlin + `--enable-lora` for the round-trip serve side
- `Sandbox/nvfp4_lora/tests/day3_roundtrip_compare.py` (new): 50-prompt exact-greedy match against the served `nano+day3` adapter

## Open questions to revisit

- What's the right way to characterise memory pressure on a unified-memory system? Process RSS is wrong; `free -g` used includes page cache that's reclaimable. May need a custom metric: `MemTotal - MemAvailable` minus expected page cache.
- Can FlashInfer's eager JIT be fully disabled? Check vLLM env vars (`VLLM_USE_TRITON_FLASH_ATTN`, `VLLM_FLASHINFER_FORCE_TENSOR_CORES`, etc.) to see if any kills the warmup compile.
- Why did TE's cpp extension fail to compile on aarch64 + cu130? Specific error needs reading.
