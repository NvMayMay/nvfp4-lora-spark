# Troubleshooting recipes for NVFP4 + LoRA on DGX Spark

A playbook of common errors we hit during development, mapped to causes
and fixes. If you hit one of these, you can shortcut your debugging.

## vLLM serve errors

### `ValueError: NvFp4 MoE backend 'FLASHINFER_TRTLLM' does not support the deployment configuration since kernel does not support current device cuda`

**Cause**: vLLM 0.21's FlashInfer NVFP4 MoE kernels are not compiled for
sm_121 (Blackwell consumer / DGX Spark). Only `MARLIN`, `EMULATION`, and
`VLLM_CUTLASS` accept this device in 0.21.

**Fix**: pass `--moe-backend cutlass` (recommended for Super; gets
~12-14 tok/s with native NVFP4 path). If you hit memory issues, use
`--enforce-eager` to skip the CUDA graph capture phase.

### `ValueError: NvFp4 MoE backend 'VLLM_CUTLASS' does not support the deployment configuration since kernel does not support LoRA`

**Cause**: vLLM 0.21's CUTLASS MoE kernel (`CutlassExpertsFp4`) has
`supports_lora() = False`. The kernel itself doesn't have a LoRA-aware
forward path. Same for `FlashInferExperts`. Only `MARLIN` and `EMULATION`
claim LoRA support, but Marlin OOMs on Spark and EMULATION's LoRA Triton
kernel has a known bug.

**Fix**: use the merge-then-serve workflow.
`scripts/merge_lora_into_nvfp4.py` bakes your LoRA into the NVFP4 base
weights; then serve the merged checkpoint via the CUTLASS recipe. See
`serve/run_super_ft_merged.sh`.

For dynamic adapter swap (no merge), see `docs/PHASE2.md` for the
upstream-PR work in progress.

### `RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered ... _fused_moe_lora_expand`

**Cause**: known vLLM 0.21 bug. The Triton `_fused_moe_lora_expand`
kernel assumes Marlin-format MoE weight layout. With the EMULATION
backend, weights are still in raw NVFP4 packed form; the kernel
dereferences off-end and crashes during dummy warmup.

**Fix**: don't combine `--enable-lora` with `--moe-backend emulation`.
The merge-then-serve workflow above avoids this entirely. Upstream issue
draft: `Research/nvfp4_lora_spark/vllm_issue_draft_fused_moe_lora_emulation_bug.md`.

### `ValueError: No available memory for the cache blocks. Try increasing gpu_memory_utilization`

**Cause**: `--gpu-memory-utilization` is too low. After loading the
~69 GiB Super weights, the remaining budget for KV cache went negative.

**Fix**: bump to at least 0.70 for Super with our tight knobs. The
working number is 0.70 (gives ~12 GiB KV budget) to 0.92 (~37 GiB,
default).

### `NVRM NV_ERR_NO_MEMORY` in dmesg, vLLM hangs

**Cause**: the only situation I've seen where vLLM exceeds the physical
memory ceiling on Spark is with `--moe-backend marlin` on a model whose
per-expert Marlin repack pushes transient memory above 130 GiB. This is
unrecoverable; the process becomes unkillable until a hard reboot.

**Fix**: do not use `--moe-backend marlin` for Super (or any model
where this happens). Use `cutlass` or `emulation` instead. See
`serve/diagnostics/` for the safety-thread harness that prevents this
from happening to a diagnostic process.

### `safety_emergency RAM avail X < Y GB` in diagnostic harness logs

**Not a vLLM bug.** Our diagnostic wrapper (`serve/diagnostics/diag_vllm_safe.py`)
has a safety thread that calls `killpg(SIGKILL)` if `psutil.virtual_memory().available`
drops below a configurable floor (default 4 GB). This prevents NVRM kernel
thrash. If you see this, vLLM was approaching real OOM; the diagnostic
wrapper saved you from a hard reboot.

### Memory metrics look stale or inconsistent

`torch.cuda.mem_get_info()` on Spark UMA reports MemFree-equivalent
which **excludes** reclaimable OS page cache. After loading a large
model and then exiting the process, `cuda_free` will look ~40 GB lower
than the boot baseline even though no process is using it. This is the
page cache from the safetensors mmap.

**The CUDA driver reclaims it on demand**, so it's not actually a
leak. Verify with `torch.empty(100_000_000_000, device='cuda')` - it'll
succeed even when `cuda_free` reports < 100 GB.

For correct UMA memory readings, prefer `psutil.virtual_memory().available`
(MemAvailable) over `cuda_free`.

## Training errors

### LoRA adapter loaded but model output is base-like (no FT signal)

**Cause**: in Super-120B, shared expert MLPs and Mamba projections are
FP8, not NVFP4. The training loader silently demotes any LoRA target
on those modules to frozen (with a count printed at load time).

**Fix**: target only `up_proj` and `down_proj` on the routed (NVFP4)
MoE experts. If the printed "frozen LoRA modules" count at load time
is suspiciously high, double-check `target_modules` in adapter config.

### `causal-conv1d not found` or "naive Python scan" warning during training

**Cause**: `causal-conv1d` C++ extension didn't build against your
CUDA toolchain. Without it, Mamba2 falls back to a naive Python scan
and training is effectively infeasible at any useful sequence length.

**Fix**: rebuild with `MAX_JOBS=1 pip install --no-build-isolation causal-conv1d==1.6.2.post1`.
The `MAX_JOBS=1` cap prevents nvcc from being OOM-killed during
parallel compilation on the 128 GB unified pool.

## Merge script (`scripts/merge_lora_into_nvfp4.py`)

### `shape '[X, Y]' is invalid for input of size Z` in `NVFP4QTensor.dequantize`

**Cause**: passed packed shape instead of logical shape when constructing
`NVFP4QTensor`. Packed shape is `(out, in//2)`; logical shape is
`(out, in)` since NVFP4 stores 2 elements per byte.

**Fix**: when constructing `NVFP4QTensor(input_shape, ...)`, pass
`(packed.shape[0], packed.shape[1] * 2)`. Our merge script does this
correctly via `get_nvfp4_dequant_then_merge`.

### Merge runs but `validate_merge.py` reports high no-op fraction

**Cause**: the LoRA delta magnitude is below the NVFP4 quant step for
many weights, so the delta gets rounded away during requantization. This
is a real risk for very small LoRA deltas.

**Fix**: re-train with higher `lora_alpha` (current is 16, try 32 or
64) to amplify the effective delta. OR accept the loss and document.
Verify FT signal still shows via `distinguish_ft.py`.
