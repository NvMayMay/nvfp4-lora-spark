# Diagnostic tools for serving Nemotron-3-Super-NVFP4 on DGX Spark

A small toolkit developed during the investigation of memory and
configuration issues when serving `Nemotron-3-Super-120B-A12B-NVFP4`
via vLLM 0.21 on a single DGX Spark (GB10, sm_121, 128 GB UMA).

## Files

### `diag_vllm_safe.py`
Wrapper that launches `vllm serve` in-process with:
- A background **safety thread** that polls `psutil.virtual_memory().available`
  (the correct UMA metric - see `LESSONS.md`) every 100 ms and calls
  `killpg(SIGKILL)` if RAM available drops below `SAFETY_RAM_FLOOR_BYTES`
  (default 4 GB). Prevents the NVRM kernel-level OOM thrash that requires
  a hard reboot.
- Monkey-patch hooks around `PunicaWrapperGPU.__init__`, `FusedMoEWithLoRA.__init__`,
  `_create_lora_a_weights`, `_create_lora_b_weights`, and
  `FusedMoEModularMethod.process_weights_after_loading` for per-call memory
  snapshot logging.
- Multiple named profiles: `minimal`, `conservative`, `original`,
  `minimal-no-lora`, `lora-emul-eager-tight`, `p1a-baseline`. See the
  PROFILES dict in the script.
- `--moe-backend-override` flag to swap the backend without editing a profile.

### `diag_alloc_microrepro.py`
Pure-Python micro-allocation repro. Reserves a staged base-model GPU
footprint via `torch.empty(uint8)` slabs sized from safetensors shard
metadata, then attempts the 32 LoRA contiguous slabs at exact vLLM
`_create_lora_a/b_weights` shapes. Used to falsify the "LoRA-slab
contiguous-VA failure" hypothesis: all 32 slabs allocated cleanly even
with 80 GB of base reserved. Does NOT load vLLM - purely allocator-level
test.

### `bench_vllm.py`
Throughput benchmark for the local vLLM server. Sends sequential
`/v1/completions` requests at varying prompt/output sizes and writes
per-request latency + tok/s to a JSONL file. See
`bench_base_eager_emul_noblock_*.jsonl` for an example result.

### `release_cuda.sh`
Best-effort recovery script that drops page cache and restarts
`nvidia-persistenced` after a vLLM crash. On Spark's integrated UMA GPU,
the only fully reliable way to restore the boot baseline is `sudo reboot`
(see `LESSONS.md` for why - the display server holds refs to the nvidia
kernel modules so they can't be unloaded). This script is a best-effort
soft option.

## Why these exist

They were written during a multi-day diagnostic campaign. The TL;DR of
what was learned:

1. **vLLM 0.21 + Super-NVFP4 + MARLIN backend** doesn't fit on Spark - Marlin
   per-expert weight repack has a transient memory peak that exceeds the 130
   GB physical ceiling.
2. **vLLM 0.21 + Super-NVFP4 + EMULATION backend** DOES work for base inference
   with the right knobs (see `serve/run_super_base_inference.sh` or DECISION_LOG
   D018 for the exact command). Throughput is impractically slow (~0.7 tok/s).
3. **vLLM 0.21 native-FP4 backends** (FLASHINFER_*, VLLM_CUTLASS) are NOT
   compiled for sm_121 (Blackwell consumer / Spark) in 0.21.0. All five
   reject at oracle stage with *"kernel does not support current device cuda"*.
4. **vLLM 0.21 Triton MoE LoRA kernel** crashes with `illegal memory access`
   on the EMULATION backend. So LoRA serving via vLLM is blocked on Spark
   regardless of memory.

See the project DECISION_LOG (D016-D020) for the full investigation.
