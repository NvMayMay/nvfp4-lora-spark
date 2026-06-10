"""GB10 (DGX Spark) UMA-specific preparation helpers for NVFP4 training.

Hard-won lessons consolidated from the Mistral-Small-4-119B and Qwen3.5-122B
bring-up (2026-06). On GB10 the CPU and GPU share one ~131 GB DRAM pool, and
three failure modes follow from that:

1. CPU-resident weight buffers permanently starve CUDA. Anything sized like
   model weights MUST be allocated with an explicit cuda device. The kernel
   reclaims page cache under *anon* memory pressure but NVRM allocations fail
   with NV_ERR_NO_MEMORY instead of waiting for reclaim.

2. Shard page cache competes with CUDA for the same DRAM. After assembling
   weights, ~70 GB of safetensors pages linger in the page cache;
   `drop_shard_page_cache` releases them via posix_fadvise (no sudo needed).

3. The default CUDA caching-allocator strategy fragments under the
   dequant-workspace access pattern; expandable segments avoids step-12-style
   mid-run OOMs. `set_alloc_conf` must run BEFORE the first CUDA allocation
   (i.e. before importing code that touches the GPU), so call it at the very
   top of trainer scripts.

NVML/nvidia-smi report Not-Supported on GB10 UMA; use
`torch.cuda.mem_get_info()` + `psutil.virtual_memory()` for observability.
"""
from __future__ import annotations

import os
from pathlib import Path


def set_alloc_conf() -> None:
    """Default PYTORCH_CUDA_ALLOC_CONF to expandable_segments if unset.

    No-op when the variable is already set (the launch environment wins).
    Must be called before the first CUDA allocation to have any effect.
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def drop_shard_page_cache(model_dir: str | Path) -> tuple[float, float]:
    """Drop OS page cache for every *.safetensors shard under model_dir.

    Call after weight assembly is complete. Returns (cuda_free_before_gb,
    cuda_free_after_gb). Requires no privileges: POSIX_FADV_DONTNEED evicts
    clean page-cache pages for the advised range regardless of which process
    faulted them in.
    """
    import torch

    free_before = torch.cuda.mem_get_info()[0] / 1e9
    for shard in sorted(Path(model_dir).glob("*.safetensors")):
        fd = os.open(str(shard), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    free_after = torch.cuda.mem_get_info()[0] / 1e9
    return free_before, free_after


def memory_snapshot() -> dict:
    """UMA-correct memory readings (NVML lies on GB10)."""
    import psutil
    import torch

    free, total = torch.cuda.mem_get_info()
    vm = psutil.virtual_memory()
    return {
        "cuda_free_gb": round(free / 1e9, 2),
        "cuda_total_gb": round(total / 1e9, 2),
        "ram_available_gb": round(vm.available / 1e9, 2),
        "process_rss_gb": round(psutil.Process().memory_info().rss / 1e9, 2),
    }
