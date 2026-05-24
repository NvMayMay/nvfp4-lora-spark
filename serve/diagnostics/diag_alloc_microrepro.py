"""
Step A' microrepro: vLLM-faithful LoRA contiguous allocation pressure test.

Goal: without loading vLLM, determine whether the 32 LoRA contiguous slabs
that vLLM would allocate at `create_lora_weights` time can be made
on the Spark GB10 (128 GB UMA LPDDR5x) AFTER a base-model-sized
GPU/UMA footprint has been reserved.

Three outcomes:
  - OOM during LoRA slabs: LoRA-slab pressure hypothesis strongly supported.
  - All 32 succeed with >= 10 GB system RAM headroom: hypothesis not reproduced.
  - All 32 succeed but headroom < 10 GB: danger zone or pivot zone.

Safety: a background thread polls pynvml + psutil every 100 ms and calls
os._exit(1) if system RAM available drops below SAFETY_RAM_FLOOR_BYTES or if
GPU free memory drops below SAFETY_GPU_FLOOR_BYTES, so we abort before
NVRM kernel-level OOM thrash.

Usage:
  python diag_alloc_microrepro.py --base-fraction 0.0
  python diag_alloc_microrepro.py --base-fraction 0.5
  python diag_alloc_microrepro.py --base-fraction 0.7
  python diag_alloc_microrepro.py --base-fraction 0.85
  python diag_alloc_microrepro.py --base-fraction 1.0

Always start with 0.0 to confirm the baseline succeeds; only escalate stage
by stage. The script never loads vLLM, never starts a model server, and
never touches anything that calls into Punica or Triton.
"""

import argparse
import gc
import json
import os
import sys
import threading
import time
from pathlib import Path

import psutil
import pynvml
import torch

# ---------------------------------------------------------------------------
# Hard-coded constants derived from
#   /path/to/Models/Nemotron-3-Super-120B-A12B-NVFP4/
#   config.json   (verified 2026-05-23)
# and
#   /path/to/venvs/serve/lib/python3.12/
#   site-packages/vllm/lora/layers/fused_moe.py:84-144  (verified 2026-05-23)
#
# Key facts:
#   - hidden_size = 4096
#   - moe_intermediate_size = 2688
#   - n_routed_experts = 512
#   - num_hidden_layers = 88, of which 8 are MoE (*) per hybrid_override_pattern
#   - NemotronHMLP is NON-gated (up_proj + down_proj only, relu2 activation),
#     so vLLM's `_w13_slices = 2 if is_act_and_mul else 1` resolves to 1.
#     This is HALF the slab count vs a gated MoE like Mixtral/Qwen.
#   - max_loras = 1, max_lora_rank = 8 (as configured in the failing serve cmd).
#   - LoRA dtype = bf16 (2 bytes/elt) - vLLM default for lora_dtype.
# ---------------------------------------------------------------------------

HIDDEN_SIZE = 4096
MOE_INTERMEDIATE_SIZE = 2688
N_ROUTED_EXPERTS = 512
N_MOE_LAYERS = 8  # count of '*' in hybrid_override_pattern
MAX_LORAS = 1
MAX_LORA_RANK = 8
LORA_DTYPE = torch.bfloat16
W13_SLICES = 1  # is_act_and_mul=False for NemotronH

SUPER_MODEL_DIR = Path(
    "/path/to/Models/Nemotron-3-Super-120B-A12B-NVFP4"
)

SAFETY_RAM_FLOOR_BYTES = 4 * 1024**3  # 4 GB system RAM (UMA = GPU memory on Spark)
SAFETY_GPU_FLOOR_BYTES = 2 * 1024**3  # 2 GB CUDA-visible free (mem_get_info)
NVML_SUPPORTED = True  # set to False if nvmlDeviceGetMemoryInfo is unsupported (UMA)

POLL_INTERVAL_S = 0.1
LOG_DIR = Path(__file__).parent

# vLLM's actual creation order from fused_moe.py:
#   _create_lora_a_weights() then _create_lora_b_weights(), each interleaving
#   w13 (tuple of _w13_slices tensors) then w2 (single-element tuple). At
#   max_loras=1, _w13_slices=1, the per-layer order is therefore exactly:
#     1. w13_lora_a   shape (1, 512, 8, 4096)        bf16 ~ 33.55 MB
#     2. w2_lora_a    shape (1, 512, 8, 2688)        bf16 ~ 22.02 MB
#     3. w13_lora_b   shape (1, 512, 2688, 8)        bf16 ~ 22.02 MB
#     4. w2_lora_b    shape (1, 512, 4096, 8)        bf16 ~ 33.55 MB
# repeated across 8 MoE layers = 32 allocs total, ~ 890 MB total LoRA memory.


def make_lora_shapes():
    """Return list of (name, shape, bytes) in vLLM's create order."""
    elt = torch.finfo(LORA_DTYPE).bits // 8
    shapes = []
    for layer in range(N_MOE_LAYERS):
        # _create_lora_a_weights: w13 then w2
        for slice_idx in range(W13_SLICES):
            s = (MAX_LORAS, N_ROUTED_EXPERTS, MAX_LORA_RANK, HIDDEN_SIZE)
            shapes.append(
                (f"L{layer:02d}.w13_lora_a.s{slice_idx}", s, _nelem(s) * elt)
            )
        s = (MAX_LORAS, N_ROUTED_EXPERTS, MAX_LORA_RANK, MOE_INTERMEDIATE_SIZE)
        shapes.append((f"L{layer:02d}.w2_lora_a", s, _nelem(s) * elt))
        # _create_lora_b_weights: w13 then w2
        for slice_idx in range(W13_SLICES):
            s = (MAX_LORAS, N_ROUTED_EXPERTS, MOE_INTERMEDIATE_SIZE, MAX_LORA_RANK)
            shapes.append(
                (f"L{layer:02d}.w13_lora_b.s{slice_idx}", s, _nelem(s) * elt)
            )
        s = (MAX_LORAS, N_ROUTED_EXPERTS, HIDDEN_SIZE, MAX_LORA_RANK)
        shapes.append((f"L{layer:02d}.w2_lora_b", s, _nelem(s) * elt))
    return shapes


def _nelem(shape):
    n = 1
    for d in shape:
        n *= d
    return n


# ---------------------------------------------------------------------------
# Safety thread
# ---------------------------------------------------------------------------

_safety_state = {
    "stop": False,
    "ram_kill": False,
    "gpu_kill": False,
    "last_ram_bytes": None,
    "last_gpu_free_bytes": None,
}


def _safety_loop(nvml_handle):
    global NVML_SUPPORTED
    while not _safety_state["stop"]:
        try:
            vm = psutil.virtual_memory()
            _safety_state["last_ram_bytes"] = vm.available
            if vm.available < SAFETY_RAM_FLOOR_BYTES:
                _safety_state["ram_kill"] = True
                _emergency_log(
                    f"SAFETY: RAM available {vm.available/1e9:.2f} GB < "
                    f"floor {SAFETY_RAM_FLOOR_BYTES/1e9:.2f} GB. os._exit(1)."
                )
                os._exit(1)
            cuda_free, _cuda_total = torch.cuda.mem_get_info()
            _safety_state["last_gpu_free_bytes"] = cuda_free
            if cuda_free < SAFETY_GPU_FLOOR_BYTES:
                _safety_state["gpu_kill"] = True
                _emergency_log(
                    f"SAFETY: CUDA mem_get_info free {cuda_free/1e9:.2f} GB < "
                    f"floor {SAFETY_GPU_FLOOR_BYTES/1e9:.2f} GB. os._exit(1)."
                )
                os._exit(1)
            if NVML_SUPPORTED:
                try:
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
                    _safety_state["last_nvml_free_bytes"] = meminfo.free
                except pynvml.NVMLError:
                    NVML_SUPPORTED = False
        except Exception as e:
            _emergency_log(f"SAFETY thread exception: {e!r}")
        time.sleep(POLL_INTERVAL_S)


def _emergency_log(msg):
    path = LOG_DIR / "diag_alloc_safety.log"
    try:
        with open(path, "a") as f:
            f.write(f"[{time.time():.3f}] {msg}\n")
    except Exception:
        pass
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Base-model footprint reservation
# ---------------------------------------------------------------------------


def shard_sizes_bytes():
    """Return list of safetensors shard file sizes in bytes."""
    shards = sorted(SUPER_MODEL_DIR.glob("model-*-of-*.safetensors"))
    return [p.stat().st_size for p in shards]


def reserve_base_footprint(fraction: float):
    """
    Reserve GPU memory approximating the base model's post-load footprint.

    Strategy: take the 17 safetensors shard file sizes, scale each by
    `fraction`, and torch.empty() a uint8 slab of that size. This emulates
    the "base model has consumed N shards of GPU VA pool" condition
    without actually loading vLLM.

    Returns the list of reserved tensors (kept alive until caller drops).
    """
    if fraction <= 0.0:
        return []
    sizes = shard_sizes_bytes()
    reserved = []
    total = 0
    for i, sz in enumerate(sizes):
        nbytes = int(sz * fraction)
        if nbytes < 1024:
            continue
        try:
            t = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
            reserved.append(t)
            total += nbytes
            print(
                f"  base-shard-{i:02d}: reserved {nbytes/1e9:.2f} GB "
                f"(running total {total/1e9:.2f} GB)"
            )
        except torch.cuda.OutOfMemoryError as e:
            print(
                f"  base-shard-{i:02d}: OOM at {nbytes/1e9:.2f} GB "
                f"(running total {total/1e9:.2f} GB)\n  err: {e!s}",
                file=sys.stderr,
            )
            raise
    return reserved


# ---------------------------------------------------------------------------
# Main allocation experiment
# ---------------------------------------------------------------------------


def run(base_fraction: float):
    print(f"=== diag_alloc_microrepro: base_fraction={base_fraction} ===")
    print(f"PID={os.getpid()}  CUDA={torch.version.cuda}  torch={torch.__version__}")

    global NVML_SUPPORTED
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    try:
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
        print(
            f"start NVML: total={meminfo.total/1e9:.2f} GB  "
            f"free={meminfo.free/1e9:.2f} GB  used={meminfo.used/1e9:.2f} GB"
        )
    except pynvml.NVMLError as e:
        NVML_SUPPORTED = False
        print(f"NVML mem_info NOT supported (UMA-style GPU): {e!s}. "
              "Falling back to torch.cuda.mem_get_info + psutil.")
    cuda_free, cuda_total = torch.cuda.mem_get_info()
    print(
        f"start CUDA mem_get_info: total={cuda_total/1e9:.2f} GB  "
        f"free={cuda_free/1e9:.2f} GB"
    )
    vm = psutil.virtual_memory()
    print(
        f"start psutil: total={vm.total/1e9:.2f} GB  "
        f"available={vm.available/1e9:.2f} GB"
    )

    # Start safety thread BEFORE any allocations.
    t = threading.Thread(target=_safety_loop, args=(nvml_handle,), daemon=True)
    t.start()
    time.sleep(0.25)  # let it take a baseline reading

    log_path = LOG_DIR / f"diag_alloc_bf{int(base_fraction*100):03d}.jsonl"
    log_f = open(log_path, "w")

    def emit(rec):
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    # Touch CUDA before reserving.
    _ = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

    # Stage 1: reserve base footprint
    print(f"\n--- reserving base-model footprint @ {base_fraction*100:.0f}% ---")
    reserved = []
    try:
        reserved = reserve_base_footprint(base_fraction)
    except torch.cuda.OutOfMemoryError as e:
        emit({"stage": "base_reserve", "status": "oom", "err": repr(e)})
        log_f.close()
        print(f"OUTCOME: OOM during base reserve at fraction={base_fraction}")
        return 2
    torch.cuda.synchronize()
    cuda_free_post, _ = torch.cuda.mem_get_info()
    vm = psutil.virtual_memory()
    print(
        f"  post-reserve CUDA free={cuda_free_post/1e9:.2f} GB  "
        f"psutil available={vm.available/1e9:.2f} GB"
    )
    emit(
        {
            "stage": "base_reserve_done",
            "fraction": base_fraction,
            "cuda_free_bytes": cuda_free_post,
            "psutil_available_bytes": vm.available,
        }
    )

    # Stage 2: 32 LoRA contiguous slabs in vLLM order
    print("\n--- attempting 32 vLLM LoRA contiguous slabs ---")
    lora_tensors = []
    shapes = make_lora_shapes()
    for idx, (name, shape, nbytes) in enumerate(shapes):
        pre_cuda_free, _ = torch.cuda.mem_get_info()
        pre_vm = psutil.virtual_memory()
        try:
            tensor = torch.zeros(shape, dtype=LORA_DTYPE, device="cuda")
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError as e:
            emit(
                {
                    "stage": "lora_alloc",
                    "idx": idx,
                    "name": name,
                    "shape": list(shape),
                    "nbytes": nbytes,
                    "status": "oom",
                    "pre_cuda_free": pre_cuda_free,
                    "pre_psutil_available": pre_vm.available,
                    "err": repr(e),
                }
            )
            print(
                f"  [{idx:02d}/{len(shapes)}] {name}: OOM at shape={shape} "
                f"nbytes={nbytes/1e6:.2f} MB  "
                f"pre_cuda_free={pre_cuda_free/1e9:.2f} GB",
                file=sys.stderr,
            )
            log_f.close()
            _safety_state["stop"] = True
            return 3
        lora_tensors.append(tensor)
        post_cuda_free, _ = torch.cuda.mem_get_info()
        post_vm = psutil.virtual_memory()
        emit(
            {
                "stage": "lora_alloc",
                "idx": idx,
                "name": name,
                "shape": list(shape),
                "nbytes": nbytes,
                "status": "ok",
                "pre_cuda_free": pre_cuda_free,
                "post_cuda_free": post_cuda_free,
                "pre_psutil_available": pre_vm.available,
                "post_psutil_available": post_vm.available,
            }
        )
        if idx % 4 == 0 or idx == len(shapes) - 1:
            print(
                f"  [{idx:02d}/{len(shapes)}] {name}: OK  "
                f"shape={shape}  nbytes={nbytes/1e6:.2f} MB  "
                f"cuda_free_after={post_cuda_free/1e9:.2f} GB  "
                f"ram_avail_after={post_vm.available/1e9:.2f} GB"
            )

    # All 32 succeeded
    final_cuda_free, _ = torch.cuda.mem_get_info()
    final_vm = psutil.virtual_memory()
    emit(
        {
            "stage": "complete",
            "status": "all_ok",
            "fraction": base_fraction,
            "n_lora_allocs": len(shapes),
            "final_cuda_free": final_cuda_free,
            "final_psutil_available": final_vm.available,
        }
    )
    print(
        f"\n=== OUTCOME: all {len(shapes)} LoRA allocs succeeded at "
        f"base_fraction={base_fraction} ===\n"
        f"  final CUDA mem_get_info free = {final_cuda_free/1e9:.2f} GB\n"
        f"  final psutil available RAM = {final_vm.available/1e9:.2f} GB"
    )
    log_f.close()
    _safety_state["stop"] = True

    # Decision hint
    if final_vm.available >= 10 * 1024**3:
        print("  HEADROOM >= 10 GB: hypothesis NOT reproduced at this stage.")
    elif final_vm.available >= 5 * 1024**3:
        print("  HEADROOM 5-10 GB: danger zone.")
    else:
        print("  HEADROOM < 5 GB: pivot zone.")

    # Free everything cleanly
    del lora_tensors
    del reserved
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-fraction",
        type=float,
        default=0.0,
        help="Fraction of base-model safetensors total to reserve as torch.empty slabs on GPU before LoRA allocs.",
    )
    args = ap.parse_args()
    if not (0.0 <= args.base_fraction <= 1.0):
        print("--base-fraction must be in [0, 1]", file=sys.stderr)
        sys.exit(2)
    rc = run(args.base_fraction)
    sys.exit(rc)


if __name__ == "__main__":
    main()
