"""
Step B: instrumented vLLM startup with safety thread.

Launches the EXACT failing Super-FT serve command but in-process, with:
  - A background safety thread that os._exit(1)'s on UMA pressure before
    NVRM kernel-level OOM thrash begins.
  - Monkey-patches around suspected alloc choke points so the JSONL log
    captures shape/bytes/free-before/free-after for every candidate site.
  - Global torch.cuda.OutOfMemoryError trap so we can record where the
    first cuda OOM fires and exit cleanly (vs the kernel-level NVRM thrash
    which is what causes the hard crashes).

Assumption: in-process invocation puts the monkey-patches in the right
namespace before vLLM imports the patched targets. If vLLM has already
cached a reference to the original (e.g. via `from foo import bar`),
the patch must hit `foo.bar` BEFORE the importing module runs. So we
do all patching before importing any vllm subpackage.

This script CAN crash the box. Do not run unless you have rebooted
recently and the box is otherwise idle. Tail dmesg in another terminal:
    sudo dmesg -wT | tee dmesg_stepB.log

Usage:
    /path/to/venvs/serve/bin/python \\
        diag_vllm_safe.py [--profile minimal|conservative|original]

  minimal       : --max-num-seqs 1, --max-num-batched-tokens 256,
                  --enforce-eager, --gpu-memory-utilization 0.55
  conservative  : --max-num-seqs 1, --max-num-batched-tokens 512,
                  --enforce-eager, --gpu-memory-utilization 0.60
  original      : matches lessons_super_lora_serve_oom.md verbatim
                  (max_num_seqs=4, max_num_batched_tokens=4096, no eager)

Default is `minimal`, which is the lowest-pressure variant most likely
to succeed; escalate if it succeeds, fail-fast if it doesn't.
"""

import argparse
import datetime
import gc
import json
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

import psutil
import torch

# ---------------------------------------------------------------------------
# Safety env vars MUST be set before any cuda allocator activity.
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# CUDA_LAUNCH_BLOCKING removed: was set for crash-site visibility during
# diagnostic phase. Now causes 2-5x slowdown of inference and is unnecessary
# now that vLLM startup is known to succeed.
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")  # was DEBUG (very verbose)
os.environ.setdefault("VLLM_NVFP4_GEMM_BACKEND", "marlin")
os.environ.setdefault("MAX_JOBS", "1")
os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")

LOG_DIR = Path(__file__).parent
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = LOG_DIR / f"diag_vllm_safe_{TS}.jsonl"
SUMMARY_PATH = LOG_DIR / f"diag_vllm_safe_{TS}_summary.txt"

# Pressure floors (UMA on Spark; system RAM IS GPU memory).
SAFETY_RAM_FLOOR_BYTES = 2 * 1024**3  # abort if MemAvailable < 2 GB (PRIMARY metric on UMA)
SAFETY_CUDA_FREE_FLOOR_BYTES = 100 * 1024**2  # 100 MB cuda_free - secondary metric only
# CRITICAL CORRECTION (2026-05-23 ~00:00): on Spark UMA, the correct
# "available memory" metric is psutil.virtual_memory().available (= MemAvailable
# from /proc/meminfo, which is free + reclaimable page cache). The earlier
# floor on torch.cuda.mem_get_info() free was firing on a metric that excludes
# reclaimable page cache from prior safetensors mmaps, causing premature
# killpg of processes that had 20-30 GB of reclaimable memory still available.
# Verified by torch.empty(100 GB) succeeding when cuda_free reported 83 GB.
POLL_INTERVAL_S = 0.1

MODEL_DIR = (
    "/path/to/Models/Nemotron-3-Super-120B-A12B-NVFP4"
)
ADAPTER_DIR = (
    "/path/to/adapters/"
    "nemotron_3_super_nvfp4_lora_ichv31_1epoch"
)

# ---------------------------------------------------------------------------
# JSONL emit
# ---------------------------------------------------------------------------

_log_lock = threading.Lock()
_log_f = open(LOG_PATH, "w", buffering=1)


def emit(event: str, **kwargs):
    rec = {"ts": time.time(), "event": event, **kwargs}
    with _log_lock:
        _log_f.write(json.dumps(rec, default=str) + "\n")
        _log_f.flush()


# ---------------------------------------------------------------------------
# Safety thread
# ---------------------------------------------------------------------------


def _emergency_log(msg):
    sys.stderr.write(f"[SAFETY] {msg}\n")
    sys.stderr.flush()
    try:
        emit("safety_emergency", msg=msg)
    except Exception:
        pass


def _kill_process_group_and_exit():
    """
    Kill all descendants AND self, then os._exit(1).

    vLLM spawns subprocess workers (e.g. VLLM::EngineCore) via multiprocessing.
    A bare `os._exit(1)` in the parent leaves those children running, which on
    Spark UMA means they continue holding ~120 GB of cuda memory until manually
    killed. We send SIGKILL to the whole process group so cleanup is atomic.
    """
    try:
        pgid = os.getpgrp()
        # Send SIGKILL to entire group. This includes the current process,
        # which is why we follow with os._exit in case the signal handling
        # races; SIGKILL cannot be trapped so this is effectively belt-and-braces.
        os.killpg(pgid, signal.SIGKILL)
    except Exception as e:
        _emergency_log(f"killpg failed: {e!r}; falling back to os._exit(1)")
    os._exit(1)


def _safety_loop():
    while True:
        try:
            vm = psutil.virtual_memory()
            cuda_free, cuda_total = torch.cuda.mem_get_info()
            if vm.available < SAFETY_RAM_FLOOR_BYTES:
                _emergency_log(
                    f"RAM avail {vm.available/1e9:.3f} GB < "
                    f"{SAFETY_RAM_FLOOR_BYTES/1e9:.3f} GB. killpg+exit."
                )
                _kill_process_group_and_exit()
            if cuda_free < SAFETY_CUDA_FREE_FLOOR_BYTES:
                _emergency_log(
                    f"CUDA free {cuda_free/1e9:.3f} GB < "
                    f"{SAFETY_CUDA_FREE_FLOOR_BYTES/1e9:.3f} GB. killpg+exit."
                )
                _kill_process_group_and_exit()
        except Exception as e:
            _emergency_log(f"safety thread exception: {e!r}")
        time.sleep(POLL_INTERVAL_S)


def _start_safety_thread():
    t = threading.Thread(target=_safety_loop, daemon=True, name="diag-safety")
    t.start()
    return t


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------


def snapshot():
    vm = psutil.virtual_memory()
    cuda_free, cuda_total = torch.cuda.mem_get_info()
    return {
        "psutil_available_bytes": vm.available,
        "psutil_total_bytes": vm.total,
        "cuda_free_bytes": cuda_free,
        "cuda_total_bytes": cuda_total,
        "torch_cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "torch_cuda_reserved_bytes": torch.cuda.memory_reserved(),
    }


# ---------------------------------------------------------------------------
# Monkey-patches: install BEFORE importing vllm.
# ---------------------------------------------------------------------------


def install_patches():
    """
    Patch suspected choke points to log a before/after snapshot per call.
    Each wrapped function logs:
      - the call site name
      - the snapshot before any allocation
      - the snapshot after the original function returns (or on exception)
      - the exception class+message if it raised

    Also installs the Marlin NVFP4 MoE repack memory-fix patch (chunked +
    preallocated streaming repack) to keep peak memory under the Spark
    UMA ceiling during process_weights_after_loading.
    """
    # First: install the Marlin repack memory fix (the publishable surgery).
    # This MUST happen before vLLM's loader runs.
    try:
        from marlin_repack_patch import apply_patch as apply_marlin_patch

        apply_marlin_patch()
        emit("marlin_repack_patch_applied", snap=snapshot())
    except Exception as e:
        emit("marlin_repack_patch_failed", err=repr(e), snap=snapshot())
        print(f"[diag] WARN: marlin_repack_patch failed to apply: {e!r}")

    import vllm.lora.layers.fused_moe as fm_mod
    import vllm.lora.punica_wrapper.punica_gpu as pg_mod

    # ---- FusedMoEWithLoRA._create_lora_a_weights / _create_lora_b_weights
    orig_create_a = fm_mod.FusedMoEWithLoRA._create_lora_a_weights
    orig_create_b = fm_mod.FusedMoEWithLoRA._create_lora_b_weights

    def wrapped_create_a(self, max_loras, lora_config):
        emit("create_lora_a.enter", layer=id(self), snap=snapshot())
        try:
            r = orig_create_a(self, max_loras, lora_config)
        except BaseException as e:
            emit(
                "create_lora_a.exception",
                err=repr(e),
                tb=traceback.format_exc(limit=8),
                snap=snapshot(),
            )
            raise
        emit("create_lora_a.exit", snap=snapshot())
        return r

    def wrapped_create_b(self, max_loras, lora_config):
        emit("create_lora_b.enter", layer=id(self), snap=snapshot())
        try:
            r = orig_create_b(self, max_loras, lora_config)
        except BaseException as e:
            emit(
                "create_lora_b.exception",
                err=repr(e),
                tb=traceback.format_exc(limit=8),
                snap=snapshot(),
            )
            raise
        emit("create_lora_b.exit", snap=snapshot())
        return r

    fm_mod.FusedMoEWithLoRA._create_lora_a_weights = wrapped_create_a
    fm_mod.FusedMoEWithLoRA._create_lora_b_weights = wrapped_create_b

    # ---- FusedMoEWithLoRA.__init__ (where _replace_quant_method happens)
    orig_fm_init = fm_mod.FusedMoEWithLoRA.__init__

    def wrapped_fm_init(self, base_layer):
        emit("FusedMoEWithLoRA.init.enter", snap=snapshot())
        try:
            orig_fm_init(self, base_layer)
        except BaseException as e:
            emit(
                "FusedMoEWithLoRA.init.exception",
                err=repr(e),
                tb=traceback.format_exc(limit=12),
                snap=snapshot(),
            )
            raise
        emit("FusedMoEWithLoRA.init.exit", snap=snapshot())

    fm_mod.FusedMoEWithLoRA.__init__ = wrapped_fm_init

    # ---- FusedMoEWithLoRA.create_lora_weights (orchestrator)
    orig_create_w = fm_mod.FusedMoEWithLoRA.create_lora_weights

    def wrapped_create_w(self, max_loras, lora_config, model_config=None):
        emit("create_lora_weights.enter", snap=snapshot())
        try:
            r = orig_create_w(self, max_loras, lora_config, model_config)
        except BaseException as e:
            emit(
                "create_lora_weights.exception",
                err=repr(e),
                tb=traceback.format_exc(limit=12),
                snap=snapshot(),
            )
            raise
        emit("create_lora_weights.exit", snap=snapshot())
        return r

    fm_mod.FusedMoEWithLoRA.create_lora_weights = wrapped_create_w

    # ---- PunicaWrapperGPU.__init__
    orig_pg_init = pg_mod.PunicaWrapperGPU.__init__

    def wrapped_pg_init(self, max_num_batched_tokens, max_batches, device, **kwargs):
        emit(
            "PunicaWrapperGPU.init.enter",
            max_num_batched_tokens=max_num_batched_tokens,
            max_batches=max_batches,
            snap=snapshot(),
        )
        try:
            orig_pg_init(self, max_num_batched_tokens, max_batches, device, **kwargs)
        except BaseException as e:
            emit(
                "PunicaWrapperGPU.init.exception",
                err=repr(e),
                tb=traceback.format_exc(limit=12),
                snap=snapshot(),
            )
            raise
        emit("PunicaWrapperGPU.init.exit", snap=snapshot())

    pg_mod.PunicaWrapperGPU.__init__ = wrapped_pg_init

    # ---- Optional: instrument process_weights_after_loading on quant method
    # NVFP4 Marlin path may allocate dequant buffers there. We patch when
    # we can import the class; if absent we skip silently.
    try:
        from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
            FusedMoEModularMethod,
        )

        if hasattr(FusedMoEModularMethod, "process_weights_after_loading"):
            orig_pw = FusedMoEModularMethod.process_weights_after_loading

            def wrapped_pw(self, layer):
                emit("FusedMoEModularMethod.process_weights.enter", snap=snapshot())
                try:
                    r = orig_pw(self, layer)
                except BaseException as e:
                    emit(
                        "FusedMoEModularMethod.process_weights.exception",
                        err=repr(e),
                        tb=traceback.format_exc(limit=12),
                        snap=snapshot(),
                    )
                    raise
                emit("FusedMoEModularMethod.process_weights.exit", snap=snapshot())
                return r

            FusedMoEModularMethod.process_weights_after_loading = wrapped_pw
    except Exception as e:
        emit("patch.process_weights.skipped", err=repr(e))

    # ---- Wrap torch.cuda.OutOfMemoryError catch-all via a faulthandler-style log
    # If anything raises CudaOOM, we want a final snapshot.
    real_excepthook = sys.excepthook

    def diag_excepthook(exc_type, exc, tb):
        try:
            emit(
                "uncaught_exception",
                exc_type=exc_type.__name__,
                exc=repr(exc),
                snap=snapshot(),
                tb="".join(traceback.format_tb(tb, limit=20)),
            )
        finally:
            real_excepthook(exc_type, exc, tb)

    sys.excepthook = diag_excepthook


# ---------------------------------------------------------------------------
# Serve command profiles
# ---------------------------------------------------------------------------

PROFILES = {
    "minimal": [
        # Run-1 (2026-05-23 22:31) crashed at weight-load shard 8/17 because
        # gpu_memory_utilization=0.55 set budget at 66.93 GiB < 74.80 GiB model.
        # Corrected: budget must be >= model size; reduce KV cache instead via
        # max_model_len + max_num_batched_tokens to leave headroom for Punica.
        "vllm",
        "serve",
        MODEL_DIR,
        "--served-model-name", "nemotron-3-super-a12b-nvfp4",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--tensor-parallel-size", "1",
        "--dtype", "bfloat16",
        "--max-model-len", "2048",
        "--max-num-batched-tokens", "128",
        "--max-num-seqs", "1",
        "--gpu-memory-utilization", "0.85",
        "--enforce-eager",
        "--moe-backend", "marlin",
        "--enable-lora",
        "--lora-modules", f"nemotron-3-super-a12b-nvfp4+ich_v1_0={ADAPTER_DIR}",
        "--max-lora-rank", "8",
        "--max-loras", "1",
        "--max-cpu-loras", "1",
    ],
    "conservative": [
        "vllm",
        "serve",
        MODEL_DIR,
        "--served-model-name", "nemotron-3-super-a12b-nvfp4",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--tensor-parallel-size", "1",
        "--dtype", "bfloat16",
        "--max-model-len", "4096",
        "--max-num-batched-tokens", "512",
        "--max-num-seqs", "1",
        "--gpu-memory-utilization", "0.60",
        "--enforce-eager",
        "--moe-backend", "marlin",
        "--enable-lora",
        "--lora-modules", f"nemotron-3-super-a12b-nvfp4+ich_v1_0={ADAPTER_DIR}",
        "--max-lora-rank", "8",
        "--max-loras", "1",
        "--max-cpu-loras", "1",
    ],
    "original": [
        "vllm",
        "serve",
        MODEL_DIR,
        "--served-model-name", "nemotron-3-super-a12b-nvfp4",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--tensor-parallel-size", "1",
        "--dtype", "bfloat16",
        "--max-model-len", "4096",
        "--max-num-batched-tokens", "4096",
        "--max-num-seqs", "4",
        "--moe-backend", "marlin",
        "--enable-lora",
        "--lora-modules", f"nemotron-3-super-a12b-nvfp4+ich_v1_0={ADAPTER_DIR}",
        "--max-lora-rank", "8",
        "--max-loras", "1",
    ],
    # Option A control: identical tight profile to "minimal" but with all
    # LoRA flags removed. If this succeeds and "minimal" failed, the LoRA-
    # aware code path is the per-shard memory amplifier.
    "minimal-no-lora": [
        "vllm",
        "serve",
        MODEL_DIR,
        "--served-model-name", "nemotron-3-super-a12b-nvfp4",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--tensor-parallel-size", "1",
        "--dtype", "bfloat16",
        "--max-model-len", "2048",
        "--max-num-batched-tokens", "128",
        "--max-num-seqs", "1",
        "--gpu-memory-utilization", "0.70",
        "--enforce-eager",
        "--moe-backend", "marlin",
    ],
    # Step 2 of autonomous campaign: same as minimal-no-lora but with the
    # Super-FT LoRA adapter attached. Tests whether LoRA fits on top of the
    # working EMULATION+eager+tight+gpu_util=0.70 baseline. Needs --max-loras
    # + --max-cpu-loras + --max-lora-rank + --lora-modules.
    "lora-emul-eager-tight": [
        "vllm",
        "serve",
        MODEL_DIR,
        "--served-model-name", "nemotron-3-super-a12b-nvfp4",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--tensor-parallel-size", "1",
        "--dtype", "bfloat16",
        "--max-model-len", "2048",
        "--max-num-batched-tokens", "128",
        "--max-num-seqs", "1",
        "--gpu-memory-utilization", "0.70",
        "--enforce-eager",
        "--moe-backend", "emulation",
        "--enable-lora",
        "--lora-modules", f"nemotron-3-super-a12b-nvfp4+ich_v1_0={ADAPTER_DIR}",
        "--max-lora-rank", "8",
        "--max-loras", "1",
        "--max-cpu-loras", "1",
    ],
    # D015a/D015e (round-4 consensus): EXACT P-1a C1 baseline. No LoRA, no
    # tight knobs, no --enforce-eager, no explicit --gpu-memory-utilization
    # (defaults to 0.9). Tests whether base Super-NVFP4 marlin inference
    # still loads as it did three days ago in the P-1a sprint. If this
    # reaches "Application startup complete", the prior 3 crashes were
    # caused by something specific to our tight-profile changes. If this
    # crashes too, vLLM 0.21's base load path no longer fits on Spark and
    # we file the upstream bug + pivot to FastAPI.
    "p1a-baseline": [
        "vllm",
        "serve",
        MODEL_DIR,
        "--served-model-name", "nemotron-3-super-a12b-nvfp4",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--tensor-parallel-size", "1",
        "--dtype", "bfloat16",
        "--max-model-len", "4096",
        "--moe-backend", "marlin",
    ],
}


def main():
    # Move into our own process group so killpg from the safety thread
    # reliably reaches all vLLM-spawned subprocesses (e.g. VLLM::EngineCore)
    # without touching the parent bash. setpgrp must happen BEFORE any
    # multiprocessing.spawn calls (which copy the current pgid into children).
    try:
        os.setpgrp()
    except OSError as e:
        sys.stderr.write(f"[diag] warning: setpgrp failed: {e!r}\n")

    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=list(PROFILES), default="minimal")
    ap.add_argument(
        "--moe-backend-override",
        default=None,
        help="If set, replaces the --moe-backend value in the chosen profile. "
        "Useful for sweeping over {marlin, flashinfer-trtllm, flashinfer-cutlass, "
        "vllm-cutlass, flashinfer-cutedsl, flashinfer-cutedsl-batched, emulation}.",
    )
    args = ap.parse_args()

    emit("startup", pid=os.getpid(), profile=args.profile, env={
        k: os.environ.get(k) for k in [
            "PYTORCH_CUDA_ALLOC_CONF", "CUDA_LAUNCH_BLOCKING", "VLLM_LOGGING_LEVEL",
            "VLLM_NVFP4_GEMM_BACKEND", "MAX_JOBS"
        ]
    }, snap=snapshot())

    # Start safety thread BEFORE any heavy import.
    _start_safety_thread()
    time.sleep(0.25)

    # Touch CUDA before patching so torch.cuda is initialized.
    _ = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    emit("cuda_warm", snap=snapshot())

    install_patches()
    emit("patches_installed", snap=snapshot())

    # Now invoke vllm CLI's main() with sys.argv set.
    sys.argv = list(PROFILES[args.profile])  # copy so override does not mutate dict
    if args.moe_backend_override is not None:
        # Replace the value following --moe-backend in sys.argv
        try:
            idx = sys.argv.index("--moe-backend")
            sys.argv[idx + 1] = args.moe_backend_override
        except ValueError:
            sys.argv += ["--moe-backend", args.moe_backend_override]
        # Also override the env var that controls the NVFP4 GEMM backend
        # (separate from --moe-backend which controls the FusedMoE backend);
        # for the non-marlin paths we likely need to clear or change this.
        if args.moe_backend_override != "marlin":
            os.environ.pop("VLLM_NVFP4_GEMM_BACKEND", None)
        print(f"[diag_vllm_safe] override --moe-backend = {args.moe_backend_override}")
    print(f"[diag_vllm_safe] launching: {' '.join(sys.argv)}")
    print(f"[diag_vllm_safe] log: {LOG_PATH}")

    try:
        from vllm.entrypoints.cli.main import main as vllm_main
        rc = vllm_main()
    except SystemExit as e:
        rc = e.code
        emit("vllm_systemexit", code=str(rc))
    except BaseException as e:
        emit(
            "vllm_exception",
            err=repr(e),
            tb=traceback.format_exc(limit=40),
            snap=snapshot(),
        )
        rc = 99
    finally:
        emit("shutdown", snap=snapshot())
        _log_f.close()
        with open(SUMMARY_PATH, "w") as f:
            f.write(f"Profile: {args.profile}\nLog: {LOG_PATH}\nExit: {rc}\n")

    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
