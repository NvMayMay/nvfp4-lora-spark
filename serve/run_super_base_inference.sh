#!/usr/bin/env bash
# Serve Nemotron-3-Super-120B-A12B-NVFP4 base inference on DGX Spark via vLLM.
#
# This is the WORKING recipe discovered after the diagnostic campaign documented
# in serve/diagnostics/README.md. Every flag below is load-bearing on this hardware
# (Spark GB10, sm_121, 128 GB UMA); changing them risks OOM or vLLM internal
# errors. See LESSONS.md for the why-of-each-flag explanation.
#
# CAVEAT: This serves the BASE model only. LoRA serving is currently blocked
# by a vLLM Triton MoE LoRA kernel bug. For Super-FT LoRA serving, use the
# merge-then-serve CUTLASS workflow documented in serve/README.md.
#
# Expected timing:
#   - Weight load: ~8 min (74.80 GiB of NVFP4 weights from disk)
#   - Post-load setup: ~1 min
#   - Total time to "Application startup complete": ~9 min
#
# Expected throughput: ~0.7 tok/s. EMULATION dequantizes weights every forward
# pass via Triton kernels; this is impractically slow for production but
# functionally correct.
#
# Usage: ./run_super_base_inference.sh

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/path/to/Models/Nemotron-3-Super-120B-A12B-NVFP4}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# Env vars
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_JOBS=1  # required for FlashInfer JIT on Spark to avoid OOM-kill during nvcc

# Optional: apply the chunked Marlin repack patch (irrelevant for EMULATION
# backend but won't hurt).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR/vllm_patches:${PYTHONPATH:-}"

vllm serve "$MODEL_DIR" \
    --served-model-name nemotron-3-super-a12b-nvfp4 \
    --host "$HOST" --port "$PORT" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --max-num-batched-tokens 128 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.70 \
    --enforce-eager \
    --moe-backend emulation
