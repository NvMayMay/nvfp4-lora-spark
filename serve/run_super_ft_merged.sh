#!/usr/bin/env bash
# Serve a LoRA-merged Nemotron-3-Super NVFP4 model on DGX Spark via vLLM CUTLASS.
#
# This is the recommended path for Super-FT serving (training-side LoRA
# adapter merged into the NVFP4 base, then re-emitted as a new NVFP4
# checkpoint, then served via VLLM_CUTLASS at full ~12-14 tok/s).
#
# Use:
#   1. Train a LoRA adapter via the recipes in train/
#   2. Merge it into the base: scripts/merge_lora_into_nvfp4.py
#   3. Point MODEL_DIR at the merged output, run this script.
#
# Throughput: same as base CUTLASS, ~12-14 tok/s on Spark.
#
# Trade-off vs dynamic LoRA: this approach BAKES the FT behavior into the
# served checkpoint. If you need multiple adapters at runtime, either
# produce N merged variants OR see docs/PHASE2.md for the dynamic-LoRA
# upstream work.

set -euo pipefail

# Default to the example merged checkpoint; override via env.
MODEL_DIR="${MODEL_DIR:-/path/to/Models/Nemotron-3-Super-120B-A12B-NVFP4-ich-v1.0}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
SERVED_NAME="${SERVED_NAME:-nemotron-3-super-a12b-nvfp4+ich_v1_0}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_JOBS=1

vllm serve "$MODEL_DIR" \
    --served-model-name "$SERVED_NAME" \
    --host "$HOST" --port "$PORT" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --max-num-batched-tokens 128 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.70 \
    --enforce-eager \
    --moe-backend cutlass
