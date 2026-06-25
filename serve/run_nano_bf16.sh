#!/usr/bin/env bash
# Serve Nemotron-3-Nano-30B-A3B-BF16 on DGX Spark / GB10 via vLLM.
# Vanilla bf16 path — no FP4 backend, no marlin patches needed.
set -euo pipefail

# Machine-local roots (models / adapters / serve venv). Set the NVFP4_* env
# vars, or create serve/local_env.sh from serve/local_env.example.sh.
SERVE_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SERVE_ENV_DIR}/local_env.sh" ]; then source "${SERVE_ENV_DIR}/local_env.sh"; fi
: "${NVFP4_MODELS_DIR:?Set NVFP4_MODELS_DIR or create serve/local_env.sh (see serve/local_env.example.sh)}"
: "${NVFP4_SERVE_VENV:?Set NVFP4_SERVE_VENV or create serve/local_env.sh (see serve/local_env.example.sh)}"

MODEL_DIR="${MODEL_DIR:-${NVFP4_MODELS_DIR}/Nemotron-3-Nano-30B-A3B-BF16}"
SERVED_NAME="${SERVED_NAME:-nemotron-3-nano-30b-a3b-bf16}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MAX_JOBS="${MAX_JOBS:-1}"

source "${NVFP4_SERVE_VENV}/bin/activate"

vllm serve "$MODEL_DIR" \
    --served-model-name "$SERVED_NAME" \
    --host "$HOST" --port "$PORT" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enforce-eager
