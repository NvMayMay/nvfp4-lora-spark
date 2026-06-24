#!/usr/bin/env bash
# Serve Mistral Small 4 119B A6B NVFP4 on DGX Spark / GB10 via vLLM.
#
# This checkpoint uses Mistral's native consolidated safetensors + params.json
# layout, so vLLM needs the Mistral config/tokenizer/load-format switches.
# Defaults are text-only, single-GPU, and four concurrent sequences.

set -euo pipefail

# Machine-local roots (models / adapters / serve venv). Set the NVFP4_* env
# vars, or create serve/local_env.sh from serve/local_env.example.sh.
SERVE_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SERVE_ENV_DIR}/local_env.sh" ]; then source "${SERVE_ENV_DIR}/local_env.sh"; fi
: "${NVFP4_MODELS_DIR:?Set NVFP4_MODELS_DIR or create serve/local_env.sh (see serve/local_env.example.sh)}"
: "${NVFP4_SERVE_VENV:?Set NVFP4_SERVE_VENV or create serve/local_env.sh (see serve/local_env.example.sh)}"

MODEL_DIR="${MODEL_DIR:-${NVFP4_MODELS_DIR}/Mistral-Small-4-119B-2603-NVFP4}"
SERVED_NAME="${SERVED_NAME:-mistral-small-4-119b-a6b-nvfp4}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MAX_JOBS="${MAX_JOBS:-1}"

source "${NVFP4_SERVE_VENV}/bin/activate"

vllm serve "$MODEL_DIR" \
    --served-model-name "$SERVED_NAME" \
    --host "$HOST" --port "$PORT" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --config-format mistral \
    --tokenizer-mode mistral \
    --load-format mistral \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enforce-eager \
    --language-model-only \
    --attention-backend TRITON_MLA \
    --reasoning-parser mistral \
    --enable-auto-tool-choice \
    --tool-call-parser mistral \
    --moe-backend cutlass
