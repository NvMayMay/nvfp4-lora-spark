#!/usr/bin/env bash
# Mistral-Small-4-119B (RedHatAI NVFP4-HF) + ICH v3.5 LoRA, OpenAI-compatible.
#
# ============================================================================
# DO NOT RUN UNTIL TRAINING COMPLETES (the 13h Qwen run owns the GPU until
# ~01:30). This loads ~119B of NVFP4+BF16 weights into the 131 GB UMA.
# ============================================================================
#
# Path C, transformers server on the proven training load path. vLLM cannot
# serve this checkpoint (no mistral4 HF text backbone in 0.22.1, plus the
# stale text_config.architectures recursion), and request-time LoRA over MLA
# kv_b_proj is unsound in vLLM regardless. The adapter targets are BF16
# attention linears, so the server merges them in memory (exact update).
# Full reasoning: docs/plans/SERVE_PATH_QWEN35_MISTRAL.md.
#
# Expectations: 10-20 min load, low single-digit tok/s, single stream.
# Sharp edge: peft 0.19.1 needs the in-place WeightConverter patch in the
# qwen-serve venv (already applied; see the memory note if attach fails with
# TypeError about 'distributed_operation').
#
# Smoke test once up:
#   curl -s localhost:${PORT:-8001}/health
#   curl -s localhost:${PORT:-8001}/v1/chat/completions -H 'Content-Type: application/json' \
#     -d '{"messages":[{"role":"user","content":"Summarize ICH E6(R3) section 1 scope."}],"max_tokens":120,"temperature":0}'

set -euo pipefail

# Machine-local roots (models / adapters / serve venv). Set the NVFP4_* env
# vars, or create serve/local_env.sh from serve/local_env.example.sh.
SERVE_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SERVE_ENV_DIR}/local_env.sh" ]; then source "${SERVE_ENV_DIR}/local_env.sh"; fi
: "${NVFP4_MODELS_DIR:?Set NVFP4_MODELS_DIR or create serve/local_env.sh (see serve/local_env.example.sh)}"
: "${NVFP4_SERVE_VENV:?Set NVFP4_SERVE_VENV or create serve/local_env.sh (see serve/local_env.example.sh)}"
: "${NVFP4_ADAPTERS_DIR:?Set NVFP4_ADAPTERS_DIR or create serve/local_env.sh (see serve/local_env.example.sh)}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="${NVFP4_SERVE_VENV}/bin/python"

MODEL_DIR="${MODEL_DIR:-${NVFP4_MODELS_DIR}/RedHatAI-Mistral-Small-4-119B-2603-NVFP4-HF}"
ADAPTER_DIR="${ADAPTER_DIR:-${NVFP4_ADAPTERS_DIR}/mistral_small_4_119b_rh_nvfp4_lora_ich_v3_5}"
HOST="${HOST:-0.0.0.0}"
# 8001 by default so it can coexist with the Qwen vLLM server on 8000
# (do NOT run both at once on this box; UMA will not fit two 120B models).
PORT="${PORT:-8001}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ ! -f "$ADAPTER_DIR/adapter_config.json" ]; then
  echo "ERROR: no adapter_config.json in $ADAPTER_DIR" >&2
  exit 1
fi

cd "$REPO_DIR"
exec "$VENV_PY" -u serve/serve_mistral_rh_nvfp4_lora_openai.py \
    --model-dir "$MODEL_DIR" \
    --adapter-path "$ADAPTER_DIR" \
    --host "$HOST" --port "$PORT" \
    "$@"
