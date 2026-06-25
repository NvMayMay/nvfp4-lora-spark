#!/usr/bin/env bash
# Mistral-Small-3.2-24B-Instruct-2506 (compressed-tensors NVFP4) + full-LM LoRA
# (attention + MLP), OpenAI-compatible via vLLM.
#
# Unlike the 119B (mistral4, MLA, MoE -> transformers server), the 24B is dense
# mistral3 with STANDARD attention, and vLLM 0.22.1 supports
# Mistral3ForConditionalGeneration. The adapter is native-NVFP4 LoRA on q/k/v/o +
# gate/up/down (all NVFP4 in the LM), so it must be served by vLLM's punica LoRA
# over the NVFP4 base -- the transformers merge_and_unload path can't merge a
# bf16 LoRA into 4-bit weights. No MoE -> no attention_only_lora_cutlass_moe patch.
#
# The model is multimodal (vision tower); we serve text-only. If vLLM rejects the
# adapter keys (base_model.model.model.language_model.*) as unexpected, a key
# rewrite would be needed -- watch startup for "unexpected modules".
set -euo pipefail

SERVE_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SERVE_ENV_DIR}/local_env.sh" ]; then source "${SERVE_ENV_DIR}/local_env.sh"; fi
: "${NVFP4_MODELS_DIR:?Set NVFP4_MODELS_DIR or create serve/local_env.sh}"
: "${NVFP4_SERVE_VENV:?Set NVFP4_SERVE_VENV or create serve/local_env.sh}"
: "${NVFP4_ADAPTERS_DIR:?Set NVFP4_ADAPTERS_DIR or create serve/local_env.sh}"

BASE_DIR="${BASE_DIR:-${NVFP4_MODELS_DIR}/Mistral-Small-3.2-24B-Instruct-2506-NVFP4}"
ADAPTER_DIR="${ADAPTER_DIR:-${NVFP4_ADAPTERS_DIR}/mistral24b_lora_ich_v4_1_8k_r64a128/best}"
ADAPTER_NAME="${ADAPTER_NAME:-ich_v4_1}"
SERVED_NAME="${SERVED_NAME:-mistral-small-3.2-24b-nvfp4}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"   # adapter is r=64 alpha=128 (full LM attn+MLP)
MAX_LORAS="${MAX_LORAS:-2}"

if [ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]; then
  echo "ERROR: no adapter_model.safetensors in $ADAPTER_DIR" >&2; exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Runtime LoRA hot-swap (POST /v1/load_lora_adapter) is OPT-IN: export
# VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 only when ALLOW_RUNTIME_LORA_UPDATES=1.
if [ "${ALLOW_RUNTIME_LORA_UPDATES:-0}" = "1" ]; then
  export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
fi

source "${NVFP4_SERVE_VENV}/bin/activate"
exec vllm serve "$BASE_DIR" \
    --served-model-name "$SERVED_NAME" \
    --host "$HOST" --port "$PORT" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --tokenizer-mode "${TOKENIZER_MODE:-mistral}" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enforce-eager \
    --enable-lora \
    --max-lora-rank "$MAX_LORA_RANK" \
    --max-loras "$MAX_LORAS" \
    --lora-modules "$ADAPTER_NAME=$ADAPTER_DIR"
