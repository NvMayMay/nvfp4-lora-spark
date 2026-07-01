#!/usr/bin/env bash
# Mistral-Small-3.2-24B-Instruct-2506 (compressed-tensors NVFP4) + full-LM LoRA,
# OpenAI-compatible via the NGC vLLM docker image.
#
# Why docker (not the host qwen-serve venv): this model's HF LlamaTokenizerFast
# is broken (wrong token ids, can't round-trip -> garbage "Ġ" output), so it MUST
# be served with --tokenizer-mode mistral (native tekken.json). On host vLLM 0.22.1
# that path crashes in the Pixtral vision processor during startup profiling. The
# NGC vllm:26.04 build serves it with --tokenizer-mode mistral + --language-model-only
# (skips the vision tower entirely; we serve text only).
#
# The adapter is native-NVFP4 LoRA on the full LM (q/k/v/o + gate/up/down); it is
# applied via vLLM punica over the NVFP4 base. The 24B is dense (no MoE) so no
# attention_only_lora_cutlass_moe patch is needed.
#
# Needs sudo for docker. Verify clean output after startup:
#   curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
#     -d '{"model":"myft","messages":[{"role":"user","content":"hello"}],"max_tokens":40}'
set -euo pipefail

SERVE_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SERVE_ENV_DIR}/local_env.sh" ]; then source "${SERVE_ENV_DIR}/local_env.sh"; fi

IMAGE="${IMAGE:-nvcr.io/nvidia/vllm:26.04-py3}"
MODELS="${MODELS:-${NVFP4_MODELS_DIR:-/path/to/models}}"
ADAPTERS="${ADAPTERS:-${NVFP4_ADAPTERS_DIR:-/path/to/adapters}}"
BASE="${BASE:-Mistral-Small-3.2-24B-Instruct-2506-NVFP4}"
ADAPTER="${ADAPTER:-my_adapter}"
ADAPTER_NAME="${ADAPTER_NAME:-myft}"
NAME="${NAME:-mistral24b}"
HOST="${HOST:-127.0.0.1}"   # --network host: bind loopback by default, override for LAN
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"   # raise if your adapter rank is larger

# Runtime LoRA hot-swap is OPT-IN: pass it to the container only when
# ALLOW_RUNTIME_LORA_UPDATES=1.
RUNTIME_LORA_ENV=()
if [ "${ALLOW_RUNTIME_LORA_UPDATES:-0}" = "1" ]; then
  RUNTIME_LORA_ENV=(-e VLLM_ALLOW_RUNTIME_LORA_UPDATING=1)
fi

sudo docker rm -f "$NAME" >/dev/null 2>&1 || true
exec sudo docker run -d --name "$NAME" \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --network host \
  -v "$MODELS":/models -v "$ADAPTERS":/adapters \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${RUNTIME_LORA_ENV[@]}" \
  "$IMAGE" \
  vllm serve "/models/$BASE" \
    --served-model-name mistral-small-3.2-24b-nvfp4 \
    --host "$HOST" --port "$PORT" --tensor-parallel-size 1 \
    --dtype bfloat16 --tokenizer-mode mistral --language-model-only \
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 4 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enforce-eager --no-enable-prefix-caching \
    --enable-lora --max-lora-rank "$MAX_LORA_RANK" --max-loras 2 \
    --lora-modules "$ADAPTER_NAME=/adapters/$ADAPTER"
