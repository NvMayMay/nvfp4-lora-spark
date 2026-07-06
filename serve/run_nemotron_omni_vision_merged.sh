#!/usr/bin/env bash
# Serve a MERGED vision-fine-tuned Nemotron-3-Nano-Omni-30B-A3B NVFP4 VLM via vLLM
# (OpenAI-compatible), image input enabled. Same merge-to-bf16-base story as
# serve/run_mistral24b_vision_merged.sh (a --train-target vision adapter targets the bf16
# RADIO tower + mlp1 projector, which vLLM's runtime-LoRA path does NOT touch), so we serve
# the merged checkpoint as a plain VLM -- the fine-tune lives in the merged tower weights.
#
# Optional runtime LLM LoRA:
#   NEMOTRON_ENABLE_LLM_LORA=1 LLM_LORA_ADAPTER_DIR=/path/to/exported/llm ./serve/run_...
#
# THIS ARCH'S GB10 GOTCHAS (learned the hard way -- see gb10_serving_recipes memory / docs):
#  * FIRST serve compiles the Mamba2 SSD Triton kernels (58 autotune configs) -> ~16 min the
#    FIRST time on a cold ~/.triton/cache, then cached -> ~2.5 min on later serves. Be patient;
#    it is NOT hung. TRITON_PRINT_AUTOTUNING=1 shows progress.
#  * --skip-mm-profiling: the omni model has a VIDEO path whose dummy mm-profiling hangs; skip it
#    (image input is preserved). --limit-mm-per-prompt disallows video/audio for an image demo.
#  * --gpu-memory-utilization 0.55 (NOT 0.85): image preprocessing (bicubic resize) runs on GPU;
#    an aggressive util leaves no headroom and the FIRST image request 400s with a CUDA OOM.
#  * --mm-processor-kwargs max_num_tiles: MATCH the tiling the adapter was TRAINED at
#    (--max-image-tiles). Training at 1 tile then serving at up-to-12 is a distribution shift on
#    the exact weights you trained. Default 1 here mirrors the vqa-rad demo.
#  * NGC Docker vllm:26.04 does NOT support this arch; use the venv vLLM 0.22.1 (native
#    nano_nemotron_vl path). Needs `ninja` on PATH for the flashinfer JIT.
#  * It IS a reasoner: clients pass chat_template_kwargs {"enable_thinking": false} (/no_think)
#    for direct answers; strip any <think>...</think> for a clean short-answer metric.
set -euo pipefail

SERVE_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SERVE_ENV_DIR}/local_env.sh" ]; then source "${SERVE_ENV_DIR}/local_env.sh"; fi
: "${NVFP4_MODELS_DIR:?Set NVFP4_MODELS_DIR or create serve/local_env.sh}"
: "${NVFP4_SERVE_VENV:?Set NVFP4_SERVE_VENV or create serve/local_env.sh}"

MERGED_DIR="${MERGED_DIR:-${NVFP4_MODELS_DIR}/Nemotron-Omni-vision-merged}"
SERVED_NAME="${SERVED_NAME:-nemotron-omni-vision-merged}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
MAX_IMAGE_TILES="${MAX_IMAGE_TILES:-1}"

NEMOTRON_ENABLE_LLM_LORA="${NEMOTRON_ENABLE_LLM_LORA:-0}"
LLM_LORA_ADAPTER_DIR="${LLM_LORA_ADAPTER_DIR:-}"
LLM_LORA_NAME="${LLM_LORA_NAME:-nemotron-omni-llm-lora}"

if [ ! -f "$MERGED_DIR/model.safetensors.index.json" ]; then
  echo "ERROR: no model.safetensors.index.json in $MERGED_DIR" >&2
  echo "       Run scripts/merge_vision_lora.py to produce the merged VLM first." >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITON_PRINT_AUTOTUNING="${TRITON_PRINT_AUTOTUNING:-1}"

LORA_FLAGS=()

append_vllm_plugin() {
  local plugin="$1"
  if [ -z "${VLLM_PLUGINS:-}" ]; then
    export VLLM_PLUGINS="$plugin"
    return
  fi
  case ",${VLLM_PLUGINS}," in
    *",${plugin},"*) ;;
    *) export VLLM_PLUGINS="${VLLM_PLUGINS},${plugin}" ;;
  esac
}

if [ "$NEMOTRON_ENABLE_LLM_LORA" = "1" ]; then
  : "${LLM_LORA_ADAPTER_DIR:?Set LLM_LORA_ADAPTER_DIR to scripts/export_llm_lora.py output}"
  if [ ! -f "${LLM_LORA_ADAPTER_DIR}/adapter_config.json" ]; then
    echo "ERROR: no adapter_config.json in LLM_LORA_ADAPTER_DIR=${LLM_LORA_ADAPTER_DIR}" >&2
    exit 1
  fi
  if ! compgen -G "${LLM_LORA_ADAPTER_DIR}/adapter_model*.safetensors" >/dev/null; then
    echo "ERROR: no adapter_model*.safetensors in LLM_LORA_ADAPTER_DIR=${LLM_LORA_ADAPTER_DIR}" >&2
    exit 1
  fi

  append_vllm_plugin "nemotron_vl_lora"
  LORA_FLAGS+=(--enable-lora --lora-modules "${LLM_LORA_NAME}=${LLM_LORA_ADAPTER_DIR}")
  echo "Runtime LLM LoRA enabled: ${LLM_LORA_NAME}=${LLM_LORA_ADAPTER_DIR}" >&2
  echo "VLLM_PLUGINS=${VLLM_PLUGINS}" >&2
fi

source "${NVFP4_SERVE_VENV}/bin/activate"
export PATH="${NVFP4_SERVE_VENV}/bin:$PATH"   # so the spawned EngineCore finds ninja
exec vllm serve "$MERGED_DIR" \
    --served-model-name "$SERVED_NAME" \
    --host "$HOST" --port "$PORT" \
    --trust-remote-code --tokenizer-mode auto --enforce-eager --skip-mm-profiling \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --mm-processor-kwargs "{\"max_num_tiles\":${MAX_IMAGE_TILES}}" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --kv-cache-dtype fp8 \
    "${LORA_FLAGS[@]}"
