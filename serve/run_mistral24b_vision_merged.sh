#!/usr/bin/env bash
# Serve a MERGED vision-fine-tuned Mistral-Small-3.2-24B NVFP4 VLM via vLLM
# (OpenAI-compatible), multimodal enabled.
#
# WHY NO --enable-lora / --lora-modules here (this is the whole point):
# vLLM runtime-LoRA applies the adapter delta to the LLM BACKBONE only. A
# --train-target vision adapter targets the bf16 vision tower + multimodal
# projector, which vLLM's LoRA path does NOT touch -- so a vision adapter has NO
# runtime-LoRA path. The supported vision serve story is merge-to-bf16-base:
# bake the vision delta into a copy of the base checkpoint with
#   scripts/merge_vision_lora.py --base-model-dir <base> --adapter-dir <vision-adapter> \
#       --out-dir <merged>
# then serve that merged dir as a PLAIN VLM (below). The fine-tune lives in the
# merged tower weights; no adapter is loaded at serve time. See docs/SERVING.md
# section 6 for the full rationale + the runtime-LoRA-vision probe.
#
# Contrast serve/run_mistral24b_nvfp4_lora.sh, which serves the base + a TEXT
# (LLM-backbone) LoRA live via --lora-modules -- that path exists because text
# targets are on the backbone; the vision path does not have that option.
set -euo pipefail

SERVE_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SERVE_ENV_DIR}/local_env.sh" ]; then source "${SERVE_ENV_DIR}/local_env.sh"; fi
: "${NVFP4_MODELS_DIR:?Set NVFP4_MODELS_DIR or create serve/local_env.sh}"
: "${NVFP4_SERVE_VENV:?Set NVFP4_SERVE_VENV or create serve/local_env.sh}"

# MERGED_DIR is the output of scripts/merge_vision_lora.py (NVFP4 backbone kept
# byte-for-byte, bf16 tower carries the fine-tune). Default name mirrors the base
# with a -vision-merged suffix; override MERGED_DIR to point at your own merge.
MERGED_DIR="${MERGED_DIR:-${NVFP4_MODELS_DIR}/Mistral-Small-3.2-24B-Instruct-2506-NVFP4-vision-merged}"
SERVED_NAME="${SERVED_NAME:-mistral-small-3.2-24b-vision-merged}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

if [ ! -f "$MERGED_DIR/model.safetensors.index.json" ]; then
  echo "ERROR: no model.safetensors.index.json in $MERGED_DIR" >&2
  echo "       Run scripts/merge_vision_lora.py to produce the merged VLM first." >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

source "${NVFP4_SERVE_VENV}/bin/activate"
# Multimodal serve: DO NOT pass --language-model-only (we want the vision tower
# active so images fuse). No --enable-lora: the fine-tune is merged into the base.
exec vllm serve "$MERGED_DIR" \
    --served-model-name "$SERVED_NAME" \
    --host "$HOST" --port "$PORT" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --tokenizer-mode "${TOKENIZER_MODE:-mistral}" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enforce-eager
