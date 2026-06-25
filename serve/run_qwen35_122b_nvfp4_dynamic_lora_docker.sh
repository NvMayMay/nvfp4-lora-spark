#!/usr/bin/env bash
# Serve RedHatAI-Qwen3.5-122B-A10B-NVFP4 (compressed-tensors NVFP4; the model the
# attention_only_lora_cutlass_moe patch was built and validated for) with DYNAMIC
# attention-only LoRA, via the NGC vLLM docker image on box B.
#
# Docker analogue of run_qwen35_122b_rh_ct_dynamic_lora.sh (host-venv version).
# CUTLASS NVFP4 MoE stays LoRA-free (the patch pins is_lora_enabled=False); dense
# attention q/k/v/o LoRA is applied via punica; flat PEFT keys are remapped to the
# language_model.model.layers.* tree (the binding contract / re-key).
#
# Two adapters registered to cross-check the runtime re-key against an offline
# re-keyed copy entirely inside vLLM:
#   ich_orig  = trained PEFT keys, flat layout  -> patch remaps at load
#   ich_rekey = same weights, language_model.*   -> patch remap is a no-op
set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/vllm:26.05-py3}"
REPO="${REPO:-/home/veritan-spark-02/repos/nvfp4-lora-spark}"
MODELS="${MODELS:-/home/veritan-spark-02/Model}"
ADAPTERS="${ADAPTERS:-/home/veritan-spark-02/Model/adapters}"
BASE="${BASE:-RedHatAI-Qwen3.5-122B-A10B-NVFP4}"
ORIG="${ORIG:-qwen3_5_122b_a10b_rh_nvfp4_lora_ich_v3_5/best}"
REKEY="${REKEY:-qwen3_5_122b_a10b_rh_nvfp4_lora_ich_v3_5_vllm_rekey}"

NAME="${NAME:-qwen35122b}"
HOST="${HOST:-127.0.0.1}"   # --network host: bind loopback by default, override for LAN
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"   # matches the validated 122B recipe
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"   # this adapter is r=16
MAX_LORAS="${MAX_LORAS:-2}"

# Runtime LoRA hot-swap is OPT-IN: pass it to the container only when
# ALLOW_RUNTIME_LORA_UPDATES=1.
RUNTIME_LORA_ENV=()
if [ "${ALLOW_RUNTIME_LORA_UPDATES:-0}" = "1" ]; then
  RUNTIME_LORA_ENV=(-e VLLM_ALLOW_RUNTIME_LORA_UPDATING=1)
fi

sudo docker rm -f "$NAME" >/dev/null 2>&1 || true

exec sudo docker run -d --name "$NAME" \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v "$REPO":/repo \
  -v "$MODELS":/models \
  -v "$ADAPTERS":/adapters \
  -e PYTHONPATH=/repo/serve/vllm_patches \
  -e VLLM_PATCH_ATTN_ONLY_LORA_CUTLASS_MOE=1 \
  "${RUNTIME_LORA_ENV[@]}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$IMAGE" \
  vllm serve "/models/$BASE" \
    --served-model-name qwen35-122b-base \
    --host "$HOST" --port "$PORT" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs 4 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enforce-eager \
    --no-enable-prefix-caching \
    --language-model-only \
    --moe-backend cutlass \
    --enable-lora \
    --max-lora-rank "$MAX_LORA_RANK" \
    --max-loras "$MAX_LORAS" \
    --lora-modules "ich_orig=/adapters/$ORIG" "ich_rekey=/adapters/$REKEY"
