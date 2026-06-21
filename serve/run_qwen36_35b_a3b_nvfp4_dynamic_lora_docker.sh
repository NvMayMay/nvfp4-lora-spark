#!/usr/bin/env bash
# Serve nvidia/Qwen3.6-35B-A3B-NVFP4 (ModelOpt NVFP4; HF arch class
# Qwen3_5MoeForConditionalGeneration, model_type qwen3_5_moe, multimodal with
# the LM nested under language_model.) with DYNAMIC attention + shared_expert
# LoRA, via the NGC vLLM docker image on a box whose only vLLM is the container.
#
# This is the box-B (docker) analogue of run_qwen35_122b_rh_ct_dynamic_lora.sh
# (which assumes a host venv). It loads the attention_only_lora_cutlass_moe
# monkeypatch the same way (PYTHONPATH -> serve/vllm_patches + sitecustomize +
# VLLM_PATCH_ATTN_ONLY_LORA_CUTLASS_MOE=1), so:
#   - the FusedMoE routed experts stay on the CUTLASS NVFP4 kernel (no LoRA),
#   - dense attention + shared_expert LoRA is applied via punica,
#   - flat PEFT keys (base_model.model.model.layers.*) are remapped to the
#     language_model.model.layers.* tree (the binding contract / re-key), and
#   - any adapter that targets routed experts is hard-rejected.
#
# Two adapters are registered so the runtime re-key can be cross-checked against
# an offline-rekeyed copy entirely inside vLLM:
#   ich_orig  = trained PEFT keys, flat layout  -> patch remaps at load
#   ich_rekey = same weights, language_model.*   -> patch remap is a no-op
set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/vllm:26.04-py3}"
REPO="${REPO:-/home/veritan-spark-02/repos/nvfp4-lora-spark}"
MODELS="${MODELS:-/home/veritan-spark-02/Model}"
ADAPTERS="${ADAPTERS:-/home/veritan-spark-02/Model/adapters}"
BASE="${BASE:-Qwen3.6-35B-A3B-NVFP4}"
ORIG="${ORIG:-qwen3_6_35b_a3b_lora_reh2_ich_v4_1_r128_a256}"
REKEY="${REKEY:-qwen3_6_35b_a3b_lora_reh2_ich_v4_1_r128_a256_vllm_rekey}"

NAME="${NAME:-qwen36lora}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
MAX_LORA_RANK="${MAX_LORA_RANK:-128}"   # adapter is r=128 (the 122B recipe was 16)
MAX_LORAS="${MAX_LORAS:-2}"

sudo docker rm -f "$NAME" >/dev/null 2>&1 || true

# NGC-recommended flags (--ipc=host --ulimit ...), GPU, host network for :PORT.
exec sudo docker run -d --name "$NAME" \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v "$REPO":/repo \
  -v "$MODELS":/models \
  -v "$ADAPTERS":/adapters \
  -e PYTHONPATH=/repo/serve/vllm_patches \
  -e VLLM_PATCH_ATTN_ONLY_LORA_CUTLASS_MOE=1 \
  -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$IMAGE" \
  vllm serve "/models/$BASE" \
    --served-model-name qwen36-base \
    --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs 4 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enforce-eager \
    --no-enable-prefix-caching \
    --language-model-only \
    --enable-lora \
    --max-lora-rank "$MAX_LORA_RANK" \
    --max-loras "$MAX_LORAS" \
    --lora-modules "ich_orig=/adapters/$ORIG" "ich_rekey=/adapters/$REKEY"
