#!/usr/bin/env bash
# GLM-4.5-Air-106B-A12B (compressed-tensors NVFP4) + DYNAMIC attention-only LoRA.
#
# Host-venv analogue of run_qwen35_122b_rh_ct_dynamic_lora.sh, for the GLM-4.5-Air
# family (model_type glm4_moe, arch Glm4MoeForCausalLM). The trained adapter targets
# ONLY dense attention projections (q/k/v/o_proj), so routed-expert MoE LoRA is not
# needed. The attention_only_lora_cutlass_moe patch makes every FusedMoE report
# LoRA-disabled (stays on the CUTLASS NVFP4 kernel) while dense-attention LoRA is
# applied via punica, and hard-rejects any adapter that targets expert modules.
#
# Unlike the Qwen multimodal serve, GLM-4.5-Air is a text-only causal LM whose keys
# are already model.layers.* (matching the vLLM Glm4MoeForCausalLM tree), so the
# patch's language_model.* key rewrite is a no-op here. No --language-model-only and
# no --reasoning-parser by default (GLM emits its own reasoning markup; set
# REASONING_PARSER if your vLLM build ships a glm parser).
#
# Patch transport: PYTHONPATH -> serve/vllm_patches so sitecustomize.py loads in both
# the APIServer and the VLLM::EngineCore subprocess; gated on
# VLLM_PATCH_ATTN_ONLY_LORA_CUTLASS_MOE=1. Verify at startup:
#   [sitecustomize pid=...] applied attention_only_lora_cutlass_moe patch  (one per process)
#   Using 'VLLM_CUTLASS' NvFp4 MoE backend
set -euo pipefail

SERVE_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SERVE_ENV_DIR}/local_env.sh" ]; then source "${SERVE_ENV_DIR}/local_env.sh"; fi
: "${NVFP4_MODELS_DIR:?Set NVFP4_MODELS_DIR or create serve/local_env.sh}"
: "${NVFP4_SERVE_VENV:?Set NVFP4_SERVE_VENV or create serve/local_env.sh}"
: "${NVFP4_ADAPTERS_DIR:?Set NVFP4_ADAPTERS_DIR or create serve/local_env.sh}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_DIR="$REPO_DIR/serve/vllm_patches"

BASE_DIR="${BASE_DIR:-${NVFP4_MODELS_DIR}/GLM-4.5-Air-106B-A12B-NVFP4}"
ADAPTER_ROOT="${NVFP4_ADAPTERS_DIR}/glm45air_lora_ich_v4_1_8k_r32a64"
ADAPTER_DIR="${ADAPTER_DIR:-$ADAPTER_ROOT/best}"
ADAPTER_NAME="${ADAPTER_NAME:-ich_v4_1}"

SERVED_NAME="${SERVED_NAME:-glm-4.5-air-nvfp4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
# Serve context is independent of the 8192 training seq_len; GLM-4.5-Air's native
# context is far larger. 32768 because REH-2 needs_input prompts reach ~9.4k tokens
# and with the <think> block + up to ~7k output a request can exceed 16384 (the base
# was evaluated at 16384 but tripped a 1-row edge case at 16385). 32768 is safe and
# well within native range. Lower it only if KV-cache memory is tight.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"   # adapter is r=32 alpha=64
MAX_LORAS="${MAX_LORAS:-2}"
REASONING_PARSER="${REASONING_PARSER:-}"

if [ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ] || \
   [ ! -f "$ADAPTER_DIR/adapter_config.json" ]; then
  echo "ERROR: no adapter_model.safetensors/adapter_config.json in $ADAPTER_DIR" >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MAX_JOBS="${MAX_JOBS:-1}"
export PYTHONPATH="$PATCH_DIR${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_PATCH_ATTN_ONLY_LORA_CUTLASS_MOE=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1

ENABLE_PREFIX_CACHING_FLAG="${ENABLE_PREFIX_CACHING_FLAG:---no-enable-prefix-caching}"
ENABLE_CHUNKED_PREFILL_FLAG="${ENABLE_CHUNKED_PREFILL_FLAG:---enable-chunked-prefill}"
REASONING_FLAG=""
[ -n "$REASONING_PARSER" ] && REASONING_FLAG="--reasoning-parser $REASONING_PARSER"

source "${NVFP4_SERVE_VENV}/bin/activate"
exec vllm serve "$BASE_DIR" \
    --served-model-name "$SERVED_NAME" \
    --host "$HOST" --port "$PORT" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enforce-eager \
    $ENABLE_CHUNKED_PREFILL_FLAG \
    $ENABLE_PREFIX_CACHING_FLAG \
    $REASONING_FLAG \
    --moe-backend cutlass \
    --enable-lora \
    --max-lora-rank "$MAX_LORA_RANK" \
    --max-loras "$MAX_LORAS" \
    --lora-modules "$ADAPTER_NAME=$ADAPTER_DIR"
