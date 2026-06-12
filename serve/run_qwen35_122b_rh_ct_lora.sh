#!/usr/bin/env bash
# Qwen3.5-122B-A10B (RedHatAI compressed-tensors NVFP4) + ICH v3.5 LoRA.
#
# ============================================================================
# DO NOT RUN UNTIL TRAINING COMPLETES (the 13h run owns the GPU until ~01:30).
# Both subcommands load CUDA. Check `nvidia-smi`/trainer logs first.
# ============================================================================
#
# Path B, merge-then-serve. Runtime LoRA (--enable-lora) is BLOCKED for this
# model in vLLM 0.22.1: enabling LoRA globally flips is_lora_enabled for every
# FusedMoE layer, the CUTLASS NVFP4 MoE kernel reports supports_lora()=False,
# and the only LoRA-capable NVFP4 MoE backend left (MARLIN) cannot fit a
# 120B-class repack on Spark. See docs/plans/SERVE_PATH_QWEN35_MISTRAL.md.
#
# Usage:
#   ./run_qwen35_122b_rh_ct_lora.sh merge   # bake the adapter into a new CT
#                                           # NVFP4 checkpoint (minutes; needs
#                                           # ~40 GB host RAM per shard)
#   ./run_qwen35_122b_rh_ct_lora.sh serve   # serve the merged checkpoint via
#                                           # the proven VLLM_CUTLASS recipe
#
# Cheap pre-checks (CPU only, still avoid while the trainer is alive):
#   CUDA_VISIBLE_DEVICES= ../scripts/merge_lora_into_ct_nvfp4.py --self-test
#   CUDA_VISIBLE_DEVICES= ... --dry-run --base-model-dir ... --lora-adapter-dir ...
#
# Serve flags are a clone of the validated run_qwen35_122b_nvfp4.sh recipe
# (May 31 log: VLLM_CUTLASS backend, 11.8-14.5 tok/s generation). Conservative
# for 131 GB UMA: gpu-memory-utilization 0.70, max-model-len 4096, 4 seqs,
# enforce-eager. Note vLLM was upgraded 0.21 -> 0.22.1 on Jun 8; if the merged
# serve misbehaves, serve the unmerged base (run_qwen35_122b_nvfp4.sh with
# MODEL_DIR pointed at the RedHatAI dir) to bisect upgrade vs merge.

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

BASE_DIR="${BASE_DIR:-${NVFP4_MODELS_DIR}/RedHatAI-Qwen3.5-122B-A10B-NVFP4}"
# Final adapter is written to the adapter root by the trainer; best-val copy
# lives in best/. Override ADAPTER_DIR=...:/best to use the best checkpoint.
ADAPTER_DIR="${ADAPTER_DIR:-${NVFP4_ADAPTERS_DIR}/qwen3_5_122b_a10b_rh_nvfp4_lora_ich_v3_5}"
MERGED_DIR="${MERGED_DIR:-${NVFP4_MODELS_DIR}/RedHatAI-Qwen3.5-122B-A10B-NVFP4-ich-v3.5}"

SERVED_NAME="${SERVED_NAME:-qwen3.5-122b-a10b-nvfp4+ich_v3_5}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MAX_JOBS="${MAX_JOBS:-1}"

cmd="${1:-}"

case "$cmd" in
  merge)
    if [ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]; then
      echo "ERROR: no adapter_model.safetensors in $ADAPTER_DIR" >&2
      echo "Training may not have finished, or you want ADAPTER_DIR=\$ADAPTER_DIR/best" >&2
      exit 1
    fi
    exec "$VENV_PY" "$REPO_DIR/scripts/merge_lora_into_ct_nvfp4.py" \
        --base-model-dir "$BASE_DIR" \
        --lora-adapter-dir "$ADAPTER_DIR" \
        --output-dir "$MERGED_DIR"
    ;;
  serve)
    if [ ! -f "$MERGED_DIR/model.safetensors.index.json" ]; then
      echo "ERROR: merged checkpoint not found at $MERGED_DIR; run 'merge' first" >&2
      exit 1
    fi
    source "${NVFP4_SERVE_VENV}/bin/activate"
    exec vllm serve "$MERGED_DIR" \
        --served-model-name "$SERVED_NAME" \
        --host "$HOST" --port "$PORT" \
        --tensor-parallel-size 1 \
        --dtype bfloat16 \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --enforce-eager \
        --language-model-only \
        --reasoning-parser qwen3 \
        --moe-backend cutlass
    ;;
  *)
    echo "usage: $0 {merge|serve}" >&2
    echo "DO NOT RUN until the training run completes (~01:30)." >&2
    exit 2
    ;;
esac
