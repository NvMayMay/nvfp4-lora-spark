#!/usr/bin/env bash
# Qwen3.5-122B-A10B (RedHatAI compressed-tensors NVFP4) + DYNAMIC ICH v3.5 LoRA.
#
# ============================================================================
# DO NOT RUN UNTIL TRAINING COMPLETES (the 13h run owns the GPU until ~01:30).
# This loads the full model onto CUDA. Check `nvidia-smi`/trainer logs first.
# ============================================================================
#
# Path A revived: request-time LoRA (--enable-lora --lora-modules) over the
# CUTLASS NVFP4 MoE backend, enabled by the runtime monkeypatch
# serve/vllm_patches/attention_only_lora_cutlass_moe.py. Stock vLLM 0.22.1
# blocks this combination: --enable-lora flips is_lora_enabled for every
# FusedMoE (fused_moe/layer.py:334), the oracle rejects CUTLASS for LoRA
# (modular_kernel.py:580), and the LoRA-capable NVFP4 MoE backends are
# unusable on Spark (MARLIN repack OOM, EMULATION lost LoRA in 0.22.1).
# The adapter targets ONLY dense attention projections, so MoE LoRA support
# is not actually needed; the patch makes the MoE report LoRA-disabled while
# dense-layer LoRA stays fully enabled, and hard-rejects any adapter that
# targets expert modules. Full trace, safety argument and validation
# sequence: docs/plans/DYNAMIC_LORA_CUTLASS_PATCH.md.
#
# Patch transport: PYTHONPATH points at serve/vllm_patches so Python's site
# machinery imports sitecustomize.py in BOTH the APIServer process and the
# spawned VLLM::EngineCore subprocess; the patch itself is gated on
# VLLM_PATCH_ATTN_ONLY_LORA_CUTLASS_MOE=1. Verify at startup: one
#   [sitecustomize pid=...] applied attention_only_lora_cutlass_moe patch
# line per process, and the engine log must say
#   Using 'VLLM_CUTLASS' NvFp4 MoE backend
#
# Serve flags are a clone of the validated CUTLASS recipe in
# run_qwen35_122b_rh_ct_lora.sh / run_qwen35_122b_nvfp4.sh (May 31 log:
# 11.8-14.5 tok/s generation). Port 8000; the Mistral LoRA server owns 8001.
#
# Adapter selection:
#   ADAPTER_DIR defaults to the trainer's output root (final adapter).
#   ADAPTER_DIR=$ADAPTER_ROOT/best  -> best-by-val-loss copy.
# Hot-swap testing: set ALLOW_RUNTIME_LORA_UPDATES=1 to export
# VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 so a second adapter (e.g. a
# checkpoint_step_* dir) can be loaded at runtime via POST
# /v1/load_lora_adapter without restarting (validation step e in the plan
# doc). It is OFF by default.

set -euo pipefail

# Machine-local roots (models / adapters / serve venv). Set the NVFP4_* env
# vars, or create serve/local_env.sh from serve/local_env.example.sh.
SERVE_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SERVE_ENV_DIR}/local_env.sh" ]; then source "${SERVE_ENV_DIR}/local_env.sh"; fi
: "${NVFP4_MODELS_DIR:?Set NVFP4_MODELS_DIR or create serve/local_env.sh (see serve/local_env.example.sh)}"
: "${NVFP4_SERVE_VENV:?Set NVFP4_SERVE_VENV or create serve/local_env.sh (see serve/local_env.example.sh)}"
: "${NVFP4_ADAPTERS_DIR:?Set NVFP4_ADAPTERS_DIR or create serve/local_env.sh (see serve/local_env.example.sh)}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_DIR="$REPO_DIR/serve/vllm_patches"

BASE_DIR="${BASE_DIR:-${NVFP4_MODELS_DIR}/RedHatAI-Qwen3.5-122B-A10B-NVFP4}"
ADAPTER_ROOT="${NVFP4_ADAPTERS_DIR}/qwen3_5_122b_a10b_rh_nvfp4_lora_ich_v3_5"
ADAPTER_DIR="${ADAPTER_DIR:-$ADAPTER_ROOT}"
ADAPTER_NAME="${ADAPTER_NAME:-ich_v3_5}"

SERVED_NAME="${SERVED_NAME:-qwen3.5-122b-a10b-nvfp4}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"   # 8001 is the Mistral LoRA server; do not collide.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"   # adapter is r=16 alpha=32
MAX_LORAS="${MAX_LORAS:-2}"            # 2 slots so hot-swap A/B runs co-active

if [ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ] || \
   [ ! -f "$ADAPTER_DIR/adapter_config.json" ]; then
  echo "ERROR: no adapter_model.safetensors/adapter_config.json in $ADAPTER_DIR" >&2
  echo "Training may not have finished, or you want ADAPTER_DIR=$ADAPTER_ROOT/best" >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MAX_JOBS="${MAX_JOBS:-1}"

# --- patch transport (see header) ---
export PYTHONPATH="$PATCH_DIR${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_PATCH_ATTN_ONLY_LORA_CUTLASS_MOE=1
# Allow POST /v1/load_lora_adapter for the hot-swap validation step.
# OPT-IN: export only when ALLOW_RUNTIME_LORA_UPDATES=1.
if [ "${ALLOW_RUNTIME_LORA_UPDATES:-0}" = "1" ]; then
  export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
fi

# --- 0.21 -> 0.22.1 default-drift pins (2nd-pass de-risk, 2026-06-10) ---
# The base CUTLASS recipe was validated on vLLM 0.21. These flags are pinned
# explicitly so 0.22.1 default changes cannot silently alter the proven serve:
#   * --enforce-eager (already below): REQUIRED on Spark. CUDA-graph capture
#     consumes ~3 GiB extra and trips the free-memory safety floor
#     (serve/README.md). Not a drift item, but load-bearing; keep it.
#   * --moe-backend cutlass (already below): still a valid value in 0.22.1
#     (oracle/nvfp4.py map_nvfp4_backend "cutlass" -> VLLM_CUTLASS). The model
#     does NOT set swiglu_limit, so the new explicit-backend swiglu guard
#     (oracle/nvfp4.py:246-255) does not fire.
#   * --no-enable-prefix-caching: 0.22.1 flips the default to True
#     (config/cache.py:91) and, for this hybrid GDN+attention model, would
#     engage the mamba prefix-caching path that the proven 0.21 base serve
#     never exercised. Pin OFF to keep behavior identical for the demo.
#   * --enable-chunked-prefill: 0.22.1 resolves this from the model's
#     is_chunked_prefill_supported (True for this generative model,
#     config/model.py:1817), i.e. it defaults ON. Pin ON explicitly: a future
#     default flip to OFF would trip verify_max_model_len (scheduler.py:260)
#     since MAX_BATCHED_TOKENS(16384) >= MAX_MODEL_LEN(4096) but the engine
#     also warns that disabling it on a supporting model can crash
#     (arg_utils.py:2383). Do NOT pass --no-enable-chunked-prefill here.
#   * mamba-cache-mode left at the 0.22.1 default "none" (config/cache.py:132).
#     The model only rejects "all" (qwen3_5.py:459); "align" would add
#     block-size constraints (vllm.py:2101) not needed at this concurrency.
ENABLE_PREFIX_CACHING_FLAG="${ENABLE_PREFIX_CACHING_FLAG:---no-enable-prefix-caching}"
ENABLE_CHUNKED_PREFILL_FLAG="${ENABLE_CHUNKED_PREFILL_FLAG:---enable-chunked-prefill}"

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
    --language-model-only \
    --reasoning-parser qwen3 \
    --moe-backend cutlass \
    --enable-lora \
    --max-lora-rank "$MAX_LORA_RANK" \
    --max-loras "$MAX_LORAS" \
    --lora-modules "$ADAPTER_NAME=$ADAPTER_DIR"
