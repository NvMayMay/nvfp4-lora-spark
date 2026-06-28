#!/usr/bin/env bash
# Serve EXPERT-LoRA (LoRA on the routed MoE experts) over a frozen NVFP4 base on ONE GB10.
#
# This is the expert-adapting sibling of run_glm45_air_nvfp4_dynamic_lora.sh (which is
# ATTENTION-ONLY via the cutlass patch). It uses the EMULATION NVFP4-MoE backend, the
# only one whose fused experts apply a LoRA delta and that LOADS on a single 121 GiB
# UMA box (cutlass/flashinfer experts are not LoRA-capable; marlin IS but its load
# repack does not fit one box for GLM-Air-sized MoEs). VALIDATED on GLM-4.5-Air-NVFP4,
# single GB10/sm_121, 2026-06-28: base vs adapter outputs diverge => the expert delta
# is genuinely applied (see docs/plans/expert_lora_scope.md).
#
# The adapter must be in vLLM per-expert format. If you trained with this repo's
# --expert-lora-r, first rekey the native stacked adapter:
#   python scripts/rekey_expert_lora_for_vllm.py --in <native_adapter> --out <vllm_adapter>
#
# Tradeoff: emulation dequantizes the experts on every forward -> correct but slow
# (serving-for-iteration, not max throughput). Attention-only LoRA should keep using the
# faster cutlass launcher; this path is for when you actually need to adapt the experts.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the NVFP4 MoE base (e.g. .../GLM-4.5-Air-106B-A12B-NVFP4)}"
ADAPTER_DIR="${ADAPTER_DIR:?set ADAPTER_DIR to a vLLM per-expert adapter (rekeyed)}"
ADAPTER_NAME="${ADAPTER_NAME:-myft}"
VLLM="${VLLM:-vllm}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"

# Load-bearing on GB10 UMA: without expandable_segments the big-model load over-commits
# the shared CPU+GPU pool and can hard-reboot the box.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PATH=/usr/local/cuda/bin:$PATH   # flashinfer JIT needs nvcc

if [ "${ALLOW_RUNTIME_LORA_UPDATES:-0}" = "1" ]; then
  export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
fi

exec env MAX_JOBS=1 "$VLLM" serve "$MODEL_DIR" \
  --served-model-name base --host 127.0.0.1 --port "$PORT" \
  --moe-backend emulation \
  --enable-lora --max-lora-rank "$MAX_LORA_RANK" --max-loras 2 \
  --lora-modules "$ADAPTER_NAME=$ADAPTER_DIR" \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" --enforce-eager \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --kv-cache-dtype fp8 \
  --trust-remote-code
