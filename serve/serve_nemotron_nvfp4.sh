#!/usr/bin/env bash
# Nemotron-3 NVFP4 deployment script for GB10 / sm_121 (NVIDIA DGX Spark and equivalents).
#
# Brings up vLLM marlin against a Nemotron-3-{Nano,Super}-NVFP4 model, optionally with a LoRA
# adapter attached. Exposes the base as one model ID and (if an adapter is given) the FT version
# as `<base-id>+<adapter-tag>` via vLLM's --lora-modules, so a single serve process answers
# both base and FT calls without needing to restart.
#
# Usage:
#   serve_nemotron_nvfp4.sh <variant> [adapter_dir] [adapter_tag]
#
# Examples:
#   # base only:
#   serve_nemotron_nvfp4.sh nano
#   serve_nemotron_nvfp4.sh super
#
#   # base + adapter:
#   serve_nemotron_nvfp4.sh nano /path/to/nano_adapter ich_v1_0
#   serve_nemotron_nvfp4.sh super /path/to/super_adapter ich_v1_0
#
# Variants resolve to:
#   nano  -> Nemotron-3-Nano-30B-A3B-NVFP4   served as nemotron-3-nano-nvfp4
#   super -> Nemotron-3-Super-120B-A12B-NVFP4 served as nemotron-3-super-a12b-nvfp4
#
# GB10-required flags (per LESSONS.md dependency inventory):
#   VLLM_NVFP4_GEMM_BACKEND=marlin   - sm_121 has no native FP4 compute; weight-only marlin
#   MAX_JOBS=1                        - cap FlashInfer JIT parallelism; >1 OOMs the 128 GB pool
#   --moe-backend marlin              - matched FP4 path for routed MoE experts
#   --dtype bfloat16                  - compute dtype (weights stay FP4; activations bf16)
#
# Binds 0.0.0.0:8000 so an eval host on the LAN can reach it.
set -euo pipefail

VARIANT="${1:-}"
ADAPTER_DIR="${2:-}"
ADAPTER_TAG="${3:-ich_v1_0}"

case "$VARIANT" in
    nano)
        MODEL_DIR=/path/to/Models/Nemotron-3-Nano-30B-A3B-NVFP4
        SERVED_NAME=nemotron-3-nano-nvfp4
        ;;
    super)
        MODEL_DIR=/path/to/Models/Nemotron-3-Super-120B-A12B-NVFP4
        SERVED_NAME=nemotron-3-super-a12b-nvfp4
        ;;
    *)
        echo "Usage: $0 <nano|super> [adapter_dir] [adapter_tag]"
        echo ""
        echo "  variant       'nano' or 'super'"
        echo "  adapter_dir   optional; path to a PEFT-format LoRA adapter dir to attach"
        echo "  adapter_tag   optional; suffix for the FT model id (default: ich_v1_0)"
        exit 2
        ;;
esac

SERVE_LOG=/path/to/research/serve_${VARIANT}.log
PIDFILE=/tmp/serve_nemotron_${VARIANT}.pid

LORA_ARGS=()
if [ -n "$ADAPTER_DIR" ]; then
    if [ ! -d "$ADAPTER_DIR" ]; then
        echo "ERROR: adapter_dir does not exist: $ADAPTER_DIR" >&2
        exit 3
    fi
    if [ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]; then
        echo "ERROR: $ADAPTER_DIR is not a PEFT-format adapter (missing adapter_model.safetensors)" >&2
        exit 3
    fi
    FT_NAME="${SERVED_NAME}+${ADAPTER_TAG}"
    LORA_ARGS=(
        --enable-lora
        --lora-modules "${FT_NAME}=${ADAPTER_DIR}"
        --max-lora-rank 8
        --max-loras 1
    )
fi

source /path/to/venvs/serve/bin/activate

VLLM_NVFP4_GEMM_BACKEND=marlin \
MAX_JOBS=1 \
nohup vllm serve "$MODEL_DIR" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 8192 --max-num-batched-tokens 8192 --max-num-seqs 1 \
    --moe-backend marlin \
    "${LORA_ARGS[@]}" \
    > "$SERVE_LOG" 2>&1 &

echo $! > "$PIDFILE"

echo "vllm serve PID=$(cat "$PIDFILE") variant=${VARIANT}"
echo "log:    tail -f $SERVE_LOG"
echo "url:    http://$(hostname -I | awk '{print $1}'):8000"
echo "models:"
echo "  - $SERVED_NAME (base)"
if [ -n "$ADAPTER_DIR" ]; then
    echo "  - ${FT_NAME} (adapter at $ADAPTER_DIR)"
fi
echo ""
echo "Wait for 'Application startup complete' in the log before sending requests."
echo "Stop with: kill -TERM \$(cat $PIDFILE)"
