#!/usr/bin/env bash
# One-command reproduction of the Spider text-to-SQL runtime-LoRA before/after.
# Automates the step-by-step walkthrough in REPRODUCE_SPIDER.md (the authoritative ref).
# Public base (nvidia/Llama-3.1-8B-Instruct-NVFP4) + public Spider dataset; deterministic.
#
# Run from the repo root. prep/train need the training venv; serve needs the vLLM 0.22.1
# host venv (point $VLLM at it); eval is stdlib-only. Paths are env-overridable.
#
#   bash scripts/repro_spider.sh
#   VLLM=/path/to/qwen-serve/bin/vllm N=1034 EPOCHS=2 bash scripts/repro_spider.sh
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-models/Llama-3.1-8B-Instruct-NVFP4}
DATA_DIR=${DATA_DIR:-data/spider}
ADAPTER_DIR=${ADAPTER_DIR:-adapters/spider_llama8b_r32}
EPOCHS=${EPOCHS:-1}
N=${N:-200}
PORT=${PORT:-8000}
OUT=${OUT:-spider_retention.json}
VLLM=${VLLM:-vllm}          # the vLLM 0.22.1 host-venv binary
PYTHON=${PYTHON:-python}
SERVE_LOG=${SERVE_LOG:-/tmp/repro_spider_serve.log}

step(){ echo; echo "=== $* ==="; }

step "1/5 base model"
if [ -f "$MODEL_DIR/config.json" ]; then echo "present: $MODEL_DIR"; else
  hf download nvidia/Llama-3.1-8B-Instruct-NVFP4 --local-dir "$MODEL_DIR"
fi

step "2/5 prep Spider data"
if [ -f "$DATA_DIR/spider.dev.chat.jsonl" ]; then echo "present: $DATA_DIR"; else
  "$PYTHON" scripts/prep_spider.py --out-dir "$DATA_DIR"
fi

step "3/5 train (epochs=$EPOCHS, ~1.8h for 1 epoch)"
if [ -d "$ADAPTER_DIR/best" ]; then echo "adapter present: $ADAPTER_DIR/best"; else
  "$PYTHON" -u scripts/train_nvfp4_lora.py \
    --model-dir "$MODEL_DIR" \
    --train-file "$DATA_DIR/spider.train.chat.jsonl" \
    --val-file   "$DATA_DIR/spider.dev.chat.jsonl" \
    --output-dir "$ADAPTER_DIR" \
    --max-length 2048 --epochs "$EPOCHS" --batch-size 1 --grad-accum 16 \
    --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
    --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --learning-rate 1e-4 --warmup-ratio 0.03 \
    --checkpoint-every 50 --eval-every 200 --eval-subset 64
fi

step "4/5 serve base + adapter (runtime-LoRA)"
export PATH=/usr/local/cuda/bin:$PATH    # flashinfer JIT needs nvcc
MAX_JOBS=1 "$VLLM" serve "$MODEL_DIR" \
  --served-model-name base --host 127.0.0.1 --port "$PORT" \
  --enable-lora --max-lora-rank 32 --max-loras 2 \
  --lora-modules myft="$ADAPTER_DIR/best" \
  --max-model-len 8192 --enforce-eager \
  --gpu-memory-utilization 0.6 --kv-cache-dtype fp8 > "$SERVE_LOG" 2>&1 &
SERVE_PID=$!
cleanup(){ kill "$SERVE_PID" 2>/dev/null || true; pkill -9 EngineCor 2>/dev/null || true; }
trap cleanup EXIT
echo "waiting for serve READY (first run JITs flashinfer, a few minutes)..."
for _ in $(seq 1 180); do
  grep -q "Application startup complete" "$SERVE_LOG" && { echo "READY"; break; }
  kill -0 "$SERVE_PID" 2>/dev/null || { echo "serve exited early:"; tail -n 20 "$SERVE_LOG"; exit 1; }
  sleep 10
done

step "5/5 eval before/after (n=$N, deterministic)"
"$PYTHON" scripts/eval_retention.py \
  --dev-file "$DATA_DIR/spider.dev.chat.jsonl" \
  --models base myft --n "$N" --out "$OUT"
echo; echo "done. result written to $OUT"
