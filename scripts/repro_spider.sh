#!/usr/bin/env bash
# One-command reproduction of the Spider text-to-SQL runtime-LoRA before/after.
# Automates the step-by-step walkthrough in REPRODUCE_SPIDER.md (the authoritative ref).
# Public base (nvidia/Llama-3.1-8B-Instruct-NVFP4) + public Spider dataset; deterministic.
#
# Run from the repo root. prep/train need the training venv; serve needs the vLLM 0.22.1
# host venv (point $VLLM at it); eval is stdlib-only. Paths are env-overridable.
#
# Defaults reproduce the README headline (full 1034-row dev, 2 epochs, ~3.6h train).
#   bash scripts/repro_spider.sh
# Faster smoke (does NOT reproduce the headline numbers; ~1.8h, 200-row eval):
#   N=200 EPOCHS=1 bash scripts/repro_spider.sh
#   VLLM=/path/to/qwen-serve/bin/vllm bash scripts/repro_spider.sh   # point at the serve venv
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-models/Llama-3.1-8B-Instruct-NVFP4}
DATA_DIR=${DATA_DIR:-data/spider}
ADAPTER_DIR=${ADAPTER_DIR:-adapters/spider_llama8b_r32}
EPOCHS=${EPOCHS:-2}      # headline = 2 epochs; set EPOCHS=1 for a faster smoke
N=${N:-1034}            # headline = full 1034-row dev; set N=200 for a faster smoke
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

step "3/5 train (epochs=$EPOCHS, ~1.8h per epoch on one GB10)"
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
# Serve-the-right-tokenizer + text-only-VLM handling (validated on Mistral-Small-3.2):
#   * Tekken/mistral_common repacks ship an HF tokenizer that mis-tokenizes vs the
#     tekken one the model trained on, which also makes the NLL eval's echo/logprob
#     offsets unusable. If tekken.json is present we serve --tokenizer-mode mistral so
#     the served tokenizer MATCHES training (override/disable via TOKENIZER_MODE).
#   * VLM repacks (a vision tower, e.g. Pixtral on Mistral-Small-3.2) crash a text-only
#     serve in the image processor unless image inputs are disabled, so we add
#     --limit-mm-per-prompt '{"image":0}' when the config declares a vision component
#     (override via LIMIT_MM). Both are auto-detected from the model dir.
EXTRA_ARGS=()
TOKMODE="${TOKENIZER_MODE:-}"
if [ -z "$TOKMODE" ] && { [ -f "$MODEL_DIR/tekken.json" ] || [ -f "$MODEL_DIR/tekken.model" ]; }; then
  TOKMODE="mistral"
fi
if [ -n "$TOKMODE" ] && [ "$TOKMODE" != "none" ]; then
  EXTRA_ARGS+=(--tokenizer-mode "$TOKMODE")
  echo "serving with --tokenizer-mode $TOKMODE"
fi
LIMITMM="${LIMIT_MM:-}"
if [ -z "$LIMITMM" ] && grep -qiE '"vision_config"|"image_token|ForConditionalGeneration' "$MODEL_DIR/config.json" 2>/dev/null; then
  LIMITMM='{"image":0}'
fi
if [ -n "$LIMITMM" ] && [ "$LIMITMM" != "none" ]; then
  EXTRA_ARGS+=(--limit-mm-per-prompt "$LIMITMM")
  echo "text-only VLM serve: --limit-mm-per-prompt $LIMITMM"
fi
# DYNAMIC=1: serve the bare base and HOT-LOAD the adapter at runtime via
# POST /v1/load_lora_adapter (instead of the launch-time --lora-modules attach).
# Demonstrates runtime adapter loading; the eval is identical (hits model name "myft").
DYNAMIC="${DYNAMIC:-0}"
BASE_URL="http://127.0.0.1:$PORT"
if [ "$DYNAMIC" = "1" ]; then
  echo "DYNAMIC=1: serving base only; will hot-load '$ADAPTER_DIR/best' as 'myft' after READY"
  RUNTIME_ENV=(VLLM_ALLOW_RUNTIME_LORA_UPDATING=1)   # security: localhost-bound; off unless set here
  LORA_ATTACH=()                                      # no launch-time attach
else
  RUNTIME_ENV=()
  LORA_ATTACH=(--lora-modules myft="$ADAPTER_DIR/best")
fi
MAX_JOBS=1 env "${RUNTIME_ENV[@]}" "$VLLM" serve "$MODEL_DIR" \
  --served-model-name base --host 127.0.0.1 --port "$PORT" \
  --enable-lora --max-lora-rank 32 --max-loras 2 \
  "${LORA_ATTACH[@]}" \
  --max-model-len 8192 --enforce-eager "${EXTRA_ARGS[@]}" \
  --gpu-memory-utilization 0.6 --kv-cache-dtype fp8 > "$SERVE_LOG" 2>&1 &
SERVE_PID=$!
cleanup(){
  if [ "$DYNAMIC" = "1" ]; then
    curl -s -X POST "$BASE_URL/v1/unload_lora_adapter" -H 'Content-Type: application/json' \
      -d '{"lora_name":"myft"}' >/dev/null 2>&1 || true
  fi
  kill "$SERVE_PID" 2>/dev/null || true; pkill -9 EngineCor 2>/dev/null || true
}
trap cleanup EXIT
echo "waiting for serve READY (first run JITs flashinfer, a few minutes)..."
for _ in $(seq 1 180); do
  grep -q "Application startup complete" "$SERVE_LOG" && { echo "READY"; break; }
  kill -0 "$SERVE_PID" 2>/dev/null || { echo "serve exited early:"; tail -n 20 "$SERVE_LOG"; exit 1; }
  sleep 10
done
if [ "$DYNAMIC" = "1" ]; then
  echo "hot-loading adapter via POST /v1/load_lora_adapter ..."
  curl -s -X POST "$BASE_URL/v1/load_lora_adapter" -H 'Content-Type: application/json' \
    -d "{\"lora_name\":\"myft\",\"lora_path\":\"$ADAPTER_DIR/best\"}" ; echo
  # confirm it registered, else the eval would silently fall back to base behavior
  if "$PYTHON" - "$BASE_URL" <<'PYCHK'
import json, sys, urllib.request
url = sys.argv[1] + "/v1/models"
ids = [m["id"] for m in json.load(urllib.request.urlopen(url, timeout=30))["data"]]
print("served models:", ids)
sys.exit(0 if "myft" in ids else 1)
PYCHK
  then echo "myft hot-loaded OK"; else echo "ERROR: myft did not register after hot-load"; exit 1; fi
fi

step "5/5 eval before/after (n=$N, deterministic)"
"$PYTHON" scripts/eval_retention.py \
  --base-url "http://127.0.0.1:$PORT" \
  --dev-file "$DATA_DIR/spider.dev.chat.jsonl" \
  --models base myft --n "$N" --out "$OUT"
echo; echo "done. result written to $OUT"
