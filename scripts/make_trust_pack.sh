#!/usr/bin/env bash
# make_trust_pack.sh -- assemble a credibility "trust pack" for a (base, adapter) pair.
#
# A trust pack is ONE self-contained directory a reviewer can open to answer
# "is this fine-tune real, and is the number honest?" without re-running anything:
#
#   trust_pack/<name>/
#     manifest.json        provenance: base+adapter fingerprints, pkg versions, git sha
#                          (written via nybbloris.manifest.build_manifest -- CPU only)
#     contamination.json   train<->eval overlap report (scripts/check_contamination.py)
#     apply_check.json     runtime-LoRA APPLY proof: prompt-echo logprob delta,
#                          base vs adapter (scripts/serve_apply_check.py)  [GPU/serve]
#     retention.json       before/after retention w/ bootstrap CIs
#                          (scripts/eval_retention.py)                     [GPU/serve]
#     SUMMARY.md           human-readable index of the above, auto-generated
#
# This is an ORCHESTRATOR SCAFFOLD. The CPU-only pieces (manifest, contamination,
# directory + SUMMARY) always run. The GPU/serve-dependent pieces (apply_check,
# retention) are OPTIONAL and are SKIPPED with a clear message unless you pass a
# served base URL + model names, because this box is frequently busy holding the GPU
# for a training run. It does not need to run end-to-end to be useful: run it CPU-only
# now to lay out the pack + provenance + contamination, then re-run later with the
# serve flags to fill in the GPU proofs (existing files are not clobbered).
#
# Usage:
#   scripts/make_trust_pack.sh \
#       --name spider_llama8b_r32 \
#       --base-model  models/Llama-3.1-8B-Instruct-NVFP4 \
#       --adapter-dir adapters/spider_llama8b_r32 \
#       --train data/spider/spider.train.chat.jsonl \
#       --eval  data/spider/spider.dev.chat.jsonl \
#       [--base-url http://localhost:8000] \
#       [--base-name <served-base-name>] [--adapter-name <served-adapter-name>] \
#       [--eval-n 200] [--out-root trust_pack] [--python /path/to/venv/python]
#
# CPU-only (no serve): omit --base-url/--base-name/--adapter-name; the GPU steps skip.
set -euo pipefail

# --- defaults --------------------------------------------------------------
NAME=""
BASE_MODEL=""
ADAPTER_DIR=""
TRAIN=""
EVAL=""
BASE_URL=""
BASE_NAME=""
ADAPTER_NAME=""
EVAL_N=200
OUT_ROOT="trust_pack"
PYTHON="${PYTHON:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}"; exit "${1:-0}"; }

# --- arg parse -------------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --name)         NAME="$2"; shift 2 ;;
    --base-model)   BASE_MODEL="$2"; shift 2 ;;
    --adapter-dir)  ADAPTER_DIR="$2"; shift 2 ;;
    --train)        TRAIN="$2"; shift 2 ;;
    --eval)         EVAL="$2"; shift 2 ;;
    --base-url)     BASE_URL="$2"; shift 2 ;;
    --base-name)    BASE_NAME="$2"; shift 2 ;;
    --adapter-name) ADAPTER_NAME="$2"; shift 2 ;;
    --eval-n)       EVAL_N="$2"; shift 2 ;;
    --out-root)     OUT_ROOT="$2"; shift 2 ;;
    --python)       PYTHON="$2"; shift 2 ;;
    -h|--help)      usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 1 ;;
  esac
done

# --- validate required (CPU) args ------------------------------------------
missing=""
[ -n "$NAME" ]        || missing="$missing --name"
[ -n "$BASE_MODEL" ]  || missing="$missing --base-model"
[ -n "$ADAPTER_DIR" ] || missing="$missing --adapter-dir"
[ -n "$TRAIN" ]       || missing="$missing --train"
[ -n "$EVAL" ]        || missing="$missing --eval"
if [ -n "$missing" ]; then
  echo "ERROR: missing required args:$missing" >&2
  usage 1
fi

PACK="$OUT_ROOT/$NAME"
mkdir -p "$PACK"
echo "=== trust pack: $PACK ==="

step() { echo; echo "--- $* ---"; }

# --- 1. provenance manifest (CPU only) -------------------------------------
step "1/4 provenance manifest (nybbloris.manifest)"
if [ -f "$PACK/manifest.json" ]; then
  echo "present, skipping: $PACK/manifest.json"
else
  "$PYTHON" - "$REPO_ROOT" "$BASE_MODEL" "$ADAPTER_DIR" "$PACK/manifest.json" <<'PY'
import json, sys
repo_root, base_model, adapter_dir, out = sys.argv[1:5]
sys.path.insert(0, repo_root)
from nybbloris.manifest import build_manifest
m = build_manifest(base_model, adapter_dir, repo_hint=repo_root)
with open(out, "w") as f:
    json.dump(m, f, indent=2)
print(f"[manifest] wrote {out}")
PY
fi

# --- 2. contamination report (CPU only) ------------------------------------
step "2/4 train<->eval contamination (scripts/check_contamination.py)"
if [ -f "$PACK/contamination.json" ]; then
  echo "present, skipping: $PACK/contamination.json"
else
  "$PYTHON" "$SCRIPT_DIR/check_contamination.py" \
    --train "$TRAIN" --eval "$EVAL" --out "$PACK/contamination.json"
fi

# --- 3. runtime-apply proof (GPU/serve -- optional) ------------------------
step "3/4 runtime-LoRA apply proof (scripts/serve_apply_check.py)"
APPLY_STATUS="SKIPPED (no serve)"
if [ -n "$BASE_URL" ] && [ -n "$BASE_NAME" ] && [ -n "$ADAPTER_NAME" ]; then
  if "$PYTHON" "$SCRIPT_DIR/serve_apply_check.py" \
        --base-url "$BASE_URL" --base-model "$BASE_NAME" \
        --adapter-model "$ADAPTER_NAME" --out "$PACK/apply_check.json"; then
    APPLY_STATUS="APPLIES"
  else
    # serve_apply_check exits 1 on NO-OP, 2 on error; either way record + continue.
    rc=$?
    APPLY_STATUS="RAN (exit $rc -- see apply_check.json; 1=NO-OP, 2=error)"
  fi
else
  echo "SKIP: need --base-url, --base-name, --adapter-name (GPU serve not provided)."
fi

# --- 4. before/after retention (GPU/serve -- optional) ---------------------
step "4/4 before/after retention w/ CIs (scripts/eval_retention.py)"
RETENTION_STATUS="SKIPPED (no serve)"
if [ -n "$BASE_URL" ] && [ -n "$BASE_NAME" ] && [ -n "$ADAPTER_NAME" ]; then
  "$PYTHON" "$SCRIPT_DIR/eval_retention.py" \
    --base-url "$BASE_URL" --dev-file "$EVAL" \
    --models "$BASE_NAME" "$ADAPTER_NAME" --n "$EVAL_N" \
    --out "$PACK/retention.json" \
    && RETENTION_STATUS="RAN (see retention.json)" \
    || RETENTION_STATUS="FAILED (see console)"
else
  echo "SKIP: need --base-url, --base-name, --adapter-name (GPU serve not provided)."
fi

# --- SUMMARY.md ------------------------------------------------------------
step "SUMMARY.md"
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
  echo "# Trust pack: \`$NAME\`"
  echo
  echo "_Generated ${NOW} (repo ${GIT_SHA})._"
  echo
  echo "Answers, without re-running anything: is this fine-tune real, and is the number honest?"
  echo
  echo "| artifact | file | what it proves | status |"
  echo "|---|---|---|---|"
  echo "| provenance manifest | \`manifest.json\` | base+adapter fingerprints, pkg versions, git sha | $([ -f "$PACK/manifest.json" ] && echo present || echo MISSING) |"
  echo "| contamination | \`contamination.json\` | train<->eval overlap (exact + 8/13-gram + db_id) | $([ -f "$PACK/contamination.json" ] && echo present || echo MISSING) |"
  echo "| runtime-apply proof | \`apply_check.json\` | adapter actually moves the forward pass (not a silent no-op) | ${APPLY_STATUS} |"
  echo "| before/after retention | \`retention.json\` | quality delta vs base w/ bootstrap CIs | ${RETENTION_STATUS} |"
  echo
  echo "## Inputs"
  echo
  echo "- base model: \`$BASE_MODEL\`"
  echo "- adapter:    \`$ADAPTER_DIR\`"
  echo "- train:      \`$TRAIN\`"
  echo "- eval:       \`$EVAL\`"
  if [ -n "$BASE_URL" ]; then
    echo "- serve:      \`$BASE_URL\` (base=\`$BASE_NAME\`, adapter=\`$ADAPTER_NAME\`, eval-n=$EVAL_N)"
  else
    echo "- serve:      not provided -- GPU steps skipped (re-run with --base-url/--base-name/--adapter-name to fill in)"
  fi
  echo
  echo "## How to read it"
  echo
  echo "1. \`manifest.json\` pins WHAT was trained (base fingerprint the adapter is bound to)."
  echo "2. \`contamination.json\` shows the eval was not (heavily) trained on -- see \`warnings\`."
  echo "3. \`apply_check.json\` verdict must be APPLIES (max per-token |logprob delta| > threshold);"
  echo "   NO-OP means the adapter loaded but did nothing (the silent-no-op class this project guards)."
  echo "4. \`retention.json\` \`summary\` carries the before/after deltas with bootstrap 95% CIs."
  echo
  echo "GPU steps not run here re-run cleanly: existing JSON files are not overwritten."
} > "$PACK/SUMMARY.md"

echo
echo "=== done. trust pack at: $PACK ==="
ls -1 "$PACK"
