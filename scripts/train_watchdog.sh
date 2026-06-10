#!/usr/bin/env bash
# Crash watchdog + auto-resume for NVFP4 LoRA training runs on GB10.
#
# Watches a trainer process and its log. On crash (process gone without a
# "done" event) or stall (no log writes for STALL_SECS while alive), it:
#   1. stops the auto-starting Qwen3.6 serve unit if it grabbed memory
#   2. waits for CUDA memory to recover (dropping shard page cache each poll)
#   3. relaunches the trainer with --resume-from the latest checkpoint that
#      has train_state.pt (or fresh if none exists yet)
# Exits cleanly (and disables its own systemd unit) when the run completes,
# or after MAX_RELAUNCHES failed recoveries.
#
# Kill switch: touch "$OUTPUT_DIR/.watchdog_disabled" to make it stand down
# without stopping the trainer (use this before intentional restarts).
#
# Required environment (set in the systemd unit):
#   TRAINER_CMD   full trainer command WITHOUT --resume-from (it is appended)
#   TRAINER_GREP  pgrep -f pattern uniquely matching the trainer process
#   OUTPUT_DIR    trainer --output-dir (checkpoints + metrics.jsonl live here)
#   RUN_LOG       trainer stdout/stderr log path (relaunch appends to it)
#   MODEL_DIR     model dir whose *.safetensors get page-cache-dropped
#   VENV_PY       python interpreter with torch for memory probing
# Optional:
#   STALL_SECS (default 1500), CHECK_EVERY (60), MAX_RELAUNCHES (3),
#   MIN_FREE_GB (100), MEM_WAIT_SECS (1800), UNIT_NAME (for self-disable),
#   SERVE_UNIT (qwen36-ich-v35-serve.service), WATCHDOG_ONESHOT (health check only)

set -u

STALL_SECS="${STALL_SECS:-1500}"
CHECK_EVERY="${CHECK_EVERY:-60}"
MAX_RELAUNCHES="${MAX_RELAUNCHES:-3}"
MIN_FREE_GB="${MIN_FREE_GB:-100}"
MEM_WAIT_SECS="${MEM_WAIT_SECS:-1800}"
SERVE_UNIT="${SERVE_UNIT:-qwen36-ich-v35-serve.service}"
UNIT_NAME="${UNIT_NAME:-}"
WDLOG="$OUTPUT_DIR/watchdog.log"
COUNT_FILE="$OUTPUT_DIR/.watchdog_relaunch_count"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$WDLOG"; }

run_done() {
    grep -q '"event": "done"' "$OUTPUT_DIR/metrics.jsonl" 2>/dev/null
}

trainer_pid() {
    pgrep -f "$TRAINER_GREP" | head -1
}

log_age_secs() {
    local mt
    mt=$(stat -c %Y "$RUN_LOG" 2>/dev/null) || { echo 999999; return; }
    echo $(( $(date +%s) - mt ))
}

cuda_free_gb() {
    "$VENV_PY" - <<'EOF' 2>/dev/null || echo 0
import torch
print(int(torch.cuda.mem_get_info()[0] / 1e9))
EOF
}

drop_model_page_cache() {
    "$VENV_PY" - "$MODEL_DIR" <<'EOF' 2>/dev/null || true
import os, sys, glob
for shard in glob.glob(os.path.join(sys.argv[1], "*.safetensors")):
    fd = os.open(shard, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
EOF
}

latest_checkpoint() {
    local best="" n=0 cand
    for cand in "$OUTPUT_DIR"/checkpoint_step_*/; do
        [ -d "$cand" ] || continue
        [ -f "$cand/train_state.pt" ] || continue
        local step="${cand%/}"; step="${step##*_}"
        if [ "$step" -gt "$n" ]; then n="$step"; best="${cand%/}"; fi
    done
    echo "$best"
}

self_disable_and_exit() {
    if [ -n "$UNIT_NAME" ]; then
        systemctl --user disable "$UNIT_NAME" 2>/dev/null || true
    fi
    exit 0
}

recover() {
    local count=0
    [ -f "$COUNT_FILE" ] && count=$(cat "$COUNT_FILE")
    count=$((count + 1))
    echo "$count" > "$COUNT_FILE"
    if [ "$count" -gt "$MAX_RELAUNCHES" ]; then
        log "FATAL: relaunch budget exhausted ($MAX_RELAUNCHES); standing down. Inspect $RUN_LOG and resume manually."
        self_disable_and_exit
    fi
    log "recovery $count/$MAX_RELAUNCHES starting"

    if systemctl --user is-active --quiet "$SERVE_UNIT" 2>/dev/null; then
        log "stopping $SERVE_UNIT (memory competitor)"
        systemctl --user stop "$SERVE_UNIT" || true
    fi

    local waited=0 free
    while :; do
        drop_model_page_cache
        free=$(cuda_free_gb)
        log "memory wait: cuda_free=${free}GB (need ${MIN_FREE_GB})"
        [ "$free" -ge "$MIN_FREE_GB" ] && break
        waited=$((waited + 30))
        if [ "$waited" -ge "$MEM_WAIT_SECS" ]; then
            log "FATAL: memory did not recover within ${MEM_WAIT_SECS}s (NVRM may need a reboot); standing down."
            self_disable_and_exit
        fi
        sleep 30
    done

    local ckpt resume_arg=""
    ckpt=$(latest_checkpoint)
    if [ -n "$ckpt" ]; then
        resume_arg="--resume-from $ckpt"
        log "relaunching with $resume_arg"
    else
        log "no checkpoint with train_state.pt found; relaunching fresh"
    fi

    # shellcheck disable=SC2086
    nohup bash -c "$TRAINER_CMD $resume_arg" >> "$RUN_LOG" 2>&1 &
    log "relaunched (shell pid $!)"
    sleep 120
}

log "watchdog armed: grep='$TRAINER_GREP' output='$OUTPUT_DIR' stall=${STALL_SECS}s max_relaunch=$MAX_RELAUNCHES"

if [ "${WATCHDOG_ONESHOT:-0}" = "1" ]; then
    pid=$(trainer_pid)
    log "oneshot: pid=${pid:-none} log_age=$(log_age_secs)s done=$(run_done && echo yes || echo no) cuda_free=$(cuda_free_gb)GB latest_ckpt=$(latest_checkpoint)"
    exit 0
fi

while :; do
    if [ -f "$OUTPUT_DIR/.watchdog_disabled" ]; then
        sleep "$CHECK_EVERY"
        continue
    fi
    if run_done; then
        log "run completed cleanly; watchdog standing down"
        self_disable_and_exit
    fi
    pid=$(trainer_pid)
    if [ -z "$pid" ]; then
        log "trainer process not found and run not done: treating as crash"
        recover
    else
        age=$(log_age_secs)
        if [ "$age" -gt "$STALL_SECS" ]; then
            log "trainer pid $pid alive but log silent for ${age}s (> $STALL_SECS): killing as hung"
            kill -9 "$pid" 2>/dev/null || true
            sleep 10
            recover
        fi
    fi
    sleep "$CHECK_EVERY"
done
