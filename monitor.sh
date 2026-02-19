#!/usr/bin/env bash
# Periodic monitoring for GRPO training
# Usage: bash monitor.sh [interval_seconds]
# Output goes to monitor.log

INTERVAL=${1:-60}
LOG="/home/rlu/Code/rft/monitor.log"
RAY_LOG_DIR="/var/tmp/ray/session_latest/logs"

echo "=== Monitor started at $(date) ===" | tee -a "$LOG"
echo "Interval: ${INTERVAL}s" | tee -a "$LOG"

while true; do
    echo "" >> "$LOG"
    echo "========== $(date '+%Y-%m-%d %H:%M:%S') ==========" >> "$LOG"

    # 1. GPU status
    echo "--- GPU Status ---" >> "$LOG"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader >> "$LOG" 2>&1

    # 2. Training process check
    echo "--- Training Process ---" >> "$LOG"
    TRAIN_PID=$(pgrep -f "train_ppo_ray" | head -1)
    if [ -z "$TRAIN_PID" ]; then
        echo "WARNING: Training process NOT running!" >> "$LOG"
    else
        ps -p "$TRAIN_PID" -o pid,state,pcpu,pmem,etime --no-headers >> "$LOG" 2>&1
    fi

    # 3. Actor/Ref model status
    echo "--- Actor/Ref Processes ---" >> "$LOG"
    ps aux 2>/dev/null | grep -E "PolicyModel|ReferenceModel" | grep -v grep | awk '{printf "PID=%s CPU=%s MEM=%s CMD=%s\n", $2, $3, $4, $11}' >> "$LOG"

    # 4. Check for errors in Ray worker logs (last INTERVAL seconds)
    echo "--- Recent Errors ---" >> "$LOG"
    ERRORS=$(find "$RAY_LOG_DIR" -name "worker-*.err" -newer /tmp/.monitor_last_check 2>/dev/null | xargs grep -l "Error\|Exception\|Traceback\|FAILED\|OOM\|killed" 2>/dev/null | grep -v "__pycache__")
    if [ -n "$ERRORS" ]; then
        for f in $ERRORS; do
            echo "Errors in: $(basename $f)" >> "$LOG"
            grep -A2 "Error\|Exception\|Traceback\|FAILED\|OOM\|killed" "$f" 2>/dev/null | grep -v "FutureWarning\|deprecated\|CompiledFxGraph\|UnpicklingError\|torch._inductor\|pickle\|codecache\|output_code" | tail -10 >> "$LOG"
        done
    else
        echo "No new errors found." >> "$LOG"
    fi

    # 5. Check for 429 / rate limit errors
    echo "--- API Rate Limits ---" >> "$LOG"
    RATE_ERRORS=$(grep -r "429\|RESOURCE_EXHAUSTED\|rate.limit" "$RAY_LOG_DIR"/worker-*.err "$RAY_LOG_DIR"/worker-*.out 2>/dev/null | grep -v "__pycache__" | tail -5)
    if [ -n "$RATE_ERRORS" ]; then
        echo "ALERT: Rate limit errors detected:" >> "$LOG"
        echo "$RATE_ERRORS" >> "$LOG"
    else
        echo "No rate limit errors." >> "$LOG"
    fi

    # 6. Training log progress (last few lines of train.log)
    echo "--- Train Log (last 5 lines) ---" >> "$LOG"
    tail -5 /home/rlu/Code/rft/train.log 2>/dev/null >> "$LOG"

    # 7. System memory
    echo "--- System Memory ---" >> "$LOG"
    free -h | grep Mem >> "$LOG"

    # 8. Check if py-spy can detect what phase we're in
    echo "--- Current Phase ---" >> "$LOG"
    if [ -n "$TRAIN_PID" ]; then
        ACTOR_PID=$(pgrep -f "PolicyModelActor" | head -1)
        if [ -n "$ACTOR_PID" ]; then
            PHASE=$(py-spy dump --pid "$ACTOR_PID" 2>/dev/null | head -3 | tail -1)
            echo "Actor phase: $PHASE" >> "$LOG"
        fi
    fi

    # Touch marker file for incremental error checking
    touch /tmp/.monitor_last_check

    # Print summary to stdout
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | paste -sd, -)
    echo "[$(date '+%H:%M:%S')] GPU util: $GPU_UTIL | Train PID: ${TRAIN_PID:-DEAD}"

    sleep "$INTERVAL"
done
