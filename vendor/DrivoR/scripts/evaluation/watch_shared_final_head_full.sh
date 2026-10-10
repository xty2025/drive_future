#!/usr/bin/env bash
set -u

ROOT=/mnt/workspace/project/DrivoR
OUT=/mnt/workspace/project/original_eval_results/drivor/shared_final_head_full_navtest
STATUS="$OUT/watchdog.log"
RETRIES=0

while true; do
  {
    echo "===== $(date -Is) ====="
    if pgrep -af 'experiment_name=shared_final_head_full_navtest' >/dev/null; then
      echo RUNNING
    else
      echo NOT_RUNNING
    fi
    for step in 1 2 3 4; do
      csv=$(find "$OUT/step$step" -maxdepth 1 -name '*.csv' 2>/dev/null | sort | tail -1)
      if [[ -n "$csv" ]]; then tail -1 "$csv" | sed "s/^/step$step: /"; else echo "step$step: pending"; fi
    done
    echo "visualizations: $(find /mnt/workspace/project/original_eval_results/drivor/visualizations/requested -name '*.png' 2>/dev/null | wc -l)"
  } >>"$STATUS"

  if [[ -f "$OUT/COMPLETE" ]]; then
    echo "COMPLETE $(date -Is)" >>"$STATUS"
    exit 0
  fi
  if ! pgrep -af 'experiment_name=shared_final_head_full_navtest' >/dev/null; then
    RETRIES=$((RETRIES + 1))
    if (( RETRIES > 3 )); then
      echo "FAILED_AFTER_RETRIES $(date -Is)" >>"$STATUS"
      exit 1
    fi
    tmux kill-session -t drivor_shared_head_full 2>/dev/null || true
    tmux new-session -d -s drivor_shared_head_full "cd $ROOT && ./scripts/evaluation/run_shared_final_head_full.sh"
    echo "RESTARTED attempt=$RETRIES $(date -Is)" >>"$STATUS"
  fi
  sleep 600
done
