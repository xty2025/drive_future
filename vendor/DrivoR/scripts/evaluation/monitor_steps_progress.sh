#!/usr/bin/env bash
set -u
OUT=/mnt/workspace/project/original_eval_results/drivor
REPORT="$OUT/periodic_reports.log"
while true; do
  {
    echo "===== $(date -Is) ====="
    ps -ef | grep 'run_pdm_score_multi_gpu.py' | grep -E 'experiment_name=official_nav1_step[123]_full' | grep -v grep | awk '{ step="?"; for (i=1; i<=NF; i++) if ($i ~ /^experiment_name=official_nav1_step[123]_full$/) { step=$i; sub(/^.*step/, "", step); sub(/_full$/, "", step) } print "RUNNING pid=" $2 " step=" step }' || true
    for s in 1 2 3 4; do
      result_dir="$OUT/official_nav1_step${s}_full"
      [[ $s == 4 ]] && result_dir="$OUT/official_nav1_step4"
      csv=$(find "$result_dir" -maxdepth 1 -type f -name '*.csv' 2>/dev/null | sort | tail -1)
      row=""
      [[ -n "$csv" ]] && row=$(tail -1 "$csv")
      if [[ "$row" == 12146,average,* ]]; then echo "step${s}: $row"; else echo "step${s}: pending"; fi
    done
    [[ -f "$OUT/STEPS_GRID_COMPLETE" ]] && echo GRID_COMPLETE || echo GRID_PENDING
  } >> "$REPORT"
  sleep 600
done
