#!/usr/bin/env bash
set -euo pipefail
OUT=/mnt/workspace/project/original_eval_results/drivor
PY=/mnt/workspace/envs/fastwam_xty/bin/python
while :; do
  ok=1
  for s in 1 2 3; do
    if ! find "$OUT/navsim1_pdm_scores/official_nav1_step${s}_full" -type f -name '*.pkl' -size +0c | grep -q .; then ok=0; fi
  done
  [[ $ok == 1 ]] && break
  sleep 60
done
files=()
for s in 1 2 3; do files+=("$(find "$OUT/navsim1_pdm_scores/official_nav1_step${s}_full" -type f -name '*.pkl' -size +0c | sort | tail -1)"); done
files+=("$(find "$OUT/navsim1_pdm_scores/official_nav1_step4" -type f -name '*.pkl' -size +0c | sort | tail -1)")
PYTHONPATH=/mnt/workspace/project/DrivoR:/mnt/workspace/project/DrivoR/nuplan-devkit "$PY" /mnt/workspace/project/DrivoR/scripts/evaluation/plot_steps_grid.py --pkl "${files[@]}" --output "$OUT/visualizations/steps_grid_16scenes" --scenes 16
date -Is > "$OUT/STEPS_GRID_COMPLETE"
