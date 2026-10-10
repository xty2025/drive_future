#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/workspace/project/DrivoR
PY=/mnt/workspace/envs/fastwam_xty/bin/python
RESULT_ROOT=/mnt/workspace/project/original_eval_results/drivor
OUT="$RESULT_ROOT/shared_final_head_full_navtest"
LOG="$RESULT_ROOT/logs/shared_final_head_full_navtest.log"
OFFICIAL_PKL="$RESULT_ROOT/navsim1_pdm_scores/official_nav1_step4/2026.09.12.20.29.05.pkl"

mkdir -p "$OUT" "$RESULT_ROOT/logs"
export PYTHONPATH="$ROOT:$ROOT/nuplan-devkit:${PYTHONPATH:-}"
export NAVSIM_EXP_ROOT="$RESULT_ROOT" SUBSCORE_PATH="$RESULT_ROOT"
export NUPLAN_MAPS_ROOT=/mnt/data/navsim/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export DRIVOR_DIAGNOSTIC_SHARED_HEAD=1 DRIVOR_RECORD_SHARED_WINNERS=1 DRIVOR_SCORE_SHARED_STAGES=1
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1

"$PY" -u "$ROOT/navsim/planning/script/run_pdm_score_multi_gpu.py" \
  train_test_split=navtest agent=drivoR \
  agent.checkpoint_path=/mnt/workspace/project/checkpoints/DrivoR/nav1_30epochs_with_134k_simscale_bis_103ktrainval.pth \
  navsim_log_path=/mnt/data/navsim/test_navsim_logs/test \
  sensor_blobs_path=/mnt/data/navsim/test_sensor_blobs/test \
  metric_cache_path=/mnt/workspace/project/FastWAM/data/metric_cache_navtest \
  agent.config.ref_num=4 agent.config.scorer_ref_num=4 agent.config.image_backbone.model_weights=/dev/null \
  agent.config.noc=1 agent.config.dac=1 agent.config.ddc=0 agent.config.ttc=5 agent.config.ep=5 agent.config.comfort=2 \
  max_number_of_workers=8 experiment_name=shared_final_head_full_navtest output_dir="$OUT" \
  dataloader.params.batch_size=32 dataloader.params.num_workers=4 dataloader.params.prefetch_factor=1 \
  >"$LOG" 2>&1

PKL=$(find "$RESULT_ROOT/navsim1_pdm_scores/shared_final_head_full_navtest" -type f -name '*.pkl' -size +0c | sort | tail -1)
VIZ="$RESULT_ROOT/visualizations/requested"
mkdir -p "$VIZ"
find "$VIZ" -type f -delete
"$PY" "$ROOT/scripts/evaluation/plot_requested_drivor_views.py" \
  --pkl "$PKL" --diversity-pkl "$OFFICIAL_PKL" --output "$VIZ" --scenes 16 \
  >>"$LOG" 2>&1
date -Is > "$OUT/COMPLETE"
