#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/workspace/project/DrivoR
PY=/mnt/workspace/envs/fastwam_xty/bin/python
CKPT=/mnt/workspace/project/checkpoints/DrivoR/nav1_30epochs_with_134k_simscale_bis_103ktrainval.pth
OUT=/mnt/workspace/project/original_eval_results/drivor
export PYTHONPATH="$ROOT:$ROOT/nuplan-devkit:${PYTHONPATH:-}"
export NAVSIM_EXP_ROOT="$OUT" SUBSCORE_PATH="$OUT" NUPLAN_MAPS_ROOT=/mnt/data/navsim/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 DRIVOR_RECORD_CANDIDATES=1
common=(train_test_split=navtest agent=drivoR agent.checkpoint_path="$CKPT" navsim_log_path=/mnt/data/navsim/test_navsim_logs/test sensor_blobs_path=/mnt/data/navsim/test_sensor_blobs/test metric_cache_path=/mnt/workspace/project/FastWAM/data/metric_cache_navtest agent.config.ref_num=4 agent.config.scorer_ref_num=4 agent.config.image_backbone.model_weights=/dev/null agent.config.noc=1 agent.config.dac=1 agent.config.ddc=0 agent.config.ttc=5 agent.config.ep=5 agent.config.comfort=2 max_number_of_workers=8)
for step in 1 2 3; do
  name="official_nav1_step${step}_full"; out="$OUT/$name"
  "$PY" -u "$ROOT/navsim/planning/script/run_pdm_score_multi_gpu.py" "${common[@]}" experiment_name="$name" output_dir="$out" +agent.config.inference_ref_num="$step" dataloader.params.batch_size=32 dataloader.params.num_workers=4 dataloader.params.prefetch_factor=1 >"$OUT/logs/$name.log" 2>&1
  pred=$(find "$OUT/navsim1_pdm_scores/$name" -type f -name '*.pkl' -size +0c | sort | tail -1)
  "$PY" "$ROOT/scripts/evaluation/analyze_drivor_candidates.py" "$pred" --output "$OUT/visualizations/${name}" --max-plots 12 >"$OUT/logs/${name}_analysis.log" 2>&1
done
date -Is > "$OUT/FULL_STEPS123_COMPLETE"
