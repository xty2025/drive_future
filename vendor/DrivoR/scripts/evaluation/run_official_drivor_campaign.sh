#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/workspace/project/DrivoR
PY=/mnt/workspace/envs/fastwam_xty/bin/python
CKPT=/mnt/workspace/project/checkpoints/DrivoR/nav1_30epochs_with_134k_simscale_bis_103ktrainval.pth
OUT=/mnt/workspace/project/original_eval_results/drivor
LOGS=/mnt/data/navsim/test_navsim_logs/test
SENSORS=/mnt/data/navsim/test_sensor_blobs/test
METRICS=/mnt/workspace/project/FastWAM/data/metric_cache_navtest

export PYTHONPATH="$ROOT:$ROOT/nuplan-devkit:${PYTHONPATH:-}"
export NAVSIM_EXP_ROOT="$OUT" SUBSCORE_PATH="$OUT"
export NUPLAN_MAPS_ROOT=/mnt/data/navsim/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 DRIVOR_RECORD_CANDIDATES=1
mkdir -p "$OUT/logs" "$OUT/visualizations"

common=(train_test_split=navtest agent=drivoR agent.checkpoint_path="$CKPT"
  navsim_log_path="$LOGS" sensor_blobs_path="$SENSORS" metric_cache_path="$METRICS"
  agent.config.ref_num=4 agent.config.scorer_ref_num=4
  agent.config.image_backbone.model_weights=/dev/null
  agent.config.noc=1 agent.config.dac=1 agent.config.ddc=0
  agent.config.ttc=5 agent.config.ep=5 agent.config.comfort=2
  max_number_of_workers=8)

run_one() {
  local step=$1 batches=$2 batch_size=$3
  local name="official_nav1_step${step}"
  local out="$OUT/$name"
  local extra=()
  [[ "$batches" == full ]] || extra+=("+trainer.params.limit_predict_batches=$batches")
  "$PY" -u "$ROOT/navsim/planning/script/run_pdm_score_multi_gpu.py" \
    "${common[@]}" experiment_name="$name" output_dir="$out" \
    +agent.config.inference_ref_num="$step" dataloader.params.batch_size="$batch_size" \
    dataloader.params.num_workers=4 dataloader.params.prefetch_factor=1 "${extra[@]}" \
    >"$OUT/logs/$name.log" 2>&1
  local pred
  pred=$(find "$OUT/navsim1_pdm_scores/$name" -type f -name '*.pkl' -size +0c | sort | tail -1)
  "$PY" "$ROOT/scripts/evaluation/analyze_drivor_candidates.py" "$pred" \
    --output "$OUT/visualizations/step${step}" --max-plots 12 \
    >"$OUT/logs/${name}_analysis.log" 2>&1
}

# Paper-matching full PDMS first; bounded step ablations follow on the same scene order.
run_one 4 full 32
run_one 2 8 8
run_one 3 8 8
run_one 1 8 8

date -Is > "$OUT/CAMPAIGN_COMPLETE"
