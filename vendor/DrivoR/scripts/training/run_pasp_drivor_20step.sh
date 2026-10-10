#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN="${DRIOV_DPO_PYTHON:-/mnt/zhaozc_workspace/envs/driov_dpo/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing driov_dpo Python: $PYTHON_BIN" >&2; exit 2; }
# This environment is relocatable.  PYTHONHOME would pin the build-time
# prefix and can break stdlib extension loading (notably pyexpat).
unset PYTHONHOME
export PYTHONPATH="$PWD:$PWD/nuplan-devkit:$PWD/nuplan-devkit/nuplan:${PYTHONPATH:-}"
export PASP_NEGATIVE_BANK="${PASP_NEGATIVE_BANK:-}"
export PASP_SMOKE_DIR="${PASP_SMOKE_DIR:-$PWD/outputs/pasp_drivor_20step/diagnostics}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$PWD/data_runtime}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/data/navsim}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/data/navsim/maps}"
export SUBSCORE_PATH="${SUBSCORE_PATH:-$PWD/data_runtime}"

for required_path in \
  /mnt/data/navsim/trainval_navsim_logs/trainval \
  /mnt/data/navsim/trainval_all/trainval_sensor_blobs/trainval \
  "$NUPLAN_MAPS_ROOT" \
  "$NAVSIM_EXP_ROOT/train_metric_cache" \
  "$PWD/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors"; do
  [[ -e "$required_path" ]] || { echo "Missing required path: $required_path" >&2; exit 2; }
done
"$PYTHON_BIN" navsim/planning/script/run_training.py \
  --config-path=config/training \
  --config-name=pasp_drivor \
  use_cache_without_dataset=false \
  cache_path=null \
  navsim_log_path=/mnt/data/navsim/trainval_navsim_logs/trainval \
  sensor_blobs_path=/mnt/data/navsim/trainval_all/trainval_sensor_blobs/trainval \
  train_test_split/scene_filter=pasp_smoke20 \
  train_logs='[2021.08.17.17.17.01_veh-45_00207_00594]' \
  val_logs='[2021.08.17.17.17.01_veh-45_00207_00594]' \
  trainer.params.fast_dev_run=false \
  trainer.params.limit_train_batches=20 \
  trainer.params.limit_val_batches=0 \
  trainer.params.max_epochs=1 \
  trainer.params.devices=1 \
  agent.num_gpus=1 \
  dataloader.params.batch_size=1 \
  dataloader.params.num_workers=0 \
  dataloader.params.prefetch_factor=null \
  trainer.params.strategy=auto \
  trainer.params.precision=32-true \
  agent.checkpoint_path='' \
  train_ckpt_path='' \
  agent.config.trajectory_prior_weight=0.0 \
  agent.config.trajectory_prior_mode=constant_velocity \
  output_dir="${PASP_OUTPUT_DIR:-$PWD/outputs/pasp_drivor_20step}" \
  "$@"
