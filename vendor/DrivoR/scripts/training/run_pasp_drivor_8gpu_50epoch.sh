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
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$PWD/data_runtime}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/data/navsim}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/data/navsim/maps}"
export SUBSCORE_PATH="${SUBSCORE_PATH:-$PWD/data_runtime}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PASP_TEXT_LOG_EVERY="${PASP_TEXT_LOG_EVERY:-10}"
export DRIVOR_SHARED_RAY=1
export DRIVOR_RAY_CONNECT_TIMEOUT="${DRIVOR_RAY_CONNECT_TIMEOUT:-300}"

DEVICES="${PASP_DEVICES:-8}"
MAX_EPOCHS="${PASP_MAX_EPOCHS:-50}"
STRATEGY="${PASP_STRATEGY:-ddp}"

for required_path in \
  /mnt/data/navsim/trainval_navsim_logs/trainval \
  /mnt/data/navsim/trainval_all/trainval_sensor_blobs/trainval \
  "$NUPLAN_MAPS_ROOT" \
  "$NAVSIM_EXP_ROOT/train_metric_cache" \
  "$PWD/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors"; do
  [[ -e "$required_path" ]] || { echo "Missing required path: $required_path" >&2; exit 2; }
done

AVAILABLE_GPUS="$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')"
(( AVAILABLE_GPUS >= DEVICES )) || {
  echo "Requested $DEVICES GPUs, but only $AVAILABLE_GPUS are visible" >&2
  exit 2
}

"$PYTHON_BIN" navsim/planning/script/run_training.py \
  --config-path=config/training \
  --config-name=pasp_drivor \
  use_cache_without_dataset=false \
  cache_path=null \
  navsim_log_path=/mnt/data/navsim/trainval_navsim_logs/trainval \
  sensor_blobs_path=/mnt/data/navsim/trainval_all/trainval_sensor_blobs/trainval \
  trainer.params.max_epochs="$MAX_EPOCHS" \
  trainer.params.devices="$DEVICES" \
  trainer.params.strategy="$STRATEGY" \
  trainer.params.accumulate_grad_batches=1 \
  trainer.params.log_every_n_steps=1 \
  dataloader.params.batch_size=24 \
  agent.num_gpus="$DEVICES" \
  agent.checkpoint_path='' \
  train_ckpt_path='' \
  agent.config.trajectory_prior_weight=0.0 \
  agent.config.trajectory_prior_mode=constant_velocity \
  agent.config.scorer_worker=ray \
  agent.config.scorer_threads="${PASP_SCORER_THREADS:-8}" \
  output_dir="${PASP_OUTPUT_DIR:-$PWD/outputs/pasp_drivor_50ep}" \
  "$@"
