#!/usr/bin/env bash
# Offline PDMS evaluation of a PASP-DrivoR checkpoint on the full navtest split.
#
# Train/test consistency contract:
#   - same agent config as training (navsim/planning/script/config/common/agent/drivoR.yaml):
#     trajectory_prior_mode=constant_velocity, trajectory_prior_weight=0.0,
#     policy_selection_weight=0.0 --> learned-scorer argmax over the 64 proposals,
#     i.e. the official DrivoR inference rule used by run_training.py validation.
#   - same coordinate contract (8 ego-local rear-axle poses @ 0.5 s).
#   - GT side comes from a navsim MetricCache built for train_test_split=navtest.
#
# Usage:
#   CKPT=<checkpoint.ckpt> OUTPUT_DIR=<dir> [THREADS=n] [METRIC_CACHE_PATH=<dir>] \
#     bash scripts/evaluation/run_pasp_navtest_pdms.sh [extra hydra overrides]
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON_BIN="${DRIVOR_EVAL_PYTHON:-/mnt/zhaozc_workspace/envs/driov_dpo/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 2; }
# Relocatable env: PYTHONHOME would pin the build-time prefix.
unset PYTHONHOME
export PYTHONPATH="$PWD:$PWD/nuplan-devkit:$PWD/nuplan-devkit/nuplan:${PYTHONPATH:-}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$PWD/data_runtime}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/data/navsim}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/data/navsim/maps}"
# Ray spawns one worker process per thread slot; keep total threads ~core count.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

CKPT="${CKPT:-$PWD/outputs/pasp_drivor_dpo_8gpu_50ep_fixed/lightning_logs/version_1/checkpoints/last.ckpt}"
[[ -e "$CKPT" ]] || { echo "Missing checkpoint: $CKPT" >&2; exit 2; }
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/pasp_navtest_pdms}"
THREADS="${THREADS:-4}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/mnt/zhaozc_workspace/project/FastWAM/data/metric_cache_navtest}"

for required_path in \
  /mnt/data/navsim/test_navsim_logs/test \
  /mnt/data/navsim/test_sensor_blobs/test \
  "$NUPLAN_MAPS_ROOT" \
  "$METRIC_CACHE_PATH" \
  "$PWD/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors"; do
  [[ -e "$required_path" ]] || { echo "Missing required path: $required_path" >&2; exit 2; }
done
mkdir -p "$OUTPUT_DIR"

# The pdm_scoring config tree has no trainer/dataloader groups, so the two
# interpolations drivoR.yaml borrows from the training tree
# (scheduler_args.num_epochs, batch_size) are resolved here explicitly. Both are
# inert at inference time (scheduler / per-gpu batch bookkeeping only).
"$PYTHON_BIN" navsim/planning/script/run_pdm_score.py \
  train_test_split=navtest \
  agent.checkpoint_path="$CKPT" \
  agent.scheduler_args.num_epochs=1 \
  agent.batch_size=1 \
  metric_cache_path="$METRIC_CACHE_PATH" \
  navsim_log_path=/mnt/data/navsim/test_navsim_logs/test \
  sensor_blobs_path=/mnt/data/navsim/test_sensor_blobs/test \
  output_dir="$OUTPUT_DIR" \
  worker.threads_per_node="$THREADS" \
  "$@"
