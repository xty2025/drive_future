#!/usr/bin/env bash
set -euo pipefail
DRIVOR_HALF_VARIANT="${1:?Specify a23, a27, a28, or an a28_* ablation}"
shift
DRIVOR_HALF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${DRIVOR_HALF_DIR}/drivor_half_ablation_common.sh"
export DRIVOR_TRAINING_ENTRY=run_training.py
export DRIVOR_TRAINING_CONFIG="half_${DRIVOR_HALF_VARIANT}"
export DRIVOR_EXPERIMENT="${DRIVOR_HALF_TRAIN_EXP}"

DRIVOR_HALF_ARGS=()
for DRIVOR_HALF_ARG in "$@"; do
    DRIVOR_HALF_KEY="${DRIVOR_HALF_ARG%%=*}"
    DRIVOR_HALF_KEY="${DRIVOR_HALF_KEY#+}"
    DRIVOR_HALF_KEY="${DRIVOR_HALF_KEY#+}"
    case "${DRIVOR_HALF_KEY}" in
        agent|train_test_split|cache_path|use_cache_without_dataset|runtime.scheduler_dataset_size_from_data|agent.scheduler_args.dataset_size)
            continue ;;
    esac
    DRIVOR_HALF_ARGS+=("${DRIVOR_HALF_ARG}")
done
exec bash "${DRIVOR_HALF_DIR}/run_drivor_training.sh" \
    "agent=${DRIVOR_HALF_AGENT}" \
    train_test_split=navtrain_half cache_path=null use_cache_without_dataset=false \
    ++runtime.scheduler_dataset_size_from_data=true agent.scheduler_args.dataset_size=42555 \
    agent.config.long_trajectory_additional_poses=0 agent.config.multi_gt.enabled=false \
    agent.config.num_scene_tokens=0 agent.config.visual_feature_mode=native_patches \
    "trainer.params.strategy=${DRIVOR_HALF_STRATEGY}" \
    "trainer.params.precision=${DRIVOR_HALF_PRECISION}" \
    "${DRIVOR_HALF_ARGS[@]}" --config-name="half_${DRIVOR_HALF_VARIANT}"
