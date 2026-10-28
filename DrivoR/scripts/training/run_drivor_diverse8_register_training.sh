#!/usr/bin/env bash
set -euo pipefail
DRIVOR_DIVERSE_VARIANT="${1:?Specify the diverse8 register variant}"
shift
DRIVOR_DIVERSE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${DRIVOR_DIVERSE_DIR}/drivor_diverse8_register_common.sh"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-${DRIVOR_DIVERSE_DEFAULT_TRAIN}}"
exec bash "${DRIVOR_DIVERSE_DIR}/run_drivor_training.sh" \
    "agent=${DRIVOR_DIVERSE_AGENT}" \
    agent.config.long_trajectory_additional_poses=0 \
    agent.config.multi_gt.enabled=false \
    agent.config.num_scene_tokens=0 \
    agent.config.visual_feature_mode=native_patches \
    "agent.config.ref_num=${DRIVOR_QUERY_LAYERS:-4}" \
    "$@"
