#!/usr/bin/env bash
set -euo pipefail

DRIVOR_QUERY_GT_VARIANT="${1:?Specify single_gt_long, single_gt_long_register or human_diverse8_l1}"
shift
case "${DRIVOR_QUERY_GT_VARIANT}" in
    single_gt_long|single_gt_long_register) DRIVOR_QUERY_LONG_POSES=2 ;;
    human_diverse8_l1) DRIVOR_QUERY_LONG_POSES=0 ;;
    *) printf 'Unknown query GT variant: %s\n' "${DRIVOR_QUERY_GT_VARIANT}" >&2; exit 2 ;;
esac
DRIVOR_QUERY_GT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_query_dinov3_vitl_navtrain_${DRIVOR_QUERY_GT_VARIANT}}"
exec bash "${DRIVOR_QUERY_GT_DIR}/run_drivor_training.sh" \
    "agent=drivoR_query_dinov3_vitl_${DRIVOR_QUERY_GT_VARIANT}" \
    "agent.config.long_trajectory_additional_poses=${DRIVOR_QUERY_LONG_POSES}" \
    "agent.config.ref_num=${DRIVOR_QUERY_LAYERS:-4}" "$@"
