#!/usr/bin/env bash
set -euo pipefail

DRIVOR_QUERY_VARIANT="${1:?Specify single_gt, multigt, scorer_single_gt or scorer_multigt}"
shift
case "${DRIVOR_QUERY_VARIANT}" in
    single_gt|multigt|scorer_single_gt|scorer_multigt) ;;
    *) printf 'Unknown query-register variant: %s\n' "${DRIVOR_QUERY_VARIANT}" >&2; exit 2 ;;
esac
DRIVOR_QUERY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_query_register_dinov3_vitl_navtrain_${DRIVOR_QUERY_VARIANT}}"
exec bash "${DRIVOR_QUERY_DIR}/run_drivor_training.sh" \
    "agent=drivoR_query_register_dinov3_vitl_${DRIVOR_QUERY_VARIANT}" \
    agent.config.long_trajectory_additional_poses=0 \
    "agent.config.ref_num=${DRIVOR_QUERY_LAYERS:-4}" \
    "$@"
