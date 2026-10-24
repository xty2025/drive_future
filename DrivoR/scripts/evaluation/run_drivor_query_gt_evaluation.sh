#!/usr/bin/env bash
set -euo pipefail

DRIVOR_QUERY_GT_VARIANT="${1:?Specify single_gt, single_gt_long, single_gt_long_register or human_diverse8_l1}"
DRIVOR_QUERY_GT_CHECKPOINT="${2:?Specify a complete checkpoint}"
shift 2
case "${DRIVOR_QUERY_GT_VARIANT}" in
    single_gt|single_gt_long|single_gt_long_register|human_diverse8_l1) ;;
    *) printf 'Unknown query GT variant: %s\n' "${DRIVOR_QUERY_GT_VARIANT}" >&2; exit 2 ;;
esac
DRIVOR_QUERY_GT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-eval_navtest_query_dinov3_vitl_${DRIVOR_QUERY_GT_VARIANT}}"
export DRIVOR_EVAL_WORKER="${DRIVOR_EVAL_WORKER:-single_machine_thread_pool}"
export DRIVOR_SCORE_WORKERS="${DRIVOR_SCORE_WORKERS:-32}"
exec bash "${DRIVOR_QUERY_GT_DIR}/run_drivor_pdm_score_evaluation.sh" "${DRIVOR_QUERY_GT_CHECKPOINT}" \
    "agent=drivoR_query_dinov3_vitl_${DRIVOR_QUERY_GT_VARIANT}" \
    agent.config.multi_gt.enabled=false \
    agent.config.long_trajectory_additional_poses=0 \
    "agent.config.ref_num=${DRIVOR_QUERY_LAYERS:-4}" \
    trainer.params.precision=32-true dataloader.params.batch_size=1 "$@"
