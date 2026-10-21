#!/usr/bin/env bash
set -euo pipefail

DRIVOR_QUERY_CHECKPOINT="${1:?Specify a complete query-register checkpoint}"
shift
DRIVOR_QUERY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-eval_navtest_query_register_dinov3_vitl}"
export DRIVOR_EVAL_WORKER="${DRIVOR_EVAL_WORKER:-single_machine_thread_pool}"
export DRIVOR_SCORE_WORKERS="${DRIVOR_SCORE_WORKERS:-32}"
exec bash "${DRIVOR_QUERY_DIR}/run_drivor_pdm_score_evaluation.sh" "${DRIVOR_QUERY_CHECKPOINT}" \
    agent=drivoR_query_register_dinov3_vitl \
    agent.config.multi_gt.enabled=false \
    agent.config.long_trajectory_additional_poses=0 \
    agent.config.future_register.init_checkpoint=null \
    "agent.config.ref_num=${DRIVOR_QUERY_LAYERS:-4}" \
    trainer.params.precision=32-true \
    dataloader.params.batch_size=1 "$@"
