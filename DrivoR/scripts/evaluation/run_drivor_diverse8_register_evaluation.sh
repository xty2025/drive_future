#!/usr/bin/env bash
set -euo pipefail
DRIVOR_DIVERSE_VARIANT="${1:?Specify the diverse8 register variant}"
DRIVOR_DIVERSE_CHECKPOINT="${2:?Specify a complete checkpoint of this variant}"
shift 2
DRIVOR_DIVERSE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${DRIVOR_DIVERSE_DIR}/../training/drivor_diverse8_register_common.sh"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-${DRIVOR_DIVERSE_DEFAULT_EVAL}}"
export DRIVOR_EVAL_WORKER="${DRIVOR_EVAL_WORKER:-single_machine_thread_pool}"
export DRIVOR_SCORE_WORKERS="${DRIVOR_SCORE_WORKERS:-32}"
exec bash "${DRIVOR_DIVERSE_DIR}/run_drivor_pdm_score_evaluation.sh" "${DRIVOR_DIVERSE_CHECKPOINT}" \
    "agent=${DRIVOR_DIVERSE_AGENT}" \
    agent.config.multi_gt.enabled=false \
    agent.config.long_trajectory_additional_poses=0 \
    agent.config.future_register.init_checkpoint=null \
    "agent.config.ref_num=${DRIVOR_QUERY_LAYERS:-4}" \
    trainer.params.precision=32-true dataloader.params.batch_size=1 "$@"
