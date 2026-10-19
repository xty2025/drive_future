#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_drivor_pseudo_expert_evaluation.sh <checkpoint.ckpt> [Hydra overrides]
# Coverage changes only the training loss, so inference uses the original DrivoR.
DRIVOR_PE_CHECKPOINT="${1:?Specify the trained DrivoR pseudo-expert checkpoint}"
shift
DRIVOR_PE_EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-eval_navtest_drivor_pseudo_expert}"
export DRIVOR_EVAL_WORKER="${DRIVOR_EVAL_WORKER:-single_machine_thread_pool}"
export DRIVOR_SCORE_WORKERS="${DRIVOR_SCORE_WORKERS:-32}"
exec bash "${DRIVOR_PE_EVAL_DIR}/run_drivor_pdm_score_evaluation.sh" "${DRIVOR_PE_CHECKPOINT}" \
    agent=drivoR \
    "trainer.params.precision=${DRIVOR_EVAL_PRECISION:-32-true}" \
    "dataloader.params.batch_size=${DRIVOR_EVAL_BATCH_SIZE:-1}" \
    "$@"
