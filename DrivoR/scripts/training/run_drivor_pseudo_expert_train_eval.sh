#!/usr/bin/env bash
set -euo pipefail

# DLC: train across all nodes, then evaluate the resulting last.ckpt on node 0.
DRIVOR_PE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_PE_ROOT="$(cd -- "${DRIVOR_PE_DIR}/../.." && pwd)"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${DRIVOR_PE_ROOT}/exp}"
export DRIVOR_RUN_ID="${DRIVOR_RUN_ID:-${PAI_JOB_ID:-drivor_pseudo_expert_seed2_run1}}"
export DRIVOR_PRECISION="${DRIVOR_PRECISION:-bf16-mixed}"
export DRIVOR_EVAL_PRECISION="${DRIVOR_EVAL_PRECISION:-32-true}"
DRIVOR_PE_TRAIN_EXPERIMENT="${DRIVOR_TRAIN_EXPERIMENT:-${DRIVOR_EXPERIMENT:-training_drivoR_pseudo_expert_navtrain}}"
DRIVOR_PE_RUN_DIR="${NAVSIM_EXP_ROOT}/ke/${DRIVOR_PE_TRAIN_EXPERIMENT}/${DRIVOR_RUN_ID}"

for DRIVOR_PE_ARG in "$@"; do
    case "${DRIVOR_PE_ARG}" in
        --cfg|--cfg=*|-c|--help|-h|--hydra-help|--info|--info=*|--version)
            DRIVOR_EXPERIMENT="${DRIVOR_PE_TRAIN_EXPERIMENT}" \
            exec bash "${DRIVOR_PE_DIR}/run_drivor_pseudo_expert_training.sh" "$@" ;;
    esac
done

DRIVOR_EXPERIMENT="${DRIVOR_PE_TRAIN_EXPERIMENT}" \
bash "${DRIVOR_PE_DIR}/run_drivor_pseudo_expert_training.sh" \
    auto_resume=false train_ckpt_path=null runtime.require_resume=false \
    runtime.console_log_interval=25 "$@"

if [[ "${NODE_RANK:-${RANK:-${SLURM_NODEID:-0}}}" == 0 ]]; then
    DRIVOR_PE_LAST_CKPT="$(/usr/bin/python3 - "${DRIVOR_PE_RUN_DIR}" <<'PY'
from pathlib import Path
import sys
checkpoints = list(Path(sys.argv[1]).glob('lightning_logs/version_*/checkpoints/last.ckpt'))
if not checkpoints:
    raise FileNotFoundError('Training completed without last.ckpt in ' + sys.argv[1])
print(max(checkpoints, key=lambda path: path.stat().st_mtime_ns))
PY
    )"
    DRIVOR_EXPERIMENT="${DRIVOR_EVAL_EXPERIMENT:-eval_navtest_drivor_pseudo_expert}" \
    NPROC_PER_NODE="${DRIVOR_EVAL_GPUS:-${NPROC_PER_NODE:-${GPU_NUM:-8}}}" \
    bash "${DRIVOR_PE_ROOT}/scripts/evaluation/run_drivor_pseudo_expert_evaluation.sh" \
        "${DRIVOR_PE_LAST_CKPT}"
fi
