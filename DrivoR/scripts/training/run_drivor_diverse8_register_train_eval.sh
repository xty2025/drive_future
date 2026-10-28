#!/usr/bin/env bash
set -euo pipefail
DRIVOR_DIVERSE_VARIANT="${1:?Specify the diverse8 register variant}"
shift
DRIVOR_DIVERSE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_DIVERSE_ROOT="$(cd -- "${DRIVOR_DIVERSE_DIR}/../.." && pwd)"
source "${DRIVOR_DIVERSE_DIR}/drivor_diverse8_register_common.sh"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${DRIVOR_DIVERSE_ROOT}/exp}"
export DRIVOR_RUN_ID="${DRIVOR_RUN_ID:-${PAI_JOB_ID:-${DRIVOR_DIVERSE_VARIANT}_seed2_run1}}"
export DRIVOR_PRECISION="${DRIVOR_PRECISION:-bf16-mixed}"
DRIVOR_DIVERSE_TRAIN_EXP="${DRIVOR_TRAIN_EXPERIMENT:-${DRIVOR_DIVERSE_DEFAULT_TRAIN}}"
DRIVOR_DIVERSE_EVAL_ARGS=()
for DRIVOR_DIVERSE_ARG in "$@"; do
    case "${DRIVOR_DIVERSE_ARG}" in
        --cfg|--cfg=*|-c|--help|-h|--hydra-help|--info|--info=*|--version)
            DRIVOR_EXPERIMENT="${DRIVOR_DIVERSE_TRAIN_EXP}" exec bash \
                "${DRIVOR_DIVERSE_DIR}/run_drivor_diverse8_register_training.sh" "${DRIVOR_DIVERSE_VARIANT}" "$@" ;;
    esac
    DRIVOR_DIVERSE_KEY="${DRIVOR_DIVERSE_ARG#+}"
    DRIVOR_DIVERSE_KEY="${DRIVOR_DIVERSE_KEY#+}"
    case "${DRIVOR_DIVERSE_KEY}" in
        agent.config.*) DRIVOR_DIVERSE_EVAL_ARGS+=("${DRIVOR_DIVERSE_ARG}") ;;
    esac
done

DRIVOR_EXPERIMENT="${DRIVOR_DIVERSE_TRAIN_EXP}" \
bash "${DRIVOR_DIVERSE_DIR}/run_drivor_diverse8_register_training.sh" "${DRIVOR_DIVERSE_VARIANT}" \
    auto_resume=false train_ckpt_path=null runtime.require_resume=false "$@"

if [[ "${NODE_RANK:-${RANK:-${SLURM_NODEID:-0}}}" == 0 ]]; then
    DRIVOR_DIVERSE_CHECKPOINT="$(/usr/bin/python3 - "${NAVSIM_EXP_ROOT}/ke/${DRIVOR_DIVERSE_TRAIN_EXP}/${DRIVOR_RUN_ID}" <<'PY'
from pathlib import Path
import sys
paths = list(Path(sys.argv[1]).glob('lightning_logs/version_*/checkpoints/last.ckpt'))
if not paths:
    raise FileNotFoundError('Training completed without last.ckpt in ' + sys.argv[1])
print(max(paths, key=lambda path: path.stat().st_mtime_ns))
PY
    )"
    DRIVOR_EXPERIMENT="${DRIVOR_EVAL_EXPERIMENT:-${DRIVOR_DIVERSE_DEFAULT_EVAL}}" \
    NPROC_PER_NODE="${DRIVOR_EVAL_GPUS:-${NPROC_PER_NODE:-${GPU_NUM:-8}}}" \
    bash "${DRIVOR_DIVERSE_ROOT}/scripts/evaluation/run_drivor_diverse8_register_evaluation.sh" \
        "${DRIVOR_DIVERSE_VARIANT}" "${DRIVOR_DIVERSE_CHECKPOINT}" "${DRIVOR_DIVERSE_EVAL_ARGS[@]}"
fi

# DLC_TRAINVAL_FULL_CHAIN_chain_trainval_full_20261024_012508: append only; preserve the running script prefix.
case "${DRIVOR_RUN_ID:-}" in
    action_dit_human_xl1_diverse8_l1_register_seed2_dlc16mhamjyg8qtw_20261022_210127)
        /usr/bin/python3 /cpfs29jb/data/zhaozc/project/drive_future/environment/dlc_tail_once.py --control /cpfs29jb/data/zhaozc/project/drive_future/artifacts/dlc_workspace_26585/chain_trainval_full_20261024_012508/control_stage0.json
        /usr/bin/python3 /cpfs29jb/data/zhaozc/project/drive_future/environment/dlc_tail_once.py --control /cpfs29jb/data/zhaozc/project/drive_future/artifacts/dlc_workspace_26585/chain_trainval_full_20261024_012508/control_stage1.json
        ;;
    action_dit_human_diverse8_l1_register_seed2_20261022_210127)
        /usr/bin/python3 /cpfs29jb/data/zhaozc/project/drive_future/environment/dlc_tail_once.py --control /cpfs29jb/data/zhaozc/project/drive_future/artifacts/dlc_workspace_26585/chain_trainval_full_20261024_012508/control_stage0.json
        ;;
    action_dit_human_diverse8_l1_register_scorer_seed2_20261022_210127)
        /usr/bin/python3 /cpfs29jb/data/zhaozc/project/drive_future/environment/dlc_tail_once.py --control /cpfs29jb/data/zhaozc/project/drive_future/artifacts/dlc_workspace_26585/chain_trainval_full_20261024_012508/control_stage0.json
        ;;
    action_dit_human_xl1_diverse8_l1_register_scorer_seed2_20261022_210127)
        /usr/bin/python3 /cpfs29jb/data/zhaozc/project/drive_future/environment/dlc_tail_once.py --control /cpfs29jb/data/zhaozc/project/drive_future/artifacts/dlc_workspace_26585/chain_trainval_full_20261024_012508/control_stage0.json
        ;;
esac

# DLC_HALF_CHAIN_dlc_deployment_20261024_231341: append only; exact predecessor RunIDs.
case "${DRIVOR_RUN_ID:-}" in
    action_dit_human_xl1_diverse8_l1_register_seed2_dlc16mhamjyg8qtw_20261022_210127|\
    action_dit_human_diverse8_l1_register_seed2_20261022_210127|\
    action_dit_human_diverse8_l1_register_scorer_seed2_20261022_210127|\
    action_dit_human_xl1_diverse8_l1_register_scorer_seed2_20261022_210127|\
    a28_register_human_only_seed2_20261024_135904)
        /usr/bin/python3 /cpfs29jb/data/zhaozc/project/drive_future/environment/dlc_tail_once.py --control /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/navtrain_half_ablations/dlc_deployment_20261024_231341/control.json
        ;;
esac
