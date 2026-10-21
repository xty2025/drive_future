#!/usr/bin/env bash
set -euo pipefail

DRIVOR_QUERY_VARIANT="${1:?Specify single_gt, multigt, scorer_single_gt or scorer_multigt}"
shift
case "${DRIVOR_QUERY_VARIANT}" in
    single_gt|multigt|scorer_single_gt|scorer_multigt) ;;
    *) printf 'Unknown query-register variant: %s\n' "${DRIVOR_QUERY_VARIANT}" >&2; exit 2 ;;
esac
DRIVOR_QUERY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_QUERY_ROOT="$(cd -- "${DRIVOR_QUERY_DIR}/../.." && pwd)"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${DRIVOR_QUERY_ROOT}/exp}"
export DRIVOR_RUN_ID="${DRIVOR_RUN_ID:-${PAI_JOB_ID:-query_register_dinov3_vitl_${DRIVOR_QUERY_VARIANT}_seed2_train_eval_run1}}"
export DRIVOR_PRECISION="${DRIVOR_PRECISION:-bf16-mixed}"
DRIVOR_QUERY_TRAIN_EXP="${DRIVOR_TRAIN_EXPERIMENT:-training_drivoR_query_register_dinov3_vitl_navtrain_${DRIVOR_QUERY_VARIANT}}"
DRIVOR_QUERY_EVAL_ARGS=()
for DRIVOR_QUERY_ARG in "$@"; do
    case "${DRIVOR_QUERY_ARG}" in
        --cfg|--cfg=*|-c|--help|-h|--hydra-help|--info|--info=*|--version)
            DRIVOR_EXPERIMENT="${DRIVOR_QUERY_TRAIN_EXP}" \
            exec bash "${DRIVOR_QUERY_DIR}/run_drivor_query_register_training.sh" "${DRIVOR_QUERY_VARIANT}" "$@" ;;
    esac
    # Keep architecture overrides consistent when evaluating the resulting checkpoint.
    DRIVOR_QUERY_KEY="${DRIVOR_QUERY_ARG#+}"
    DRIVOR_QUERY_KEY="${DRIVOR_QUERY_KEY#+}"
    case "${DRIVOR_QUERY_KEY}" in
        agent.config.query_register.*|agent.config.ref_num=*|agent.config.scorer_ref_num=*|\
        agent.config.tf_d_model=*|agent.config.tf_d_ffn=*|agent.config.refiner_num_heads=*|\
        agent.config.refiner_ls_values=*|agent.config.proposal_num=*|\
        agent.config.image_backbone.*|agent.config.image_size=*|agent.config.visual_feature_mode=*|\
        agent.config.num_scene_tokens=*|agent.config.cam_*=*|agent.config.future_register.scorer.*|\
        agent.config.noc=*|agent.config.dac=*|agent.config.ddc=*|agent.config.ttc=*|\
        agent.config.ep=*|agent.config.comfort=*)
            DRIVOR_QUERY_EVAL_ARGS+=("${DRIVOR_QUERY_ARG}") ;;
    esac
done

DRIVOR_EXPERIMENT="${DRIVOR_QUERY_TRAIN_EXP}" \
bash "${DRIVOR_QUERY_DIR}/run_drivor_query_register_training.sh" "${DRIVOR_QUERY_VARIANT}" \
    auto_resume=false train_ckpt_path=null runtime.require_resume=false "$@"

if [[ "${NODE_RANK:-${RANK:-${SLURM_NODEID:-0}}}" == 0 ]]; then
    DRIVOR_QUERY_CHECKPOINT="$(/usr/bin/python3 - "${NAVSIM_EXP_ROOT}/ke/${DRIVOR_QUERY_TRAIN_EXP}/${DRIVOR_RUN_ID}" <<'PY'
from pathlib import Path
import sys
paths=list(Path(sys.argv[1]).glob('lightning_logs/version_*/checkpoints/last.ckpt'))
if not paths:
    raise FileNotFoundError('Training completed without last.ckpt in '+sys.argv[1])
print(max(paths,key=lambda path:path.stat().st_mtime_ns))
PY
    )"
    DRIVOR_EXPERIMENT="${DRIVOR_EVAL_EXPERIMENT:-eval_navtest_query_register_dinov3_vitl_${DRIVOR_QUERY_VARIANT}}" \
    NPROC_PER_NODE="${DRIVOR_EVAL_GPUS:-${NPROC_PER_NODE:-${GPU_NUM:-8}}}" \
    bash "${DRIVOR_QUERY_ROOT}/scripts/evaluation/run_drivor_query_register_evaluation.sh" \
        "${DRIVOR_QUERY_CHECKPOINT}" \
        "agent=drivoR_query_register_dinov3_vitl_${DRIVOR_QUERY_VARIANT}" \
        "${DRIVOR_QUERY_EVAL_ARGS[@]}"
fi

# DLC_QUERY_GT_CHAIN_chain_query_gt_20261021_164710: append only, exact predecessor RunIDs.
case "${DRIVOR_RUN_ID:-}" in
    query_register_scorer_single_gt_seed2_dlc1stkt1n8i1ctp_20261021_run1|query_register_scorer_multigt_seed2_dlcudc5qjukb34ak_20261021_run1)
        /usr/bin/python3 /cpfs29jb/data/zhaozc/project/drive_future/environment/dlc_tail_once.py --control /cpfs29jb/data/zhaozc/project/drive_future/artifacts/dlc_workspace_26585/chain_query_gt_20261021_164710/control.json
        ;;
esac
