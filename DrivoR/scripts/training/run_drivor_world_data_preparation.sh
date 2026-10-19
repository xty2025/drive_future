#!/usr/bin/env bash
set -euo pipefail

# DLC preprocessing job: real (group3) or cosmos (group4). Each GPU handles a
# disjoint set of scenes. Cosmos images are extracted before teacher encoding.
DRIVOR_PREP_MODE="${1:?Specify real or cosmos}"
shift
case "${DRIVOR_PREP_MODE}" in real|cosmos) ;; *) exit 2 ;; esac
DRIVOR_PREP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_ROOT="$(cd -- "${DRIVOR_PREP_DIR}/../.." && pwd)"
DRIVOR_PROJECT_ROOT="$(cd -- "${DRIVOR_ROOT}/.." && pwd)"
DRIVOR_CONDA_ROOT="${DRIVOR_CONDA_ROOT:-/cpfs29jb/data/zhaozc/miniconda3}"
DRIVOR_CONDA_ENV="${DRIVOR_CONDA_ENV:-${DRIVOR_CONDA_ROOT}/envs/drive_future}"
if [[ -n "${DRIVOR_PYTHON:-}" ]]; then
    DRIVOR_PYBIN="${DRIVOR_PYTHON}"
else
    source "${DRIVOR_CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${DRIVOR_CONDA_ENV}"
    DRIVOR_PYBIN="${CONDA_PREFIX}/bin/python"
fi
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${DRIVOR_ROOT}/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${DRIVOR_PROJECT_ROOT}/data/navsim}"
export NAVSIM_OSS_CONFIG_PATH="${NAVSIM_OSS_CONFIG_PATH:-/mnt/human/shixi/human/oss_config.json}"
export NAVSIM_OSS_SENSOR_BLOBS_PATH="${NAVSIM_OSS_SENSOR_BLOBS_PATH-oss://ofasys-multimodal-wlcb-5-toshanghai/anonymous_driving/autonomousvision_navsim/1/download_trainval/openscene-v1.1/sensor_blobs/trainval}"
export PYTHONPATH="${DRIVOR_ROOT}:${DRIVOR_ROOT}/nuplan-devkit${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
DRIVOR_PREP_ROOT="${DRIVOR_WORLD_DATA_ROOT:-${NAVSIM_EXP_ROOT}/world4scorer_data}"
DRIVOR_PREP_NODES="${NNODES:-${WORLD_SIZE:-1}}"
DRIVOR_PREP_RANK="${NODE_RANK:-${RANK:-0}}"
DRIVOR_PREP_GPUS="${NPROC_PER_NODE:-${GPU_NUM:-8}}"
mkdir -p -- "${DRIVOR_PREP_ROOT}"
if [[ "${DRIVOR_PREP_MODE}" == real ]]; then
    DRIVOR_REAL_INDEX="${DRIVOR_PREP_ROOT}/real_index.jsonl"
    if [[ "${DRIVOR_PREP_RANK}" == 0 && ! -f "${DRIVOR_REAL_INDEX}" ]]; then
        "${DRIVOR_PYBIN}" -u "${DRIVOR_ROOT}/tools/prepare_world4scorer_data.py" index-real \
            --navsim "${OPENSCENE_DATA_ROOT}" --output "${DRIVOR_REAL_INDEX}"
    fi
    DRIVOR_PREP_WAIT=0
    while [[ ! -f "${DRIVOR_REAL_INDEX}" ]]; do
        sleep 3
        DRIVOR_PREP_WAIT=$((DRIVOR_PREP_WAIT + 3))
        if (( DRIVOR_PREP_WAIT > 3600 )); then
            printf '%s\n' 'Timed out waiting for the NAVSIM future-image index.' >&2
            exit 1
        fi
    done
    DRIVOR_PREP_ARGS=(encode-real --index "${DRIVOR_REAL_INDEX}" \
                     --output "${DRIVOR_PREP_ROOT}/future_real_vjepa21")
else
    DRIVOR_PREP_ARGS=(encode-cosmos --extract-frames \
                     --frames-output "${DRIVOR_PREP_ROOT}/frames_cosmos" \
                     --output "${DRIVOR_PREP_ROOT}/future_cosmos_vjepa21")
fi
DRIVOR_PREP_LAUNCH=("${DRIVOR_PYBIN}" -u -m torch.distributed.run "--nproc_per_node=${DRIVOR_PREP_GPUS}")
if (( DRIVOR_PREP_NODES == 1 )); then
    DRIVOR_PREP_LAUNCH+=(--standalone --nnodes=1)
else
    DRIVOR_PREP_LAUNCH+=("--nnodes=${DRIVOR_PREP_NODES}" "--node_rank=${DRIVOR_PREP_RANK}" \
        "--master_addr=${MASTER_ADDR:?DLC must supply MASTER_ADDR}" "--master_port=${MASTER_PORT:-29500}")
fi
exec "${DRIVOR_PREP_LAUNCH[@]}" "${DRIVOR_ROOT}/tools/prepare_world4scorer_data.py" \
    "${DRIVOR_PREP_ARGS[@]}" --navsim "${OPENSCENE_DATA_ROOT}" \
    --workers "${DRIVOR_PREP_WORKERS:-2}" --batch-size "${DRIVOR_PREP_BATCH:-128}" "$@"
