#!/usr/bin/env bash
set -euo pipefail

# 全量 navtest 上对比两个 DrivoR 权重：被选中轨迹的 PDMS 与同一组 64 条候选里最优的 PDMS，
# 差值即换成完美打分器最多能提升的分数。64 条候选和人类轨迹都用官方 NAVSIM-v1 PDMS 打分。
# 默认对比单 GT 与多 GT 监督训练的 epoch=24；单卡推理，打分用多进程 CPU。
# 用法（任意目录执行）：
#   bash DrivoR/scripts/evaluation/run_scorer_oracle_gap_analysis.sh [--max-scenes 64] [其他参数...]
# 只重出报告：bash DrivoR/scripts/evaluation/run_scorer_oracle_gap_analysis.sh --report-only <输出目录>
# 输出：DrivoR/exp/analysis/scorer_oracle_gap_<时间戳>/{summary.md,summary.json,scenes.csv,oracle_gap_summary.png,arrays.npz}

DRIVOR_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_ROOT="$(cd -- "${DRIVOR_SCRIPT_DIR}/../.." && pwd)"
DRIVOR_PROJECT_ROOT="$(cd -- "${DRIVOR_ROOT}/.." && pwd)"

DRIVOR_CONDA_ROOT="${DRIVOR_CONDA_ROOT:-/cpfs29jb/data/zhaozc/miniconda3}"
DRIVOR_CONDA_ENV="${DRIVOR_CONDA_ENV:-${DRIVOR_CONDA_ROOT}/envs/drive_future}"
export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
if [[ -n "${DRIVOR_PYTHON:-}" ]]; then
    DRIVOR_PYBIN="${DRIVOR_PYTHON}"
else
    source "${DRIVOR_CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${DRIVOR_CONDA_ENV}"
    DRIVOR_PYBIN="${CONDA_PREFIX}/bin/python"
fi

export NAVSIM_DEVKIT_ROOT="${DRIVOR_ROOT}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${DRIVOR_PROJECT_ROOT}/data/navsim}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${DRIVOR_ROOT}/exp}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
# PDM 打分在多个 CPU 进程里并行，每个进程只用一个线程。
export OMP_NUM_THREADS="${DRIVOR_CPU_THREADS:-1}"
export MKL_NUM_THREADS="${DRIVOR_CPU_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${DRIVOR_CPU_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${DRIVOR_CPU_THREADS:-1}"
# 本地 sensor_blobs/test 只有 CAM_F0，其余相机从 OSS 的 test 目录读取并缓存（前缀由 Python 设置）。
export NAVSIM_OSS_CONFIG_PATH="${NAVSIM_OSS_CONFIG_PATH:-/mnt/human/shixi/human/oss_config.json}"
export NAVSIM_OSS_IMAGE_CACHE_DIR="${NAVSIM_OSS_IMAGE_CACHE_DIR-${NAVSIM_EXP_ROOT}/oss_image_cache}"
export PYTHONPATH="${DRIVOR_ROOT}:${DRIVOR_ROOT}/nuplan-devkit${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# YAML 中的 weights/... 相对于 DrivoR 根目录解析。
cd -- "${DRIVOR_ROOT}"
# DRIVOR_ANALYSIS_SCRIPT：同目录下的其他分析脚本（run_cross_scorer_analysis.sh 用它）。
exec "${DRIVOR_PYBIN}" -u "${DRIVOR_SCRIPT_DIR}/${DRIVOR_ANALYSIS_SCRIPT:-analyze_scorer_oracle_gap.py}" "$@"
