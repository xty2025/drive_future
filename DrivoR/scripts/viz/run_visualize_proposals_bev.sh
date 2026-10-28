#!/usr/bin/env bash
set -euo pipefail

# 在 BEV 上对比两个 DrivoR 权重：每个场景的 64 条候选、打分器预测分数和逐条官方 PDMS，
# 并突出显示打分器选中的轨迹。默认对比单 GT 与多 GT 监督训练的 epoch=24，
# navtest 和训练集各挑 8 个场景（B 更好 / A 更好 / 都差 / 都好，各 2 个）。
# 用法（任意目录执行，单卡即可）：
#   bash DrivoR/scripts/viz/run_visualize_proposals_bev.sh [--per-group 2] [--test-tokens t1 t2 ...] [--train-tokens ...]
# 全部参数见：bash DrivoR/scripts/viz/run_visualize_proposals_bev.sh --help
# 输出：DrivoR/exp/viz/proposals_bev_<时间戳>/{test,train}/*.png、summary.csv、index.html。

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
# 本地缺失的相机图像从 OSS 读取并缓存；test / trainval 的 OSS 前缀由 Python 脚本按 split 设置。
export NAVSIM_OSS_CONFIG_PATH="${NAVSIM_OSS_CONFIG_PATH:-/mnt/human/shixi/human/oss_config.json}"
export NAVSIM_OSS_IMAGE_CACHE_DIR="${NAVSIM_OSS_IMAGE_CACHE_DIR-${NAVSIM_EXP_ROOT}/oss_image_cache}"
export PYTHONPATH="${DRIVOR_ROOT}:${DRIVOR_ROOT}/nuplan-devkit${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# YAML 中的 weights/... 相对于 DrivoR 根目录解析；--output-dir、--reuse 等相对路径仍按调用时的目录解析。
export DRIVOR_CALLER_CWD="${PWD}"
cd -- "${DRIVOR_ROOT}"
# DRIVOR_VIZ_SCRIPT：同目录下的其他可视化脚本（run_visualize_action_dit_proposals.sh 用它）。
exec "${DRIVOR_PYBIN}" -u "${DRIVOR_SCRIPT_DIR}/${DRIVOR_VIZ_SCRIPT:-visualize_proposals_bev.py}" "$@"
