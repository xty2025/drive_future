#!/usr/bin/env bash
set -euo pipefail

# 按 README 的 NAVSIM-v1 参数在 navtest 上评测 DrivoR 权重，输出 PDMS。
# 单机多卡：torchrun 每卡一个进程推理，rank 0 收齐轨迹后计算 PDM 分数。
# DRIVOR_EVAL_WORKER=single_machine_thread_pool 使用本机 CPU 进程池；sequential 为串行。
# 用法（任意目录执行）：
#   bash DrivoR/scripts/evaluation/run_drivor_pdm_score_evaluation.sh <权重 .pth/.ckpt> [Hydra 参数...]
# 冒烟测试只评前 N 个场景：追加 train_test_split.scene_filter.max_scenes=64
# 查看配置：bash DrivoR/scripts/evaluation/run_drivor_pdm_score_evaluation.sh <权重> --cfg job --package agent --resolve
# 官方发布的两个 Nav1 权重可直接用 run_drivor_release_nav1_evaluation.sh。
# 结果写入 $NAVSIM_EXP_ROOT/ke/$DRIVOR_EXPERIMENT/$DRIVOR_RUN_ID/：逐场景 CSV（最后一行 average）和 pdms_summary.txt。

# 第一个参数必须是权重；训练产出的 epoch=24-step=33250.ckpt 含 '='，存在的文件不当作 Hydra 参数。
if (( $# < 1 )) || [[ ! -f "$1" && ( "$1" == -* || "$1" == *=* ) ]]; then
    printf '%s\n' '用法：run_drivor_pdm_score_evaluation.sh <权重 .pth/.ckpt> [Hydra 参数...]' >&2
    exit 2
fi
# 在切换目录前解析，相对路径以调用时的当前目录为准。
DRIVOR_CHECKPOINT="$(realpath -m -- "$1")"
shift
if [[ ! -f "${DRIVOR_CHECKPOINT}" ]]; then
    printf '找不到权重文件：%s\n' "${DRIVOR_CHECKPOINT}" >&2
    exit 2
fi

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
# 评测入口把全部预测轨迹存到 $SUBSCORE_PATH/navsim1_pdm_scores/<实验名>/。
export SUBSCORE_PATH="${SUBSCORE_PATH:-${NAVSIM_EXP_ROOT}}"
# 不沿用训练用的 NAVSIM_METRIC_CACHE_PATH（那是 navtrain 缓存）。
DRIVOR_METRIC_CACHE="${NAVSIM_TEST_METRIC_CACHE_PATH:-${DRIVOR_PROJECT_ROOT}/data/metric_cache_navtest}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${DRIVOR_CPU_THREADS:-1}"
export MKL_NUM_THREADS="${DRIVOR_CPU_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${DRIVOR_CPU_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${DRIVOR_CPU_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false

for DRIVOR_PATH in "${OPENSCENE_DATA_ROOT}/navsim_logs/test" "${OPENSCENE_DATA_ROOT}/sensor_blobs/test" \
        "${DRIVOR_METRIC_CACHE}/metadata"; do
    if [[ ! -d "${DRIVOR_PATH}" ]]; then
        printf '缺少目录：%s\n' "${DRIVOR_PATH}" >&2
        exit 2
    fi
done

# 本地 sensor_blobs/test 只有 CAM_F0；左、右、后视按相同相对路径从 OSS 的 test 目录读取。
# 训练脚本的 NAVSIM_OSS_SENSOR_BLOBS_PATH 指向 trainval，这里不沿用；
# 用 NAVSIM_TEST_OSS_SENSOR_BLOBS_PATH 覆盖，设为空字符串可关闭 OSS 回退。凭据仅从 JSON 加载。
export NAVSIM_OSS_CONFIG_PATH="${NAVSIM_OSS_CONFIG_PATH:-/mnt/human/shixi/human/oss_config.json}"
export NAVSIM_OSS_SENSOR_BLOBS_PATH="${NAVSIM_TEST_OSS_SENSOR_BLOBS_PATH-oss://ofasys-multimodal-wlcb-5-toshanghai/anonymous_driving/autonomousvision_navsim/1/download_test/unzip/test_sensor_blobs/test}"
# 与训练共用下载缓存，再次评测直接读缓存；设为空字符串可关闭。
export NAVSIM_OSS_IMAGE_CACHE_DIR="${NAVSIM_OSS_IMAGE_CACHE_DIR-${NAVSIM_EXP_ROOT}/oss_image_cache}"

# 优先导入本目录的源码，覆盖环境里 vendor/DrivoR 的 editable 路径。
export PYTHONPATH="${DRIVOR_ROOT}:${DRIVOR_ROOT}/nuplan-devkit${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a DRIVOR_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
DRIVOR_GPUS="${NPROC_PER_NODE:-${#DRIVOR_GPU_IDS[@]}}"
# 推理结束后只剩 rank 0，PDM 打分默认用满本机 CPU（nproc 会受上面 OMP_NUM_THREADS=1 影响）。
DRIVOR_SCORE_WORKERS="${DRIVOR_SCORE_WORKERS:-$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)}"
if ! [[ "${DRIVOR_GPUS}" =~ ^[1-9][0-9]*$ && "${DRIVOR_SCORE_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s\n' 'NPROC_PER_NODE 和 DRIVOR_SCORE_WORKERS 必须是正整数。' >&2
    exit 2
fi
case "${DRIVOR_EVAL_WORKER:-ray_distributed_no_torch}" in
    single_machine_thread_pool)
        DRIVOR_WORKER_ARGS=(worker=single_machine_thread_pool worker.use_process_pool=true
                           "worker.max_workers=${DRIVOR_SCORE_WORKERS}") ;;
    sequential)
        DRIVOR_WORKER_ARGS=(worker=sequential) ;;
    ray_distributed_no_torch)
        DRIVOR_WORKER_ARGS=(worker=ray_distributed_no_torch
                           "worker.threads_per_node=${DRIVOR_SCORE_WORKERS}") ;;
    *)
        printf '%s\n' 'DRIVOR_EVAL_WORKER 必须是 single_machine_thread_pool、sequential 或 ray_distributed_no_torch。' >&2
        exit 2 ;;
esac

DRIVOR_CHECKPOINT_TAG="$(basename -- "${DRIVOR_CHECKPOINT}")"
DRIVOR_CHECKPOINT_TAG="${DRIVOR_CHECKPOINT_TAG%.*}"
# 训练产出的 epoch=24-step=33250.ckpt、last.ckpt 等文件名在不同训练间重复，默认实验名再加上训练实验、run 和版本。
if [[ "${DRIVOR_CHECKPOINT}" =~ /ke/([^/]+)/([^/]+)/lightning_logs/([^/]+)/checkpoints/[^/]+$ ]]; then
    DRIVOR_CHECKPOINT_TAG="${BASH_REMATCH[1]}_${BASH_REMATCH[2]}_${BASH_REMATCH[3]}_${DRIVOR_CHECKPOINT_TAG}"
fi
# 默认实验名只保留字母、数字、点、下划线和连字符，'=' 等其他字符换成下划线。
DRIVOR_CHECKPOINT_TAG="${DRIVOR_CHECKPOINT_TAG//[^A-Za-z0-9._-]/_}"
DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-eval_navtest_${DRIVOR_CHECKPOINT_TAG}}"
# 所有 rank 共用一个输出目录；默认按启动时间区分每次评测。
DRIVOR_RUN_ID="${DRIVOR_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
# 二者用作目录名，并在下面以引号传给 Hydra，防止 20261014_225815 被解析成整数。
if ! [[ "${DRIVOR_EXPERIMENT}" =~ ^[A-Za-z0-9._-]+$ && "${DRIVOR_RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    printf '%s\n' 'DRIVOR_EXPERIMENT 和 DRIVOR_RUN_ID 只能包含字母、数字、点、下划线和连字符。' >&2
    exit 2
fi
DRIVOR_OUTPUT_DIR="${NAVSIM_EXP_ROOT}/ke/${DRIVOR_EXPERIMENT}/${DRIVOR_RUN_ID}"

# Hydra 配置/帮助查询直接运行单进程，不启动 torchrun。
DRIVOR_CONFIG_ONLY=false
for DRIVOR_ARG in "$@"; do
    case "${DRIVOR_ARG}" in
        --cfg|--cfg=*|-c|--help|-h|--hydra-help|--info|--info=*|--version)
            DRIVOR_CONFIG_ONLY=true ;;
    esac
done
DRIVOR_LAUNCH=("${DRIVOR_PYBIN}" -u)
if [[ "${DRIVOR_CONFIG_ONLY}" == false ]]; then
    if [[ -n "${LOCAL_RANK+x}" || -n "${TORCHELASTIC_RUN_ID:-}" ]]; then
        printf '%s\n' '此脚本内部已调用 torchrun；请直接用 bash 执行一次，不要放在 torchrun worker 内启动。' >&2
        exit 2
    fi
    # /cpfs29jb 是 virtiofs：宿主机守护进程为内核缓存的每个 inode 占一个 fd，上限约 100 万。
    # 接近上限时新打开 CPFS 文件会报 Too many open files，import 也会失败；容器内无法释放缓存，需重启实例。
    DRIVOR_FUSE_INODES="$(awk '$1 == "fuse_inode" {print $2}' /proc/slabinfo 2>/dev/null || true)"
    if [[ "${DRIVOR_FUSE_INODES}" =~ ^[0-9]+$ ]] && (( DRIVOR_FUSE_INODES > 1000000 )); then
        printf '警告：内核已缓存 %s 个 virtiofs inode，接近守护进程约 100 万个 fd 的上限，可能报 Too many open files / No module named；建议重启实例或改用 DLC 运行。\n' \
            "${DRIVOR_FUSE_INODES}" >&2
    fi
    # 单机评测：--standalone 自动选择空闲端口，不使用平台注入的多机地址。
    DRIVOR_LAUNCH+=(-m torch.distributed.run --standalone --nnodes=1 "--nproc_per_node=${DRIVOR_GPUS}")
    printf 'DrivoR eval: checkpoint=%s\n' "${DRIVOR_CHECKPOINT}" >&2
    printf 'GPUs=%s (CUDA_VISIBLE_DEVICES=%s); PDM score workers=%s\n' \
        "${DRIVOR_GPUS}" "${CUDA_VISIBLE_DEVICES}" "${DRIVOR_SCORE_WORKERS}" >&2
    printf 'NAVSIM data=%s; metric cache=%s; output=%s\n' \
        "${OPENSCENE_DATA_ROOT}" "${DRIVOR_METRIC_CACHE}" "${DRIVOR_OUTPUT_DIR}" >&2
fi

# YAML 中的 weights/... 相对于 DrivoR 根目录解析。
cd -- "${DRIVOR_ROOT}"

# agent.score_worker=null：推理时 Agent 不计算 Oracle；最终 PDM 打分由 rank 0 的 worker 完成。
# 评测不使用合成训练集；避免展开及向评分进程传输其庞大的 token 过滤列表。
# logger=false：predict 不需要 Lightning 日志，也不会在 DrivoR 根目录生成 lightning_logs/。
DRIVOR_STATUS=0
"${DRIVOR_LAUNCH[@]}" "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_multi_gpu.py" \
    train_test_split=navtest \
    train_test_split_synthetic/scene_filter=all_scenes \
    agent=drivoR \
    "agent.checkpoint_path='${DRIVOR_CHECKPOINT}'" \
    "experiment_name='${DRIVOR_EXPERIMENT}'" \
    "experiment_uid='${DRIVOR_RUN_ID}'" \
    "metric_cache_path='${DRIVOR_METRIC_CACHE}'" \
    "${DRIVOR_WORKER_ARGS[@]}" \
    agent.score_worker=null \
    "++trainer.params.devices=${DRIVOR_GPUS}" \
    trainer.params.num_nodes=1 \
    trainer.params.strategy=ddp \
    ++trainer.params.logger=false \
    agent.config.proposal_num=64 \
    agent.config.refiner_ls_values=0.0 \
    agent.config.image_backbone.focus_front_cam=false \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.config.noc=1 \
    agent.config.dac=1 \
    agent.config.ddc=0.0 \
    agent.config.ttc=5 \
    agent.config.ep=5 \
    agent.config.comfort=2 \
    "$@" || DRIVOR_STATUS=$?

if [[ "${DRIVOR_CONFIG_ONLY}" == true || "${DRIVOR_STATUS}" != 0 ]]; then
    exit "${DRIVOR_STATUS}"
fi

DRIVOR_CSV=""
for DRIVOR_FILE in "${DRIVOR_OUTPUT_DIR}"/*.csv; do
    if [[ -f "${DRIVOR_FILE}" && ( -z "${DRIVOR_CSV}" || "${DRIVOR_FILE}" -nt "${DRIVOR_CSV}" ) ]]; then
        DRIVOR_CSV="${DRIVOR_FILE}"
    fi
done
if [[ -z "${DRIVOR_CSV}" ]]; then
    printf '评测进程已退出，但 %s 下没有结果 CSV。\n' "${DRIVOR_OUTPUT_DIR}" >&2
    exit 1
fi

# README 的 PDMS 按百分制报告，这里同样乘以 100。
"${DRIVOR_PYBIN}" - "${DRIVOR_CSV}" "${DRIVOR_CHECKPOINT}" <<'PY' | tee "${DRIVOR_OUTPUT_DIR}/pdms_summary.txt"
import sys

import pandas as pd

csv_path, checkpoint = sys.argv[1:3]
table = pd.read_csv(csv_path, index_col=0)
average = table[table["token"] == "average"].iloc[-1]
scenes = table[table["token"] != "average"]
valid = int(scenes["valid"].astype(str).str.lower().eq("true").sum())
metrics = [
    ("NC", "no_at_fault_collisions"),
    ("DAC", "drivable_area_compliance"),
    ("TTC", "time_to_collision_within_bound"),
    ("C", "comfort"),
    ("EP", "ego_progress"),
]
parts = "  ".join(f"{name}={100 * float(average[column]):.2f}" for name, column in metrics)
print(f"PDMS={100 * float(average['score']):.2f}  {parts}  valid={valid}/{len(scenes)}  "
      f"checkpoint={checkpoint}  csv={csv_path}")
PY
