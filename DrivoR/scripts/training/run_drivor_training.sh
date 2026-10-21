#!/usr/bin/env bash
set -euo pipefail

# README 的 NAVSIM-v1 参数，使用 run_training.py 的 train/val 分离方式。
# 默认 8 机 × 8 卡 × 每卡 batch 1 × 累积 1，有效总 batch 保持 64。
# PAI-DLC：创建 PyTorch 分布式任务，配置 8 个节点、每节点 8 卡，在启动命令中填一次：
# bash DrivoR/scripts/training/run_drivor_training.sh [Hydra 参数...]
# DLC 分发命令并注入 WORLD_SIZE / RANK / MASTER_ADDR / MASTER_PORT。
# 查看配置：bash DrivoR/scripts/training/run_drivor_training.sh --cfg job --package agent --resolve
# NNODES / NPROC_PER_NODE 可显式覆盖机器数、每机卡数；无需手填节点地址或编号。
# 评分缓存默认读取 drive_future/data/metric_cache_navtrain。
# 当前默认：BF16、fused AdamW、数据 workers=4、Oracle 进程=2、CPU threads=1。
# 模型随机流在 DDP 就绪后使用 seed + global_rank；数据采样保持公共 seed。
# 新训练：DRIVOR_RUN_ID=drivor_nav1_bf16_seed2_fast_run1 bash DrivoR/scripts/training/run_drivor_training.sh auto_resume=false train_ckpt_path=null runtime.require_resume=false
# 续训当前目录：DRIVOR_RUN_ID=multinode bash DrivoR/scripts/training/run_drivor_training.sh auto_resume=true runtime.require_resume=true
# 完整可复制的两个模型命令见同目录 README.md。

DRIVOR_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_ROOT="$(cd -- "${DRIVOR_SCRIPT_DIR}/../.." && pwd)"
DRIVOR_PROJECT_ROOT="$(cd -- "${DRIVOR_ROOT}/.." && pwd)"

DRIVOR_CONDA_ROOT="${DRIVOR_CONDA_ROOT:-/cpfs29jb/data/zhaozc/miniconda3}"
DRIVOR_CONDA_ENV="${DRIVOR_CONDA_ENV:-${DRIVOR_CONDA_ROOT}/envs/drive_future}"
# Activation does not need external Conda plugins; avoid scanning their metadata on CPFS.
export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
if [[ -n "${DRIVOR_PYTHON:-}" ]]; then
    # DLC images may already contain the runtime; do not require another user's Conda.
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
export NAVSIM_METRIC_CACHE_PATH="${NAVSIM_METRIC_CACHE_PATH:-${DRIVOR_PROJECT_ROOT}/data/metric_cache_navtrain}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
# Each GPU process runs a CPU oracle and DataLoader workers. Prevent each from
# spawning a full BLAS/OpenMP pool on top of all the other ranks on this node.
export OMP_NUM_THREADS="${DRIVOR_CPU_THREADS:-1}"
export MKL_NUM_THREADS="${DRIVOR_CPU_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${DRIVOR_CPU_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${DRIVOR_CPU_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export NAVSIM_ORACLE_WORKERS="${NAVSIM_ORACLE_WORKERS:-2}"
export NAVSIM_ORACLE_FAST_PATH="${NAVSIM_ORACLE_FAST_PATH:-1}"
DRIVOR_WORKERS="${DRIVOR_NUM_WORKERS:-4}"
DRIVOR_PREFETCH="${DRIVOR_PREFETCH_FACTOR:-2}"
if ! [[ "${DRIVOR_WORKERS}" =~ ^(0|[1-9][0-9]*)$ && "${DRIVOR_PREFETCH}" =~ ^[1-9][0-9]*$ && "${OMP_NUM_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s\n' 'DRIVOR_NUM_WORKERS 必须非负，DRIVOR_PREFETCH_FACTOR 和 DRIVOR_CPU_THREADS 必须是正整数。' >&2
    exit 2
fi

# 本地图像缺失时，读取同一 trainval 目录下的 OSS 对象；凭据仅从 JSON 加载。
export NAVSIM_OSS_CONFIG_PATH="${NAVSIM_OSS_CONFIG_PATH:-/mnt/human/shixi/human/oss_config.json}"
export NAVSIM_OSS_SENSOR_BLOBS_PATH="${NAVSIM_OSS_SENSOR_BLOBS_PATH-oss://ofasys-multimodal-wlcb-5-toshanghai/anonymous_driving/autonomousvision_navsim/1/download_trainval/openscene-v1.1/sensor_blobs/trainval}"
# 显式设为空字符串可关闭下载缓存。
export NAVSIM_OSS_IMAGE_CACHE_DIR="${NAVSIM_OSS_IMAGE_CACHE_DIR-${NAVSIM_EXP_ROOT}/oss_image_cache}"

# 优先导入本目录的源码，覆盖环境里 vendor/DrivoR 的 editable 路径。
export PYTHONPATH="${DRIVOR_ROOT}:${DRIVOR_ROOT}/nuplan-devkit${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a DRIVOR_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
# 此处仍在 torchrun 外层：DLC 的 WORLD_SIZE / RANK 对应节点数 / 节点编号。
# torchrun 启动后，训练进程的 WORLD_SIZE 才是全局 GPU 进程数。
DRIVOR_NNODES="${NNODES:-${WORLD_SIZE:-8}}"
DRIVOR_GPUS_PER_NODE="${NPROC_PER_NODE:-${GPU_NUM:-${#DRIVOR_GPU_IDS[@]}}}"
if ! [[ "${DRIVOR_NNODES}" =~ ^[1-9][0-9]*$ && "${DRIVOR_GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s\n' 'NNODES 和 NPROC_PER_NODE 必须是正整数。' >&2
    exit 2
fi
DRIVOR_WORLD_SIZE="$((DRIVOR_NNODES * DRIVOR_GPUS_PER_NODE))"
DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_Nav1_navtrain_traj_long_25epochs}"
# 不使用各节点独立生成的时间戳，保证 Hydra/Lightning 的输出目录一致。
DRIVOR_RUN_ID="${DRIVOR_RUN_ID:-${PAI_JOB_ID:-${SLURM_JOB_ID:-multinode}}}"

# Hydra 配置/帮助查询直接运行单进程，不进入 torchrun 的跨机 rendezvous。
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
        printf '%s\n' '此脚本内部已调用 torchrun；请让 DLC 每节点执行一次 shell 命令，不要在 torchrun worker 内再次启动。' >&2
        exit 2
    fi
    DRIVOR_NODE_RANK="${NODE_RANK:-${RANK:-${SLURM_NODEID:-}}}"
    if [[ "${DRIVOR_NNODES}" == 1 ]]; then
        DRIVOR_NODE_RANK="${DRIVOR_NODE_RANK:-0}"
        export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    else
        : "${MASTER_ADDR:?未检测到 MASTER_ADDR；请通过 DLC 的 PyTorch 分布式任务提交，平台会自动注入节点信息}"
        : "${DRIVOR_NODE_RANK:?未检测到 DLC 的 RANK（或 NODE_RANK）；请使用 PyTorch 分布式任务启动}"
    fi
    if ! [[ "${DRIVOR_NODE_RANK}" =~ ^(0|[1-9][0-9]*)$ ]] || (( DRIVOR_NODE_RANK >= DRIVOR_NNODES )); then
        printf 'NODE_RANK 必须在 0 到 %s 之间。\n' "$((DRIVOR_NNODES - 1))" >&2
        exit 2
    fi
    export MASTER_PORT="${MASTER_PORT:-29500}"
    DRIVOR_LAUNCH+=(
        -m torch.distributed.run
        "--nnodes=${DRIVOR_NNODES}"
        "--nproc_per_node=${DRIVOR_GPUS_PER_NODE}"
        "--node_rank=${DRIVOR_NODE_RANK}"
        "--master_addr=${MASTER_ADDR}"
        "--master_port=${MASTER_PORT}"
        --rdzv_backend=static
        "--rdzv_id=${DRIVOR_RUN_ID}"
    )
    printf 'DrivoR: node %s/%s, GPUs/node=%s, world_size=%s, master=%s:%s, run=%s\n' \
        "${DRIVOR_NODE_RANK}" "${DRIVOR_NNODES}" "${DRIVOR_GPUS_PER_NODE}" \
        "${DRIVOR_WORLD_SIZE}" "${MASTER_ADDR}" "${MASTER_PORT}" "${DRIVOR_RUN_ID}" >&2
    printf 'NAVSIM data=%s; metric cache=%s\n' "${OPENSCENE_DATA_ROOT}" "${NAVSIM_METRIC_CACHE_PATH}" >&2
    printf 'Launcher defaults (Hydra arguments may override): CPU threads/process=%s; data workers/rank=%s; prefetch/worker=%s; precision=%s\n' \
        "${OMP_NUM_THREADS}" "${DRIVOR_WORKERS}" "${DRIVOR_PREFETCH}" "${DRIVOR_PRECISION:-bf16-mixed}" >&2
    printf 'Per-node DataLoader workers=%s per loader; oracle workers/rank=%s; diagnostics=%s\n' \
        "$((DRIVOR_GPUS_PER_NODE * DRIVOR_WORKERS))" "${NAVSIM_ORACLE_WORKERS:-1}" \
        "${DRIVOR_PERF_ENABLED:-false}" >&2
fi

# YAML 中的 weights/... 相对于 DrivoR 根目录解析。
cd -- "${DRIVOR_ROOT}"

exec "${DRIVOR_LAUNCH[@]}" "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training.py" \
    agent=drivoR \
    "experiment_name=${DRIVOR_EXPERIMENT}" \
    "experiment_uid=${DRIVOR_RUN_ID}" \
    "auto_resume=${DRIVOR_AUTO_RESUME:-true}" \
    train_test_split=navtrain \
    worker=sequential \
    cache_path=null \
    use_cache_without_dataset=false \
    trainer.params.max_epochs=25 \
    trainer.params.check_val_every_n_epoch=1 \
    trainer.params.val_check_interval=1.0 \
    trainer.params.limit_val_batches=1.0 \
    "++trainer.params.devices=${DRIVOR_GPUS_PER_NODE}" \
    "trainer.params.num_nodes=${DRIVOR_NNODES}" \
    trainer.params.strategy=ddp \
    trainer.params.accumulate_grad_batches=1 \
    "trainer.params.precision=${DRIVOR_PRECISION:-bf16-mixed}" \
    "dataloader.params.num_workers=${DRIVOR_WORKERS}" \
    "dataloader.params.prefetch_factor=${DRIVOR_PREFETCH}" \
    "runtime.performance_diagnostics.enabled=${DRIVOR_PERF_ENABLED:-false}" \
    "runtime.performance_diagnostics.sample_every=${DRIVOR_PERF_SAMPLE_EVERY:-25}" \
    "runtime.performance_diagnostics.cuda_events=${DRIVOR_PERF_CUDA_EVENTS:-false}" \
    "runtime.require_resume=${DRIVOR_REQUIRE_RESUME:-false}" \
    "runtime.rank_seeded_training=${DRIVOR_RANK_SEEDED_TRAINING:-true}" \
    "runtime.console_log_interval=${DRIVOR_CONSOLE_LOG_INTERVAL:-25}" \
    "runtime.validation_fp32=${DRIVOR_VAL_FP32:-true}" \
    dataloader.params.persistent_workers=true \
    dataloader.params.pin_memory=true \
    dataloader.params.batch_size=1 \
    agent.lr_args.name=AdamW \
    agent.lr_args.base_lr=0.0002 \
    "+agent.lr_args.fused=${DRIVOR_FUSED_ADAMW:-true}" \
    "agent.num_gpus=${DRIVOR_WORLD_SIZE}" \
    agent.progress_bar=false \
    agent.config.refiner_ls_values=0.0 \
    agent.config.image_backbone.focus_front_cam=false \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.loss.prev_weight=0.0 \
    agent.config.long_trajectory_additional_poses=2 \
    seed=2 \
    "$@"
