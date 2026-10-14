#!/usr/bin/env bash
# =============================================================================
# PAI-DLC 16-GPU RESUME launcher  --  Wan ActionDiT e2e, ALL-NATIVE protocol
# =============================================================================
# 用途：复用已经跑完的 Phase A（generator.pt）和已建好的 100GB Phase-B cache，
#       跳过 Phase A（~9.3h）和 cache build（~48min），只跑：
#         STAGE 1R (resume-cache)  run_navtrain_wan_e2e_twophase_ddp.py --resume-cache
#                                  -> load generator.pt + 16 shard + 现取 gt(load_navtrain_gt)
#                                  -> Phase B: scorer-ONLY (psi), 感知冻结, 50 epochs
#                                  -> 生成 ${STAGED_DIR}/checkpoint.pt
#         STAGE 2  (joint)         run_navtrain_wan_e2e_ddp.py --resume checkpoint.pt
#                                  -> 感知梯度打开, eta+theta+psi 联合微调, 50 epochs
#
# 背景：上一次完整跑（dlc_navtrain_wan_e2e_allnative.sh）在 Phase B 崩溃，
#       根因是 run_navtrain_wan_e2e_twophase_ddp.py 首次运行路径读完 cache 后
#       未给 shard 注入 gt/gt_ok（只有 --resume-cache 路径注入了），导致
#       phase_b 读 shard['gt'] -> KeyError，16 卡全崩。该 bug 已修复（首次路径
#       现与 --resume-cache 路径逐字一致地调用 load_navtrain_gt 注入 gt）。
#       本脚本直接用已修复的 --resume-cache 路径复用成果，无需重跑 Phase A/cache。
#
# 前置：${STAGED_DIR}/generator.pt 和 ${STAGED_DIR}/phase_b_cache/shard_0..15.pt
#       必须存在（本脚本 preflight 会检查）。恢复必须用与建 cache 相同的 16 卡，
#       因为 --resume-cache 多卡路径是每个 rank 读自己的 shard_{rank}.pt。
#
# 拓扑：1 node x 16 GPU, per-GPU batch 12 => global batch 192。
# 启动：python -m torch.distributed.run --standalone --nnodes=1（不要用 torchrun shim）。
# =============================================================================

set -o pipefail

# ---- paths (all overridable via env) --------------------------------------- #
PROJECT_ROOT="${NVS_PROJECT_ROOT:-/mnt/zhaozc_workspace/project/NVS}"
PYBIN="${NVS_PYTHON:-/mnt/zhaozc_workspace/envs/driov_dpo/bin/python}"
PPU_ENV="${NVS_PPU_ENV:-/usr/local/PPU_SDK/envsetup.sh}"
VENV_ACTIVATE="${NVS_VENV_ACTIVATE:-/mnt/zhaozc_workspace/envs/driov_dpo/bin/activate}"

export NVS_MULTIGT_ROOT="${NVS_MULTIGT_ROOT:-/mnt/zhaozc_workspace/project/drivezero_goal_gen/outputs/navsim_multigt_0p5s_8step_16gpu}"
export NVS_SENSOR_ROOT="${NVS_SENSOR_ROOT:-/mnt/data/navsim/trainval_all/trainval_sensor_blobs/trainval}"
export NVS_LOG_ROOT="${NVS_LOG_ROOT:-/mnt/data/navsim/trainval_navsim_logs/trainval}"
export NVS_CACHE_ROOT="${NVS_CACHE_ROOT:-/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtrain_20261005/metric_cache}"
export NVS_DINO_CHECKPOINT="${NVS_DINO_CHECKPOINT:-/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors}"

STAGED_DIR="${NVS_STAGED_DIR:-${PROJECT_ROOT}/artifacts/navtrain_wan_e2e_twophase_ddp}"
# 新目录：避免覆盖正在跑的“回归版”Joint（artifacts/navtrain_wan_e2e_ddp）。
JOINT_DIR="${NVS_OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/navtrain_wan_e2e_ddp_gentle}"
CACHE_DIR="${STAGED_DIR}/phase_b_cache"
GEN_CKPT="${STAGED_DIR}/generator.pt"

# ---- preflight: 通用路径 + 恢复专用产物 ------------------------------------ #
missing=0
for p in "$PROJECT_ROOT" "$PYBIN" "$PPU_ENV" "$VENV_ACTIVATE" \
         "$NVS_MULTIGT_ROOT" "$NVS_SENSOR_ROOT" "$NVS_LOG_ROOT" \
         "$NVS_CACHE_ROOT" "$NVS_DINO_CHECKPOINT" "$GEN_CKPT" "$CACHE_DIR"; do
  if [ ! -e "$p" ]; then echo "[PREFLIGHT] MISSING: $p" >&2; missing=1; fi
done
shard_count=$(ls -1 "${CACHE_DIR}"/shard_*.pt 2>/dev/null | wc -l)
if [ "$shard_count" -eq 0 ]; then
  echo "[PREFLIGHT] MISSING: no shard_*.pt under ${CACHE_DIR}" >&2; missing=1
fi
if [ "$missing" -ne 0 ]; then
  echo "[PREFLIGHT] Aborting. 需要已完成的 generator.pt 和 phase_b_cache/shard_*.pt。" >&2
  echo "[PREFLIGHT] 若要从头重跑（Phase A + cache + Phase B + Joint），改用 dlc_navtrain_wan_e2e_allnative.sh。" >&2
  exit 2
fi

# ---- runtime env ------------------------------------------------------------ #
# shellcheck disable=SC1090
source "$PPU_ENV" ppu
# shellcheck disable=SC1090
source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT" || { echo "[PREFLIGHT] cannot cd to $PROJECT_ROOT" >&2; exit 3; }

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/vendor/DrivoR:${PROJECT_ROOT}/vendor/DrivoR/nuplan-devkit"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${NVS_OMP_THREADS:-4}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK

NPROC="${NVS_NPROC_PER_NODE:-16}"
BATCH="${NVS_BATCH:-12}"
EPOCHS_B="${NVS_PHASE_B_EPOCHS:-50}"
EPOCHS_JOINT="${NVS_JOINT_EPOCHS:-50}"
LAMBDA_RANK="${NVS_LAMBDA_RANK:-0.1}"
mkdir -p "$STAGED_DIR" "$JOINT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_A="${STAGED_DIR}/train_resumeB_${TS}.log"
LOG_J="${JOINT_DIR}/train_joint_${TS}.log"

echo "[RESUME] e2e ALL-NATIVE resume | nproc=${NPROC} | batch/GPU=${BATCH} | global_batch=$((BATCH * NPROC)) | shards=${shard_count}"
echo "[RESUME] reuse generator.pt=${GEN_CKPT}"
echo "[RESUME] reuse cache=${CACHE_DIR} (${shard_count} shards)"
echo "[RESUME] epochs: PhaseB(score-only)=${EPOCHS_B} (lambda_rank=${LAMBDA_RANK}) -> Joint(perception-grad-open)=${EPOCHS_JOINT}"
echo "[RESUME] stage1R log=${LOG_A}"
echo "[RESUME] stage2  log=${LOG_J}"
echo "[RESUME] python=$("$PYBIN" -c "import sys,timm; print(sys.executable, timm.__version__)")"

# ==================== STAGE 1R: resume-cache -> Phase B ====================== #
# 若 Phase B 已产出 checkpoint.pt（上次 resume 已成功，selected_pdms~0.978），
# 默认跳过 Phase B，直接进 Joint。要强制重跑 Phase B：export NVS_FORCE_PHASE_B=1。
if [ -f "${STAGED_DIR}/checkpoint.pt" ] && [ "${NVS_FORCE_PHASE_B:-0}" != "1" ]; then
  echo "[RESUME] found ${STAGED_DIR}/checkpoint.pt -> SKIP Phase B, go straight to Joint."
  echo "[RESUME] (force re-run Phase B with: export NVS_FORCE_PHASE_B=1)"
else
"$PYBIN" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  scripts/run_navtrain_wan_e2e_twophase_ddp.py \
  --use-all-native-tokens \
  --resume-cache \
  --phase-b-epochs "$EPOCHS_B" \
  --batch-size "$BATCH" \
  --grad-accum 1 \
  --workers 4 \
  --lr 2e-4 \
  --scorer-lr 5e-4 \
  --weight-decay 0.05 \
  --warmup 500 \
  --lambda-action 1.0 \
  --lambda-score 1.0 \
  --lambda-rank "$LAMBDA_RANK" \
  --future-target none \
  --depth 8 \
  --candidates 64 \
  --seed 17 \
  --log-every 10 \
  --save-every 500 \
  --output-dir "$STAGED_DIR" \
  ${NVS_STAGED_EXTRA_ARGS:-} \
  2>&1 | tee "$LOG_A"
status_a=${PIPESTATUS[0]}
echo "[RESUME] STAGE1R Phase B finished with exit=${status_a}"
if [ "$status_a" -ne 0 ]; then
  echo "[RESUME] STAGE1R failed; NOT starting the joint finetune." >&2
  exit "$status_a"
fi
if [ ! -f "${STAGED_DIR}/checkpoint.pt" ]; then
  echo "[RESUME] STAGE1R produced no ${STAGED_DIR}/checkpoint.pt; cannot resume joint finetune." >&2
  exit 4
fi
fi

# ======================= STAGE 2: joint finetune ============================ #
# 方案A（对齐 DrivoR 感知容量，目标超过 Phase B 的 0.978）：
#   上一版温和联合用 lambda_score=0.2 防回归，但 scorer utility 的训练信号被压到
#   DrivoR 的 1/5，sel 平在 0.95 上不去。根因：我们比 DrivoR 多训了大容量的
#   scene_blocks（8 层 1007-token self-attn），未 detach 的 source_loss 会把它冲离
#   Phase A 学到的 GT 表征（最早 sel 跌到 0.85 的根因）。
#   方案A：在 run_navtrain_wan_e2e_ddp.py 的 build_optimizers 里冻结 scene_blocks，
#   感知可训容量回到 DrivoR 的 LoRA 级别，于是 source 可安全提回 DrivoR 的
#   final_score_weight=1.0，让 scorer utility 满强度训练。rank 保持 0.1（=DrivoR
#   preference_weight 1.0 * preference_rank_weight 0.1）。lr 仍 2e-5 保护已收敛 eta。
#   回退到温和联合：把下面 --lambda-score 改回 0.2，并删除 build_optimizers 里冻结
#   scene_blocks 的 for 循环（详见 AGENTS.md「方案A」节）。
"$PYBIN" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  scripts/run_navtrain_wan_e2e_ddp.py \
  --use-all-native-tokens \
  --resume "${STAGED_DIR}/checkpoint.pt" \
  --batch-size "$BATCH" \
  --epochs "$EPOCHS_JOINT" \
  --grad-accum 1 \
  --workers 4 \
  --lr 2e-5 \
  --scorer-lr 5e-4 \
  --weight-decay 0.05 \
  --warmup 500 \
  --lambda-action 1.0 \
  --lambda-score 1.0 \
  --lambda-rank "$LAMBDA_RANK" \
  --unsafe-threshold 0.2 \
  --future-target none \
  --depth 8 \
  --candidates 64 \
  --seed 17 \
  --log-every 10 \
  --save-every 500 \
  --output-dir "$JOINT_DIR" \
  ${NVS_EXTRA_ARGS:-} \
  2>&1 | tee "$LOG_J"
status_j=${PIPESTATUS[0]}
echo "[RESUME] STAGE2 joint finetune finished with exit=${status_j}"
exit "$status_j"
