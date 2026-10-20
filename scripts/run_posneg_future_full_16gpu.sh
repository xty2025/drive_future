#!/usr/bin/env bash
# =============================================================================
# 16-GPU 完整未来监督训练：正负样本一起当 GT、各自预测自己的未来
#
#   Stage0-pos : 正样本 Future Register cache（rank0 fit standardizer，全 rank 提取）
#   Stage0-neg : 负样本 Future Register cache（复用正样本 standardizer.pt，同一 target 空间）
#   Stage1     : 未来表征训练。负样本 future 行追加进 GT 行，source=gt_trajectory 每行
#                clean action @ sigma=0，identity_future_l1 让第 k 行预测第 k 行自己的未来
#                （负样本行 -> 负样本未来），paired_contrast_future_l1 在正负行之间做差，
#                消掉 scene 背景、只留轨迹差异。verdict 保持 positive-only 以可比。
#   Stage2     : joint 训练。--negative-target-cache-root 让负样本 future 行同样追加进 GT，
#                于是 gather_flow_data / hungarian_xy / multigt_trajectory_loss /
#                dense_future_l1 全部覆盖正负样本：每个 candidate 被拉向它最近的 GT
#                （无论好坏）的未来，模型由此学到"坏轨迹会导致坏未来"。
#                不启用 rd_loss / neg_score —— 负样本已当作 GT 在前面匹配，不再只在
#                score 头里被排斥（这正是用户验证过更优的那条路线）。
#
# 负样本 = drivezero_goal_gen 挖掘的 poor_negative：通过硬门槛（无责碰撞/DAC/TTC 可行）
# 但官方 PDMS 比该场景最弱 top4 GT 低至少一个 margin 的轨迹。其 Cosmos 生成未来视频经
# 同一个 frozen DINOv3 query-readout teacher 提取 register target，与正样本共用 standardizer。
#
# 一切产物都落在本 clone（drive_future_zhaozc_1019）下；NVS 原仓库只读借 WAN backbone，
# 绝不修改。d_model=512/heads=8/depth=8/ffn=2048 与 WAN512 payload 的 meta 完全一致
# （512 宽是 1024-wide register target 所需：256 宽的 future_head 子空间会把
#  gain_over_scene_mean_target 的天花板压到 -0.18，低于 stage1_eval 要求的 +0.02）。
#
# 用法：  bash -lc 'bash scripts/run_posneg_future_full_16gpu.sh'
# 旋钮：  NVS_NPROC_PER_NODE(16)  NVS_BATCH_SIZE(12)
#         NVS_STAGE1_EPOCHS(20)   NVS_STAGE2_EPOCHS(50)
#         NVS_STAGE1_DIR  NVS_STAGE2_DIR
#         NVS_SKIP_STAGE0  NVS_SKIP_STAGE1  NVS_SKIP_STAGE2   （置 1 跳过对应阶段）
#         NVS_STAGE1_EXTRA_ARGS  NVS_STAGE2_EXTRA_ARGS         （追加到对应 trainer 末尾）
# =============================================================================
set -o pipefail

PROJECT_ROOT=/mnt/zhaozc_workspace/project/drive_future_zhaozc_1019
NVS_ROOT=/mnt/zhaozc_workspace/project/NVS            # 只读：借 WAN backbone，绝不修改

# ---- 环境（与 NVS 生产一致）----
PPU_ENV=/usr/local/PPU_SDK/envsetup.sh
VENV_ACTIVATE=/mnt/zhaozc_workspace/envs/driov_dpo/bin/activate
[ -f "$PPU_ENV" ] && source "$PPU_ENV" ppu
[ -f "$VENV_ACTIVATE" ] && source "$VENV_ACTIVATE"
PYBIN="${PYBIN:-python}"

cd "$PROJECT_ROOT" || exit 99
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/vendor/DrivoR:$PROJECT_ROOT/vendor/DrivoR/nuplan-devkit"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${NVS_OMP_THREADS:-4}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset MASTER_ADDR MASTER_PORT

NPROC="${NVS_NPROC_PER_NODE:-16}"

# ---- 数据根（全局共享，只读）----
POS_VIDEO=/mnt/data/multi_future_view/navtrain_full_allcosmos_cuda_v1_gpu64/navtrain_full_allcosmos_cuda_v1_gpu64
NEG_VIDEO=/mnt/data/multi_future_view/navtrain_poor4_full_allcosmos_cuda_v1/navtrain_poor4_full_allcosmos_cuda_v1
LOG_ROOT=/mnt/data/navsim/trainval_navsim_logs/trainval
SENSOR_ROOT=/mnt/data/navsim/trainval_all/trainval_sensor_blobs/trainval
METRIC_CACHE=/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtrain_20261005/metric_cache
MULTIGT_ROOT=/mnt/zhaozc_workspace/project/drivezero_goal_gen/outputs/navsim_multigt_0p5s_8step_16gpu
DINO=/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors

# ---- 本 clone 下的产物 ----
mkdir -p "$PROJECT_ROOT/checkpoints" "$PROJECT_ROOT/artifacts"
# WAN backbone（512 宽）：本 clone 没有，软链 NVS 的（只读借，不改 NVS）。
WAN512="$PROJECT_ROOT/checkpoints/ActionDiT_Wan22_512hdim_3outdim.pt"
[ -e "$WAN512" ] || ln -sf "$NVS_ROOT/checkpoints/ActionDiT_Wan22_512hdim_3outdim.pt" "$WAN512"

POS_CACHE="$PROJECT_ROOT/artifacts/future_target_cache_reg32_full"
NEG_CACHE="$PROJECT_ROOT/artifacts/future_target_cache_reg32_neg"
STAGE1_DIR="${NVS_STAGE1_DIR:-$PROJECT_ROOT/artifacts/stage1_reg32_d512_posneg}"
STAGE2_DIR="${NVS_STAGE2_DIR:-$PROJECT_ROOT/artifacts/stage2_reg32_d512_posneg}"
STAMP=$(date +%Y%m%d_%H%M%S)

DIM_ARGS="--target-dim 512 --heads 8 --depth 8"
BATCH="${NVS_BATCH_SIZE:-12}"
S1_EPOCHS="${NVS_STAGE1_EPOCHS:-20}"
S2_EPOCHS="${NVS_STAGE2_EPOCHS:-50}"

# ---- preflight：宁可在烧 16 卡前失败，也不要 NCCL init 几分钟后才说卡数不对 ----
gpu_count=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU ")
if [ "$gpu_count" -gt 0 ] && [ "$gpu_count" -ne "$NPROC" ]; then
  echo "[PREFLIGHT] FATAL: NPROC=$NPROC 但本节点只有 $gpu_count 张卡（设 NVS_NPROC_PER_NODE=\$gpu_count 可单机自测）" >&2
  exit 92
fi
echo "[PREFLIGHT] devices=${gpu_count:-unknown} nproc=$NPROC batch=$BATCH dim=512 heads=8"
for p in "$PYBIN" "$POS_VIDEO" "$NEG_VIDEO" "$LOG_ROOT" "$SENSOR_ROOT" "$METRIC_CACHE" "$MULTIGT_ROOT" "$DINO" "$WAN512" \
         "$PROJECT_ROOT/scripts/build_future_target_cache.py" \
         "$PROJECT_ROOT/scripts/run_stage1_cache_ddp.py" \
         "$PROJECT_ROOT/scripts/run_navtrain_wan_e2e_ddp.py"; do
  [ -e "$p" ] || { echo "[PREFLIGHT] FATAL missing: $p" >&2; exit 91; }
done

# =============================================================================
# Stage0-pos：正样本 Future Register cache
# =============================================================================
if [ -z "${NVS_SKIP_STAGE0:-}" ] && [ ! -e "$POS_CACHE/standardizer.pt" ]; then
  echo "[STAGE0-POS] 构建正样本 future cache -> $POS_CACHE"
  "$PYBIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$NPROC \
    scripts/build_future_target_cache.py \
    --target-cache-root "$POS_CACHE" \
    --multiview-root "$POS_VIDEO" \
    --log-root "$LOG_ROOT" --sensor-root "$SENSOR_ROOT" --cache-root "$METRIC_CACHE" \
    --dino-checkpoint "$DINO" --skip-degenerate \
    2>&1 | tee "$PROJECT_ROOT/artifacts/stage0_pos_$STAMP.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[STAGE0-POS] FATAL builder exit=${PIPESTATUS[0]}" >&2; exit 93; }
else
  echo "[STAGE0-POS] 复用已存在的 $POS_CACHE"
fi

# =============================================================================
# Stage0-neg：负样本 Future Register cache（复用正样本 standardizer，同一 target 空间）
# =============================================================================
if [ -z "${NVS_SKIP_STAGE0:-}" ]; then
  mkdir -p "$NEG_CACHE"
  if [ ! -e "$NEG_CACHE/standardizer.pt" ]; then
    cp "$POS_CACHE/standardizer.pt" "$NEG_CACHE/standardizer.pt" || exit 97
  fi
  # fingerprint 必须一致：两个 cache 标准化进同一空间，行才能拼在同一个 future_head 后面。
  "$PYBIN" -c "import sys;sys.path.insert(0,sys.argv[1]);from nvs_v6_wan_e2e.target_cache import load_standardizer as L;a,_=L(sys.argv[2]);b,_=L(sys.argv[3]);print('pos',a.fingerprint());print('neg',b.fingerprint());sys.exit(0 if a.fingerprint()==b.fingerprint() else 96)" \
    "$PROJECT_ROOT/src" "$POS_CACHE" "$NEG_CACHE" || { echo "[STAGE0-NEG] FATAL: 正负 cache 的 standardizer fingerprint 不一致" >&2; exit 96; }
  if [ ! -e "$NEG_CACHE/_manifest.json" ]; then
    echo "[STAGE0-NEG] 构建负样本 future cache -> $NEG_CACHE"
    "$PYBIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$NPROC \
      scripts/build_future_target_cache.py \
      --target-cache-root "$NEG_CACHE" \
      --multiview-root "$NEG_VIDEO" \
      --log-root "$LOG_ROOT" --sensor-root "$SENSOR_ROOT" --cache-root "$METRIC_CACHE" \
      --dino-checkpoint "$DINO" --skip-degenerate \
      2>&1 | tee "$PROJECT_ROOT/artifacts/stage0_neg_$STAMP.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[STAGE0-NEG] FATAL builder exit=${PIPESTATUS[0]}" >&2; exit 94; }
  else
    echo "[STAGE0-NEG] 复用已存在的 $NEG_CACHE"
  fi
  neg_payloads=$(find "$NEG_CACHE/targets" -name "*.pt" 2>/dev/null | wc -l)
  echo "[STAGE0-NEG] 负样本 payload 数: $neg_payloads"
  if [ "$neg_payloads" -le 1000 ]; then
    echo "[STAGE0-NEG] FATAL: 负样本 cache 几乎为空（$neg_payloads 个 payload），拒绝训练" >&2; exit 95
  fi
fi

# =============================================================================
# Stage1：未来表征训练（正负样本一起预测各自的未来）
# =============================================================================
if [ -z "${NVS_SKIP_STAGE1:-}" ]; then
  echo "[STAGE1] d_model=512 heads=8 | 正负样本一起预测各自的未来 (identity_future_l1 + paired_contrast)"
  mkdir -p "$STAGE1_DIR"
  "$PYBIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$NPROC \
    scripts/run_stage1_cache_ddp.py \
    --target-cache-root "$POS_CACHE" \
    --negative-target-cache-root "$NEG_CACHE" \
    --output-dir "$STAGE1_DIR" \
    --dino-checkpoint "$DINO" \
    --wan-backbone "$WAN512" \
    $DIM_ARGS \
    --lambda-contrast 1.0 \
    --batch-size "$BATCH" --epochs "$S1_EPOCHS" --workers 4 \
    --lr 2e-4 --warmup 500 --precision bf16 --grad-checkpointing \
    --skip-degenerate \
    --log-every 20 --save-every 500 --seed 17 \
    ${NVS_STAGE1_EXTRA_ARGS:-} \
    2>&1 | tee "$STAGE1_DIR/train_$STAMP.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[STAGE1] FATAL exit=${PIPESTATUS[0]}" >&2; exit 95; }
fi

# =============================================================================
# Stage2：joint 训练（dense_future_l1 + action 匹配都吃正负样本）
# =============================================================================
if [ -z "${NVS_SKIP_STAGE2:-}" ]; then
  echo "[STAGE2] d_model=512 heads=8 | dense_future_l1 + action 匹配都吃正负样本"
  mkdir -p "$STAGE2_DIR"
  S1_CKPT="$STAGE1_DIR/stage1_checkpoint.pt"
  [ -e "$S1_CKPT" ] || { echo "[STAGE2] FATAL: 缺少 Stage1 checkpoint $S1_CKPT（先跑 Stage1）" >&2; exit 96; }
  "$PYBIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$NPROC \
    scripts/run_navtrain_wan_e2e_ddp.py \
    --future-target cache \
    --target-cache-root "$POS_CACHE" \
    --negative-target-cache-root "$NEG_CACHE" \
    --lambda-future 0.1 \
    --stage1-checkpoint "$S1_CKPT" \
    --output-dir "$STAGE2_DIR" \
    --dino-checkpoint "$DINO" \
    --wan-backbone "$WAN512" \
    $DIM_ARGS --candidates 64 \
    --multigt-root "$MULTIGT_ROOT" \
    --sensor-root "$SENSOR_ROOT" --log-root "$LOG_ROOT" --cache-root "$METRIC_CACHE" \
    --batch-size "$BATCH" --epochs "$S2_EPOCHS" --grad-accum 1 --workers 4 \
    --lr 2e-4 --scorer-lr 5e-4 --weight-decay 0.05 --warmup 500 \
    --lambda-action 1.0 --lambda-velocity 1.0 --lambda-score 1.0 --lambda-rank 0.1 \
    --dense-weight 0.1 --diversity-weight 1.0 \
    --precision bf16 --grad-checkpointing --seed 17 \
    --log-every 10 --save-every 500 \
    ${NVS_STAGE2_EXTRA_ARGS:-} \
    2>&1 | tee "$STAGE2_DIR/train_$STAMP.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[STAGE2] FATAL exit=${PIPESTATUS[0]}" >&2; exit 97; }
fi

echo "[DONE] Stage1=$STAGE1_DIR  Stage2=$STAGE2_DIR"
