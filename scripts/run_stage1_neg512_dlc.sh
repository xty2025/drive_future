#!/usr/bin/env bash
# Stage 0(neg) + Stage 1 at d_model=512 with mined negatives merged into the training rows.
#
# Why this script exists rather than a flag on dlc_navtrain_wan_e2e.sh:
#
#   * The negative Future Register targets do not exist yet.  They have to be extracted
#     by the same Teacher over the negative video corpus, and the builder refuses to run
#     without a CUDA/PPU device, so this is a 16-card, multi-hour stage of its own.
#   * It must reuse the POSITIVE cache's standardizer.pt instead of fitting its own.
#     TargetStandardizer.fingerprint() hashes the fitted mean/scale, so two separately
#     fitted caches live in two target spaces: their rows cannot be concatenated behind
#     one future_head, and the Stage-1 loader's per-payload target_fingerprint check
#     would reject every negative scene.  Step 2 below copies the file and then hard-fails
#     (exit 96) if the two fingerprints differ.
#   * dlc_navtrain_wan_e2e.sh hardcodes --target-dim 256 --heads 4, and 256 is too narrow
#     for a 1024-wide register target: future_head is one Linear(d_model -> 1024) shared
#     by all 32 register tokens, so every prediction lives in a fixed d_model-dimensional
#     subspace.  Measured over 384 cached scenes / 49152 target vectors, the best rank-256
#     subspace leaves future_relative_l1 at 0.48773 while the scene-mean anchor sits at
#     0.41216 -- i.e. gain_over_scene_mean_target can never rise above -0.1833, and
#     stage1_eval's learned_trajectory_conditional_future requires +0.02.  Rank 512 moves
#     that ceiling to +0.2665.  The width is injected through NVS_STAGE1_EXTRA_ARGS, which
#     the launcher expands after its own literals, so argparse's last-value-wins overrides
#     them and the launcher itself needs no edit.
#
# Resumable: the builder skips scenes that already have a cached target and reuses its
# _ready_scenes.json sidecar, so re-running this script after an interruption continues.
#
# Knobs: NVS_NPROC_PER_NODE (default 16, must equal the node's device count),
# NVS_STAGE1_EPOCHS (default 20; lowering it shortens the run but changes the cosine
# schedule's total step count, which breaks comparability with the 20-epoch baseline).
set -o pipefail

cd /mnt/zhaozc_workspace/project/NVS || exit 99
PROJECT_ROOT=$PWD
PYBIN=/mnt/zhaozc_workspace/envs/driov_dpo/bin/python
PPU_ENV=/usr/local/PPU_SDK/envsetup.sh
VENV_ACTIVATE=/mnt/zhaozc_workspace/envs/driov_dpo/bin/activate
NPROC="${NVS_NPROC_PER_NODE:-16}"
POS_CACHE=$PROJECT_ROOT/artifacts/future_target_cache_reg32_full
NEG_CACHE=$PROJECT_ROOT/artifacts/future_target_cache_reg32_neg
NEG_VIDEO=/mnt/data/multi_future_view/navtrain_poor4_full_allcosmos_cuda_v1/navtrain_poor4_full_allcosmos_cuda_v1
LOG_ROOT=/mnt/data/navsim/trainval_navsim_logs/trainval
SENSOR_ROOT=/mnt/data/navsim/trainval_all/trainval_sensor_blobs/trainval
METRIC_CACHE=/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtrain_20261005/metric_cache
DINO=/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors
WAN512=$PROJECT_ROOT/checkpoints/ActionDiT_Wan22_512hdim_3outdim.pt
STAGE1_DIR="${NVS_STAGE1_DIR:-$PROJECT_ROOT/artifacts/stage1_reg32_d512_neg_b12x16_20ep}"
STAMP=$(date +%Y%m%d_%H%M%S)

# --------------------------------------------------------------------------- #
# 1. preflight: fail before burning a 16-card allocation on a missing mount
# --------------------------------------------------------------------------- #
echo "[PREFLIGHT] inputs"
for p in "$PROJECT_ROOT" "$PYBIN" "$PPU_ENV" "$VENV_ACTIVATE" "$POS_CACHE/standardizer.pt" \
         "$NEG_VIDEO" "$LOG_ROOT" "$SENSOR_ROOT" "$METRIC_CACHE" "$DINO" "$WAN512" \
         "$PROJECT_ROOT/scripts/build_future_target_cache.py" "$PROJECT_ROOT/scripts/dlc_navtrain_wan_e2e.sh"; do
  [ -e "$p" ] || { echo "[PREFLIGHT] FATAL missing: $p" >&2; exit 91; }
done
# Same probe the launcher uses: --standalone starts NPROC ranks on however many devices
# exist and dies inside NCCL init minutes later without ever saying "too many ranks".
gpu_count=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU ")
if [ "$gpu_count" -gt 0 ] && [ "$gpu_count" -ne "$NPROC" ]; then
  echo "[PREFLIGHT] FATAL: NPROC=$NPROC but this node exposes $gpu_count device(s)" >&2
  exit 92
fi
echo "[PREFLIGHT] devices=${gpu_count:-unknown} nproc=$NPROC stage1_dir=$STAGE1_DIR"

source "$PPU_ENV" ppu
source "$VENV_ACTIVATE"
export PYTHONPATH=$PROJECT_ROOT/src:$PROJECT_ROOT/vendor/DrivoR:$PROJECT_ROOT/vendor/DrivoR/nuplan-devkit
export NCCL_DEBUG=WARN OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset MASTER_ADDR MASTER_PORT

# --------------------------------------------------------------------------- #
# 2. one target space: reuse the positive standardizer, never refit
# --------------------------------------------------------------------------- #
echo "[STAGE0-NEG] reusing the positive standardizer so both caches share one target space"
mkdir -p "$NEG_CACHE"
if [ ! -e "$NEG_CACHE/standardizer.pt" ]; then
  cp "$POS_CACHE/standardizer.pt" "$NEG_CACHE/standardizer.pt" || exit 97
fi
"$PYBIN" -c "import sys;sys.path.insert(0,sys.argv[1]);from nvs_v6_wan_e2e.target_cache import load_standardizer as L;a,_=L(sys.argv[2]);b,_=L(sys.argv[3]);print(a.fingerprint(),b.fingerprint());sys.exit(0 if a.fingerprint()==b.fingerprint() else 96)" "$PROJECT_ROOT/src" "$POS_CACHE" "$NEG_CACHE" || exit 96

# --------------------------------------------------------------------------- #
# 3. build the negative Future Register cache
# --------------------------------------------------------------------------- #
# The trajectory needs no separate loader: load_multiview_sample derives gt_trajectory
# from each trajectory_i/ego_pose.npy at FUTURE_POSE_INDICES, and on three real negative
# scenes that is bit-identical (max abs diff 0) to the negative label root's
# gt_trajectories.npy[i], which is the join POOR4_DOWNLOAD_README.md documents.
echo "[STAGE0-NEG] building negative future targets into $NEG_CACHE (resumable)"
"$PYBIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$NPROC \
  scripts/build_future_target_cache.py \
  --target-cache-root "$NEG_CACHE" \
  --multiview-root "$NEG_VIDEO" \
  --log-root "$LOG_ROOT" \
  --sensor-root "$SENSOR_ROOT" \
  --cache-root "$METRIC_CACHE" \
  --dino-checkpoint "$DINO" \
  --skip-degenerate 2>&1 | tee "$NEG_CACHE/build_$STAMP.log"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
  echo "[STAGE0-NEG] FATAL: builder exit=$status" >&2
  exit "$status"
fi

neg_payloads=$(find "$NEG_CACHE/targets" -name "*.pt" 2>/dev/null | wc -l)
echo "[STAGE0-NEG] cached locally: $neg_payloads negative payloads under $NEG_CACHE/targets"
if [ "$neg_payloads" -le 1000 ]; then
  echo "[STAGE0-NEG] FATAL: only $neg_payloads payloads; refusing to train on an empty negative cache" >&2
  exit 95
fi

# --------------------------------------------------------------------------- #
# 4. Stage 1 at 512 with negatives merged into the training rows
# --------------------------------------------------------------------------- #
# A fresh NVS_STAGE1_DIR matters: the launcher auto-passes --resume whenever
# ${STAGE1_DIR}/stage1_checkpoint.pt exists, and the trainer loads it with strict=True,
# so pointing at a 256-wide checkpoint dir would abort on shape mismatch.
echo "[STAGE1] d_model=512 heads=8 | negatives in training rows, verdict positive-only"
NVS_NPROC_PER_NODE=$NPROC NVS_SKIP_STAGE0=1 NVS_SKIP_STAGE2=1 \
NVS_WAN_BACKBONE=$WAN512 \
NVS_STAGE1_DIR=$STAGE1_DIR \
NVS_STAGE1_EXTRA_ARGS="--target-dim 512 --heads 8 --negative-target-cache-root $NEG_CACHE" \
  bash scripts/dlc_navtrain_wan_e2e.sh
