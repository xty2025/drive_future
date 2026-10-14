#!/usr/bin/env bash
# =============================================================================
# PAI-DLC single-node launcher -- the mandated two-stage pipeline
# =============================================================================
#   STAGE 0 (once)   build_future_target_cache.py   -> compressor.pt + targets/
#   STAGE 1          run_stage1_multiview.py        -> stage1_checkpoint.pt
#   STAGE 2 (joint)  run_navtrain_wan_e2e_ddp.py    -> checkpoint.pt
#
# Stage 1 learns the future representation only: clean (sigma = 0) Multi-GT
# trajectories in, 4 compressed register tokens out, theta frozen.  Stage 2
# restores that checkpoint and trains eta + theta + psi jointly, with the anchor
# prior denoised at a randomly sampled flow timestep per forward.
#
# Stage 0 must run before Stage 1/2 can use --future-target cache: Stage 2 needs
# the same compressor.pt Stage 1 fitted, or its register targets are not in the
# basis the model was trained to predict.
#
# Topology : Stage 0/2 = 1 node x 16 GPU (torch.distributed.run --standalone);
#            Stage 1 is single-process.
# Launcher : python -m torch.distributed.run, NEVER the `torchrun` shim -- its
#            shebang is broken on this image and falls back to a system python
#            whose timm has no dinov3 weights.
#
# Skip a stage by exporting NVS_SKIP_STAGE0=1 / NVS_SKIP_STAGE1=1 / NVS_SKIP_STAGE2=1.
# Override ANY path by exporting the matching NVS_* var before running.
# =============================================================================

# NOTE: no `set -euo pipefail` here on purpose. The PPU envsetup.sh and the venv
# activate script reference unbound vars and can return non-zero; a global -e/-u
# would abort the launcher for harmless reasons. We preflight explicitly instead.
set -o pipefail

# ---- paths (all overridable via env; defaults = the mounts used locally) ----- #
PROJECT_ROOT="${NVS_PROJECT_ROOT:-/mnt/zhaozc_workspace/project/NVS}"
PYBIN="${NVS_PYTHON:-/mnt/zhaozc_workspace/envs/driov_dpo/bin/python}"
PPU_ENV="${NVS_PPU_ENV:-/usr/local/PPU_SDK/envsetup.sh}"
VENV_ACTIVATE="${NVS_VENV_ACTIVATE:-/mnt/zhaozc_workspace/envs/driov_dpo/bin/activate}"

export NVS_MULTIGT_ROOT="${NVS_MULTIGT_ROOT:-/mnt/zhaozc_workspace/project/drivezero_goal_gen/outputs/navsim_multigt_0p5s_8step_16gpu}"
export NVS_MULTIVIEW_ROOT="${NVS_MULTIVIEW_ROOT:-/mnt/zhaozc_workspace/project/multitra2egoview/cosmos3/batch_native10/rollouts/single_gpu_test_gtfallback_v2}"
export NVS_SENSOR_ROOT="${NVS_SENSOR_ROOT:-/mnt/data/navsim/trainval_all/trainval_sensor_blobs/trainval}"
export NVS_LOG_ROOT="${NVS_LOG_ROOT:-/mnt/data/navsim/trainval_navsim_logs/trainval}"
export NVS_CACHE_ROOT="${NVS_CACHE_ROOT:-/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtrain_20261005/metric_cache}"
export NVS_DINO_CHECKPOINT="${NVS_DINO_CHECKPOINT:-/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors}"
export NVS_WAN_BACKBONE="${NVS_WAN_BACKBONE:-${PROJECT_ROOT}/checkpoints/ActionDiT_Wan22_256hdim_3outdim.pt}"

NVS_OUTPUT_ROOT="${NVS_OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts}"
TARGET_CACHE_ROOT="${NVS_TARGET_CACHE_ROOT:-${NVS_OUTPUT_ROOT}/future_target_cache}"
STAGE1_DIR="${NVS_STAGE1_DIR:-${NVS_OUTPUT_ROOT}/stage1_multiview}"
STAGE2_DIR="${NVS_STAGE2_DIR:-${NVS_OUTPUT_ROOT}/navtrain_wan_e2e_ddp}"
STAGE1_CKPT="${NVS_STAGE1_CHECKPOINT:-${STAGE1_DIR}/stage1_checkpoint.pt}"

# ---- Stage 1 / Stage 2 scale (env-overridable) ------------------------------ #
# Stage 1 holds every scene resident as float32 RGB, so --scenes is RAM-bound, not
# GPU-bound: future_rgb is MAX_GT(4) x 8 frames x 3 x 464 x 832 x 4B = 148 MB and
# rgb_history is 5 x 3 x 672 x 1152 x 4B = 46 MB, i.e. ~195 MB per scene.  64
# scenes is ~12.5 GB, which fits this 50 GB host with room for the Teacher; 256
# would need ~50 GB and will not.  Stage 2 has no such limit -- it reads the
# compressed [K,4,256] cache (~66 KB/scene) instead of pixels.
STAGE1_SCENES="${NVS_STAGE1_SCENES:-64}"
STAGE1_EVAL_SCENES="${NVS_STAGE1_EVAL_SCENES:-8}"
STAGE1_STEPS="${NVS_STAGE1_STEPS:-20000}"
STAGE1_BATCH="${NVS_STAGE1_BATCH:-4}"
NPROC="${NVS_NPROC_PER_NODE:-16}"
STAGE2_BATCH="${NVS_STAGE2_BATCH:-12}"
STAGE2_EPOCHS="${NVS_STAGE2_EPOCHS:-50}"

# ---- preflight: fail fast with a clear, actionable message ----------------- #
missing=0
for p in "$PROJECT_ROOT" "$PYBIN" "$PPU_ENV" "$VENV_ACTIVATE" \
         "$NVS_MULTIGT_ROOT" "$NVS_MULTIVIEW_ROOT" "$NVS_SENSOR_ROOT" \
         "$NVS_LOG_ROOT" "$NVS_CACHE_ROOT" "$NVS_DINO_CHECKPOINT"; do
  if [ ! -e "$p" ]; then echo "[PREFLIGHT] MISSING: $p" >&2; missing=1; fi
done
if [ -z "${NVS_SKIP_STAGE2:-}" ] && [ ! -e "$STAGE1_CKPT" ] && [ -n "${NVS_SKIP_STAGE1:-}" ]; then
  echo "[PREFLIGHT] MISSING: $STAGE1_CKPT (Stage 1 is skipped but Stage 2 restores it)" >&2
  missing=1
fi
if [ "$missing" -ne 0 ]; then
  echo "[PREFLIGHT] Aborting. Export the matching NVS_* env var(s) to your DLC mount path(s) and re-run." >&2
  exit 2
fi
if [ -z "${NVS_SKIP_STAGE1:-}" ]; then
  # ~195 MB of resident float32 RGB per scene (see the scale block above).
  need_gb=$(( (STAGE1_SCENES + STAGE1_EVAL_SCENES) * 195 / 1024 ))
  have_gb=$(awk '/MemTotal/ {printf "%d", $2 / 1048576}' /proc/meminfo)
  echo "[PREFLIGHT] Stage 1 resident RGB estimate: ~${need_gb} GB for $((STAGE1_SCENES + STAGE1_EVAL_SCENES)) scenes; host MemTotal ${have_gb} GB"
  if [ "$need_gb" -gt $((have_gb / 2)) ]; then
    echo "[PREFLIGHT] Aborting: that leaves under half the host RAM for the Teacher, the DataLoader and the page cache." >&2
    echo "[PREFLIGHT] Lower NVS_STAGE1_SCENES (Stage 1 is RAM-bound) or raise the host memory." >&2
    exit 2
  fi
fi
if [ ! -e "$NVS_WAN_BACKBONE" ]; then
  # Not fatal: both stages fall back to a random theta and log a warning.  Say so
  # loudly, because a random-init ActionDiT will not reach the anchor grid's
  # oracle-best PDMS and the run would look like a modelling failure instead.
  echo "[PREFLIGHT] WARNING: Wan backbone $NVS_WAN_BACKBONE not found; theta starts random." >&2
  echo "[PREFLIGHT]          Build it with: $PYBIN scripts/preprocess_wan_action_dit.py" >&2
fi

# ---- runtime env ----------------------------------------------------------- #
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
# Single-node --standalone rendezvous is authoritative; drop any job-level
# distributed vars DLC may have injected so they cannot confuse the agent.
unset MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK

mkdir -p "$TARGET_CACHE_ROOT" "$STAGE1_DIR" "$STAGE2_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "[LAUNCH] two-stage e2e | nproc=${NPROC} | stage2 batch/GPU=${STAGE2_BATCH} => global $((STAGE2_BATCH * NPROC))"
echo "[LAUNCH] target_cache=${TARGET_CACHE_ROOT}"
echo "[LAUNCH] stage1=${STAGE1_DIR}  stage2=${STAGE2_DIR}"
echo "[LAUNCH] python=$("$PYBIN" -c "import sys,timm; print(sys.executable, timm.__version__)")"

# ---- STAGE 0: Future Register target cache -------------------------------- #
if [ -z "${NVS_SKIP_STAGE0:-}" ]; then
  if [ -e "${TARGET_CACHE_ROOT}/compressor.pt" ] && [ -z "${NVS_FORCE_STAGE0:-}" ]; then
    echo "[STAGE0] compressor.pt already present; skipping (set NVS_FORCE_STAGE0=1 to rebuild)"
  else
    LOG0="${TARGET_CACHE_ROOT}/build_${STAMP}.log"
    echo "[STAGE0] building Future Register targets | log=${LOG0}"
    "$PYBIN" -m torch.distributed.run \
      --standalone --nnodes=1 --nproc_per_node="$NPROC" \
      scripts/build_future_target_cache.py \
      --target-cache-root "$TARGET_CACHE_ROOT" \
      --multiview-root "$NVS_MULTIVIEW_ROOT" \
      --log-root "$NVS_LOG_ROOT" \
      --sensor-root "$NVS_SENSOR_ROOT" \
      --cache-root "$NVS_CACHE_ROOT" \
      --dino-checkpoint "$NVS_DINO_CHECKPOINT" \
      --target-dim 256 \
      --skip-degenerate \
      ${NVS_STAGE0_EXTRA_ARGS:-} \
      2>&1 | tee "$LOG0"
    status=${PIPESTATUS[0]}
    echo "[STAGE0] finished with exit=${status}"
    [ "$status" -ne 0 ] && exit "$status"
  fi
else
  echo "[STAGE0] skipped (NVS_SKIP_STAGE0 set)"
fi

# ---- STAGE 1: future representation only ---------------------------------- #
if [ -z "${NVS_SKIP_STAGE1:-}" ]; then
  LOG1="${STAGE1_DIR}/train_${STAMP}.log"
  echo "[STAGE1] future representation | scenes=${STAGE1_SCENES} steps=${STAGE1_STEPS} | log=${LOG1}"
  "$PYBIN" -u scripts/run_stage1_multiview.py \
    --multiview-root "$NVS_MULTIVIEW_ROOT" \
    --log-root "$NVS_LOG_ROOT" \
    --sensor-root "$NVS_SENSOR_ROOT" \
    --cache-root "$NVS_CACHE_ROOT" \
    --dino-checkpoint "$NVS_DINO_CHECKPOINT" \
    --wan-backbone "$NVS_WAN_BACKBONE" \
    --scenes "$STAGE1_SCENES" \
    --eval-scenes "$STAGE1_EVAL_SCENES" \
    --batch-size "$STAGE1_BATCH" \
    --steps "$STAGE1_STEPS" \
    --target-dim 256 \
    --depth 8 \
    --heads 4 \
    --candidates 64 \
    --skip-degenerate \
    --grad-checkpointing \
    --output-dir "$STAGE1_DIR" \
    ${NVS_STAGE1_EXTRA_ARGS:-} \
    2>&1 | tee "$LOG1"
  status=${PIPESTATUS[0]}
  echo "[STAGE1] finished with exit=${status}"
  [ "$status" -ne 0 ] && exit "$status"
else
  echo "[STAGE1] skipped (NVS_SKIP_STAGE1 set)"
fi

# ---- STAGE 2: joint generator + scorer ------------------------------------ #
if [ -z "${NVS_SKIP_STAGE2:-}" ]; then
  if [ ! -e "$STAGE1_CKPT" ]; then
    echo "[STAGE2] MISSING $STAGE1_CKPT -- Stage 2 must restore Stage 1's eta+phi+theta." >&2
    exit 4
  fi
  FUTURE_ARGS=(--future-target none)
  if [ -e "${TARGET_CACHE_ROOT}/compressor.pt" ]; then
    # Stage 2 continues to align the register to the same compressed future Stage 1
    # was trained on, at the auxiliary scale FastWAM's ema_joint uses.
    FUTURE_ARGS=(--future-target cache --target-cache-root "$TARGET_CACHE_ROOT" --lambda-future 0.1)
  else
    echo "[STAGE2] WARNING: no compressor.pt under ${TARGET_CACHE_ROOT}; running with --future-target none" >&2
    echo "[STAGE2]          phi still runs (Stage 1 weights) but receives no future gradient." >&2
  fi
  LOG2="${STAGE2_DIR}/train_${STAMP}.log"
  echo "[STAGE2] joint eta+theta+psi | stage1=${STAGE1_CKPT} | future=${FUTURE_ARGS[*]} | log=${LOG2}"
  "$PYBIN" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    scripts/run_navtrain_wan_e2e_ddp.py \
    --stage1-checkpoint "$STAGE1_CKPT" \
    "${FUTURE_ARGS[@]}" \
    --multigt-root "$NVS_MULTIGT_ROOT" \
    --sensor-root "$NVS_SENSOR_ROOT" \
    --log-root "$NVS_LOG_ROOT" \
    --cache-root "$NVS_CACHE_ROOT" \
    --dino-checkpoint "$NVS_DINO_CHECKPOINT" \
    --wan-backbone "$NVS_WAN_BACKBONE" \
    --target-dim 256 \
    --batch-size "$STAGE2_BATCH" \
    --epochs "$STAGE2_EPOCHS" \
    --grad-accum 1 \
    --workers 4 \
    --lr 2e-4 \
    --scorer-lr 5e-4 \
    --weight-decay 0.05 \
    --warmup 500 \
    --lambda-action 1.0 \
    --lambda-velocity 1.0 \
    --lambda-score 1.0 \
    --lambda-rank 0.1 \
    --dense-weight 0.1 \
    --diversity-weight 1.0 \
    --depth 8 \
    --heads 4 \
    --candidates 64 \
    --seed 17 \
    --log-every 10 \
    --save-every 500 \
    --grad-checkpointing \
    --output-dir "$STAGE2_DIR" \
    ${NVS_STAGE2_EXTRA_ARGS:-} \
    2>&1 | tee "$LOG2"
  status=${PIPESTATUS[0]}
  echo "[STAGE2] joint run finished with exit=${status}"
  exit "$status"
else
  echo "[STAGE2] skipped (NVS_SKIP_STAGE2 set)"
  exit 0
fi
