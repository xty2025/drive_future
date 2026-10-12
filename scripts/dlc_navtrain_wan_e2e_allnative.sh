#!/usr/bin/env bash
# =============================================================================
# PAI-DLC single-node 16-GPU launcher  --  Wan ActionDiT e2e, ALL-NATIVE protocol
# =============================================================================
# Retrains FROM SCRATCH with the DrivoR-exact end-to-end scorer package
# (nvs_v6_wan_e2e: Scene K/V is NOT detached, so the score loss shapes
# perception/eta) and the FULL all-native Scene K/V (5x201+2 = 1007 tokens,
# every DINOv3 token retained instead of the compressed 66).
#
# Protocol (user: "先分阶段再做score的训练，最后联合进行微调"):
#   STAGE 1 (staged)  run_navtrain_wan_e2e_twophase_ddp.py
#                     Phase A: trajectory generator (eta+theta), phi frozen.
#                     Phase B: scorer-only (psi) on a frozen per-scene cache.
#   STAGE 2 (joint)   run_navtrain_wan_e2e_ddp.py --resume <stage1>/checkpoint.pt
#                     eta+theta+psi trained TOGETHER; the un-detached score loss
#                     backprops into perception.  This is the "联合微调".
#
# Topology : 1 node x 16 GPU, per-GPU batch 12 => global batch 192.
#            All-native (1007-token scene self-attention) is ~heavier than the
#            66-token run; measured ~23GB/GPU at batch 4 locally, so batch 12
#            fits a 96GB card with margin.  Lower NVS_BATCH if you see OOM.
# Launcher : python -m torch.distributed.run --standalone --nnodes=1
#            (NEVER the `torchrun` shim: broken shebang on this image.)
#
# Override ANY path on DLC by exporting the NVS_* vars before running.
# =============================================================================

# NOTE: no `set -euo pipefail` on purpose (PPU envsetup / venv activate reference
# unbound vars). We preflight explicitly instead.
set -o pipefail

# ---- paths (all overridable via env; defaults = the mounts used locally) ----- #
PROJECT_ROOT="${NVS_PROJECT_ROOT:-/mnt/zhaozc_workspace/project/NVS}"
PYBIN="${NVS_PYTHON:-/mnt/zhaozc_workspace/envs/driov_dpo/bin/python}"
PPU_ENV="${NVS_PPU_ENV:-/usr/local/PPU_SDK/envsetup.sh}"
VENV_ACTIVATE="${NVS_VENV_ACTIVATE:-/mnt/zhaozc_workspace/envs/driov_dpo/bin/activate}"

export NVS_MULTIGT_ROOT="${NVS_MULTIGT_ROOT:-/mnt/zhaozc_workspace/project/drivezero_goal_gen/outputs/navsim_multigt_0p5s_8step_16gpu}"
export NVS_SENSOR_ROOT="${NVS_SENSOR_ROOT:-/mnt/data/navsim/trainval_all/trainval_sensor_blobs/trainval}"
export NVS_LOG_ROOT="${NVS_LOG_ROOT:-/mnt/data/navsim/trainval_navsim_logs/trainval}"
export NVS_CACHE_ROOT="${NVS_CACHE_ROOT:-/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtrain_20261005/metric_cache}"
export NVS_DINO_CHECKPOINT="${NVS_DINO_CHECKPOINT:-/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors}"

# Two NON-CLASHING output dirs: staged (two-phase) then joint finetune.
STAGED_DIR="${NVS_STAGED_DIR:-${PROJECT_ROOT}/artifacts/navtrain_wan_e2e_twophase_ddp}"
JOINT_DIR="${NVS_OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/navtrain_wan_e2e_ddp}"

# ---- preflight: fail fast with a clear, actionable message ----------------- #
missing=0
for p in "$PROJECT_ROOT" "$PYBIN" "$PPU_ENV" "$VENV_ACTIVATE" \
         "$NVS_MULTIGT_ROOT" "$NVS_SENSOR_ROOT" "$NVS_LOG_ROOT" \
         "$NVS_CACHE_ROOT" "$NVS_DINO_CHECKPOINT"; do
  if [ ! -e "$p" ]; then echo "[PREFLIGHT] MISSING: $p" >&2; missing=1; fi
done
if [ "$missing" -ne 0 ]; then
  echo "[PREFLIGHT] Aborting. Export the matching NVS_* env var(s) to your DLC mount path(s) and re-run." >&2
  exit 2
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

NPROC="${NVS_NPROC_PER_NODE:-16}"
BATCH="${NVS_BATCH:-12}"
EPOCHS_A="${NVS_PHASE_A_EPOCHS:-50}"
EPOCHS_B="${NVS_PHASE_B_EPOCHS:-50}"
EPOCHS_JOINT="${NVS_JOINT_EPOCHS:-50}"
mkdir -p "$STAGED_DIR" "$JOINT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_A="${STAGED_DIR}/train_staged_${TS}.log"
LOG_J="${JOINT_DIR}/train_joint_${TS}.log"

echo "[LAUNCH] e2e ALL-NATIVE protocol | Wan ActionDiT | nproc=${NPROC} | batch/GPU=${BATCH} | global_batch=$((BATCH * NPROC))"
echo "[LAUNCH] stage1(staged A+B)=${STAGED_DIR}  log=${LOG_A}"
echo "[LAUNCH] stage2(joint finetune)=${JOINT_DIR}  log=${LOG_J}"
echo "[LAUNCH] python=$("$PYBIN" -c "import sys,timm; print(sys.executable, timm.__version__)")"

# ============================ STAGE 1: staged ============================== #
"$PYBIN" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  scripts/run_navtrain_wan_e2e_twophase_ddp.py \
  --use-all-native-tokens \
  --phase-a-epochs "$EPOCHS_A" \
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
echo "[LAUNCH] STAGE1 staged finished with exit=${status_a}"
if [ "$status_a" -ne 0 ]; then
  echo "[LAUNCH] STAGE1 failed; NOT starting the joint finetune." >&2
  exit "$status_a"
fi
if [ ! -f "${STAGED_DIR}/checkpoint.pt" ]; then
  echo "[LAUNCH] STAGE1 produced no ${STAGED_DIR}/checkpoint.pt; cannot resume joint finetune." >&2
  exit 4
fi

# ======================= STAGE 2: joint finetune =========================== #
"$PYBIN" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  scripts/run_navtrain_wan_e2e_ddp.py \
  --use-all-native-tokens \
  --resume "${STAGED_DIR}/checkpoint.pt" \
  --batch-size "$BATCH" \
  --epochs "$EPOCHS_JOINT" \
  --grad-accum 1 \
  --workers 4 \
  --lr 2e-4 \
  --scorer-lr 5e-4 \
  --weight-decay 0.05 \
  --warmup 500 \
  --lambda-action 1.0 \
  --lambda-score 1.0 \
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
echo "[LAUNCH] STAGE2 joint finetune finished with exit=${status_j}"
exit "$status_j"
