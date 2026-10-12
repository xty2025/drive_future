#!/usr/bin/env bash
# =============================================================================
# PAI-DLC single-node 16-GPU launcher  --  NVS v6 SPLIT training (all-native tokens)
# =============================================================================
# Visual stream: every DINOv3 native token of the current + 4 history frames
#   (5 x 201 = 1005 visual + 2 ego-fused = 1007 scene tokens) goes through the
#   Action DiT.  The scorer consumes the SAME 1007-token scene_context K/V.
#
# TWO STAGES ONLY (trajectory, then scorer).  The register/future branch (phi)
# is NOT trained: it is frozen for the whole run and only supplies *detached*
# action-attention memory inside the directed pass.
#
# Phase A (Stage 1) : eta + theta, Action(residual-vel MSE vs multi-GT via
#                     Hungarian).  NO oracle, NO scorer, phi frozen -> the
#                     generator learns 64 trajectory proposals.
# Freeze + Cache    : freeze generator, one pass over navtrain persists per-rank
#                     shards (scene_context[1007], 64 proposals, oracle raw7,
#                     valid, padded multi-GT).  PDM oracle runs ONCE per scene.
# Phase B (Stage 2) : train psi ONLY (ScratchDrivoRScorer, random init) on the
#                     stationary cache via official DrivoRLoss.score_loss
#                     (+ DrivoR-style rank/margin loss).  Reports selected /
#                     oracle-best GT-ADE.  == scorer_only_fixed_proposals
#                     (PID 1663817 method).
#
# Topology : 1 node x 16 GPU, per-GPU batch 12, grad-accum 1 (global 192).
#            all-native scene self-attn is O(1007^2); batch 12 was measured to
#            fit the 97GB device (~61/73GB).  On OOM lower --batch-size or
#            raise --grad-accum; raise it only if the device has headroom.
# Launcher : python -m torch.distributed.run --standalone --nnodes=1
#            (NEVER the `torchrun` shim -- broken shebang -> system python w/o dinov3.)
#            --standalone auto-picks a free rendezvous port, so this never clashes
#            with another run already on the node.
# Logging  : prints step / loss / eval every --log-every (10) optimizer-steps.
#
# Restart Phase B only (after Phase A + cache already succeeded):
#   NVS_RESUME_CACHE=1 bash dlc_navtrain_v6_twophase.sh
# Override ANY path/epoch on DLC by exporting the NVS_* vars before running.
# =============================================================================

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
export NVS_OUTPUT_DIR="${NVS_OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/navtrain_v6_twophase_allnative_ddp}"

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
unset MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK

NPROC="${NVS_NPROC_PER_NODE:-16}"
BATCH="${NVS_BATCH_SIZE:-12}"
PHASE_A_EPOCHS="${NVS_PHASE_A_EPOCHS:-50}"
PHASE_B_EPOCHS="${NVS_PHASE_B_EPOCHS:-50}"
LAMBDA_RANK="${NVS_LAMBDA_RANK:-0.1}"
UNSAFE_THRESHOLD="${NVS_UNSAFE_THRESHOLD:-0.2}"
mkdir -p "$NVS_OUTPUT_DIR"
LOG="${NVS_OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

RESUME_FLAG=""
if [ "${NVS_RESUME_CACHE:-0}" = "1" ]; then
  RESUME_FLAG="--resume-cache"
  echo "[LAUNCH] resume-cache mode: skipping Phase A + cache build (needs prior generator.pt + phase_b_cache/shard_*.pt)"
fi

echo "[LAUNCH] NVS v6 SPLIT | all-native 1007 scene tokens | nproc=${NPROC} | batch/GPU=${BATCH} | global_batch=$((BATCH * NPROC))"
echo "[LAUNCH] epochs: phase_a=${PHASE_A_EPOCHS} phase_b=${PHASE_B_EPOCHS}"
echo "[LAUNCH] output=${NVS_OUTPUT_DIR}"
echo "[LAUNCH] log=${LOG}"
echo "[LAUNCH] python=$("$PYBIN" -c "import sys,timm; print(sys.executable, timm.__version__)")"

"$PYBIN" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC" \
  scripts/run_navtrain_v6_twophase_ddp.py \
  --use-all-native-tokens \
  --batch-size "$BATCH" \
  --phase-a-epochs "$PHASE_A_EPOCHS" \
  --phase-b-epochs "$PHASE_B_EPOCHS" \
  --grad-accum 1 \
  --workers 6 \
  --lr 2e-4 \
  --scorer-lr 5e-4 \
  --weight-decay 0.05 \
  --warmup 500 \
  --lambda-action 1.0 \
  --lambda-score 1.0 \
  --lambda-rank "$LAMBDA_RANK" \
  --unsafe-threshold "$UNSAFE_THRESHOLD" \
  --depth 8 \
  --candidates 64 \
  --seed 17 \
  --log-every 10 \
  --save-every 500 \
  --output-dir "$NVS_OUTPUT_DIR" \
  $RESUME_FLAG \
  ${NVS_EXTRA_ARGS:-} \
  2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}
echo "[LAUNCH] NVS v6 split run finished with exit=${status}"
exit "$status"
