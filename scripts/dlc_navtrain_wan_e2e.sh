#!/usr/bin/env bash
# =============================================================================
# PAI-DLC single-node 16-GPU launcher  --  TASK 1-E2E: JOINT stage2 -- DrivoR-exact end-to-end scorer (Scene NOT detached)
# =============================================================================
# Trains eta (DINOv3 LoRA + scene) + theta (Wan action) + psi (scorer) TOGETHER
# on the FULL navtrain set. phi (future/register branch) is FROZEN because
# --future-target none => lambda_future = 0.
#
# Topology : 1 node x 16 GPU, per-GPU batch 12  => global batch 192.
# Launcher : python -m torch.distributed.run --standalone --nnodes=1
#            (NEVER the `torchrun` shim: its shebang is broken on this image and
#             falls back to a system python whose timm has no dinov3 weights.)
# Logging  : prints step / loss / eval every --log-every (10) optimizer-steps.
#
# Override ANY path on DLC by exporting the NVS_* vars before running, e.g.
#   NVS_SENSOR_ROOT=/your/mount bash dlc_navtrain_wan_joint.sh
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
export NVS_SENSOR_ROOT="${NVS_SENSOR_ROOT:-/mnt/data/navsim/trainval_all/trainval_sensor_blobs/trainval}"
export NVS_LOG_ROOT="${NVS_LOG_ROOT:-/mnt/data/navsim/trainval_navsim_logs/trainval}"
export NVS_CACHE_ROOT="${NVS_CACHE_ROOT:-/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtrain_20261005/metric_cache}"
export NVS_DINO_CHECKPOINT="${NVS_DINO_CHECKPOINT:-/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors}"
export NVS_OUTPUT_DIR="${NVS_OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/navtrain_wan_e2e_ddp}"

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
mkdir -p "$NVS_OUTPUT_DIR"
LOG="${NVS_OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

echo "[LAUNCH] TASK1-E2E joint stage2 (un-detached scene) | Wan ActionDiT | nproc=${NPROC} | batch/GPU=12 | epochs=50 | global_batch=$((12 * NPROC))"
echo "[LAUNCH] output=${NVS_OUTPUT_DIR}"
echo "[LAUNCH] log=${LOG}"
echo "[LAUNCH] python=$("$PYBIN" -c "import sys,timm; print(sys.executable, timm.__version__)")"

"$PYBIN" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC" \
  scripts/run_navtrain_wan_e2e_ddp.py \
  --batch-size 12 \
  --epochs 50 \
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
  --output-dir "$NVS_OUTPUT_DIR" \
  ${NVS_EXTRA_ARGS:-} \
  2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}
echo "[LAUNCH] joint run finished with exit=${status}"
exit "$status"
