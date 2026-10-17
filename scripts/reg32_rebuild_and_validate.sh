#!/usr/bin/env bash
# Rebuilds the Stage-0 Future-Register cache at the 32-token readout and validates
# the Stage-1 path against it, unattended, on the one local GPU.
#
# WHY A SEPARATE CACHE ROOT
#   A remote DLC job is still building artifacts/future_target_cache with the code it
#   imported at launch, which is the *old* 8-token CLS-query readout -- its own audit
#   says so: standardizer_fit_rows 16384 = 512 scenes x 32 rows, entropy_rows 131072 =
#   512 x 4 GT x 8 frames x 8 heads, fingerprint 625c7a99...  Two builders in one root
#   would race on the same target files and on standardizer.pt, so this script writes
#   to its own root and leaves that job alone.  The payloads are not interchangeable:
#   targets_from_payload shape-checks (gt_count, REGISTER_TOKENS, dim) and the
#   trainer checks target_fingerprint, so pointing a new run at the old cache fails
#   loudly rather than training on the wrong target.
#
# WHY BOUNDED
#   --limit caps the scene index.  A full 79797-scene rebuild is ~53 GPU-hours on one
#   card; this is a validation of the new readout, not the production cache.  Raise
#   NVS_REG32_LIMIT (and drop it entirely for the real build, which belongs on the
#   multi-card launcher) once the numbers below look right.
#
# Each step is logged separately and a failing step does not abort the run, so one
# bad step still leaves the rest on the record.
set -uo pipefail

PROJECT_ROOT="${NVS_PROJECT_ROOT:-/mnt/zhaozc_workspace/project/NVS}"
PYBIN="${NVS_PYTHON:-/mnt/zhaozc_workspace/envs/driov_dpo/bin/python}"
cd "$PROJECT_ROOT"

export NVS_MULTIVIEW_ROOT="${NVS_MULTIVIEW_ROOT:-/mnt/data/multi_future_view/navtrain_full_allcosmos_cuda_v1_gpu64/navtrain_full_allcosmos_cuda_v1_gpu64}"
export NVS_SENSOR_ROOT="${NVS_SENSOR_ROOT:-/mnt/data/navsim/trainval_all/trainval_sensor_blobs/trainval}"
export NVS_LOG_ROOT="${NVS_LOG_ROOT:-/mnt/data/navsim/trainval_navsim_logs/trainval}"
export NVS_CACHE_ROOT="${NVS_CACHE_ROOT:-/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtrain_20261005/metric_cache}"
export NVS_DINO_CHECKPOINT="${NVS_DINO_CHECKPOINT:-/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors}"
export NVS_WAN_BACKBONE="${NVS_WAN_BACKBONE:-${PROJECT_ROOT}/checkpoints/ActionDiT_Wan22_256hdim_3outdim.pt}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/vendor/DrivoR:${PROJECT_ROOT}/vendor/DrivoR/nuplan-devkit"

CACHE_ROOT="${NVS_REG32_CACHE_ROOT:-${PROJECT_ROOT}/artifacts/future_target_cache_reg32}"
OUT="${NVS_REG32_OUT:-${PROJECT_ROOT}/artifacts/reg32_validation}"
LIMIT="${NVS_REG32_LIMIT:-1024}"
FIT_SCENES="${NVS_REG32_FIT_SCENES:-512}"
EVAL_SCENES="${NVS_REG32_EVAL_SCENES:-128}"
STEPS="${NVS_REG32_STEPS:-200}"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUT"
SUMMARY="${OUT}/summary_${STAMP}.txt"

# Refuse to share a root with anything else.  A half-written standardizer.pt is the
# one failure here that would silently poison both runs.
case "$CACHE_ROOT" in
  */future_target_cache) echo "refusing to build into the running job's root: $CACHE_ROOT" >&2; exit 2 ;;
esac

step() {  # step <name> <log> <command...>
  local name="$1" log="$2"; shift 2
  local started ended status
  started=$(date +%s)
  echo "[${name}] start $(date -Is)" | tee -a "$SUMMARY"
  "$@" >"$log" 2>&1
  status=$?
  ended=$(date +%s)
  echo "[${name}] exit=${status} elapsed=$((ended - started))s log=${log}" | tee -a "$SUMMARY"
}

step stage0_build "${OUT}/stage0_${STAMP}.log" \
  "$PYBIN" scripts/build_future_target_cache.py \
    --target-cache-root "$CACHE_ROOT" \
    --multiview-root "$NVS_MULTIVIEW_ROOT" \
    --log-root "$NVS_LOG_ROOT" \
    --sensor-root "$NVS_SENSOR_ROOT" \
    --cache-root "$NVS_CACHE_ROOT" \
    --dino-checkpoint "$NVS_DINO_CHECKPOINT" \
    --skip-degenerate \
    --limit "$LIMIT" --fit-scenes "$FIT_SCENES" --eval-scenes "$EVAL_SCENES" \
    --log-every 50

# The audit is the point of the build: separation_ratio and trajectory_distance_pearson
# have to land near the probe's register_query_attn row (0.458 / 0.771) and the query
# entropy has to stay under the 0.75 ceiling.  Echoed to the summary so it is readable
# without opening the log.  Read through load_standardizer rather than a raw
# torch.load, so this never needs object deserialization turned on.
if [ -f "${CACHE_ROOT}/standardizer.pt" ]; then
  "$PYBIN" - "$CACHE_ROOT" >>"$SUMMARY" 2>&1 <<'PY'
import sys
from nvs_v6_wan_e2e.target_cache import load_standardizer

_, audit = load_standardizer(sys.argv[1])
keys = ("register_target_dim", "query_temperature", "query_heads",
        "query_attention_entropy_mean", "query_attention_entropy_max",
        "query_attention_entropy_rows", "standardizer_fit_rows", "standardizer_fit_scenes",
        "standardizer_clamped_components", "standardizer_fingerprint",
        "separation_ratio", "trajectory_distance_pearson", "trajectory_distance_spearman",
        "within_scene_gt_distance", "across_scene_gt_distance")
print("[stage0_audit]")
for key in keys:
    if key in audit:
        print(f"  {key} = {audit[key]}")
PY
fi

# The structural question the last run answered "no" to: does the register's output
# depend on which trajectory it was handed?  Two scenes, deliberately an overfit, so
# the answer is about the pathway and not about capacity.
step pathway_probe "${OUT}/pathway_${STAMP}.log" \
  "$PYBIN" scripts/diagnose_trajectory_pathway.py \
    --target-cache-root "$CACHE_ROOT" \
    --scenes 2 \
    --dino-checkpoint "$NVS_DINO_CHECKPOINT" \
    --wan-backbone "$NVS_WAN_BACKBONE" \
    --json "${OUT}/pathway_${STAMP}.json"

# Short real Stage-1 on the new cache.  The number that decides anything is
# paired_future_l1 vs trajectory_blind_l1: the last run moved the first from 0.9338 to
# 0.7259 while the second stayed identical to six decimals, i.e. it learned the scene
# and ignored the trajectory.  gain_over_scene_mean_target is the verdict's own input.
step stage1_smoke "${OUT}/stage1_${STAMP}.log" \
  "$PYBIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
    scripts/run_stage1_cache_ddp.py \
    --target-cache-root "$CACHE_ROOT" \
    --output-dir "${OUT}/stage1_${STAMP}" \
    --dino-checkpoint "$NVS_DINO_CHECKPOINT" \
    --wan-backbone "$NVS_WAN_BACKBONE" \
    --grad-checkpointing \
    --max-steps "$STEPS" --batch-size 2 --workers 4 \
    --eval-every 50 --log-every 10 \
    --train-metric-scenes 16 --eval-metric-scenes 16 --mean-target-scenes 256

echo "[done] $(date -Is) summary=${SUMMARY}" | tee -a "$SUMMARY"
