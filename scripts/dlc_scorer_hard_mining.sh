#!/usr/bin/env bash
# =============================================================================
# PAI-DLC single-node 16-GPU driver -- SCORER HARD-SCENE MINING + CANDIDATE SUBSAMPLING
# =============================================================================
# Everything runs from THIS file. Base model = joint@17500
# (artifacts/navtrain_wan_joint_ddp/checkpoint.pt), navtest official PDMS 0.8990.
# The representation (eta + theta + DINOv3) is frozen; only psi (the scratch
# DrivoR scorer) is fine-tuned, with the OFFICIAL DrivoRLoss unchanged.
#
#   STAGE 1 build  16-GPU frozen-generator pass over navtrain -> per-scene
#                  scene_context / 64 proposals / DrivoR oracle raw7 / psi's
#                  current pick.  Resumable via --skip-existing.
#   STAGE 2 mine   single process: pool-vs-selection attribution + scene lists
#                  (hard_all / hard_selection / hard_pool / random_matched /
#                  heldout).  Hard scenes are mined from NAVTRAIN ONLY; navtest
#                  is never used for selection, so 0.8990 stays comparable.
#   STAGE 3 train  5 arms x 16-GPU scorer-only fine-tune, all from the same psi,
#                  same global batch (12x16=192), same LR, same 2000 steps.
#                    B0  all scenes          64 candidates   (control)
#                    E1  hard_all            64 candidates   (experiment 1)
#                    E1v random_matched      64 candidates   (size-matched control)
#                    E1s hard_selection      64 candidates   (drop pool-limited)
#                    E2  all scenes          top10+bottom10  (experiment 2)
#   STAGE 4 eval   5 arms x 16 scene-shards of the FULL navtest official
#                  pdm_score path, then merged.
#   STAGE 5 report paired diffs vs baseline + the navtrain diagnosis.
#
# Any stage can be re-run alone; completed work is detected and skipped:
#   NVS_STAGE=all|smoke|build|mine|train|eval|report   (default all)
#   NVS_ARMS="B0 E1 E1v E1s E2"   NVS_FORCE=1 to recompute finished stages
#   NVS_MAX_STEPS NVS_NPROC_PER_NODE NVS_EVAL_SHARDS NVS_BASE_CHECKPOINT ...
#
# RUN `NVS_STAGE=smoke` FIRST. It executes the 8 acceptance checks on tiny
# samples (~20 min) and aborts with [SMOKE-FAIL] if any invariant breaks, so a
# broken pipeline never burns the full 2.5 h run. Everything lands in
# ${WORK_DIR}/_smoke and nothing there is reused by the real stages.
#
# Launcher note: always `python -m torch.distributed.run --standalone --nnodes=1`
# (NEVER the `torchrun` shim -- broken shebang -> system python w/o dinov3).
# =============================================================================

# NOTE: no `set -euo pipefail` on purpose. The PPU envsetup.sh and the venv
# activate script reference unbound vars and can return non-zero; a global -e/-u
# would abort the driver for harmless reasons. We preflight explicitly instead.
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

BASE_CKPT="${NVS_BASE_CHECKPOINT:-${PROJECT_ROOT}/artifacts/navtrain_wan_joint_ddp/checkpoint.pt}"
JOINT_INDEX_DIR="${NVS_JOINT_INDEX_DIR:-${PROJECT_ROOT}/artifacts/navtrain_wan_joint_ddp}"
WORK_DIR="${NVS_WORK_DIR:-${PROJECT_ROOT}/artifacts/scorer_hard_mining}"
CACHE_DIR="${NVS_CACHE_DIR:-${WORK_DIR}/cache}"
BASELINE_RESULT="${NVS_BASELINE_RESULT:-${PROJECT_ROOT}/artifacts/navtest_official_pdms/joint17500_full/result.json}"

# navtest (evaluation only -- the official navsim pdm_score path)
NAVTEST_DRIVOR_CACHE="${NVS_NAVTEST_DRIVOR_CACHE:-/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtest_20261006/metric_cache}"
NAVTEST_OFFICIAL_CACHE="${NVS_NAVTEST_OFFICIAL_CACHE:-/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/drivor_official_navtest_full_cache_20261007/metric_cache}"
NAVTEST_LOG_ROOT="${NVS_NAVTEST_LOG_ROOT:-/mnt/data/navsim/test_navsim_logs/test}"
NAVTEST_SENSOR_ROOT="${NVS_NAVTEST_SENSOR_ROOT:-/mnt/data/navsim/test_sensor_blobs/test}"
NAVTEST_INDEX_DIR="${NVS_NAVTEST_INDEX_DIR:-${PROJECT_ROOT}/artifacts/navtest_wan_joint_eval_index}"

# ---- knobs ------------------------------------------------------------------ #
STAGE="${NVS_STAGE:-all}"
ARMS="${NVS_ARMS:-B0 E1 E1v E1s E2}"
NPROC="${NVS_NPROC_PER_NODE:-16}"
FORCE="${NVS_FORCE:-0}"
EXPECT_OPT_STEP="${NVS_EXPECT_OPT_STEP:-0}"

BUILD_BATCH="${NVS_BUILD_BATCH:-8}"
BUILD_WORKERS="${NVS_BUILD_WORKERS:-6}"
SHARD_SCENES="${NVS_SHARD_SCENES:-2048}"

HELDOUT_SCENES="${NVS_HELDOUT_SCENES:-2000}"
HARD_THRESHOLD="${NVS_HARD_THRESHOLD:-0.80}"
SELECTION_THRESHOLD="${NVS_SELECTION_THRESHOLD:-0.95}"
POOL_THRESHOLD="${NVS_POOL_THRESHOLD:-0.90}"

MAX_STEPS="${NVS_MAX_STEPS:-2000}"
TRAIN_BATCH="${NVS_TRAIN_BATCH:-12}"
SCORER_LR="${NVS_SCORER_LR:-5e-4}"
WEIGHT_DECAY="${NVS_WEIGHT_DECAY:-0.05}"
WARMUP="${NVS_WARMUP:-100}"
EVAL_EVERY="${NVS_EVAL_EVERY:-100}"

EVAL_SHARDS="${NVS_EVAL_SHARDS:-16}"
EVAL_MAX_SCENES="${NVS_EVAL_MAX_SCENES:-0}"   # 0 = the full navtest split
EVAL_BATCH="${NVS_EVAL_BATCH:-8}"
EVAL_WORKERS="${NVS_EVAL_WORKERS:-3}"

# ---- smoke knobs (NVS_STAGE=smoke only) ------------------------------------- #
SMOKE_DIR="${NVS_SMOKE_DIR:-${WORK_DIR}/_smoke}"
SMOKE_SCENES="${NVS_SMOKE_SCENES:-512}"       # builder sample; 512 so hard_all is non-trivial
SMOKE_BUILD_SHARD="${NVS_SMOKE_BUILD_SHARD:-8}"
SMOKE_NPROC="${NVS_SMOKE_NPROC:-4}"           # DDP width for the finetune smoke (capped by shard count)
SMOKE_STEPS="${NVS_SMOKE_STEPS:-20}"
SMOKE_HELDOUT="${NVS_SMOKE_HELDOUT:-8}"
SMOKE_EVAL_SCENES="${NVS_SMOKE_EVAL_SCENES:-512}"
SMOKE_EVAL_SHARDS="${NVS_SMOKE_EVAL_SHARDS:-4}"

# ---- preflight: fail fast with a clear, actionable message ----------------- #
preflight() {
  local missing=0 p
  local required=("$PROJECT_ROOT" "$PYBIN" "$PPU_ENV" "$VENV_ACTIVATE" "$BASE_CKPT" "$JOINT_INDEX_DIR")
  case "$STAGE" in
    all|smoke|build|mine|train)
      required+=("$NVS_MULTIGT_ROOT" "$NVS_SENSOR_ROOT" "$NVS_LOG_ROOT" "$NVS_CACHE_ROOT" "$NVS_DINO_CHECKPOINT") ;;
  esac
  case "$STAGE" in
    all|smoke|eval|report)
      required+=("$NAVTEST_DRIVOR_CACHE" "$NAVTEST_OFFICIAL_CACHE" "$NAVTEST_LOG_ROOT" "$NAVTEST_SENSOR_ROOT") ;;
  esac
  for p in "${required[@]}"; do
    if [ ! -e "$p" ]; then echo "[PREFLIGHT] MISSING: $p" >&2; missing=1; fi
  done
  case "$STAGE" in
    all|smoke|build|mine|train|eval|report) ;;
    *) echo "[PREFLIGHT] unknown NVS_STAGE='$STAGE' (all|smoke|build|mine|train|eval|report)" >&2; missing=1 ;;
  esac
  if [ "$missing" -ne 0 ]; then
    echo "[PREFLIGHT] Aborting. Export the matching NVS_* env var(s) to your DLC mount path(s) and re-run." >&2
    return 2
  fi
  return 0
}

# ---- helpers ---------------------------------------------------------------- #
want_stage() { [ "$STAGE" = "all" ] || [ "$STAGE" = "$1" ]; }

run_ddp_n() {
  local n="$1"; shift
  "$PYBIN" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$n" \
    "$@"
}

run_ddp() {
  run_ddp_n "$NPROC" "$@"
}

smoke() {
  # every smoke step must pass; the first failure aborts the driver
  echo "[SMOKE] $*"
  "$@"
  local st=$?
  if [ "$st" -ne 0 ]; then echo "[SMOKE-FAIL] exit=$st :: $*" >&2; return "$st"; fi
  return 0
}

die() { echo "[FATAL] $*" >&2; exit 1; }

arm_scene_list() {
  case "$1" in
    B0)  echo "all" ;;
    E1)  echo "hard_all" ;;
    E1v) echo "random_matched" ;;
    E1s) echo "hard_selection" ;;
    E2)  echo "all" ;;
    *)   echo "" ;;
  esac
}

arm_candidate_select() {
  case "$1" in
    E2) echo "top10bottom10" ;;
    *)  echo "all64" ;;
  esac
}

# ---- stages ----------------------------------------------------------------- #
stage_build() {
  echo "[STAGE 1/5] build frozen-generator scorer cache -> ${CACHE_DIR}"
  if [ -f "${CACHE_DIR}/_builder_meta.json" ] && [ "$FORCE" != "1" ]; then
    echo "[STAGE 1/5] _builder_meta.json present -> skipping (NVS_FORCE=1 to rebuild)"
    return 0
  fi
  run_ddp scripts/build_scorer_cache.py \
    --checkpoint "$BASE_CKPT" \
    --expect-opt-step "$EXPECT_OPT_STEP" \
    --output-dir "$CACHE_DIR" \
    --index-dir "$JOINT_INDEX_DIR" \
    --batch-size "$BUILD_BATCH" \
    --workers "$BUILD_WORKERS" \
    --shard-scenes "$SHARD_SCENES" \
    --candidates 64 \
    --seed 17 \
    --log-every 20 \
    --skip-existing \
    ${NVS_EXTRA_ARGS:-}
  local st=$?
  if [ "$st" -ne 0 ]; then echo "[STAGE 1/5] build FAILED exit=$st" >&2; return "$st"; fi
  echo "[STAGE 1/5] build OK ($(ls -1 "${CACHE_DIR}"/shard_*.pt 2>/dev/null | wc -l) shard files)"
  return 0
}

stage_mine() {
  echo "[STAGE 2/5] mine hard scenes + pool-vs-selection diagnosis -> ${WORK_DIR}"
  if [ -f "${WORK_DIR}/diagnosis.json" ] && [ "$FORCE" != "1" ]; then
    echo "[STAGE 2/5] diagnosis.json present -> skipping (NVS_FORCE=1 to re-mine)"
    return 0
  fi
  "$PYBIN" scripts/mine_hard_scenes.py \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$WORK_DIR" \
    --hard-threshold "$HARD_THRESHOLD" \
    --selection-threshold "$SELECTION_THRESHOLD" \
    --pool-threshold "$POOL_THRESHOLD" \
    --heldout-scenes "$HELDOUT_SCENES" \
    --seed 17
  local st=$?
  if [ "$st" -ne 0 ]; then echo "[STAGE 2/5] mine FAILED exit=$st" >&2; return "$st"; fi
  return 0
}

stage_train_one() {
  local arm="$1"
  local list sel out
  list="$(arm_scene_list "$arm")"
  sel="$(arm_candidate_select "$arm")"
  out="${WORK_DIR}/arms/${arm}"
  if [ -z "$list" ]; then echo "[STAGE 3/5] unknown arm '$arm' (B0|E1|E1v|E1s|E2)" >&2; return 1; fi
  echo "[STAGE 3/5] arm=${arm} scene_list=${list} candidate_select=${sel} -> ${out}"
  if [ -f "${out}/checkpoint_best.pt" ] && [ "$FORCE" != "1" ]; then
    echo "[STAGE 3/5] arm=${arm} already trained -> skipping (NVS_FORCE=1 to redo)"
    return 0
  fi
  mkdir -p "$out"
  run_ddp scripts/finetune_scorer_from_cache.py \
    --tag "$arm" \
    --cache-dir "$CACHE_DIR" \
    --base-checkpoint "$BASE_CKPT" \
    --mined-dir "$WORK_DIR" \
    --scene-list "$list" \
    --heldout-list heldout \
    --output-dir "$out" \
    --candidate-select "$sel" \
    --top-k 10 --bottom-k 10 \
    --batch-size "$TRAIN_BATCH" \
    --scorer-lr "$SCORER_LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --warmup "$WARMUP" \
    --max-steps "$MAX_STEPS" \
    --log-every 10 --eval-every "$EVAL_EVERY" --save-every "$EVAL_EVERY" \
    --seed 17 \
    ${NVS_TRAIN_EXTRA_ARGS:-}
  local st=$?
  if [ "$st" -ne 0 ]; then echo "[STAGE 3/5] arm=${arm} FAILED exit=$st" >&2; return "$st"; fi
  return 0
}

stage_train() {
  echo "[STAGE 3/5] scorer-only fine-tune | arms='${ARMS}' | nproc=${NPROC} | batch/GPU=${TRAIN_BATCH} | max_steps=${MAX_STEPS}"
  local arm
  for arm in $ARMS; do
    stage_train_one "$arm" || return $?
  done
  return 0
}

stage_eval_one() {
  local arm="$1"
  local ckpt="${WORK_DIR}/arms/${arm}/checkpoint_best.pt"
  local out="${WORK_DIR}/eval/${arm}"
  if [ ! -f "$ckpt" ]; then echo "[STAGE 4/5] arm=${arm}: missing $ckpt" >&2; return 1; fi
  mkdir -p "$out"
  echo "[STAGE 4/5] arm=${arm} navtest official PDMS | shards=${EVAL_SHARDS} | max_scenes=${EVAL_MAX_SCENES} -> ${out}"
  if [ -f "${out}/result.json" ] && [ "$FORCE" != "1" ]; then
    echo "[STAGE 4/5] arm=${arm} already evaluated+merged -> skipping (NVS_FORCE=1 to redo)"
    return 0
  fi
  local pids=() i st fail=0
  for (( i=0; i<EVAL_SHARDS; i++ )); do
    "$PYBIN" scripts/eval_navtest_official_pdms.py \
      --checkpoint "$ckpt" \
      --tag "$arm" \
      --output-dir "$out" \
      --device "cuda:$((i % NPROC))" \
      --shard-index "$i" \
      --shard-count "$EVAL_SHARDS" \
      --max-scenes "$EVAL_MAX_SCENES" \
      --batch-size "$EVAL_BATCH" \
      --workers "$EVAL_WORKERS" \
      --depth 0 \
      --seed 17 \
      --drivor-cache "$NAVTEST_DRIVOR_CACHE" \
      --official-cache "$NAVTEST_OFFICIAL_CACHE" \
      --log-root "$NAVTEST_LOG_ROOT" \
      --sensor-root "$NAVTEST_SENSOR_ROOT" \
      --index-dir "$NAVTEST_INDEX_DIR" \
      ${NVS_EVAL_EXTRA_ARGS:-} \
      > "${out}/shard_${i}.log" 2>&1 &
    pids+=("$!")
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}"; st=$?
    if [ "$st" -ne 0 ]; then
      echo "[STAGE 4/5] arm=${arm} shard ${i} FAILED exit=$st (see ${out}/shard_${i}.log)" >&2
      fail=1
    fi
  done
  if [ "$fail" -ne 0 ]; then return 1; fi
  "$PYBIN" scripts/merge_navtest_shards.py --shard-dir "$out" || return 1
  return 0
}

stage_eval() {
  echo "[STAGE 4/5] navtest official pdm_score evaluation"
  local arm
  for arm in $ARMS; do
    stage_eval_one "$arm" || return $?
  done
  return 0
}

stage_report() {
  echo "[STAGE 5/5] report"
  if [ ! -f "$BASELINE_RESULT" ]; then
    echo "[STAGE 5/5] baseline result missing: $BASELINE_RESULT" >&2
    echo "[STAGE 5/5] set NVS_BASELINE_RESULT to a navtest official result.json" >&2
    return 1
  fi
  "$PYBIN" scripts/report_scorer_arms.py \
    --baseline "$BASELINE_RESULT" \
    --work-dir "$WORK_DIR" \
    --arms "$ARMS" \
    --hard-threshold "$HARD_THRESHOLD" \
    --output "${WORK_DIR}/REPORT.json"
  local st=$?
  if [ "$st" -ne 0 ]; then echo "[STAGE 5/5] report FAILED exit=$st" >&2; return "$st"; fi
  return 0
}

ok() { echo "[SMOKE-OK] $*"; }

# smoke_eval <tag> <checkpoint> <out-dir> <device> <shard-index> <shard-count>
smoke_eval() {
  local tag="$1" ckpt="$2" out="$3" dev="$4" si="$5" sc="$6"
  "$PYBIN" scripts/eval_navtest_official_pdms.py \
    --checkpoint "$ckpt" --tag "$tag" --output-dir "$out" \
    --device "$dev" --shard-index "$si" --shard-count "$sc" \
    --max-scenes "$SMOKE_EVAL_SCENES" --batch-size "$EVAL_BATCH" --workers "$EVAL_WORKERS" \
    --depth 0 --seed 17 --verify-scenes 0 \
    --dino-checkpoint "$NVS_DINO_CHECKPOINT" \
    --drivor-cache "$NAVTEST_DRIVOR_CACHE" --official-cache "$NAVTEST_OFFICIAL_CACHE" \
    --log-root "$NAVTEST_LOG_ROOT" --sensor-root "$NAVTEST_SENSOR_ROOT" \
    --index-dir "$NAVTEST_INDEX_DIR" \
    > "${out}/${tag}_shard${si}.log" 2>&1
}

# ---------------------------------------------------------------------------
# NVS_STAGE=smoke -- the 8 acceptance checks from the plan, on tiny samples.
# Runs BEFORE the real pipeline so a broken invariant never costs 2.5 h.
# ---------------------------------------------------------------------------
stage_smoke() {
  local v=scripts/smoke_verify_scorer_mining.py
  echo "[SMOKE] acceptance checks -> ${SMOKE_DIR}"
  mkdir -p "$SMOKE_DIR"

  # (1) static: driver syntax, module compile, contract tests
  smoke bash -n scripts/dlc_scorer_hard_mining.sh || return $?
  smoke "$PYBIN" -m py_compile scripts/build_scorer_cache.py scripts/mine_hard_scenes.py \
        scripts/finetune_scorer_from_cache.py scripts/merge_navtest_shards.py \
        scripts/report_scorer_arms.py scripts/eval_navtest_official_pdms.py "$v" || return $?
  smoke "$PYBIN" -m pytest -q tests/contract_tests_v6_wan.py || return $?

  # (8) preflight must reject a bogus NVS_* path with exit 2
  NVS_STAGE=report NVS_BASE_CHECKPOINT=/nonexistent/nvs/preflight/probe.pth \
    bash scripts/dlc_scorer_hard_mining.sh > "${SMOKE_DIR}/preflight_negative.log" 2>&1
  local pf=$?
  if [ "$pf" -ne 2 ]; then
    echo "[SMOKE-FAIL] preflight negative test exited $pf (want 2); see ${SMOKE_DIR}/preflight_negative.log" >&2
    return 1
  fi
  smoke grep -q '\[PREFLIGHT\] MISSING:' "${SMOKE_DIR}/preflight_negative.log" || return $?
  ok "preflight aborts with exit 2 and a MISSING line on a bogus path"

  # (2) builder, single process + oracle re-score of 2 scenes
  local c1="${SMOKE_DIR}/cache_n1" c2="${SMOKE_DIR}/cache_n2"
  rm -rf "$c1" "$c2"; mkdir -p "$c1" "$c2"
  smoke run_ddp_n 1 scripts/build_scorer_cache.py \
      --checkpoint "$BASE_CKPT" --expect-opt-step "$EXPECT_OPT_STEP" \
      --output-dir "$c1" --index-dir "$JOINT_INDEX_DIR" \
      --batch-size 8 --workers 2 --shard-scenes "$SMOKE_BUILD_SHARD" \
      --limit-scenes "$SMOKE_SCENES" --candidates 64 --seed 17 --log-every 5 || return $?
  smoke "$PYBIN" "$v" builder --cache-dir "$c1" --oracle-scenes 2 --expect-opt-step "$EXPECT_OPT_STEP" \
      --limit-scenes "$SMOKE_SCENES" --index-dir "$JOINT_INDEX_DIR" \
      --multigt-root "$NVS_MULTIGT_ROOT" --cache-root "$NVS_CACHE_ROOT" --log-root "$NVS_LOG_ROOT" || return $?

  # (3) the same scenes under a 2-way DDP partition must be identical
  smoke run_ddp_n 2 scripts/build_scorer_cache.py \
      --checkpoint "$BASE_CKPT" --expect-opt-step "$EXPECT_OPT_STEP" \
      --output-dir "$c2" --index-dir "$JOINT_INDEX_DIR" \
      --batch-size 8 --workers 2 --shard-scenes "$SMOKE_BUILD_SHARD" \
      --limit-scenes "$SMOKE_SCENES" --candidates 64 --seed 17 --log-every 5 || return $?
  smoke "$PYBIN" "$v" shards --a "$c1" --b "$c2" --nproc-a 1 --nproc-b 2 || return $?

  # (4) mining: attribution + scene-list invariants
  local mined="${SMOKE_DIR}/mined"
  rm -rf "$mined"; mkdir -p "$mined"
  smoke "$PYBIN" scripts/mine_hard_scenes.py --cache-dir "$c1" --output-dir "$mined" \
      --hard-threshold "$HARD_THRESHOLD" --selection-threshold "$SELECTION_THRESHOLD" \
      --pool-threshold "$POOL_THRESHOLD" --heldout-scenes "$SMOKE_HELDOUT" --seed 17 || return $?
  smoke "$PYBIN" "$v" mining --dir "$mined" || return $?

  # (5) finetune smoke: both candidate modes; psi moves, generator stays frozen
  local nshard fnproc d
  nshard=$(ls -1 "${c1}"/shard_*.pt 2>/dev/null | wc -l)
  fnproc="$SMOKE_NPROC"
  if [ "$fnproc" -gt "$nshard" ]; then fnproc="$nshard"; fi
  if [ "$fnproc" -lt 1 ]; then echo "[SMOKE-FAIL] no shards in $c1" >&2; return 1; fi
  ok "finetune smoke DDP width = ${fnproc} (${nshard} shard files)"
  local fta="${SMOKE_DIR}/ft_all64" ftb="${SMOKE_DIR}/ft_tb" ftz="${SMOKE_DIR}/ft_zero"
  for d in "$fta" "$ftb" "$ftz"; do rm -rf "$d"; mkdir -p "$d"; done
  local hard_n list_a
  hard_n=$("$PYBIN" -c "import json,sys; print(json.load(open(sys.argv[1]))['count'])" "${mined}/scenes_hard_all.json")
  # hard_all must give every DDP rank at least one full batch, else fall back
  if [ "$hard_n" -ge "$((2 * fnproc))" ]; then list_a=hard_all; else list_a=all; fi
  ok "hard_all=${hard_n} scenes -> the all64 smoke trains on scene-list '${list_a}'"

  local sel out tag list
  for sel in all64 top10bottom10; do
    if [ "$sel" = "all64" ]; then out="$fta"; tag=smoke_all64; list="$list_a"; else out="$ftb"; tag=smoke_tb; list=all; fi
    smoke run_ddp_n "$fnproc" scripts/finetune_scorer_from_cache.py \
        --tag "$tag" --cache-dir "$c1" --base-checkpoint "$BASE_CKPT" --mined-dir "$mined" \
        --scene-list "$list" --heldout-list heldout --output-dir "$out" \
        --candidate-select "$sel" --top-k 10 --bottom-k 10 \
        --batch-size 2 --eval-batch-size 8 --scorer-lr "$SCORER_LR" \
        --weight-decay "$WEIGHT_DECAY" --warmup 2 --max-steps "$SMOKE_STEPS" \
        --log-every 2 --eval-every 10 --save-every 10 --seed 17 || return $?
    smoke "$PYBIN" "$v" finetune --dir "$out" --base-checkpoint "$BASE_CKPT" --expect-changed || return $?
  done

  # (6a) zero-step export must be a byte-faithful re-pack of the base psi
  smoke run_ddp_n "$fnproc" scripts/finetune_scorer_from_cache.py \
      --tag smoke_zero --cache-dir "$c1" --base-checkpoint "$BASE_CKPT" --mined-dir "$mined" \
      --scene-list all --heldout-list heldout --output-dir "$ftz" \
      --candidate-select all64 --batch-size 2 --eval-batch-size 8 \
      --max-steps 0 --epochs 0 --log-every 0 --eval-every 0 --save-every 0 --seed 17 || return $?
  smoke "$PYBIN" "$v" finetune --dir "$ftz" --base-checkpoint "$BASE_CKPT" || return $?

  # (6b)+(7) navtest eval: unsharded baseline first (also warms the index sidecar),
  # then the zero-step checkpoint and the 4-way sharded baseline concurrently.
  local eb="${SMOKE_DIR}/eval_base" ez="${SMOKE_DIR}/eval_zero" es="${SMOKE_DIR}/eval_base_sh"
  for d in "$eb" "$ez" "$es"; do rm -rf "$d"; mkdir -p "$d"; done
  smoke smoke_eval base "$BASE_CKPT" "$eb" "cuda:0" 0 1 || return $?
  smoke_eval zero "${ftz}/checkpoint_best.pt" "$ez" "cuda:1" 0 1 &
  local pids=("$!") i st fail=0
  for (( i=0; i<SMOKE_EVAL_SHARDS; i++ )); do
    smoke_eval base_sh "$BASE_CKPT" "$es" "cuda:$(( (2 + i) % NPROC ))" "$i" "$SMOKE_EVAL_SHARDS" &
    pids+=("$!")
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}"; st=$?
    if [ "$st" -ne 0 ]; then echo "[SMOKE-FAIL] eval process ${i} exited $st" >&2; fail=1; fi
  done
  if [ "$fail" -ne 0 ]; then return 1; fi
  ok "all smoke eval processes finished"
  smoke "$PYBIN" scripts/merge_navtest_shards.py --shard-dir "$es" --reference "${eb}/result.json" || return $?
  smoke "$PYBIN" "$v" eval-equal --a "${eb}/result.json" --b "${ez}/result.json" --tol 1e-9 || return $?

  echo "[SMOKE] ALL 8 ACCEPTANCE CHECKS PASSED"
  echo "[SMOKE] safe to launch the real run: NVS_STAGE=all bash scripts/dlc_scorer_hard_mining.sh"
  return 0
}

main() {
  echo "[DRIVER] scorer hard-mining + candidate subsampling"
  echo "[DRIVER] stage=${STAGE} arms='${ARMS}' nproc=${NPROC} force=${FORCE}"
  echo "[DRIVER] base_checkpoint=${BASE_CKPT}"
  echo "[DRIVER] work_dir=${WORK_DIR}"
  echo "[DRIVER] cache_dir=${CACHE_DIR}"
  echo "[DRIVER] python=$("$PYBIN" -c "import sys,timm; print(sys.executable, timm.__version__)")"
  local start=$SECONDS
  # smoke never runs implicitly -- launch it explicitly with NVS_STAGE=smoke
  if [ "$STAGE" = "smoke" ]; then stage_smoke || return $?; fi
  if want_stage build;  then stage_build  || return $?; fi
  if want_stage mine;   then stage_mine   || return $?; fi
  if want_stage train;  then stage_train  || return $?; fi
  if want_stage eval;   then stage_eval   || return $?; fi
  if want_stage report; then stage_report || return $?; fi
  echo "[DRIVER] all requested stages finished in $((SECONDS - start)) s"
  return 0
}

# ---- runtime env ------------------------------------------------------------ #
preflight || exit $?
# shellcheck disable=SC1090
source "$PPU_ENV" ppu
# shellcheck disable=SC1090
source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT" || { echo "[PREFLIGHT] cannot cd to $PROJECT_ROOT" >&2; exit 3; }

# NOTE: ${PROJECT_ROOT}/scripts MUST come first -- the new drivers
# `import run_navtrain_wan_joint_ddp as trainer`.
export PYTHONPATH="${PROJECT_ROOT}/scripts:${PROJECT_ROOT}/src:${PROJECT_ROOT}/vendor/DrivoR:${PROJECT_ROOT}/vendor/DrivoR/nuplan-devkit"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${NVS_OMP_THREADS:-4}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# navtest official oracle needs the nuplan map env; the eval script also
# setdefaults these, exporting them here makes DLC mount changes explicit.
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/data/navsim/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
# Single-node --standalone rendezvous is authoritative; drop any job-level
# distributed vars DLC may have injected so they cannot confuse the agent.
unset MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK

mkdir -p "$WORK_DIR" "$CACHE_DIR"
LOG="${WORK_DIR}/dlc_$(date +%Y%m%d_%H%M%S).log"
echo "[DRIVER] log=${LOG}"

main 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
echo "[DRIVER] finished with exit=${status}"
exit "$status"
