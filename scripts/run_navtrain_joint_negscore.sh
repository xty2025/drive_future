#!/usr/bin/env bash
# =============================================================================
# Joint (eta+theta+psi) navtrain training -- NO future representation, with the
# mined poor_negative used ONLY as a SCORER (psi) negative.  Single stage,
# 1 node x 16 GPU.  This is the "score-part-only" sibling of
# scripts/run_navtrain_joint_neg.sh; it changes ONLY the negative flags, so the
# running DLC job and the rd-based joint-neg run are untouched.
#
# What "score part only, not the GT-matching front" means concretely:
#   DrivoR consumes an external negative_trajectory in exactly ONE place -- its
#   preference_module rd_loss -- which repels the ORACLE-CHOSEN proposal from the
#   negative in TRAJECTORY space (rd = -||chosen_traj - negative||.clamp_max(margin)).
#   Because chosen_traj indexes the ATTACHED endpoint, that gradient shapes THETA
#   (the generator), and `chosen` comes from the GT/oracle pair mining -- i.e. it
#   lives in the front GT-matching path.  --lambda-neg-match is the same story: it
#   Hungarian-matches negatives onto proposal slots and regresses them (metric-L1,
#   gradient to theta).  BOTH are trajectory / GT-matching-front terms.
#
#   This run turns BOTH off and instead uses --lambda-neg-score, the NVS-only
#   score-part negative:
#     --lambda-neg 0 --lambda-neg-match 0 --lambda-neg-score 0.1
#   psi scores each mined poor_negative in a SECOND scorer pass over the SAME
#   attached scene_context, and a DrivoR-style utility margin pushes the
#   oracle-chosen good proposal ABOVE every valid negative:
#       loss_neg_score = mean_over_valid( softplus(neg_utility - chosen_utility) )
#   The scorer detaches the trajectory internally, so this gradient shapes PSI
#   (and eta through the attached Scene K/V, exactly like DrivoR's score loss) but
#   NEVER theta's trajectory generation.  The negative enters the SCORE head only;
#   the endpoint / Hungarian / GT-matching front sees no negative term at all.
#   Like rd, it needs the oracle to pick `chosen`, so a step whose oracle fails
#   carries no negative signal (an honest zero).
#
#   Drop --lambda-neg-score (set NVS_LAMBDA_NEG_SCORE=0) for the positives-only
#   baseline -- the trainer is then bit-identical to the pre-negative code.
#
# Everything else (DrivoR-official 1152x672 resolution, 89-token DiT sequence
# [Scene(81) | Action(8)] with register/future frozen out, --precision bf16,
# --grad-checkpointing, batch 12 x 16 = global 192, 536 steps/epoch) is identical
# to run_navtrain_joint_neg.sh, so this also runs at ~2h/epoch.
# =============================================================================
set -o pipefail

PROJECT_ROOT=/mnt/zhaozc_workspace/project/NVS
PPU_ENV=/usr/local/PPU_SDK/envsetup.sh
VENV_ACTIVATE=/mnt/zhaozc_workspace/envs/driov_dpo/bin/activate

# Data roots (identical to run_navtrain_joint_neg.sh; export to override).
export NVS_MULTIGT_ROOT="${NVS_MULTIGT_ROOT:-/mnt/zhaozc_workspace/project/drivezero_goal_gen/outputs/navsim_multigt_0p5s_8step_16gpu}"
export NVS_SENSOR_ROOT="${NVS_SENSOR_ROOT:-/mnt/data/navsim/trainval_all/trainval_sensor_blobs/trainval}"
export NVS_LOG_ROOT="${NVS_LOG_ROOT:-/mnt/data/navsim/trainval_navsim_logs/trainval}"
export NVS_CACHE_ROOT="${NVS_CACHE_ROOT:-/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtrain_20261005/metric_cache}"
export NVS_DINO_CHECKPOINT="${NVS_DINO_CHECKPOINT:-/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors}"
export NVS_WAN_BACKBONE="${NVS_WAN_BACKBONE:-${PROJECT_ROOT}/checkpoints/ActionDiT_Wan22_256hdim_3outdim.pt}"
# Mined poor_negative corpus (drivezero_goal_gen posneg output).
NEG_ROOT="${NVS_NEGATIVE_ROOT:-/mnt/zhaozc_workspace/project/drivezero_goal_gen/outputs/navtrain_poor4_multigt_dense_0p1s_f64_16gpu}"

# ---- negative routing: SCORE PART ONLY ------------------------------------- #
# rd (trajectory repulsion on the oracle-chosen slot) and neg-match (Hungarian
# regression toward negatives) are BOTH off; only the scorer-utility margin on the
# negatives is on.  Override any of the three via env to A/B other combinations.
LAMBDA_NEG="${NVS_LAMBDA_NEG:-0.0}"                # rd_loss (theta, GT-matching front) -- OFF here
LAMBDA_NEG_MATCH="${NVS_LAMBDA_NEG_MATCH:-0.0}"    # Hungarian neg regression (theta)   -- OFF here
LAMBDA_NEG_SCORE="${NVS_LAMBDA_NEG_SCORE:-0.1}"    # psi scorer-utility margin on negatives -- ON
RD_MARGIN="${NVS_RD_MARGIN:-1.0}"                  # only read by rd_loss; inert when LAMBDA_NEG=0

# Parallel online-oracle workers PER RANK (see run_navtrain_joint_neg.sh for the
# full rationale).  16 ranks x ORACLE_WORKERS must fit the node's cores.
ORACLE_WORKERS="${NVS_ORACLE_WORKERS:-8}"
# --precision bf16: the established Stage-1 mechanism (whole-net autocast around
# net(), outputs upcast to fp32 for match/loss/oracle).  DrivoR resolution keeps
# the O(tokens^2) ViT-L trunk the bottleneck; bf16 is ~4.3x faster than fp32 and
# numerics are preserved (matched_ade / oracle_best_pdms identical).  Set
# NVS_STAGE2_PRECISION=fp32 to restore the exact fp32 path.
PRECISION="${NVS_STAGE2_PRECISION:-bf16}"
VISION_BF16="${NVS_VISION_BF16:-0}"
VISION_BF16_FLAG=""
[ "$VISION_BF16" = "1" ] && VISION_BF16_FLAG="--vision-bf16"
OUT_DIR="${NVS_STAGE2_DIR:-${PROJECT_ROOT}/artifacts/navtrain_wan_e2e_ddp_negscore}"

source "$PPU_ENV" ppu
source "$VENV_ACTIVATE"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/vendor/DrivoR:${PROJECT_ROOT}/vendor/DrivoR/nuplan-devkit"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${NVS_OMP_THREADS:-4}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset MASTER_ADDR MASTER_PORT

cd "$PROJECT_ROOT" || exit 99
mkdir -p "$OUT_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="${OUT_DIR}/train_${STAMP}.log"

echo "[JOINT-NEGSCORE] precision=${PRECISION} | lambda_neg(rd)=${LAMBDA_NEG} lambda_neg_match=${LAMBDA_NEG_MATCH} lambda_neg_score=${LAMBDA_NEG_SCORE} | neg_root=${NEG_ROOT}"

python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=16 \
  scripts/run_navtrain_wan_e2e_ddp.py \
  --future-target none --lambda-future 0 \
  --negative-root "$NEG_ROOT" \
  --lambda-neg "$LAMBDA_NEG" --rd-margin "$RD_MARGIN" \
  --lambda-neg-match "$LAMBDA_NEG_MATCH" \
  --lambda-neg-score "$LAMBDA_NEG_SCORE" \
  --oracle-workers "$ORACLE_WORKERS" \
  --precision "$PRECISION" \
  $VISION_BF16_FLAG \
  --multigt-root "$NVS_MULTIGT_ROOT" \
  --sensor-root "$NVS_SENSOR_ROOT" \
  --log-root "$NVS_LOG_ROOT" \
  --cache-root "$NVS_CACHE_ROOT" \
  --dino-checkpoint "$NVS_DINO_CHECKPOINT" \
  --wan-backbone "$NVS_WAN_BACKBONE" \
  --target-dim 256 --depth 8 --heads 4 --candidates 64 \
  --batch-size 12 --epochs 50 --grad-accum 1 --workers 4 \
  --lr 2e-4 --scorer-lr 5e-4 --weight-decay 0.05 --warmup 500 \
  --lambda-action 1.0 --lambda-velocity 1.0 --lambda-score 1.0 --lambda-rank 0.1 \
  --dense-weight 0.1 --diversity-weight 1.0 \
  --grad-checkpointing --seed 17 \
  --log-every 10 --save-every 500 \
  --output-dir "$OUT_DIR" \
  ${NVS_STAGE2_EXTRA_ARGS:-} \
  2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}
echo "[JOINT-NEGSCORE] finished with exit=${status} | log=${LOG}"
exit "$status"
