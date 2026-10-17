#!/usr/bin/env bash
# =============================================================================
# Joint (eta+theta+psi) navtrain training -- NO future representation, WITH the
# mined poor_negative preference term.  Single stage, 1 node x 16 GPU.
#
# Why this bypasses scripts/dlc_navtrain_wan_e2e.sh:
#   the production launcher's Stage 2 *requires* --stage1-checkpoint and turns the
#   future cache on whenever standardizer.pt exists.  Both put the 32 register
#   tokens back into the DiT sequence.  To train trajectory + scorer ONLY, with no
#   future branch, we run torchrun directly:
#     - no --stage1-checkpoint  -> predict_future=False
#     - --future-target none --lambda-future 0
#   => the DiT sequence is [Scene(81) | Action(8)] = 89 tokens (DrivoR's 5x16+1
#      scene layout), register/future_head/register_type_embedding are frozen out
#      of the DDP reducer.
#
# Negative hook (the point of this run):
#   --negative-root <poor4 corpus> --lambda-neg 0.1 --rd-margin 1.0
#   This is DrivoR's rd_loss (repulsive distance) -- the ONLY place DrivoR consumes
#   an external negative_trajectory.  For each scene the oracle-chosen generator
#   proposal is pushed away from every mined poor_negative in trajectory space:
#       rd = -||chosen_traj - negative||.clamp_max(rd_margin)
#   chosen_traj indexes the ATTACHED endpoint, so the gradient shapes THETA (the
#   generator learns to steer off the mined bad trajectory), NOT the scorer; the
#   scorer still only ranks the model's own proposals via --lambda-rank.  Because
#   rd needs the oracle to pick `chosen`, a step whose oracle fails carries no
#   negative signal (an honest zero) -- exactly as in DrivoR.  lambda_neg=0.1 and
#   rd_margin=1.0 mirror DrivoR's preference_weight*preference_rd_weight and
#   preference_rd_margin.  Drop the three flags for the positives-only baseline --
#   the trainer is then bit-identical to the pre-negative code.
# =============================================================================
set -o pipefail

PROJECT_ROOT=/mnt/zhaozc_workspace/project/NVS
PPU_ENV=/usr/local/PPU_SDK/envsetup.sh
VENV_ACTIVATE=/mnt/zhaozc_workspace/envs/driov_dpo/bin/activate

# Data roots (identical to the launcher's NVS_* defaults; export to override).
export NVS_MULTIGT_ROOT="${NVS_MULTIGT_ROOT:-/mnt/zhaozc_workspace/project/drivezero_goal_gen/outputs/navsim_multigt_0p5s_8step_16gpu}"
export NVS_SENSOR_ROOT="${NVS_SENSOR_ROOT:-/mnt/data/navsim/trainval_all/trainval_sensor_blobs/trainval}"
export NVS_LOG_ROOT="${NVS_LOG_ROOT:-/mnt/data/navsim/trainval_navsim_logs/trainval}"
export NVS_CACHE_ROOT="${NVS_CACHE_ROOT:-/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtrain_20261005/metric_cache}"
export NVS_DINO_CHECKPOINT="${NVS_DINO_CHECKPOINT:-/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors}"
export NVS_WAN_BACKBONE="${NVS_WAN_BACKBONE:-${PROJECT_ROOT}/checkpoints/ActionDiT_Wan22_256hdim_3outdim.pt}"
# Mined poor_negative corpus (drivezero_goal_gen posneg output).
NEG_ROOT="${NVS_NEGATIVE_ROOT:-/mnt/zhaozc_workspace/project/drivezero_goal_gen/outputs/navtrain_poor4_multigt_dense_0p1s_f64_16gpu}"
# EXPERIMENT: neg-GT-matching weight (NVS-only, NOT DrivoR).  0.0 (default) = off,
# trainer is the rd-only path bit-identically.  Set NVS_LAMBDA_NEG_MATCH=0.1 to also
# regress proposals toward the mined bad trajectories so the scorer sees them as Q over
# scene_kv.  20-step local A/B: ~3% per-step overhead, no instability, directionally
# loss_rank down / sel_pdms flat-to-up at a small (+~3%) matched_ade cost -- within
# noise at 20 steps, so decide at production length.
LAMBDA_NEG_MATCH="${NVS_LAMBDA_NEG_MATCH:-0.0}"
# Parallel online-oracle workers PER RANK.  DrivoR scores its per-scene get_sub_score
# calls in parallel (ray worker_map); NVS's sequential loop idled the accelerator for
# the whole CPU oracle every step.  NOTE: phase timing later proved the oracle is only
# ~7% of the joint step (the real bottleneck is the fp32 DINOv3 backward, see PRECISION
# below), so this is a real but minor win.  Local probe: bit-identical scores, 4.06x
# oracle speedup at w=8.  16 ranks x ORACLE_WORKERS processes must fit the node's cores:
# tune to ~min(12, cores_per_node/16); lower it if the node is core-poor or another
# job (e.g. stage1_cache) contends.  0 = exact sequential path (old behavior).
ORACLE_WORKERS="${NVS_ORACLE_WORKERS:-8}"
# --precision bf16 is THE fix for the joint step's real bottleneck, and it is the
# ESTABLISHED way, not a new invention: Stage 1 (run_stage1_cache_ddp.py) has shipped
# --precision bf16 by default ("3.97x faster than fp32, peak mem 29.69 -> 19.72") and
# DrivoR trains Lightning `precision: 16-mixed`.  The joint Stage 2 trainer had been left
# on fp32 -- git history shows run_navtrain_wan_e2e_ddp.py NEVER carried an autocast --
# which is exactly why "the previous joint" was slow: it ran the SAME frozen DINOv3 ViT-L
# trunk in fp32 that Stage 1 runs in bf16.  Phase timing (1 GPU, batch 12) split the
# fp32 step: backward 45s (71%), forward 11s, oracle 4.5s, data 2.5s.  The backward is the
# O(tokens^2) attention of the ViT-L trunk over ~3k patch tokens x 5 frames x batch,
# recomputed under --grad-checkpointing (memory-required: no-ckpt OOMs at 94.6GB fp32 /
# 95.5GB bf16).  Whole-net bf16 autocast around net(), outputs upcast to fp32 for the
# match/loss/oracle:
#   backward 45.0s -> 8.8s (5.1x on the dominant phase); end-to-end warm marginal
#   ~55.8s -> ~12.9s/step (4.3x, and below Stage 1's ~15s).  Numerics preserved:
#   matched_ade 1.3418 identical, oracle_best_pdms 0.9723 identical, loss_total <0.1%
#   (eval_selected_pdms argmax flips on near-ties at some steps, as it does in fp32 too).
# Trainer default is fp32 (bit-identical pre-flag path); this production script opts into
# bf16.  Set NVS_STAGE2_PRECISION=fp32 to restore the exact fp32 path.
PRECISION="${NVS_STAGE2_PRECISION:-bf16}"
# Optional narrower lever: bf16 on the frozen trunk ONLY (theta/psi stay fp32).  Redundant
# when PRECISION=bf16 (whole-net already covers the trunk); kept for A/B, default off.
VISION_BF16="${NVS_VISION_BF16:-0}"
VISION_BF16_FLAG=""
[ "$VISION_BF16" = "1" ] && VISION_BF16_FLAG="--vision-bf16"
OUT_DIR="${NVS_STAGE2_DIR:-${PROJECT_ROOT}/artifacts/navtrain_wan_e2e_ddp_neg}"

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

python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=16 \
  scripts/run_navtrain_wan_e2e_ddp.py \
  --future-target none --lambda-future 0 \
  --negative-root "$NEG_ROOT" --lambda-neg 0.1 --rd-margin 1.0 \
  --lambda-neg-match "$LAMBDA_NEG_MATCH" \
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
echo "[JOINT-NEG] finished with exit=${status} | log=${LOG}"
exit "$status"
