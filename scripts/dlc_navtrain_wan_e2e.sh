#!/usr/bin/env bash
# =============================================================================
# PAI-DLC single-node launcher -- the mandated two-stage pipeline
# =============================================================================
#   STAGE 0 (once)   build_future_target_cache.py   -> standardizer.pt + targets/
#   STAGE 1          run_stage1_cache_ddp.py        -> stage1_checkpoint.pt
#   STAGE 2 (joint)  run_navtrain_wan_e2e_ddp.py    -> checkpoint.pt
#
# Stage 1 learns the future representation only: clean (sigma = 0) Multi-GT
# trajectories in, four register tokens per future frame out (REGISTER_QUERIES_PER_FRAME
# x FUTURE_FRAMES = 32), and only the velocity head stays frozen -- the DiT trunk
# itself is trained, so Stage 1 runs at the full 121-token [scene | action | register]
# sequence.  Stage 2 restores that checkpoint and trains eta + theta + psi jointly,
# with the anchor prior denoised at a randomly sampled flow timestep per forward.
#
# Stage 0 must run before BOTH Stage 1 and Stage 2.  Stage 1 reads the cached
# [K,32,1024] query-extracted targets and the frozen standardizer rather than
# re-decoding future.mp4 through the Teacher, which is what makes it a 16-GPU job
# instead of a single process: the Teacher forward over 32 future frames is the same
# work at step 1 and at step 50000, and paying it inside Stage 1 would repeat it on
# every epoch.  Measured on the DLC node, Stage 0 costs ~2.26 s of GPU time per scene
# per rank, so the ~80k-scene ready corpus is ~4-7 h on 16 ranks -- against a Stage 1
# that would otherwise re-pay ~5 s/scene on all 20 of its epochs.  Stage 1 also
# inherits the builder's train/eval split from _ready_scenes.json instead of
# recomputing one from CLI flags -- the standardizer was fit on the builder's train
# scenes, so a split chosen here could hand the verdict a "held-out" scene that helped
# define the scaling its own targets were standardized by.  Stage 2 resolves the
# register width from the cache standardizer and the Stage-1 payload and refuses to
# run if they disagree.
#
# The cache is versioned by its *readout*, not by its directory: the token count is a
# property of how the Teacher's future frames are read (one CLS query per frame was
# retired in favour of four prefix-register queries per frame) and it is NOT part of
# target_fingerprint, which only locks the Teacher checkpoint, the query temperature
# and heads, and the standardizer's mean/scale.  A cache built under the old readout
# therefore has a fingerprint the current code accepts and a payload shape it cannot
# use -- Stage 1 would die on PairTargets.validate at its first batch, after Stage 0
# had "succeeded" by skipping every already-cached scene.  The preflight below checks
# the token count directly so that cannot happen.
#
# Topology : all three stages = 1 node x 16 GPU (torch.distributed.run --standalone).
# Launcher : python -m torch.distributed.run, NEVER the `torchrun` shim -- its
#            shebang is broken on this image and falls back to a system python
#            whose timm has no dinov3 weights.
#
# Skip a stage by exporting NVS_SKIP_STAGE0=1 / NVS_SKIP_STAGE1=1 / NVS_SKIP_STAGE2=1.
# Override ANY path by exporting the matching NVS_* var before running.
#
# HOW TO LAUNCH -- this matters more than it looks:
#   PAI-DLC job command : bash scripts/dlc_navtrain_wan_e2e.sh          (FOREGROUND)
#   interactive DSW box : nohup bash scripts/dlc_navtrain_wan_e2e.sh > my.log 2>&1 &
#
# Never background it inside a DLC job command.  `bash -lc '... nohup ... &'` exits 0
# the instant it forks, DLC reads that as "job succeeded", tears the container down and
# kills the child -- so the run reports success in seconds having produced nothing.  The
# launcher now tees its own log under NVS_OUTPUT_ROOT for exactly this reason: whether a
# log exists must not depend on how the caller redirected stdout.
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
export NVS_MULTIVIEW_ROOT="${NVS_MULTIVIEW_ROOT:-/mnt/data/multi_future_view/navtrain_full_allcosmos_cuda_v1_gpu64/navtrain_full_allcosmos_cuda_v1_gpu64}"
export NVS_SENSOR_ROOT="${NVS_SENSOR_ROOT:-/mnt/data/navsim/trainval_all/trainval_sensor_blobs/trainval}"
export NVS_LOG_ROOT="${NVS_LOG_ROOT:-/mnt/data/navsim/trainval_navsim_logs/trainval}"
export NVS_CACHE_ROOT="${NVS_CACHE_ROOT:-/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtrain_20261005/metric_cache}"
export NVS_DINO_CHECKPOINT="${NVS_DINO_CHECKPOINT:-/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors}"
export NVS_WAN_BACKBONE="${NVS_WAN_BACKBONE:-${PROJECT_ROOT}/checkpoints/ActionDiT_Wan22_256hdim_3outdim.pt}"

NVS_OUTPUT_ROOT="${NVS_OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts}"
# NOT future_target_cache: that root holds the retired one-CLS-query-per-frame build
# ([K,8,1024], fingerprint 625c7a99...), which the current 32-token readout cannot
# use.  Pointing at it would skip Stage 0 -- standardizer.pt is there -- and then fail
# in Stage 1.  This root is built for the four-register-query readout ([K,32,1024]).
TARGET_CACHE_ROOT="${NVS_TARGET_CACHE_ROOT:-${NVS_OUTPUT_ROOT}/future_target_cache_reg32_full}"
STAGE1_DIR="${NVS_STAGE1_DIR:-${NVS_OUTPUT_ROOT}/stage1_reg32_b12x16_20ep}"
STAGE2_DIR="${NVS_STAGE2_DIR:-${NVS_OUTPUT_ROOT}/navtrain_wan_e2e_ddp}"
STAGE1_CKPT="${NVS_STAGE1_CHECKPOINT:-${STAGE1_DIR}/stage1_checkpoint.pt}"

# ---- Stage 1 / Stage 2 scale (env-overridable) ------------------------------ #
# Stage 1 streams the cache, so it is GPU-bound rather than RAM-bound: the target is
# the 512 KB [K,32,1024] block Stage 0 already paid for, and the history stays uint8
# on the host (11.6 MB per scene in flight, against the 46.3 MB the same frames cost
# as float32) until the deferred /255 on the device.  That is the whole reason there
# are two Stage-1 scripts -- run_stage1_multiview.py holds every scene resident and
# caps out at a few hundred, which is why it stays the smoke and diagnostic harness.
#
# --grad-checkpointing is NOT optional here.  Measured on one 96 GB card: batch 4
# without it is 81 GB, and batch 8 and batch 12 both OOM (~85 GB, dying in the first
# MLP of the DINOv3 trunk).  The cost is per-layer activations of a frozen ViT-L made
# differentiable by LoRA on all 24 blocks -- ~5.8 GB/layer -- so it barely moves with
# batch size and there is no batch at which turning it off is affordable.  With it on,
# batch 12 peaks at 36.8 GB and runs 11.26 s/step.
#
# Steps per epoch = usable_train_scenes / (batch * nproc), drop_last on both.  At
# batch 12 x 16 ranks that is 192 scenes/step, so the ~80k-scene ready corpus minus
# whatever --min-gt-spread-m filters gives roughly 380-415 steps per epoch and
# 7.5k-8.3k steps over 20 epochs.  --max-steps defaults to 0 so the epoch count is the
# binding constraint; a non-zero --max-steps has to be read together with --epochs,
# because the loop AND the cosine schedule both end at whichever comes first.
STAGE1_STEPS="${NVS_STAGE1_STEPS:-0}"
STAGE1_EPOCHS="${NVS_STAGE1_EPOCHS:-20}"
STAGE1_BATCH="${NVS_STAGE1_BATCH:-12}"
STAGE1_GRAD_ACCUM="${NVS_STAGE1_GRAD_ACCUM:-1}"
STAGE1_WORKERS="${NVS_STAGE1_WORKERS:-6}"
# bf16 measured 3.97x faster than fp32 on one card and cut peak memory 29.69 -> 19.72
# GiB per scene.  It is not the script's default -- that stays fp32 so an old command
# line reproduces the run it was written for -- so the launcher has to ask for it.
STAGE1_PRECISION="${NVS_STAGE1_PRECISION:-bf16}"
# Kept at 2e-4 even though the global batch went 64 -> 192 scenes: this is the value
# the earlier 20k-step run used, so a change in the register's behaviour is attributable
# to the 32-token target and the epoch count rather than to a learning rate nobody
# measured.  sqrt-scaling for the 3x larger batch would suggest ~3.5e-4; export
# NVS_STAGE1_LR to try it.
STAGE1_LR="${NVS_STAGE1_LR:-2e-4}"
STAGE1_WARMUP="${NVS_STAGE1_WARMUP:-500}"
# 20 steps is ~20 log lines per epoch, ~400 over the run: enough for a loss/cosine
# curve, and each one costs a single 3-value all-reduce.
STAGE1_LOG_EVERY="${NVS_STAGE1_LOG_EVERY:-20}"
STAGE1_SAVE_EVERY="${NVS_STAGE1_SAVE_EVERY:-400}"
# Scenes the rank-0 verdict pass scores.  Rank 0 only, between barriers, so these
# cost wall clock on one card and nothing on the other fifteen.  A pass over
# 32 train + 32 eval scenes is two forwards each, ~2 min; 400 steps is roughly one
# pass per epoch, ~20 passes and ~40 min over the whole run.  Much denser and the
# verdict, not the optimizer, becomes the schedule.
STAGE1_TRAIN_METRIC_SCENES="${NVS_STAGE1_TRAIN_METRIC_SCENES:-32}"
STAGE1_EVAL_METRIC_SCENES="${NVS_STAGE1_EVAL_METRIC_SCENES:-32}"
STAGE1_MEAN_TARGET_SCENES="${NVS_STAGE1_MEAN_TARGET_SCENES:-512}"
STAGE1_EVAL_EVERY="${NVS_STAGE1_EVAL_EVERY:-400}"
# Metres of mean pairwise GT separation a scene needs to teach trajectory conditioning.
# Stage 0 deliberately does NOT apply this: it caches every scene it can extract, so the
# floor stays a tuning knob that costs a re-filter instead of the ~4-7 h a Stage-0
# rebuild would.  --skip-degenerate in Stage 0 is the older label-uniqueness
# test, which cannot see K distinct labels sitting centimetres apart.
STAGE1_MIN_GT_SPREAD_M="${NVS_STAGE1_MIN_GT_SPREAD_M:-0.1}"
NPROC="${NVS_NPROC_PER_NODE:-16}"
STAGE2_BATCH="${NVS_STAGE2_BATCH:-12}"
STAGE2_EPOCHS="${NVS_STAGE2_EPOCHS:-50}"
# Stage 2 shares Stage 1's frozen DINOv3 ViT-L trunk but had NO precision flag, so the
# joint backward ran that trunk in fp32 -- ~45s/step (71% of the step) recomputed under
# --grad-checkpointing.  This is the SAME lever Stage 1 already ships (bf16, "3.97x faster
# than fp32"); passing it to Stage 2 brings the joint step to ~12.9s warm marginal (4.3x),
# numerics preserved (matched_ade / oracle_best_pdms identical, loss_total <0.1%).  Set
# NVS_STAGE2_PRECISION=fp32 to restore the exact pre-flag fp32 path.
STAGE2_PRECISION="${NVS_STAGE2_PRECISION:-bf16}"

# ---- self-log: must precede every preflight abort --------------------------- #
# Tee'd by the launcher instead of redirected by the caller.  This closes two distinct
# ways a run can leave nothing behind.  One is a backgrounded DLC job command, which
# exits 0 before the child opens its redirect.  The other is the one that actually bit:
# every `exit 2` / `exit 3` below used to fire before any log file existed, so a job
# that aborted on a missing mount path wrote zero bytes to CPFS and was indistinguishable
# from a job nobody ever submitted.  Logging first means "the launcher ran" always leaves
# a record of exactly how far it got and which paths it resolved them from.
mkdir -p "$NVS_OUTPUT_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCHER_LOG="${NVS_OUTPUT_ROOT}/launcher_${STAMP}.log"
exec > >(tee -a "$LAUNCHER_LOG") 2>&1

echo "[LAUNCH] self-log=${LAUNCHER_LOG}"
echo "[LAUNCH] host=$(hostname 2>/dev/null || echo unknown) | cwd=$PWD | $(date -Is)"
echo "[LAUNCH] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[LAUNCH] PYBIN=${PYBIN} | NVS_OUTPUT_ROOT=${NVS_OUTPUT_ROOT}"
echo "[LAUNCH] mount multiview=${NVS_MULTIVIEW_ROOT}"
echo "[LAUNCH] mount sensor=${NVS_SENSOR_ROOT}"
echo "[LAUNCH] mount log=${NVS_LOG_ROOT}"
echo "[LAUNCH] mount cache=${NVS_CACHE_ROOT}"
echo "[LAUNCH] mount multigt=${NVS_MULTIGT_ROOT}"
echo "[LAUNCH] mount dino=${NVS_DINO_CHECKPOINT}"

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
# Existence of the directory is not enough.  multiview_data.scene_is_ready also
# requires completion.json plus future.mp4 and ego_pose.npy in *every* trajectory_k,
# and a rollout directory can hold scene folders that satisfy none of it -- the
# previous default pointed at exactly such a directory, so Stage 0 and Stage 1 both
# died with "no ready Multi-GT scenes" after the whole env had been sourced.
# `find -quit` stops at the first hit (0.09 s on the 103075-scene corpus); a shell
# glob over the same root expands every entry before the loop can break and costs
# ~39 s.  It checks only completion.json, so it is a cheap approximation of
# scene_is_ready, not a substitute: it catches "wrong root" and "rollout still in
# flight", not a scene whose completion.json exists but whose mp4 is missing.
if [ -z "${NVS_SKIP_STAGE0:-}" ]; then
  first_ready="$(find "$NVS_MULTIVIEW_ROOT" -mindepth 2 -maxdepth 2 -name completion.json -print -quit 2>/dev/null)"
  if [ -n "$first_ready" ]; then
    echo "[PREFLIGHT] multiview root has ready scenes, e.g. $(basename "$(dirname "$first_ready")")"
  else
    echo "[PREFLIGHT] Aborting: no scene under $NVS_MULTIVIEW_ROOT carries completion.json" >&2
    echo "[PREFLIGHT]          scene_is_ready needs completion.json + future.mp4 + ego_pose.npy per trajectory_k." >&2
    echo "[PREFLIGHT]          Point NVS_MULTIVIEW_ROOT at a finished rollout corpus." >&2
    exit 2
  fi
fi
if [ -z "${NVS_SKIP_STAGE1:-}" ]; then
  # Stage 1 reads the cache and nothing else, so what it needs is a standardizer plus
  # at least batch*nproc payloads -- below that DistributedSampler(drop_last) yields an
  # empty loader and all 16 ranks fail after the model has been built.  The scene
  # corpus itself is Stage 0's business; checking it here would only duplicate a
  # 103075-file walk that resolve_scene_sets already does once, on rank 0.
  if [ ! -e "${TARGET_CACHE_ROOT}/standardizer.pt" ] && [ -n "${NVS_SKIP_STAGE0:-}" ]; then
    echo "[PREFLIGHT] MISSING: ${TARGET_CACHE_ROOT}/standardizer.pt (Stage 0 is skipped but Stage 1 reads its cache)" >&2
    echo "[PREFLIGHT]          Build it first, or drop NVS_SKIP_STAGE0 so this run builds it." >&2
    exit 2
  fi
  cached_targets=$(find "${TARGET_CACHE_ROOT}/targets" -mindepth 2 -maxdepth 2 -name '*.pt' 2>/dev/null | head -n $((STAGE1_BATCH * NPROC + 1)) | wc -l)
  if [ -e "${TARGET_CACHE_ROOT}/standardizer.pt" ] && [ "$cached_targets" -le "$((STAGE1_BATCH * NPROC))" ]; then
    echo "[PREFLIGHT] Aborting: ${TARGET_CACHE_ROOT}/targets holds <= $((STAGE1_BATCH * NPROC)) payloads," >&2
    echo "[PREFLIGHT]          which is one global batch (batch ${STAGE1_BATCH} x ${NPROC} ranks); drop_last would leave nothing." >&2
    echo "[PREFLIGHT]          Finish the Stage-0 build, or lower NVS_STAGE1_BATCH / NVS_NPROC_PER_NODE." >&2
    exit 2
  fi
  echo "[PREFLIGHT] Stage 1 streams the cache: ~11.6 MB uint8 history per scene in flight per worker, not resident scenes."
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

echo "[LAUNCH] two-stage e2e | nproc=${NPROC} | stage2 batch/GPU=${STAGE2_BATCH} => global $((STAGE2_BATCH * NPROC))"
echo "[LAUNCH] target_cache=${TARGET_CACHE_ROOT}"
echo "[LAUNCH] stage1=${STAGE1_DIR}  stage2=${STAGE2_DIR}"
echo "[LAUNCH] python=$("$PYBIN" -c "import sys,timm; print(sys.executable, timm.__version__)")"

# NPROC comes from NVS_NPROC_PER_NODE and nothing compares it to the node.  A mismatch
# does not fail fast either: --standalone starts 16 ranks on 8 devices and dies inside
# NCCL init minutes later under a wall of output that never says "too many ranks".
gpu_count="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
if [ "$gpu_count" -gt 0 ] && [ "$gpu_count" -ne "$NPROC" ]; then
  echo "[LAUNCH] FATAL: NVS_NPROC_PER_NODE=${NPROC} but this node exposes ${gpu_count} device(s)." >&2
  echo "[LAUNCH]        Re-submit with NVS_NPROC_PER_NODE=${gpu_count}.  Stage 1's global batch" >&2
  echo "[LAUNCH]        is batch x nproc, so NVS_STAGE1_LR may want a second look as well." >&2
  exit 4
fi
echo "[LAUNCH] devices=${gpu_count:-unknown} | nproc=${NPROC}"

# ---- preflight: the cache's readout must be the one this code expects --------- #
# target_fingerprint does NOT cover the register token count -- it locks the Teacher
# checkpoint, the query temperature and heads, and the standardizer's mean/scale.  So
# a cache built under the retired one-CLS-query-per-frame readout carries a
# fingerprint this code accepts and a [K,8,1024] payload it cannot use, and the
# failure would land in Stage 1's first batch, after Stage 0 had already "succeeded"
# by skipping every scene it found cached.  Read one payload and compare instead.
"$PYBIN" - "$TARGET_CACHE_ROOT" <<'PYEOF' || exit 5
import os
import sys

from nvs_v6_wan_e2e.data import REGISTER_TOKENS
from nvs_v6_wan_e2e.target_cache import read_scene_payload

root = sys.argv[1]
if not os.path.exists(os.path.join(root, "standardizer.pt")):
    print(f"[PREFLIGHT] no standardizer under {root}; Stage 0 will build a {REGISTER_TOKENS}-token cache")
    sys.exit(0)

targets = os.path.join(root, "targets")
payload_path = None
for bucket in sorted(os.listdir(targets)) if os.path.isdir(targets) else []:
    directory = os.path.join(targets, bucket)
    if not os.path.isdir(directory):
        continue
    names = sorted(name for name in os.listdir(directory) if name.endswith(".pt"))
    if names:
        payload_path = os.path.join(directory, names[0])
        break
if payload_path is None:
    print(f"[PREFLIGHT] standardizer present, no payload yet; Stage 0 will resume into a {REGISTER_TOKENS}-token cache")
    sys.exit(0)

shape = tuple(read_scene_payload(payload_path)["future_register"].shape)
if shape[1] != REGISTER_TOKENS:
    print(
        f"[PREFLIGHT] FATAL: {payload_path} holds future_register {shape}, but this code reads "
        f"{REGISTER_TOKENS} register tokens per scene ({shape[1]} means the retired "
        "one-CLS-query-per-frame readout). The cache is not reusable: point "
        "NVS_TARGET_CACHE_ROOT at a fresh directory and let Stage 0 rebuild it.",
        file=sys.stderr,
    )
    sys.exit(5)
print(f"[PREFLIGHT] cache readout matches the code: future_register {shape}, REGISTER_TOKENS={REGISTER_TOKENS}")
PYEOF

# ---- STAGE 0: Future Register target cache -------------------------------- #
if [ -z "${NVS_SKIP_STAGE0:-}" ]; then
  # Completeness is read off the manifest, not off standardizer.pt.  The standardizer is
  # written first, so its presence used to mean "skip" even for a build killed an hour in
  # -- and Stage 1 would then abort on the payload-count preflight, needing
  # NVS_FORCE_STAGE0=1 to get going again.  The builder itself already resumes (it skips
  # any scene whose target exists), so the only decision here is whether to launch it.
  #
  # --rebuild-index is deliberately NOT passed.  Deleting _ready_scenes.json would rescan
  # the corpus and re-split it; on a resume that could move a scene the standardizer was
  # fit on into the eval set, and the held-out numbers would be leak-free in name only.
  # With the sidecar in place the builder reuses the split it fit under.
  stage0_done=$("$PYBIN" - "$TARGET_CACHE_ROOT" <<'PYEOF'
import json
import os
import sys

path = os.path.join(sys.argv[1], "_manifest.json")
if not os.path.exists(path):
    print("no")
    sys.exit(0)
manifest = json.load(open(path))
indexed = int(manifest.get("scene_index_count", 0))
accounted = int(manifest.get("cached_scene_count", 0)) + int(manifest.get("failed_scene_count", 0))
print("yes" if indexed > 0 and accounted >= indexed else "no")
PYEOF
)
  if [ "$stage0_done" = "yes" ] && [ -z "${NVS_FORCE_STAGE0:-}" ]; then
    echo "[STAGE0] manifest reports the build complete; skipping (set NVS_FORCE_STAGE0=1 to rebuild)"
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

  # Stage 1 writes a checkpoint every --save-every steps, but the launcher never passed
  # --resume and that flag defaults to None, so every restart began at step 0 and then
  # overwrote the very checkpoint it was restarting next to.  On a run measured in tens of
  # hours that turns any preemption into total loss.  Stage 0 already had the equivalent
  # behaviour (the builder skips payloads that exist); this makes Stage 1 match it.
  # NVS_STAGE1_FRESH=1 opts out when a restart from scratch is what you actually want.
  RESUME_ARGS=()
  if [ -n "${NVS_STAGE1_FRESH:-}" ]; then
    [ -e "$STAGE1_CKPT" ] && echo "[STAGE1] NVS_STAGE1_FRESH=1: ignoring ${STAGE1_CKPT}, starting from step 0."
  elif [ -e "$STAGE1_CKPT" ]; then
    resumed_step="$("$PYBIN" -c 'import sys,torch;print(int(torch.load(sys.argv[1],map_location="cpu",weights_only=True).get("opt_step",0)))' "$STAGE1_CKPT" 2>/dev/null)"
    RESUME_ARGS=(--resume "$STAGE1_CKPT")
    echo "[STAGE1] RESUMING from ${STAGE1_CKPT} at opt_step=${resumed_step:-unknown}"
    echo "[STAGE1]          it stops again at max_steps=${STAGE1_STEPS} (0 = epochs bind, ${STAGE1_EPOCHS} epochs); raise that or set NVS_STAGE1_FRESH=1 if this is not what you meant."
  fi

  echo "[STAGE1] future representation | cache=${TARGET_CACHE_ROOT} epochs=${STAGE1_EPOCHS} max_steps=${STAGE1_STEPS} (0 = epochs bind) precision=${STAGE1_PRECISION} batch/GPU=${STAGE1_BATCH} => global $((STAGE1_BATCH * NPROC)) | log=${LOG1}"
  "$PYBIN" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    scripts/run_stage1_cache_ddp.py \
    --target-cache-root "$TARGET_CACHE_ROOT" \
    --dino-checkpoint "$NVS_DINO_CHECKPOINT" \
    --wan-backbone "$NVS_WAN_BACKBONE" \
    --target-dim 256 \
    --depth 8 \
    --heads 4 \
    --candidates 64 \
    --skip-degenerate \
    --min-gt-spread-m "$STAGE1_MIN_GT_SPREAD_M" \
    --grad-checkpointing \
    --precision "$STAGE1_PRECISION" \
    --batch-size "$STAGE1_BATCH" \
    --grad-accum "$STAGE1_GRAD_ACCUM" \
    --workers "$STAGE1_WORKERS" \
    --epochs "$STAGE1_EPOCHS" \
    --max-steps "$STAGE1_STEPS" \
    --lr "$STAGE1_LR" \
    --warmup "$STAGE1_WARMUP" \
    --clip-grad-norm 1.0 \
    --train-metric-scenes "$STAGE1_TRAIN_METRIC_SCENES" \
    --eval-metric-scenes "$STAGE1_EVAL_METRIC_SCENES" \
    --mean-target-scenes "$STAGE1_MEAN_TARGET_SCENES" \
    --eval-every "$STAGE1_EVAL_EVERY" \
    --check-scorer \
    --log-every "$STAGE1_LOG_EVERY" \
    --save-every "$STAGE1_SAVE_EVERY" \
    --seed 17 \
    --output-dir "$STAGE1_DIR" \
    "${RESUME_ARGS[@]}" \
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
    echo "[STAGE2] MISSING $STAGE1_CKPT -- Stage 2 must restore Stage 1's eta+theta." >&2
    exit 4
  fi
  FUTURE_ARGS=(--future-target none)
  if [ -e "${TARGET_CACHE_ROOT}/standardizer.pt" ]; then
    # Stage 2 continues to align the register to the same query-extracted future Stage 1
    # was trained on, at the auxiliary scale FastWAM's ema_joint uses.  The register
    # width comes from this standardizer (and must agree with the Stage-1 payload), not
    # from --target-dim, which is the DiT's hidden width.
    FUTURE_ARGS=(--future-target cache --target-cache-root "$TARGET_CACHE_ROOT" --lambda-future 0.1)
  else
    echo "[STAGE2] WARNING: no standardizer.pt under ${TARGET_CACHE_ROOT}; running with --future-target none" >&2
    echo "[STAGE2]          the 32 register tokens stay in the DiT sequence (Stage 1 trained the trunk at 121)" >&2
    echo "[STAGE2]          but the branch is frozen: lambda_future=0 means no future gradient is built." >&2
  fi
  LOG2="${STAGE2_DIR}/train_${STAMP}.log"
  echo "[STAGE2] joint eta+theta+psi | stage1=${STAGE1_CKPT} | precision=${STAGE2_PRECISION} | future=${FUTURE_ARGS[*]} | log=${LOG2}"
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
    --precision "$STAGE2_PRECISION" \
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
