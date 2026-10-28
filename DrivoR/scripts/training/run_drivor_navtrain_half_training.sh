#!/usr/bin/env bash
set -euo pipefail

# Keep the selected model recipe, halve only train scenes, and keep full val.
# Default recipe: original DrivoR. For another existing training wrapper:
# DRIVOR_HALF_LAUNCHER=run_drivor_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_training.sh \
#   bash DrivoR/scripts/training/run_drivor_navtrain_half_training.sh
# This does not submit a DLC job; DLC's per-node environment works as usual.
DRIVOR_HALF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_HALF_LAUNCHER="${DRIVOR_HALF_LAUNCHER:-run_drivor_training.sh}"
if [[ "${DRIVOR_HALF_LAUNCHER}" != /* ]]; then
    DRIVOR_HALF_LAUNCHER="${DRIVOR_HALF_DIR}/${DRIVOR_HALF_LAUNCHER}"
fi
if [[ ! -f "${DRIVOR_HALF_LAUNCHER}" || "${DRIVOR_HALF_LAUNCHER}" == "${BASH_SOURCE[0]}" || "${DRIVOR_HALF_LAUNCHER##*/}" == run_drivor_navtrain_half_training.sh ]]; then
    printf 'Invalid underlying training launcher: %s\n' "${DRIVOR_HALF_LAUNCHER}" >&2
    exit 2
fi
case "${DRIVOR_HALF_LAUNCHER##*/}" in
    *_full_*|*_train_eval.sh)
        printf 'Use a train-only *_training.sh model launcher, not full or train_eval: %s\n' "${DRIVOR_HALF_LAUNCHER}" >&2
        exit 2 ;;
esac
DRIVOR_HALF_MODEL="${DRIVOR_HALF_LAUNCHER##*/}"
DRIVOR_HALF_MODEL="${DRIVOR_HALF_MODEL#run_}"
DRIVOR_HALF_MODEL="${DRIVOR_HALF_MODEL%_training.sh}"

export DRIVOR_TRAINING_ENTRY=run_training.py
export DRIVOR_TRAINING_CONFIG=half_training
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_${DRIVOR_HALF_MODEL}_navtrain_half}"
export DRIVOR_RUN_ID="${DRIVOR_RUN_ID:-${PAI_JOB_ID:-navtrain_half_seed2_run1}}"
export DRIVOR_AUTO_RESUME="${DRIVOR_AUTO_RESUME:-false}"

# Keep positional overrides before user query flags (--cfg/--package), because
# Hydra's argparse cannot resume its positional list after an option. Discard
# duplicate scope settings here so those flags cannot accidentally restore all
# navtrain scenes or cache-only loading. Model/trainer overrides remain intact.
DRIVOR_HALF_ARGS=()
for DRIVOR_HALF_ARG in "$@"; do
    DRIVOR_HALF_KEY="${DRIVOR_HALF_ARG%%=*}"
    DRIVOR_HALF_KEY="${DRIVOR_HALF_KEY#+}"
    DRIVOR_HALF_KEY="${DRIVOR_HALF_KEY#+}"
    case "${DRIVOR_HALF_KEY}" in
        train_test_split|cache_path|use_cache_without_dataset|runtime.scheduler_dataset_size_from_data|agent.scheduler_args.dataset_size)
            continue ;;
    esac
    DRIVOR_HALF_ARGS+=("${DRIVOR_HALF_ARG}")
done

# The live length also sets the scheduler, so the LR cycle is not left at 85k.
# human_only wrappers reset DRIVOR_TRAINING_CONFIG internally; the final CLI
# option wins over their earlier --config-name=default_training in Hydra.
exec bash "${DRIVOR_HALF_LAUNCHER}" \
    train_test_split=navtrain_half \
    cache_path=null \
    use_cache_without_dataset=false \
    ++runtime.scheduler_dataset_size_from_data=true \
    agent.scheduler_args.dataset_size=42555 \
    "${DRIVOR_HALF_ARGS[@]}" \
    --config-name=half_training
