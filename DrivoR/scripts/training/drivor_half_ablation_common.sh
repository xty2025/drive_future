#!/usr/bin/env bash
# Sourced only by the independent half-data experiment entries.
DRIVOR_HALF_VARIANT="${DRIVOR_HALF_VARIANT,,}"
case "${DRIVOR_HALF_VARIANT}" in
    a23) DRIVOR_HALF_AGENT=drivoR_action_dit_dinov3_vitl_human_xl1_diverse8_l1 ;;
    a27) DRIVOR_HALF_AGENT=drivoR_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register ;;
    a28) DRIVOR_HALF_AGENT=drivoR_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_scorer ;;
    a28_random_dit|a28_patch16|a28_register_first|a28_bidirectional|a28_shared_vit|a28_frozen_vit|a28_dit1024|a28_wan5b|a28_vae16|a28_grouped_noise|a28_mot|a28_wan21_1p3b)
        DRIVOR_HALF_AGENT="drivoR_half_${DRIVOR_HALF_VARIANT}" ;;
    *) printf 'Unknown half-data experiment: %s\n' "${DRIVOR_HALF_VARIANT}" >&2; exit 2 ;;
esac
DRIVOR_HALF_COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRIVOR_HALF_ROOT="$(cd -- "${DRIVOR_HALF_COMMON_DIR}/../.." && pwd)"
DRIVOR_HALF_PROJECT="$(cd -- "${DRIVOR_HALF_ROOT}/.." && pwd)"
export DRIVOR_PYTHON="${DRIVOR_PYTHON:-/cpfs29jb/data/zhaozc/miniconda3/envs/drive_future/bin/python}"
export NAVSIM_EXP_ROOT="${DRIVOR_HALF_ABLATION_ROOT:-${DRIVOR_HALF_ROOT}/exp/navtrain_half_ablations}"
export SUBSCORE_PATH="${NAVSIM_EXP_ROOT}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${DRIVOR_HALF_PROJECT}/data/navsim}"
export NAVSIM_METRIC_CACHE_PATH="${NAVSIM_METRIC_CACHE_PATH:-${DRIVOR_HALF_PROJECT}/data/metric_cache_navtrain}"
export NAVSIM_TEST_METRIC_CACHE_PATH="${NAVSIM_TEST_METRIC_CACHE_PATH:-${DRIVOR_HALF_PROJECT}/data/metric_cache_navtest}"
# Reuse the completed RGB/image caches when changing the experiment output root.
export DRIVOR_DIVERSE8_FRAMES_ROOT="${DRIVOR_DIVERSE8_FRAMES_ROOT:-${DRIVOR_HALF_ROOT}/exp/world4scorer_data/frames_diverse8}"
export NAVSIM_OSS_IMAGE_CACHE_DIR="${NAVSIM_OSS_IMAGE_CACHE_DIR-${DRIVOR_HALF_ROOT}/exp/oss_image_cache}"
export DRIVOR_RUN_ID="${DRIVOR_RUN_ID:-${PAI_JOB_ID:-half_${DRIVOR_HALF_VARIANT}_seed2_run1}}"
DRIVOR_HALF_TRAIN_EXP="${DRIVOR_TRAIN_EXPERIMENT:-training_drivoR_half_${DRIVOR_HALF_VARIANT}}"
DRIVOR_HALF_EVAL_EXP="${DRIVOR_EVAL_EXPERIMENT:-eval_navtest_drivoR_half_${DRIVOR_HALF_VARIANT}}"
export DRIVOR_AUTO_RESUME="${DRIVOR_AUTO_RESUME:-false}"
export DRIVOR_VAL_FP32=true
export DRIVOR_EVAL_PRECISION=32-true
DRIVOR_HALF_STRATEGY=ddp
DRIVOR_HALF_PRECISION="${DRIVOR_PRECISION:-bf16-mixed}"
if [[ "${DRIVOR_HALF_VARIANT}" == a28_wan5b || "${DRIVOR_HALF_VARIANT}" == a28_mot || "${DRIVOR_HALF_VARIANT}" == a28_wan21_1p3b ]]; then
    DRIVOR_HALF_STRATEGY=deepspeed_stage_2
    # The model opts into BF16 forward autocast, keeping FP32 master parameters
    # for the same FP32 epoch validation as the smaller A28 experiments.
    DRIVOR_HALF_PRECISION=32-true
fi
