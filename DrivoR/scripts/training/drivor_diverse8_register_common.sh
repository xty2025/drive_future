#!/usr/bin/env bash
# Sourced by the new train/eval entries; no running legacy launch script changes.
case "${DRIVOR_DIVERSE_VARIANT:?Specify the diverse8 register variant}" in
    action_dit_human_diverse8_l1_register|action_dit_human_diverse8_l1_register_scorer|\
    action_dit_human_xl1_diverse8_l1_register|action_dit_human_xl1_diverse8_l1_register_scorer|\
    action_dit_human_xl1_diverse8_l1_register_human_only|action_dit_human_xl1_diverse8_l1_register_scorer_human_only)
        DRIVOR_DIVERSE_FAMILY=action_dit
        DRIVOR_DIVERSE_SUFFIX="${DRIVOR_DIVERSE_VARIANT#action_dit_}" ;;
    query_human_diverse8_l1_register|query_human_diverse8_l1_register_scorer)
        DRIVOR_DIVERSE_FAMILY=query
        DRIVOR_DIVERSE_SUFFIX="${DRIVOR_DIVERSE_VARIANT#query_}" ;;
    *) printf 'Unknown diverse8 register variant: %s\n' "${DRIVOR_DIVERSE_VARIANT}" >&2; exit 2 ;;
esac
DRIVOR_DIVERSE_AGENT="drivoR_${DRIVOR_DIVERSE_FAMILY}_dinov3_vitl_${DRIVOR_DIVERSE_SUFFIX}"
DRIVOR_DIVERSE_DEFAULT_TRAIN="training_drivoR_${DRIVOR_DIVERSE_FAMILY}_dinov3_vitl_navtrain_${DRIVOR_DIVERSE_SUFFIX}"
DRIVOR_DIVERSE_DEFAULT_EVAL="eval_navtest_${DRIVOR_DIVERSE_FAMILY}_dinov3_vitl_${DRIVOR_DIVERSE_SUFFIX}"
