#!/usr/bin/env bash
set -euo pipefail

# Train the requested GT ablation, then automatically evaluate last.ckpt in FP32.
case "${1:-}" in
    multigt_poor|single_gt|single_gt_long) ;;
    *) printf '%s\n' 'Usage: run_drivor_action_dit_dinov3_vitl_gt_train_eval.sh multigt_poor|single_gt|single_gt_long [training overrides]' >&2; exit 2 ;;
esac
DRIVOR_GT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${DRIVOR_GT_DIR}/run_drivor_action_dit_dinov3_vitl_aux_train_eval.sh" "$@"
