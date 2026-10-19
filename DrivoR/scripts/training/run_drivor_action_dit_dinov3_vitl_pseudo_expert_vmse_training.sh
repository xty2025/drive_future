#!/usr/bin/env bash
set -euo pipefail

# Original multi-GT v loss + 0.5 * pseudo-expert best-of-64 v MSE.
DRIVOR_VMSE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_action_dit_dinov3_vitl_navtrain_multigt_pseudo_expert_vmse}"
exec bash "${DRIVOR_VMSE_DIR}/run_drivor_action_dit_dinov3_vitl_pseudo_expert_training.sh" \
    agent=drivoR_action_dit_dinov3_vitl_pseudo_expert_vmse \
    "$@"
