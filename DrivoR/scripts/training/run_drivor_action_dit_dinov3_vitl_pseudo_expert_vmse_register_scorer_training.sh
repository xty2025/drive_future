#!/usr/bin/env bash
set -euo pipefail

# Same joint loss; fuse compressed registers into the scoring token.
DRIVOR_PE_REGISTER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_action_dit_dinov3_vitl_navtrain_multigt_pseudo_expert_vmse_register_scorer}"
exec bash "${DRIVOR_PE_REGISTER_DIR}/run_drivor_action_dit_dinov3_vitl_pseudo_expert_vmse_register_training.sh" \
    agent=drivoR_action_dit_dinov3_vitl_pseudo_expert_vmse_register_scorer \
    "$@"
