#!/usr/bin/env bash
set -euo pipefail

# Shares data/OSS/environment setup; the agent config enables current CAM_F0 only.
# Defaults: 8 nodes x 8 GPUs x batch 1 x accumulation 1 = global batch 64.
# Submit once as the command of a PAI-DLC distributed PyTorch job (8 x 8 GPUs).
# The shared launcher reads DLC's WORLD_SIZE / RANK / MASTER_ADDR / MASTER_PORT.
# Training: one X-prediction forward, normal/long GT best-of-64 v losses.
# Scorer uses the detached 64 normal-branch predictions; evaluation uses 10-step Euler.
# BF16, fused AdamW, CPU scoring pool and global-rank RNG use shared launcher defaults.
# New run:
# DRIVOR_RUN_ID=action_dit_xpred_seed2_run1 bash DrivoR/scripts/training/run_drivor_action_dit_training.sh auto_resume=false train_ckpt_path=null runtime.require_resume=false
# Resume the existing multinode run:
# DRIVOR_RUN_ID=multinode bash DrivoR/scripts/training/run_drivor_action_dit_training.sh auto_resume=true runtime.require_resume=true
# Inspect configuration:
# bash DrivoR/scripts/training/run_drivor_action_dit_training.sh --cfg job --package agent --resolve
# Extra Hydra overrides are passed last and take precedence.
DRIVOR_ACTION_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRIVOR_EXPERIMENT="${DRIVOR_EXPERIMENT:-training_drivoR_action_dit_navtrain}"

exec bash "${DRIVOR_ACTION_SCRIPT_DIR}/run_drivor_training.sh" \
    agent=drivoR_action_dit \
    dataloader.params.batch_size=1 \
    trainer.params.accumulate_grad_batches=1 \
    agent.progress_bar=false \
    agent.config.num_scene_tokens=0 \
    agent.config.full_history_status=false \
    agent.config.long_trajectory_additional_poses=2 \
    agent.config.action_dit.num_inference_steps=10 \
    agent.config.action_dit.loss_reduction=best_of_k \
    "$@"
