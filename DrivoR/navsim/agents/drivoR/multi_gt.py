"""Optional multi-trajectory targets and GT-to-proposal winner-only losses."""

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder


def multi_gt_enabled(config):
    options = config.get("multi_gt", {}) if isinstance(config, Mapping) else getattr(config, "multi_gt", {})
    return bool(options.get("enabled", False))


class MultiTrajectoryTargetBuilder(AbstractTargetBuilder):
    """Read local 8-point GTs; padding is excluded using BOTH source masks.

    The supplied directory already contains rear-axle-relative 0.5s targets in
    gt_trajectories.npy. selected_ego_poses_global.npy is a separate dense global
    representation and must not be passed directly to the trajectory loss.
    """

    def __init__(self, config):
        self.root = Path(config.root)
        self.max_gt = int(config.get("max_gt", 4))
        self.missing_policy = str(config.get("missing_policy", "error"))
        self.weighting = str(config.get("weighting", "equal"))
        self.matching = str(config.get("matching", "nearest"))
        if self.weighting not in ("equal", "drive_jepa"):
            raise ValueError("multi_gt.weighting must be equal or drive_jepa")
        if self.matching not in ("nearest", "hungarian"):
            raise ValueError("multi_gt.matching must be nearest or hungarian")
        if self.matching == "hungarian" and self.weighting != "equal":
            raise ValueError("Hungarian multi-GT supervision requires multi_gt.weighting=equal")
        if self.max_gt < 1:
            raise ValueError("multi_gt.max_gt must be positive")
        if self.missing_policy not in ("error", "original"):
            raise ValueError("multi_gt.missing_policy must be error or original")
        if not self.root.is_dir():
            raise FileNotFoundError(f"Multi-GT root not found: {self.root}")

    def get_unique_name(self):
        # Distinct from drivor_target so old feature caches cannot silently
        # provide a single trajectory when multi-GT targets were requested.
        # Weighted supervision also needs the exported human/pseudo identity.
        # Keep existing equal-weight (including Action-DiT) caches compatible.
        return "drivor_multi_gt_human_v1" if self.weighting == "drive_jepa" else "drivor_multi_gt_v1"

    def compute_targets(self, scene):
        token = scene.scene_metadata.initial_token
        fallback = scene.get_future_trajectory(num_trajectory_frames=8).poses
        return self.load(token, fallback)

    def load(self, token, fallback):
        folder = self.root / token
        trajectory_path = folder / "gt_trajectories.npy"
        fallback_used = False
        human = None
        if not trajectory_path.is_file():
            if self.missing_policy == "error":
                raise FileNotFoundError(f"Multi-GT trajectories missing for {token}: {trajectory_path}")
            trajectories = np.asarray(fallback, dtype=np.float32)[None]
            valid = np.ones(1, dtype=bool)
            fallback_used = True
        else:
            trajectories = np.asarray(np.load(trajectory_path, allow_pickle=False), dtype=np.float32)
            if trajectories.ndim != 3 or trajectories.shape[1:] != (8, 3):
                raise ValueError(f"Expected Multi-GT [K,8,3] at {trajectory_path}, got {trajectories.shape}")
            valid = np.asarray(np.load(folder / "gt_valid_mask.npy", allow_pickle=False), dtype=bool)
            if valid.shape != (len(trajectories),):
                raise ValueError(f"Multi-GT mask shape mismatch for {token}: {valid.shape}")
            metadata = json.loads((folder / "trajectory.json").read_text())
            # This source can mark repeated padded rows valid in the npy mask.
            padded = np.asarray(metadata["gt_padded"], dtype=bool)
            if padded.shape != valid.shape:
                raise ValueError(f"Multi-GT gt_padded shape mismatch for {token}")
            if not np.allclose(metadata["target_times"], np.arange(1, 9) * 0.5, rtol=0, atol=1e-6):
                raise ValueError(f"Multi-GT target times must be 0.5,...,4.0 seconds for {token}")
            valid = valid & ~padded
            if self.weighting == "drive_jepa":
                labels = np.asarray(metadata["gt_labels"])
                if labels.shape != valid.shape:
                    raise ValueError(f"Multi-GT gt_labels shape mismatch for {token}")
                human = labels == "human_log"
            if not np.isfinite(trajectories[valid]).all():
                raise ValueError(f"Non-finite valid Multi-GT trajectory for {token}")
            if not valid.any():
                if self.missing_policy == "error":
                    raise ValueError(f"No valid non-padded Multi-GT trajectories for {token}")
                trajectories = np.asarray(fallback, dtype=np.float32)[None]
                valid = np.ones(1, dtype=bool)
                fallback_used = True

        if fallback_used:
            human = np.ones(1, dtype=bool)
        actual = trajectories[valid]
        if len(actual) > self.max_gt:
            raise ValueError(f"{token} has {len(actual)} GTs, exceeding multi_gt.max_gt={self.max_gt}")
        output = np.zeros((self.max_gt, 8, 3), dtype=np.float32)
        mask = np.zeros(self.max_gt, dtype=bool)
        output[:len(actual)] = actual
        mask[:len(actual)] = True
        result = {"trajectory_multi_gt": torch.from_numpy(output),
                  "trajectory_multi_gt_valid": torch.from_numpy(mask),
                  "multi_gt_count": torch.tensor(len(actual), dtype=torch.long),
                  "multi_gt_fallback": torch.tensor(fallback_used, dtype=torch.bool)}
        if self.weighting == "drive_jepa":
            human_mask = np.zeros(self.max_gt, dtype=bool)
            human_mask[:len(actual)] = human[valid]
            result["trajectory_multi_gt_is_human"] = torch.from_numpy(human_mask)
        return result


def valid_multi_gt(targets):
    trajectories = targets["trajectory_multi_gt"].detach().float()
    valid = targets["trajectory_multi_gt_valid"].bool()
    if trajectories.ndim != 4 or trajectories.shape[-2:] != (8, 3) or valid.shape != trajectories.shape[:2]:
        raise ValueError("Expected multi-GT [B,K,8,3] and validity [B,K]")
    # Invalid rows may be padded with arbitrary values; mask before arithmetic.
    return trajectories.masked_fill(~valid[..., None, None], 0.), valid


def mean_valid_gt_winners(cost, valid, proposal_weights=None, matching="nearest"):
    """Select proposals from cost [B,K,M], then average valid GTs inside scenes.

    nearest permits shared winners; hungarian minimizes the joint cost with one
    distinct proposal per valid GT. Only assignment indices are detached: the
    selected costs stay on the original device and keep their regression graph.
    """
    if matching == "nearest":
        winner_cost, indices = cost.min(dim=-1)
    elif matching == "hungarian":
        # Transfer one small B x K x M matrix, never individual trajectories or
        # full model features. Solve only valid rows; padding must not compete.
        cost_cpu = cost.detach().float().cpu().numpy()
        valid_cpu = valid.detach().cpu().numpy()
        assignment = np.full(valid_cpu.shape, -1, dtype=np.int64)
        for batch_index, mask in enumerate(valid_cpu):
            gt_rows = np.flatnonzero(mask)
            if len(gt_rows) > cost.shape[-1]:
                raise ValueError("Hungarian matching needs at least as many proposals as valid GTs")
            if len(gt_rows):
                rows, columns = linear_sum_assignment(cost_cpu[batch_index, gt_rows])
                assignment[batch_index, gt_rows[rows]] = columns
        indices = torch.as_tensor(assignment, device=cost.device)
        winner_cost = cost.gather(-1, indices.clamp_min(0)[..., None]).squeeze(-1)
    else:
        raise ValueError(f"Unknown multi_gt.matching: {matching}")
    if proposal_weights is not None:
        # Match by distance before weighting, including zero-weight endpoint t.
        winner_cost = winner_cost * proposal_weights.gather(1, indices.clamp_min(0))
    counts = valid.sum(dim=1)
    per_scene = winner_cost.masked_fill(~valid, 0.).sum(dim=1) / counts.clamp_min(1)
    return per_scene.mean(), indices.masked_fill(~valid, -1)


def multi_gt_l1_loss(proposals, targets, matching="nearest"):
    gt, valid = valid_multi_gt(targets)
    # Same metric as the old DrivoR/Drive-JEPA objective: sum(x,y,h), mean(time).
    cost = (proposals.float()[:, None] - gt[:, :, None]).abs().sum(-1).mean(-1)
    return mean_valid_gt_winners(cost, valid, matching=matching)


def drive_jepa_multi_gt_l1_loss(proposals, targets, human_weight=1.0, other_weight=0.5):
    """Human best-of-64 + 0.5 * mean(other GTs' best-of-64), then batch mean.

    The human term uses the original NAVSIM trajectory. Exclude all exported
    human_log rows from the auxiliary term, regardless of their row positions.
    Scenes without auxiliary GTs contribute zero auxiliary loss, as in JEPA.
    Matching remains independent; different GTs may choose the same proposal.
    """
    gt, valid = valid_multi_gt(targets)
    other_valid = valid & ~targets["trajectory_multi_gt_is_human"].bool()
    human_gt = targets["trajectory"].detach().float()
    human_cost = (proposals.float() - human_gt[:, None]).abs().sum(-1).mean(-1)
    human_values, human_indices = human_cost.min(dim=-1)
    human_loss = human_values.mean()
    other_cost = (proposals.float()[:, None] - gt[:, :, None]).abs().sum(-1).mean(-1)
    other_loss, other_indices = mean_valid_gt_winners(other_cost, other_valid)
    indices = torch.cat((human_indices[:, None], other_indices), dim=1)
    loss = float(human_weight) * human_loss + float(other_weight) * other_loss
    diagnostics = {"multi_gt_human_loss": human_loss.detach(),
                   "multi_gt_other_loss": other_loss.detach(),
                   "multi_gt_other_count": other_valid.sum(dim=1).float().mean()}
    return loss, indices, diagnostics


def multi_gt_velocity_loss(pred_x, normalized_gt, valid, timesteps, schedule, matching="nearest"):
    """All GTs compete over the SAME 64 X predictions using existing v error.

    For each proposed noisy input x_sigma and matched GT g, both v conversions
    use that x_sigma and the same bounded denominator. Their difference is
    (g - pred_x) / sigma_safe, so no candidate-by-GT noise tensor is needed.
    """
    gt = normalized_gt.detach().float().masked_fill(~valid[..., None, None], 0.)
    sigma_safe = schedule.sigma(timesteps).clamp(schedule.sigma_min, 1.)
    error = (gt[:, :, None] - pred_x.float()[:, None]) / sigma_safe[:, None, :, None, None]
    cost = error.square().mean(dim=(-2, -1))
    return mean_valid_gt_winners(cost, valid, schedule.training_weight(timesteps), matching=matching)


def drive_jepa_multi_gt_velocity_loss(pred_x, normalized_human_gt, normalized_gt, valid,
                                    is_human, timesteps, schedule,
                                    human_weight=1.0, other_weight=0.5):
    """Human v loss + weighted mean of non-human GT v losses over the SAME X predictions.

    Each group uses the existing nearest-proposal v MSE and time weight. The
    human term comes from the original NAVSIM trajectory; exported human rows
    are excluded from the auxiliary group. Empty auxiliary groups contribute
    zero before the batch mean. This function never runs the trajectory model.
    """
    other_valid = valid & ~is_human.bool()
    human_valid = torch.ones_like(valid[:, :1])
    human_loss, human_indices = multi_gt_velocity_loss(
        pred_x, normalized_human_gt[:, None], human_valid, timesteps, schedule)
    other_loss, other_indices = multi_gt_velocity_loss(
        pred_x, normalized_gt, other_valid, timesteps, schedule)
    indices = torch.cat((human_indices, other_indices), dim=1)
    loss = float(human_weight) * human_loss + float(other_weight) * other_loss
    diagnostics = {"multi_gt_human_loss": human_loss.detach(),
                   "multi_gt_other_loss": other_loss.detach(),
                   "multi_gt_other_count": other_valid.sum(dim=1).float().mean()}
    return loss, indices, diagnostics


def balanced_gt_inputs(normalized_gt, valid, candidates, fallback):
    """Assign valid GTs cyclically to 64 independent noise slots.

    With 4/2/1 GTs this gives 16/32/64 slots per GT; with 3 it gives 22/21/21.
    This assignment only constructs noisy inputs, not the loss matching: every
    GT still chooses freely from every resulting prediction.
    """
    b, k, steps, dims = normalized_gt.shape
    counts = valid.sum(dim=1)
    source_order = torch.arange(k, device=valid.device)[None].expand(b, -1)
    source_order = source_order.masked_fill(~valid, k).sort(dim=1).values
    positions = torch.arange(candidates, device=valid.device)[None] % counts.clamp_min(1)[:, None]
    indices = source_order.gather(1, positions).clamp_max(k - 1)
    inputs = normalized_gt.gather(1, indices[..., None, None].expand(-1, -1, steps, dims))
    inputs = torch.where(counts[:, None, None, None] > 0, inputs, fallback[:, None])
    return inputs, indices


def matched_proposal_count(indices, valid, candidates):
    selected = torch.nn.functional.one_hot(indices.clamp_min(0), candidates).bool()
    return (selected & valid[..., None]).any(dim=1).sum(dim=1).float().mean()
