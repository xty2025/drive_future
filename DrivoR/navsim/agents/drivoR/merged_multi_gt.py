"""Concatenate valid trajectories from the primary and additional GT exports."""

import torch
from omegaconf import OmegaConf

from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder
from .multi_gt import MultiTrajectoryTargetBuilder


class MergedMultiTrajectoryTargetBuilder(AbstractTargetBuilder):
    """Use primary GTs plus optional per-scene additions, with equal GT weights.

    Source readers retain the existing validity/padding/time-grid checks. Their
    human fallback is retained only for the primary source: an empty/missing
    additional source contributes zero GTs, not another copy of the human GT.
    """

    def __init__(self, config):
        self.max_gt = int(config.get("max_gt", 8))
        if config.get("weighting", "equal") != "equal":
            raise ValueError("Merged multi-GT supervision uses weighting=equal")
        self.primary = MultiTrajectoryTargetBuilder(config)
        self.additional = [MultiTrajectoryTargetBuilder(OmegaConf.create(dict(
            root=str(root), max_gt=self.max_gt, missing_policy="original",
            weighting="equal", matching=config.get("matching", "nearest"))))
            for root in config.get("additional_roots", [])]

    def get_unique_name(self):
        return f"drivor_multi_gt_merged_v1_k{self.max_gt}"

    def compute_targets(self, scene):
        token = scene.scene_metadata.initial_token
        fallback = scene.get_future_trajectory(num_trajectory_frames=8).poses
        return self.load(token, fallback)

    def load(self, token, fallback):
        primary = self.primary.load(token, fallback)
        pieces = [primary["trajectory_multi_gt"][primary["trajectory_multi_gt_valid"]]]
        primary_count = len(pieces[0])
        for reader in self.additional:
            result = reader.load(token, fallback)
            if not result["multi_gt_fallback"]:
                pieces.append(result["trajectory_multi_gt"][result["trajectory_multi_gt_valid"]])
        trajectories = torch.cat(pieces)
        count = len(trajectories)
        if count > self.max_gt:
            raise ValueError(f"{token} has {count} merged GTs, exceeding multi_gt.max_gt={self.max_gt}")
        output = torch.zeros(self.max_gt, 8, 3, dtype=torch.float32)
        valid = torch.zeros(self.max_gt, dtype=torch.bool)
        output[:count], valid[:count] = trajectories, True
        return {"trajectory_multi_gt": output, "trajectory_multi_gt_valid": valid,
                "multi_gt_count": torch.tensor(count, dtype=torch.long),
                "multi_gt_fallback": primary["multi_gt_fallback"],
                "multi_gt_primary_count": torch.tensor(primary_count, dtype=torch.long),
                "multi_gt_additional_count": torch.tensor(count - primary_count, dtype=torch.long)}
