"""Read the compact, scene-indexed CLOVER expert set through NumPy mmap."""

import json
from pathlib import Path

import numpy as np
import torch

from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder
from .selection import include_human_expert


class PseudoExpertTargetBuilder(AbstractTargetBuilder):
    def __init__(self, config):
        self.root = Path(config.path)
        self.top_k = int(config.top_k)
        self.score_thr = float(config.score_thr)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Pseudo-expert cache missing: {self.root}; run tools/build_clover_pseudo_experts.py first")
        manifest = json.loads(manifest_path.read_text())
        if manifest["top_k"] != self.top_k or manifest["score_thr"] != self.score_thr:
            raise ValueError("Pseudo-expert cache top_k/score_thr differ from config; rebuild the cache")
        self._pack = None

    def get_unique_name(self):
        return f"drivor_pseudo_expert_v1_k{self.top_k}_thr{self.score_thr:g}"

    def _load_pack(self):
        if self._pack is None:
            tokens = np.load(self.root / "tokens.npy", allow_pickle=False)
            self._index = {token.decode(): row for row, token in enumerate(tokens)}
            self._pack = [np.load(self.root / name, mmap_mode="r", allow_pickle=False)
                          for name in ("pseudo_experts.npy", "pseudo_expert_mask.npy")]

    def compute_targets(self, scene):
        human = scene.get_future_trajectory(num_trajectory_frames=8).poses
        return self.load(scene.scene_metadata.initial_token, human)

    def load(self, token, original_expert):
        self._load_pack()
        row = self._index.get(token)
        if row is None:
            experts = np.zeros((self.top_k, 8, 3), np.float32)
            mask = np.zeros(self.top_k, bool)
        else:
            experts, mask = (array[row] for array in self._pack)
        has_scored_expert = bool(mask.any())
        experts, mask = include_human_expert(experts, mask, original_expert)
        return {"pseudo_experts": torch.from_numpy(experts),
                "pseudo_expert_mask": torch.from_numpy(mask),
                "pseudo_expert_fallback": torch.tensor(not has_scored_expert)}
