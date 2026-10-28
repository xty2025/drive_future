"""Scene-indexed candidate bank and frozen future-feature targets, without hashing."""

from pathlib import Path

import numpy as np
import torch

from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder


class BankTargetBuilder(AbstractTargetBuilder):
    def __init__(self, config):
        self.root = Path(config.path)
        self.capacity = int(config.capacity)
        self.missing_policy = str(config.missing_policy)
        self._pack = None
        if not (self.root / 'tokens.npy').is_file() or not (self.root / 'manifest.json').is_file():
            raise FileNotFoundError(f'Candidate bank is missing: {self.root}; prepare the CLOVER bank first')

    def get_unique_name(self):
        return 'drivor_world_bank_v1'

    def _load_pack(self):
        if self._pack is None:
            tokens = np.load(self.root / 'tokens.npy', allow_pickle=False)
            self._index = {(t.decode() if isinstance(t, bytes) else str(t)): i for i, t in enumerate(tokens)}
            self._pack = [np.load(self.root / name, mmap_mode='r', allow_pickle=False)
                          for name in ('bank_candidates.npy', 'bank_subscores.npy', 'bank_mask.npy')]

    def compute_targets(self, scene):
        return self.load(scene.scene_metadata.initial_token)

    def load(self, token):
        self._load_pack()
        candidates = np.zeros((self.capacity, 8, 3), np.float32)
        labels = np.zeros((self.capacity, 7), np.float32)
        valid = np.zeros(self.capacity, bool)
        row = self._index.get(token)
        if row is None:
            if self.missing_policy == 'error':
                raise KeyError(f'Candidate bank has no scene {token}')
        else:
            c, s, m = (array[row] for array in self._pack)
            n = min(len(c), self.capacity)
            candidates[:n], labels[:n], valid[:n] = c[:n], s[:n], m[:n]
            valid &= np.isfinite(candidates).all((1, 2))
            candidates[~valid] = 0
        return {'bank_candidates': torch.from_numpy(candidates),
                'bank_subscores': torch.from_numpy(labels), 'bank_valid': torch.from_numpy(valid)}


class FutureTargetBuilder(AbstractTargetBuilder):
    def __init__(self, config):
        self.root = Path(config.path)
        self.max_gt = int(config.max_gt)
        self.times = np.asarray(config.times, np.float32)
        self.dim = int(config.embedding_dim)
        self.missing_policy = str(config.missing_policy)
        if not self.root.is_dir():
            raise FileNotFoundError(f'Future feature cache is missing: {self.root}; run prepare_world4scorer_data.py')

    def get_unique_name(self):
        return 'drivor_world_future_v2'

    def compute_targets(self, scene):
        return self.load(scene.scene_metadata.initial_token)

    def load(self, token):
        shape = (self.max_gt, len(self.times), self.dim)
        features = np.zeros(shape, np.float32)
        trajectories = np.zeros((self.max_gt, 8, 3), np.float32)
        valid = np.zeros(shape[:2], bool)
        path = self.root / token[:2] / (token + '.npz')
        if not path.is_file():
            if self.missing_policy == 'error':
                raise FileNotFoundError(f'Future features missing for {token}: {path}')
        else:
            with np.load(path, allow_pickle=False) as data:
                source_features, source_valid = data['features'], data['valid']
                source_trajectories = data['trajectories']
                if not np.allclose(data['times'], self.times, rtol=0, atol=1e-6):
                    raise ValueError(f'Future time grid mismatch for {token}')
                n = len(source_features)
                if n > self.max_gt or source_features.shape[1:] != shape[1:]:
                    raise ValueError(f'Future feature shape mismatch for {token}: {source_features.shape}')
                if source_trajectories.shape != (n, 8, 3) or source_valid.shape != (n, len(self.times)):
                    raise ValueError(f'Future trajectory/validity shape mismatch for {token}')
                features[:n], trajectories[:n], valid[:n] = source_features, source_trajectories, source_valid
        return {'future_features': torch.from_numpy(features),
                'future_gt_trajectories': torch.from_numpy(trajectories),
                'future_valid': torch.from_numpy(valid)}
