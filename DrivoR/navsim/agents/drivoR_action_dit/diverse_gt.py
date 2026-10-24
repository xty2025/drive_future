"""Human-primary Action-DiT with the selected Cosmos diverse8 trajectories.

Reuse the existing L1/v-MSE auxiliary objectives with a new data reader. No
CLOVER selection, score threshold, or extra human fallback is applied here.
"""

import json
from pathlib import Path

import numpy as np
import torch

from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder
from .agent import DrivoRActionDiTAgent
from .auxiliary import DrivoRActionDiTAuxiliaryAgent


class DiverseTrajectoryTargetBuilder(AbstractTargetBuilder):
    def __init__(self, config):
        self.root = Path(config.path)
        self.top_k = int(config.get('top_k',8))
        self.missing_policy = str(config.get('missing_policy','ignore'))
        if not self.root.is_dir():
            raise FileNotFoundError(f'Diverse8 GT root not found: {self.root}')
        # Optional extra roots (list or comma-separated string) searched when the main root lacks a scene,
        # e.g. the navtrain scenes with only 3 history frames, generated separately.
        fallback = config.get('fallback_paths', None) or []
        if isinstance(fallback, str):
            fallback = [item for item in fallback.split(',') if item.strip()]
        self.fallback_roots = [Path(str(item).strip()) for item in fallback]
        for root in self.fallback_roots:
            if not root.is_dir():
                raise FileNotFoundError(f'Diverse8 fallback GT root not found: {root}')
        if self.missing_policy not in ('ignore','error'):
            raise ValueError('Diverse8 missing_policy must be ignore or error')

    def get_unique_name(self):
        return f'drivor_diverse8_gt_v1_k{self.top_k}'

    def compute_targets(self, scene):
        return self.load(scene.scene_metadata.initial_token)

    def scene_folder(self, token):
        """The main root's scene folder, else the first fallback root that holds the complete scene."""
        folder = self.root/token
        if not (folder/'complete.json').is_file():
            for root in self.fallback_roots:
                if (root/token/'complete.json').is_file():
                    return root/token
        return folder

    def load(self, token):
        folder = self.scene_folder(token)
        trajectories = np.zeros((self.top_k,8,3),np.float32)
        valid = np.zeros(self.top_k,bool)
        gt_like = np.zeros(self.top_k,bool)
        # The producer writes this marker last. Do not consume partial scenes
        # or cache missing results: a later epoch may see newly completed data.
        missing = not (folder/'complete.json').is_file()
        if missing:
            if self.missing_policy=='error':
                raise FileNotFoundError(f'Diverse8 scene is not complete: {folder}')
        else:
            source = np.asarray(np.load(folder/'gt_trajectories.npy',allow_pickle=False),dtype=np.float32)
            mask = np.asarray(np.load(folder/'gt_valid_mask.npy',allow_pickle=False),dtype=bool)
            like = np.asarray(np.load(folder/'gt_like_mask.npy',allow_pickle=False),dtype=bool)
            metadata = json.loads((folder/'trajectory.json').read_text())
            padded = np.asarray(metadata['gt_padded'],dtype=bool)
            if source.ndim!=3 or source.shape[1:]!=(8,3) or len(source)>self.top_k:
                raise ValueError(f'Expected at most {self.top_k} ego-relative [8,3] trajectories: {folder}')
            if any(value.shape!=(len(source),) for value in (mask,like,padded)):
                raise ValueError(f'Diverse8 mask shape mismatch: {folder}')
            if not np.allclose(metadata['target_times'],np.arange(1,9)*.5,rtol=0,atol=1e-6):
                raise ValueError(f'Diverse8 times must be 0.5,...,4.0 seconds: {folder}')
            if metadata.get('sparse_coordinate_frame','current_ego_rear_axle')!='current_ego_rear_axle':
                raise ValueError(f'Diverse8 trajectories must use current-ego rear-axle coordinates: {folder}')
            mask &= ~padded
            if not np.isfinite(source[mask]).all():
                raise ValueError(f'Non-finite valid Diverse8 trajectory: {folder}')
            count = len(source)
            trajectories[:count] = np.where(mask[:,None,None],source,0.)
            valid[:count] = mask
            # This denotes generated GT-like rollouts, NOT the human itself.
            gt_like[:count] = like & mask
        return {'pseudo_experts':torch.from_numpy(trajectories),
                'pseudo_expert_mask':torch.from_numpy(valid),
                'pseudo_expert_fallback':torch.tensor(False),
                'diverse_gt_like_mask':torch.from_numpy(gt_like),
                'diverse_gt_missing':torch.tensor(missing)}


class DrivoRActionDiTDiverseGTAgent(DrivoRActionDiTAuxiliaryAgent):
    def get_target_builders(self):
        # Bypass the CLOVER reader in the parent auxiliary agent. The main
        # single-human target remains the original DrivoRTargetBuilder.
        builders = DrivoRActionDiTAgent.get_target_builders(self)
        if not self._checkpoint_path:
            builders.append(DiverseTrajectoryTargetBuilder(self._config.pseudo_expert))
        return builders

    def compute_loss(self, features, targets, pred):
        result = super().compute_loss(features,targets,pred)
        if self.training:
            result['human_trajectory_loss'] = pred['_flow_loss'].detach()
            result['diverse_gt_loss'] = result['pseudo_expert_loss']
            result['diverse_gt_count'] = result['pseudo_expert_count']
            result['diverse_gt_scene_coverage'] = targets['pseudo_expert_mask'].any(-1).float().mean()
            result['diverse_gt_missing_fraction'] = targets['diverse_gt_missing'].float().mean()
            result['diverse_gt_like_count'] = targets['diverse_gt_like_mask'].sum(-1).float().mean()
        return result
