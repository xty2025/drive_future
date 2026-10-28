"""Opt-in physical-coordinate X-prediction L1 for human and long GTs.

Network parameters, noisy inputs, scoring, and Euler inference are inherited
unchanged. Only the main training regression objective is replaced.
"""

import torch

from navsim.agents.drivoR.multi_gt import multi_gt_enabled

from .agent import DrivoRActionDiTAgent
from .diverse_gt import DrivoRActionDiTDiverseGTAgent
from .flow import denormalize_trajectory, normalize_trajectory
from .model import DrivoRActionDiTModel


def physical_x_l1_loss(predicted_x, target, reduction='best_of_k'):
    """Sum |dx|+|dy|+|dheading|, average times, then select each scene's winner.

    predicted_x: normalized [B,M,8,3]; target: physical ego poses [B,8,3].
    There is no division by sigma and no timestep-dependent loss weighting.
    """
    poses = denormalize_trajectory(predicted_x.float())
    error = (poses - target.detach().float()[:, None]).abs().sum(-1).mean(-1)
    if reduction == 'best_of_k':
        return error.min(dim=1).values.mean()
    if reduction == 'mean':
        return error.mean()
    raise ValueError(f'Unknown X L1 reduction: {reduction}')


class DrivoRActionDiTXL1Model(DrivoRActionDiTModel):
    def __init__(self, config):
        if multi_gt_enabled(config):
            raise ValueError('X L1 presets use human GT, optional long GT and separate diverse8 coverage')
        auxiliary = config.get('pseudo_expert', {})
        if auxiliary.get('enabled', False) and auxiliary.get('loss_type', 'l1') != 'l1':
            raise ValueError('X L1 presets require physical-coordinate auxiliary L1 coverage')
        super().__init__(config)

    def flow_training_outputs(self, targets, scene, ego):
        # Keep the parent's random-number order and human/long noisy inputs.
        times = self.schedule.sample_timesteps(len(scene), self.proposal_num, scene.device)
        noise = torch.randn(len(scene), self.proposal_num, self.poses_num, 3,
                            device=scene.device, dtype=torch.float32)
        names = ['trajectory']
        if self._config.long_trajectory_additional_poses > 0:
            names.append('trajectory_long')
        clean = [normalize_trajectory(targets[name].detach().float())[:, None].expand(
            -1, self.proposal_num, -1, -1) for name in names]
        noisy = [self.schedule.add_noise(value, noise, times) for value in clean]
        predicted = self.action_dit(torch.cat(noisy, dim=1),
            torch.cat([times] * len(names), dim=1), scene, ego).split(self.proposal_num, dim=1)
        losses = [physical_x_l1_loss(pred, targets[name], self.flow_reduction)
                  for pred, name in zip(predicted, names)]
        diagnostics = {'human_x_l1_loss': losses[0].detach(),
                       'flow_time_mean': times.detach().mean()}
        if len(losses) == 2:
            diagnostics['long_x_l1_loss'] = losses[1].detach()
        # Parent forward_train retains the ordinary group's differentiable
        # physical prediction for diverse8 coverage and detaches Scorer inputs.
        return predicted[0], torch.stack(losses).sum(), diagnostics


class DrivoRActionDiTXL1Agent(DrivoRActionDiTAgent):
    def _build_model(self, config):
        return DrivoRActionDiTXL1Model(config)


class DrivoRActionDiTDiverseGTXL1Agent(DrivoRActionDiTDiverseGTAgent):
    def _build_model(self, config):
        return DrivoRActionDiTXL1Model(config)
