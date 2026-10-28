"""Human + diverse8 L1 with a paired future query group for every proposal."""

import torch

from navsim.agents.drivoR.drivor_features import DrivoRTargetBuilder
from navsim.agents.drivoR_pseudo_expert.loss import multi_expert_coverage_loss
from navsim.agents.drivoR_action_dit_register.diverse_losses import physical_gt_matches, paired_future_targets
from navsim.agents.drivoR_action_dit_register.diverse_targets import DiverseFutureRegisterTargetBuilder
from .gt_variants import DrivoRQueryGTModel, DrivoRQueryGTAgent
from .losses import match_trajectories, matched_future_loss


class DrivoRQueryDiverseRegisterModel(DrivoRQueryGTModel):
    def __init__(self, config):
        if not config.future_register.enabled or config.long_trajectory_additional_poses > 0:
            raise ValueError('Use joint future registers with human + diverse8 and no long GT')
        super().__init__(config)
        if not self.use_diverse8:
            raise ValueError('These future targets must be paired with diverse8 trajectories')

    def _add_supervision(self, output, targets):
        proposals = output['proposals']
        human_valid = torch.ones(len(proposals), 1, device=proposals.device, dtype=torch.bool)
        human_loss, human_match = match_trajectories(proposals, targets['trajectory'][:, None], human_valid)
        diverse_loss = multi_expert_coverage_loss(proposals, targets['pseudo_experts'], targets['pseudo_expert_mask'])
        diverse_match = physical_gt_matches(proposals, targets['pseudo_experts'], targets['pseudo_expert_mask'])
        matched = torch.cat((human_match, diverse_match), dim=1)
        teacher, future_valid = paired_future_targets(self.ema_teacher, targets)
        future_loss = matched_future_loss(output['predicted_registers'], teacher, future_valid,
                                          matched, self._config.future_register.loss_type)
        output.update(_trajectory_loss=human_loss + float(self._config.pseudo_expert.weight) * diverse_loss,
                      _register_loss=future_loss, _register_matched_indices=matched,
                      _human_matched_indices=human_match, _diverse_matched_indices=diverse_match)
        output['_query_diagnostics'] = dict(human_trajectory_loss=human_loss.detach(),
            diverse_gt_loss=diverse_loss.detach(), diverse_gt_count=targets['pseudo_expert_mask'].sum(-1).float().mean(),
            diverse_gt_missing_fraction=targets['diverse_gt_missing'].float().mean(),
            future_register_gt_count=future_valid.any(-1).sum(-1).float().mean(),
            future_register_frame_fraction=future_valid.float().mean(),
            ema_updates=self.ema_teacher.updates.detach().float().clone())


class DrivoRQueryDiverseRegisterAgent(DrivoRQueryGTAgent):
    def _build_model(self, config):
        return DrivoRQueryDiverseRegisterModel(config)

    def get_target_builders(self):
        builders = [DrivoRTargetBuilder(self._config)]
        if not self._checkpoint_path:
            builders.append(DiverseFutureRegisterTargetBuilder(self._config.pseudo_expert,
                                                               self._config.future_register))
        return builders
