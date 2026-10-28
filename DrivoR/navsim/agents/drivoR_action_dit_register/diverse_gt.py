"""Human-primary Action-DiT plus trajectory-paired diverse8 register prediction."""

import torch

from navsim.agents.drivoR.drivor_features import DrivoRTargetBuilder
from navsim.agents.drivoR.multi_gt import multi_gt_enabled, multi_gt_velocity_loss
from navsim.agents.drivoR_action_dit.agent import DrivoRActionDiTAgent
from navsim.agents.drivoR_action_dit.flow import normalize_trajectory, denormalize_trajectory
from navsim.agents.drivoR_action_dit.x_l1 import physical_x_l1_loss
from navsim.agents.drivoR_pseudo_expert.loss import multi_expert_coverage_loss
from .agent import DrivoRActionRegisterAgent
from .model import DrivoRActionRegisterModel
from .losses import matched_register_loss
from .diverse_losses import physical_gt_matches, paired_future_targets, future_supervision_scope, grouped_anchor_indices
from .diverse_targets import DiverseFutureRegisterTargetBuilder


class DrivoRActionDiverseRegisterModel(DrivoRActionRegisterModel):
    def _validate_auxiliary_config(self, config):
        options = config.pseudo_expert
        if (multi_gt_enabled(config) or config.long_trajectory_additional_poses > 0
                or not options.enabled or options.source != 'diverse8' or options.loss_type != 'l1'
                or config.future_register.stage != 'joint'):
            raise ValueError('Use human + 0.5 diverse8 L1, joint future registers, without old M4/long GT')
        if config.future_register.human_loss_type not in ('velocity', 'l1'):
            raise ValueError('human_loss_type must be velocity or l1')
        self.register_supervision_scope = future_supervision_scope(config.future_register)
        self.input_gt_assignment = config.future_register.get('input_gt_assignment', 'original')
        if self.input_gt_assignment not in ('original', 'balanced_human_diverse'):
            raise ValueError('Future input_gt_assignment must be original or balanced_human_diverse')
        if self.input_gt_assignment != 'original' and self.register_supervision_scope != 'human_and_diverse':
            raise ValueError('Grouped noisy inputs require the paired human and D8 future targets')

    def forward_train(self, features, targets):
        scene, ego = self.encode_scene(features)
        teacher, future_valid = paired_future_targets(self.ema_teacher, targets, self.register_supervision_scope)
        human = normalize_trajectory(targets['trajectory'].detach().float())
        times = self.schedule.sample_timesteps(len(scene), self.proposal_num, scene.device)
        if self.input_gt_assignment == 'balanced_human_diverse':
            assignment = grouped_anchor_indices(targets['pseudo_expert_mask'], self.proposal_num)
            safe_gt = targets['pseudo_experts'].detach().float().masked_fill(
                ~targets['pseudo_expert_mask'].bool()[..., None, None], 0.)
            normalized_gt = torch.cat((human[:, None], normalize_trajectory(safe_gt)), dim=1)
            batch = torch.arange(len(scene), device=scene.device)[:, None]
            action_anchor = normalized_gt[batch, assignment]
            register_anchor = teacher[batch, assignment].flatten(2, 3)
        else:
            action_anchor = human[:, None].expand(-1, self.proposal_num, -1, -1)
            register_anchor = teacher[:, 0].flatten(1, 2)[:, None].expand(-1, self.proposal_num, -1, -1)
        action_input = self.schedule.add_noise(action_anchor, torch.randn_like(action_anchor), times)
        sigma = self.schedule.sigma(times)[..., None, None]
        register_input = (1 - sigma) * register_anchor + sigma * torch.randn_like(register_anchor)
        action_x, registers = self.action_dit(action_input, register_input, times, scene, ego)
        registers = registers.reshape(len(scene), self.proposal_num, self.register_frames,
                                       self.registers_per_frame, self.register_dim)
        physical = denormalize_trajectory(action_x.float())
        human_valid = torch.ones(len(scene), 1, device=scene.device, dtype=torch.bool)
        if self._config.future_register.human_loss_type == 'velocity':
            human_loss, human_match = multi_gt_velocity_loss(action_x, human[:, None], human_valid,
                                                            times, self.schedule)
        else:
            human_loss = physical_x_l1_loss(action_x, targets['trajectory'], 'best_of_k')
            human_match = physical_gt_matches(physical, targets['trajectory'][:, None], human_valid)
        diverse_loss = multi_expert_coverage_loss(physical, targets['pseudo_experts'], targets['pseudo_expert_mask'])
        diverse_match = physical_gt_matches(physical, targets['pseudo_experts'], targets['pseudo_expert_mask'])
        matched = (human_match if self.register_supervision_scope == 'human_only'
                   else torch.cat((human_match, diverse_match), dim=1))
        register_loss = matched_register_loss(registers, teacher, future_valid, matched,
                                              times, self.schedule, self.register_loss_type)
        output = self.score_proposals(physical.detach(), scene, ego, registers)
        output.update(_flow_loss=human_loss + float(self._config.pseudo_expert.weight) * diverse_loss,
            _register_loss=register_loss, predicted_registers=registers,
            _register_matched_indices=matched, _human_matched_indices=human_match,
            _diverse_matched_indices=diverse_match)
        output['_flow_diagnostics'] = dict(human_trajectory_loss=human_loss.detach(),
            diverse_gt_loss=diverse_loss.detach(), diverse_gt_count=targets['pseudo_expert_mask'].sum(-1).float().mean(),
            diverse_gt_missing_fraction=targets['diverse_gt_missing'].float().mean(),
            future_register_gt_count=future_valid.any(-1).sum(-1).float().mean(),
            future_register_frame_fraction=future_valid.float().mean(),
            flow_time_mean=times.detach().mean(), ema_updates=self.ema_teacher.updates.detach().float().clone())
        return output


class DrivoRActionDiverseRegisterAgent(DrivoRActionRegisterAgent):
    def _build_model(self, config):
        return DrivoRActionDiverseRegisterModel(config)

    def get_target_builders(self):
        builders = [DrivoRTargetBuilder(self._config)]
        if not self._checkpoint_path:
            builders.append(DiverseFutureRegisterTargetBuilder(self._config.pseudo_expert,
                                                              self._config.future_register))
        return builders

    def compute_loss(self, features, targets, pred):
        result = DrivoRActionDiTAgent.compute_loss(self, features, targets, pred)
        result['future_register_loss'] = pred['_register_loss']
        result['loss'] = result['loss'] + float(self._config.future_register.loss_weight) * pred['_register_loss']
        return result
