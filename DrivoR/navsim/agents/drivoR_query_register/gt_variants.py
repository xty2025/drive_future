"""Matched query-planner controls: human/long GT, optional future, or diverse8."""

import torch
from torch import nn

from navsim.agents.drivoR.drivor_features import DrivoRTargetBuilder
from navsim.agents.drivoR.layers.utils.mlp import MLP
from navsim.agents.drivoR.multi_gt import multi_gt_enabled
from navsim.agents.drivoR.score_module.scorer import Scorer
from navsim.agents.drivoR.transformer_decoder import TransformerDecoderScorer
from navsim.agents.drivoR_action_dit.agent import DrivoRActionDiTAgent
from navsim.agents.drivoR_action_dit.dinov3_image_encoder import DinoV3ImageEncoder
from navsim.agents.drivoR_action_dit.diverse_gt import DiverseTrajectoryTargetBuilder
from navsim.agents.drivoR_action_dit_register.targets import HumanFutureRegisterTargetBuilder
from navsim.agents.drivoR_action_dit_register.teacher import EmaDinoRegisters
from navsim.agents.drivoR_pseudo_expert.loss import multi_expert_coverage_loss

from .decoder import QueryRegisterDecoder
from .losses import match_trajectories, matched_future_loss
from .model import DrivoRQueryRegisterModel


class DrivoRQueryGTModel(DrivoRQueryRegisterModel):
    def __init__(self, config):
        # Construct all shared parameters first, in the same order for all
        # three controls. Optional future parameters cannot change their seed.
        nn.Module.__init__(self)
        self._config = config
        if config.image_backbone.family != 'dinov3' or multi_gt_enabled(config):
            raise ValueError('These controls use DINOv3 and human GT with optional long/diverse8 targets')
        if config.full_history_status or len(config.lidar_pc) or not config.one_token_per_traj:
            raise ValueError('Use current ego/cameras and one Action query per trajectory')
        if any(config[key] for key in ('double_score', 'agent_pred', 'area_pred', 'bev_map', 'bev_agent')):
            raise ValueError('Use the standard six-head trajectory Scorer')
        if config.future_register.scorer.enabled:
            raise ValueError('All three controls use the original trajectory-only Scorer')
        cameras = [list(config[name]) for name in
            ('cam_f0','cam_l0','cam_l1','cam_l2','cam_r0','cam_r1','cam_r2','cam_b0') if len(config[name])]
        if not cameras or any(indices != [3] for indices in cameras):
            raise ValueError('Enabled cameras must request only the current frame [3]')
        auxiliary = config.get('pseudo_expert', {})
        self.use_diverse8 = bool(auxiliary.get('enabled', False))
        if self.use_diverse8 and (auxiliary.source != 'diverse8' or auxiliary.loss_type != 'l1'):
            raise ValueError('The additional GT source must be diverse8 with physical L1 coverage')
        self.predict_future = bool(config.future_register.enabled)
        self.use_register_scorer = False
        self.future_stage = 'joint'
        self.num_cams = len(cameras)
        self.proposal_num, self.poses_num = int(config.proposal_num), int(config.num_poses)
        self.register_frames, self.registers_per_frame = 8, 4
        self.register_count = 32 if self.predict_future else 0
        width = int(config.tf_d_model)
        self.image_backbone = DinoV3ImageEncoder(config.image_backbone, config.image_size,
            width, self.num_cams, config.num_scene_tokens, config.visual_feature_mode)
        self.register_dim = int(self.image_backbone.num_features)
        self.hist_encoding = nn.Linear(11, width)
        self.action_queries = nn.Embedding(self.proposal_num, width)
        self.query_decoder = QueryRegisterDecoder(config, self.register_count)
        self.trajectory_head = MLP(width, config.tf_d_ffn, self.poses_num * 3)
        self.pos_embed = nn.Sequential(nn.Linear(self.poses_num * 3, config.tf_d_ffn),
                                       nn.ReLU(), nn.Linear(config.tf_d_ffn, width))
        self.scorer_attention = TransformerDecoderScorer(num_layers=config.scorer_ref_num,
            d_model=width, proj_drop=.1, drop_path=.2, config=config)
        self.scorer = Scorer(config)
        if self.predict_future:
            self.future_queries = nn.Parameter(torch.empty(self.proposal_num, 32, width))
            self.future_time_embedding = nn.Parameter(torch.empty(8, width))
            self.register_slot_embedding = nn.Parameter(torch.empty(4, width))
            for parameter in (self.future_queries, self.future_time_embedding, self.register_slot_embedding):
                nn.init.trunc_normal_(parameter, std=float(config.query_register.future_query_init_std))
            self.register_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, self.register_dim))
            self.ema_teacher = EmaDinoRegisters(self.image_backbone.native_vit,
                decay=config.future_register.ema_decay, chunk_size=config.future_register.teacher_batch_size)

    def make_queries(self, ego):
        if self.predict_future:
            return super().make_queries(ego)
        action = self.action_queries.weight[None].expand(len(ego), -1, -1) + ego[:, None]
        return action, action.new_empty(len(ego), self.proposal_num, 0, action.shape[-1])

    def forward(self, features, targets=None):
        scene, ego = self.encode_scene(features)
        action, future = self.make_queries(ego)
        action, future = self.query_decoder(action, future, scene)
        proposals = self.trajectory_head(action).reshape(len(scene), self.proposal_num, self.poses_num, 3).float()
        registers = (self.register_head(future).reshape(len(scene), self.proposal_num, 8, 4, self.register_dim).float()
                     if self.predict_future else None)
        output = self.score_proposals(proposals, registers, scene, ego)
        if registers is not None:
            output['predicted_registers'] = registers
        if targets is not None:
            self._add_supervision(output, targets)
        return output

    def _add_supervision(self, output, targets):
        proposals = output['proposals']
        valid = torch.ones(len(proposals), 1, device=proposals.device, dtype=torch.bool)
        human_loss, human_match = match_trajectories(proposals, targets['trajectory'][:, None], valid)
        trajectory_loss = human_loss
        diagnostics = {'human_trajectory_loss': human_loss.detach()}
        output['_human_matched_indices'] = human_match
        if self._config.long_trajectory_additional_poses > 0:
            long_loss, long_match = match_trajectories(proposals, targets['trajectory_long'][:, None], valid)
            trajectory_loss = trajectory_loss + long_loss
            diagnostics['long_trajectory_loss'] = long_loss.detach()
            output['_long_matched_indices'] = long_match
        if self.use_diverse8:
            coverage = multi_expert_coverage_loss(proposals, targets['pseudo_experts'], targets['pseudo_expert_mask'])
            trajectory_loss = trajectory_loss + float(self._config.pseudo_expert.weight) * coverage
            diagnostics.update(diverse_gt_loss=coverage.detach(),
                diverse_gt_count=targets['pseudo_expert_mask'].sum(-1).float().mean(),
                diverse_gt_scene_coverage=targets['pseudo_expert_mask'].any(-1).float().mean(),
                diverse_gt_missing_fraction=targets['diverse_gt_missing'].float().mean())
        future_loss = proposals.new_zeros(())
        if self.predict_future:
            pixels = targets['human_future_register_images'][:, None]
            future_valid = targets['human_future_register_valid'].bool()[:, None]
            teacher = self.ema_teacher(pixels, future_valid)
            # Long GT only supervises trajectory. Future slots remain at
            # +0.5,...,+4s and are paired with the ordinary human winner.
            future_loss = matched_future_loss(output['predicted_registers'], teacher,
                future_valid, human_match, self._config.future_register.loss_type)
            output['_register_matched_indices'] = human_match
            diagnostics.update(future_register_gt_count=future_valid.any(-1).sum(-1).float().mean(),
                future_register_frame_fraction=future_valid.float().mean(),
                ema_updates=self.ema_teacher.updates.detach().float().clone())
        output['_trajectory_loss'] = trajectory_loss
        output['_register_loss'] = future_loss
        output['_query_diagnostics'] = diagnostics


class DrivoRQueryGTAgent(DrivoRActionDiTAgent):
    def _build_model(self, config):
        return DrivoRQueryGTModel(config)

    def initialize(self):
        if self._checkpoint_path:
            payload = torch.load(self._checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
            self.load_state_dict({key.removeprefix('agent.'):value
                                  for key,value in payload['state_dict'].items()}, strict=True)

    def get_target_builders(self):
        builders = [DrivoRTargetBuilder(self._config)]
        if not self._checkpoint_path:
            if self._drivor_model.predict_future:
                builders.append(HumanFutureRegisterTargetBuilder(self._config.future_register))
            if self._drivor_model.use_diverse8:
                builders.append(DiverseTrajectoryTargetBuilder(self._config.pseudo_expert))
        return builders

    def compute_loss(self, features, targets, pred):
        return self.loss(targets, pred, self._config, self.compute_score)

    def get_optimizers(self):
        configured = super().get_optimizers()
        optimizers = configured[0] if isinstance(configured, tuple) else configured
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group['params'] = [parameter for parameter in group['params'] if parameter.requires_grad]
            if self._drivor_model.predict_future:
                optimizer.register_step_post_hook(self._update_teacher)
        return configured

    @torch.no_grad()
    def _update_teacher(self, optimizer, args, kwargs):
        model = self._drivor_model
        model.ema_teacher.update(model.image_backbone.native_vit)
