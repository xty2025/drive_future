"""One-pass planning and future-register prediction from learned queries."""

import torch
from torch import nn

from navsim.agents.drivoR.layers.utils.mlp import MLP
from navsim.agents.drivoR.multi_gt import multi_gt_enabled, valid_multi_gt, matched_proposal_count
from navsim.agents.drivoR.score_module.ranking import aggregate_score_logits
from navsim.agents.drivoR.score_module.scorer import Scorer
from navsim.agents.drivoR.transformer_decoder import TransformerDecoderScorer
from navsim.agents.drivoR_action_dit.dinov3_image_encoder import DinoV3ImageEncoder
from navsim.agents.drivoR_action_dit_register.teacher import EmaDinoRegisters

from .decoder import QueryRegisterDecoder
from .losses import match_trajectories, matched_future_loss


class DrivoRQueryRegisterModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        if config.image_backbone.family != 'dinov3':
            raise ValueError('Query-register targets use the online DINOv3 encoder EMA')
        if config.full_history_status or len(config.lidar_pc) or not config.one_token_per_traj:
            raise ValueError('Use current ego/cameras and one Action query per trajectory')
        if config.long_trajectory_additional_poses > 0:
            raise ValueError('These paired-future configs supervise the 4-second trajectory, without long GT')
        if any(config[key] for key in ('double_score', 'agent_pred', 'area_pred', 'bev_map', 'bev_agent')):
            raise ValueError('Query-register planning uses the standard six-head DrivoR scorer')
        camera_names = ('cam_f0', 'cam_l0', 'cam_l1', 'cam_l2', 'cam_r0', 'cam_r1', 'cam_r2', 'cam_b0')
        cameras = [list(config[name]) for name in camera_names if len(config[name])]
        if not cameras or any(indices != [3] for indices in cameras):
            raise ValueError('Enabled cameras must request only the current frame [3]')
        if config.future_register.stage != 'joint':
            raise ValueError('Learned-query experiments use joint training')
        if config.multi_gt.get('matching', 'nearest') != 'nearest' or config.multi_gt.get('weighting', 'equal') != 'equal':
            raise ValueError('Use independent nearest matching and equal valid-GT averaging')
        self.future_stage = 'joint'
        self.num_cams = len(cameras)
        self.proposal_num, self.poses_num = int(config.proposal_num), int(config.num_poses)
        self.register_frames, self.registers_per_frame = 8, 4
        self.register_count = self.register_frames * self.registers_per_frame
        width = int(config.tf_d_model)
        self.image_backbone = DinoV3ImageEncoder(
            config.image_backbone, config.image_size, width, self.num_cams,
            config.num_scene_tokens, config.visual_feature_mode)
        self.register_dim = int(self.image_backbone.num_features)
        self.hist_encoding = nn.Linear(11, width)
        # DrivoR's learned Action embeddings: random once, then optimized.
        self.action_queries = nn.Embedding(self.proposal_num, width)
        self.future_queries = nn.Parameter(torch.empty(self.proposal_num, self.register_count, width))
        self.future_time_embedding = nn.Parameter(torch.empty(self.register_frames, width))
        self.register_slot_embedding = nn.Parameter(torch.empty(self.registers_per_frame, width))
        for parameter in (self.future_queries, self.future_time_embedding, self.register_slot_embedding):
            nn.init.trunc_normal_(parameter, std=float(config.query_register.future_query_init_std))
        self.query_decoder = QueryRegisterDecoder(config, self.register_count)
        self.trajectory_head = MLP(width, config.tf_d_ffn, self.poses_num * 3)
        self.register_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, self.register_dim))
        self.ema_teacher = EmaDinoRegisters(self.image_backbone.native_vit,
            decay=config.future_register.ema_decay, chunk_size=config.future_register.teacher_batch_size)
        self.pos_embed = nn.Sequential(nn.Linear(self.poses_num * 3, config.tf_d_ffn),
                                       nn.ReLU(), nn.Linear(config.tf_d_ffn, width))
        self.scorer_attention = TransformerDecoderScorer(
            num_layers=config.scorer_ref_num, d_model=width, proj_drop=0.1, drop_path=0.2, config=config)
        self.scorer = Scorer(config)
        self.use_register_scorer = bool(config.future_register.scorer.enabled)
        if self.use_register_scorer:
            if config.future_register.scorer.fusion != 'project':
                raise ValueError('Register scorer uses project: compress registers, concatenate, then MLP')
            self.register_score_projection = nn.Sequential(
                nn.Linear(self.register_count * self.register_dim, width), nn.ReLU())
            self.score_fusion = nn.Sequential(nn.Linear(2 * width, config.tf_d_ffn),
                                             nn.ReLU(), nn.Linear(config.tf_d_ffn, width))

    def encode_scene(self, features):
        ego = self.hist_encoding(features['ego_status'][:, -1])
        images = features.get('image', features.get('camera_feature'))
        if images is None or images.shape[1] != self.num_cams:
            raise ValueError(f'Expected {self.num_cams} current camera views')
        return self.image_backbone(images), ego

    def make_queries(self, ego):
        batch = len(ego)
        action = self.action_queries.weight[None].expand(batch, -1, -1) + ego[:, None]
        positions = (self.future_time_embedding[:, None] + self.register_slot_embedding[None]).flatten(0, 1)
        future = self.future_queries[None].expand(batch, -1, -1, -1)
        future = future + positions[None, None] + ego[:, None, None]
        return action, future

    def score_proposals(self, proposals, predicted_registers, scene, ego):
        detached = proposals.detach().float()
        embedded = self.pos_embed(detached.flatten(-2))
        if self.use_register_scorer:
            registers = self.register_score_projection(predicted_registers.detach().float().flatten(2))
            embedded = self.score_fusion(torch.cat((embedded, registers), dim=-1))
        features = self.scorer_attention(embedded, scene) + ego[:, None]
        logits, logits2, agent_states, areas, bev_map, agents, labels = self.scorer(detached, features)
        scores = aggregate_score_logits(logits, self._config)
        chosen = scores.argmax(1)
        rows = torch.arange(len(proposals), device=proposals.device)
        return dict(proposals=proposals, proposal_list=[proposals], trajectory=proposals[rows, chosen],
                    pdm_score=scores, pred_logit=logits, pred_logit2=logits2,
                    pred_agents_states=agent_states, pred_area_logit=areas,
                    bev_semantic_map=bev_map, agent_states=agents, agent_labels=labels)

    def forward(self, features, targets=None):
        scene, ego = self.encode_scene(features)
        action, future = self.make_queries(ego)
        action, future = self.query_decoder(action, future, scene)
        proposals = self.trajectory_head(action).reshape(len(scene), self.proposal_num, self.poses_num, 3).float()
        registers = self.register_head(future).reshape(len(scene), self.proposal_num,
                          self.register_frames, self.registers_per_frame, self.register_dim).float()
        output = self.score_proposals(proposals, registers, scene, ego)
        output['predicted_registers'] = registers
        if targets is not None:
            self._add_supervision(output, targets)
        return output

    def _add_supervision(self, output, targets):
        if multi_gt_enabled(self._config):
            gt, valid = valid_multi_gt(targets)
            pixels = targets['future_register_images']
            future_valid = targets['future_register_valid'].bool() & valid[..., None]
        else:
            gt = targets['trajectory'].detach().float()[:, None]
            valid = torch.ones(gt.shape[:2], device=gt.device, dtype=torch.bool)
            pixels = targets['human_future_register_images'][:, None]
            future_valid = targets['human_future_register_valid'].bool()[:, None]
        trajectory_loss, matched = match_trajectories(output['proposals'], gt, valid)
        teacher = self.ema_teacher(pixels, future_valid)
        future_loss = matched_future_loss(output['predicted_registers'], teacher, future_valid,
                                          matched, self._config.future_register.loss_type)
        output['_trajectory_loss'], output['_register_loss'] = trajectory_loss, future_loss
        output['_register_matched_indices'] = matched
        diagnostics = {
            'multi_gt_count': valid.sum(-1).float().mean(),
            'multi_gt_matched_proposals': matched_proposal_count(matched, matched >= 0, self.proposal_num),
            'future_register_gt_count': future_valid.any(-1).sum(-1).float().mean(),
            'future_register_frame_fraction': future_valid.float().mean(),
            'ema_updates': self.ema_teacher.updates.detach().float().clone(),
        }
        if 'multi_gt_fallback' in targets:
            diagnostics['multi_gt_fallback_fraction'] = targets['multi_gt_fallback'].float().mean()
        output['_query_diagnostics'] = diagnostics
