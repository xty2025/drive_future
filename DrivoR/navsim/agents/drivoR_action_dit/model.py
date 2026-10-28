"""DrivoR's ego/scorer surrounding an Action-DiT with configurable visual features."""

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from navsim.agents.drivoR.score_module.scorer import Scorer
from navsim.agents.drivoR.transformer_decoder import TransformerDecoderScorer
from navsim.agents.drivoR.multi_gt import (
    multi_gt_enabled, valid_multi_gt, balanced_gt_inputs,
    multi_gt_velocity_loss, drive_jepa_multi_gt_velocity_loss, matched_proposal_count,
)

from .action_dit import ActionDiT
from .flow import ContinuousFlowSchedule, denormalize_trajectory, normalize_trajectory
from .image_encoder import DrivoRSceneQueryImageEncoder, NativePatchImageEncoder
from .dinov3_image_encoder import DinoV3ImageEncoder
from .vjepa2_image_encoder import VJepa2ImageEncoder
from .vjepa2_1_image_encoder import VJepa21ImageEncoder


class DrivoRActionDiTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        if config.full_history_status or len(config.lidar_pc):
            raise ValueError("Action-DiT variant uses only current ego state and cameras")
        camera_names = ("cam_f0", "cam_l0", "cam_l1", "cam_l2",
                        "cam_r0", "cam_r1", "cam_r2", "cam_b0")
        active_cameras = [list(config[name]) for name in camera_names if len(config[name])]
        if not active_cameras or any(indices != [3] for indices in active_cameras):
            raise ValueError("Each enabled camera must request only the current frame: [3]")
        if any(config[name] for name in ("double_score", "agent_pred", "area_pred", "bev_map", "bev_agent")):
            raise ValueError("Action-DiT uses the standard six DrivoR score heads")
        action = config.action_dit
        self.num_cams = len(active_cameras)
        self.poses_num = config.num_poses
        self.proposal_num = config.proposal_num
        self.visual_feature_mode = config.get("visual_feature_mode", "native_patches")
        image_family = config.image_backbone.get("family", "dinov2")
        if image_family == "dinov3":
            self.image_backbone = DinoV3ImageEncoder(
                config.image_backbone, config.image_size, config.tf_d_model,
                self.num_cams, config.num_scene_tokens, self.visual_feature_mode)
        elif image_family == "vjepa2":
            self.image_backbone = VJepa2ImageEncoder(
                config.image_backbone, config.image_size, config.tf_d_model,
                self.num_cams, config.num_scene_tokens, self.visual_feature_mode)
        elif image_family == "vjepa2_1":
            self.image_backbone = VJepa21ImageEncoder(
                config.image_backbone, config.image_size, config.tf_d_model,
                self.num_cams, config.num_scene_tokens, self.visual_feature_mode)
        elif image_family != "dinov2":
            raise ValueError(f"Unknown image backbone family: {image_family}")
        elif self.visual_feature_mode == "native_patches":
            self.image_backbone = NativePatchImageEncoder(
                config.image_backbone, config.image_size, config.tf_d_model)
        elif self.visual_feature_mode == "drivor_queries":
            self.image_backbone = DrivoRSceneQueryImageEncoder(
                config.image_backbone, config.image_size, config.tf_d_model,
                self.num_cams, config.num_scene_tokens)
        else:
            raise ValueError(f"Unknown visual_feature_mode: {self.visual_feature_mode}")
        self.hist_encoding = nn.Linear(11, config.tf_d_model)
        self.action_dit = ActionDiT(
            hidden_dim=action.hidden_dim, action_dim=3, ffn_dim=action.ffn_dim,
            freq_dim=action.freq_dim, num_heads=action.num_heads,
            attn_head_dim=action.attn_head_dim, num_layers=action.num_layers, eps=action.eps,
            use_gradient_checkpointing=action.gradient_checkpointing, context_dim=config.tf_d_model)
        if action.pretrained_path:
            if action.get('pretrained_format', 'fastwam') in ('wan22_native', 'wan21_native'):
                self.action_dit.load_wan_pretrained(action.pretrained_path,
                    checkpoint_format=action.pretrained_format)
            else:
                self.action_dit.load_pretrained(action.pretrained_path)
        self.schedule = ContinuousFlowSchedule(
            num_train_timesteps=action.num_train_timesteps, train_shift=action.train_shift,
            infer_shift=action.infer_shift, sigma_min=action.sigma_min)
        self.num_inference_steps = int(action.num_inference_steps)
        self.flow_reduction = str(action.loss_reduction)

        # Same network classes, shapes, dropout and six heads as original DrivoR.
        # No DrivoR release state_dict is loaded here.
        self.pos_embed = nn.Sequential(nn.Linear(self.poses_num * 3, config.tf_d_ffn),
                                       nn.ReLU(), nn.Linear(config.tf_d_ffn, config.tf_d_model))
        self.scorer_attention = TransformerDecoderScorer(
            num_layers=config.scorer_ref_num, d_model=config.tf_d_model,
            proj_drop=0.1, drop_path=0.2, config=config)
        self.scorer = Scorer(config)

    def encode_scene(self, features):
        ego = self.hist_encoding(features["ego_status"][:, -1])
        images = features.get("image", features.get("camera_feature"))
        if images is None or images.shape[1] != self.num_cams:
            raise ValueError(f"Expected {self.num_cams} current camera images")
        return self.image_backbone(images), ego

    def sample_normalized(self, scene, ego, noise=None):
        b = scene.shape[0]
        if noise is None:
            noise = torch.randn(b, self.proposal_num, self.poses_num, 3,
                                 device=scene.device, dtype=torch.float32)
        current = noise.float()
        timesteps, deltas = self.schedule.inference_schedule(self.num_inference_steps, scene.device)
        for index, (timestep, delta) in enumerate(zip(timesteps, deltas)):
            times = timestep.expand(b, self.proposal_num)
            pred_x = self.action_dit(current, times, scene, ego).float()
            if index == self.num_inference_steps - 1:
                # The last Euler step has delta=-sigma, so its exact result is
                # pred_x. Do not use the TRAINING loss floor in this integrator.
                return pred_x
            sigma = self.schedule.sigma(times)[..., None, None]
            velocity = (current - pred_x) / sigma
            current = current + velocity * delta
        return current

    def score_proposals(self, proposals, scene, ego):
        b, m = proposals.shape[:2]
        detached = proposals.detach()
        embedded = self.pos_embed(detached.reshape(b, m, -1))
        return self._score_embedded_proposals(proposals, embedded, scene, ego)

    def _score_embedded_proposals(self, proposals, embedded, scene, ego):
        b = proposals.shape[0]
        detached = proposals.detach()
        score_features = self.scorer_attention(embedded, scene) + ego[:, None]
        (logits, logits2, agent_states, areas, bev_map, agents, labels) = self.scorer(detached, score_features)
        config = self._config
        # Same ranking equation as DrivoR, written with log-sigmoid for stability.
        pdm_score = (config.noc * F.logsigmoid(logits["no_at_fault_collisions"].float())
                     + config.dac * F.logsigmoid(logits["drivable_area_compliance"].float())
                     + config.ddc * F.logsigmoid(logits["driving_direction_compliance"].float())
                     + (config.ttc * logits["time_to_collision_within_bound"].float().sigmoid()
                        + config.ep * logits["ego_progress"].float().sigmoid()
                        + config.comfort * logits["comfort"].float().sigmoid()).clamp_min(1e-12).log())
        selected = pdm_score.argmax(dim=1)
        return dict(proposals=proposals, proposal_list=[proposals],
                    trajectory=proposals[torch.arange(b, device=proposals.device), selected],
                    pdm_score=pdm_score, pred_logit=logits, pred_logit2=logits2,
                    pred_agents_states=agent_states, pred_area_logit=areas,
                    bev_semantic_map=bev_map, agent_states=agents, agent_labels=labels)

    def forward_train(self, features, targets):
        """Share one X-prediction forward between flow supervision and ranking."""
        scene, ego = self.encode_scene(features)
        candidate_x, flow_loss, diagnostics = self.flow_training_outputs(targets, scene, ego)
        # Legacy mode returns its ordinary-GT group; multi-GT mode returns the
        # shared set of 64 predictions used by multi-GT matching and regression.
        proposals = denormalize_trajectory(candidate_x.detach().float())
        output = self.score_proposals(proposals, scene, ego)
        output["_scene_context"] = scene
        output["_ego_embedding"] = ego
        output["_flow_loss"] = flow_loss
        output["_flow_diagnostics"] = diagnostics
        if self._config.get("pseudo_expert", {}).get("enabled", False):
            if self._config.pseudo_expert.get("loss_type", "l1") == "velocity":
                # Computed with the exact normalized X prediction and sampled
                # timesteps, before either is discarded by the flow helper.
                output["_pseudo_expert_velocity_loss"] = diagnostics.pop("_pseudo_expert_velocity_loss")
            else:
                # Keep the original physical-coordinate L1 experiment intact.
                output["_pseudo_expert_proposals"] = denormalize_trajectory(candidate_x.float())
        return output

    def forward(self, features: Dict[str, torch.Tensor], targets=None):
        if targets is not None:
            return self.forward_train(features, targets)
        # Feature-only prediction is the validation/inference API. Training
        # supplies targets and reuses its single X-prediction forward above.
        scene, ego = self.encode_scene(features)
        # Evaluation/inference integrates pure noise without a generator graph.
        # Keep no-grad parameter casts out of any surrounding autocast cache.
        with torch.no_grad(), torch.autocast(
            device_type=scene.device.type,
            enabled=torch.is_autocast_enabled(scene.device.type),
            dtype=torch.get_autocast_dtype(scene.device.type),
            cache_enabled=False,
        ):
            normalized = self.sample_normalized(scene.detach(), ego.detach())
        proposals = denormalize_trajectory(normalized)
        output = self.score_proposals(proposals, scene, ego)
        output["_scene_context"] = scene
        output["_ego_embedding"] = ego
        return output

    def flow_training_outputs(self, targets, scene, ego):
        if multi_gt_enabled(self._config):
            return self.multi_gt_flow_training_outputs(targets, scene, ego)
        b = scene.shape[0]
        timesteps = self.schedule.sample_timesteps(b, self.proposal_num, scene.device)
        noise = torch.randn(b, self.proposal_num, self.poses_num, 3,
                             device=scene.device, dtype=torch.float32)
        names = ["trajectory"]
        if self._config.long_trajectory_additional_poses > 0:
            if "trajectory_long" not in targets:
                raise ValueError("Long trajectory supervision enabled but trajectory_long is missing")
            names.append("trajectory_long")
        clean = [normalize_trajectory(targets[name].float())[:, None].expand(-1, self.proposal_num, -1, -1)
                 for name in names]
        noisy = [self.schedule.add_noise(value, noise, timesteps) for value in clean]
        # Candidates never interact in Action self-attention. Packing normal/long
        # groups as 128 queries preserves two independent 64-way sets and shares
        # the large visual K/V projection in the training forward pass.
        pred = self.action_dit(torch.cat(noisy, dim=1), torch.cat([timesteps] * len(names), dim=1),
                               scene, ego).split(self.proposal_num, dim=1)
        losses = [self.schedule.velocity_loss(p, c, x, timesteps, self.flow_reduction)[0]
                  for p, c, x in zip(pred, clean, noisy)]
        diagnostics = {"flow_normal_loss": losses[0].detach(),
                       "flow_time_mean": timesteps.detach().mean()}
        if len(losses) == 2:
            diagnostics["flow_long_loss"] = losses[1].detach()
        self._add_pseudo_expert_velocity_loss(targets, pred[0], timesteps, diagnostics)
        return pred[0], torch.stack(losses).sum(), diagnostics

    def multi_gt_flow_training_outputs(self, targets, scene, ego):
        weighting = self._config.multi_gt.get("weighting", "equal")
        matching = self._config.multi_gt.get("matching", "nearest")
        if weighting not in ("equal", "drive_jepa"):
            raise ValueError(f"Unknown multi_gt.weighting: {weighting}")
        if weighting == "drive_jepa" and matching != "nearest":
            raise ValueError("Human/other weighted v loss requires multi_gt.matching=nearest")
        gt, valid = valid_multi_gt(targets)
        normalized_gt = normalize_trajectory(gt)
        original = normalize_trajectory(targets["trajectory"].float())
        assignment = self._config.multi_gt.get("input_gt_assignment", "original")
        if assignment == "balanced":
            anchors, _ = balanced_gt_inputs(normalized_gt, valid, self.proposal_num, original)
        elif assignment == "original":
            anchors = original[:, None].expand(-1, self.proposal_num, -1, -1)
        else:
            raise ValueError(f"Unknown multi_gt.input_gt_assignment: {assignment}")
        timesteps = self.schedule.sample_timesteps(scene.shape[0], self.proposal_num, scene.device)
        noise = torch.randn(scene.shape[0], self.proposal_num, self.poses_num, 3,
                            device=scene.device, dtype=torch.float32)
        noisy = self.schedule.add_noise(anchors, noise, timesteps)
        pred_x = self.action_dit(noisy, timesteps, scene, ego)
        weighted_diagnostics = {}
        if weighting == "drive_jepa":
            loss, matched_indices, weighted_diagnostics = drive_jepa_multi_gt_velocity_loss(
                pred_x, original, normalized_gt, valid, targets["trajectory_multi_gt_is_human"],
                timesteps, self.schedule,
                human_weight=self._config.multi_gt.get("human_weight", 1.0),
                other_weight=self._config.multi_gt.get("other_weight", 0.5))
        else:
            loss, matched_indices = multi_gt_velocity_loss(
                pred_x, normalized_gt, valid, timesteps, self.schedule, matching=matching)
        diagnostics = {"flow_multi_gt_loss": loss.detach(),
                       "flow_time_mean": timesteps.detach().mean(),
                       "multi_gt_count": valid.sum(dim=1).float().mean(),
                       "multi_gt_matched_proposals": matched_proposal_count(
                           matched_indices, matched_indices >= 0, self.proposal_num)}
        diagnostics.update(weighted_diagnostics)
        if "multi_gt_fallback" in targets:
            diagnostics["multi_gt_fallback_fraction"] = targets["multi_gt_fallback"].float().mean()
        for name in ("multi_gt_primary_count", "multi_gt_additional_count"):
            if name in targets:
                diagnostics[name] = targets[name].float().mean()
        self._add_pseudo_expert_velocity_loss(targets, pred_x, timesteps, diagnostics)
        return pred_x, loss, diagnostics

    def _add_pseudo_expert_velocity_loss(self, targets, pred_x, timesteps, diagnostics):
        options = self._config.get("pseudo_expert", {})
        if not options.get("enabled", False):
            return
        loss_type = options.get("loss_type", "l1")
        if loss_type == "l1":
            return
        if loss_type != "velocity":
            raise ValueError(f"Unknown pseudo_expert.loss_type: {loss_type}")
        experts = normalize_trajectory(targets["pseudo_experts"].float())
        valid = targets["pseudo_expert_mask"].bool()
        loss, indices = multi_gt_velocity_loss(
            pred_x, experts, valid, timesteps, self.schedule, matching="nearest")
        # forward_train moves this differentiable scalar into its loss context;
        # diagnostic logging receives only the separate scalar match count.
        diagnostics["_pseudo_expert_velocity_loss"] = loss
        diagnostics["pseudo_expert_matched_proposals"] = matched_proposal_count(
            indices, indices >= 0, pred_x.shape[1])

    def flow_training_loss(self, targets, scene, ego):
        """Standalone flow-loss helper; the trainer reuses forward_train output."""
        _, loss, diagnostics = self.flow_training_outputs(targets, scene, ego)
        diagnostics.pop("_pseudo_expert_velocity_loss", None)
        return loss, diagnostics
