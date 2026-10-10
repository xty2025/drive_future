from typing import Dict
import os
import numpy as np
import torch
import torch.nn as nn
from .score_module.scorer import Scorer
from .transformer_decoder import TransformerDecoder, TransformerDecoderScorer
from .layers.image_encoder.dinov2_lora import ImgEncoder
from .layers.utils.mlp import MLP
from navsim.agents.drivoR.utils import pylogger
log = pylogger.get_pylogger(__name__)
import logging
# log.setLevel(logging.DEBUG)

class DrivoRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.poses_num=config.num_poses
        self.state_size=3
        self.embed_dims = self._config.tf_d_model

        ###########################################
        # camera embedding
        self.num_cams = 0
        if len(self._config["cam_f0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_b0"]) > 0:
            self.num_cams += 1

        ############################################
        # lidar embedding
        self.num_lidar = 0
        if len(self._config["lidar_pc"]) > 0:
            self.num_lidar += 1

        # create the image backbone
        if self.num_cams > 0:
            config_image_backbone = config["image_backbone"]
            config_image_backbone["image_size"] = config["image_size"]
            config_image_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_image_backbone["tf_d_model"] = config["tf_d_model"]
            self.image_backbone = ImgEncoder(config_image_backbone)
            self.scene_embeds = nn.Parameter(torch.randn(1, self.num_cams, self._config.num_scene_tokens, self.image_backbone.num_features)*1e-6, requires_grad=True)

            # print("self.scene_embeds ", self.scene_embeds)

        # create the lidar backbone
        if self.num_lidar > 0:
            config_lidar_backbone = config["lidar_backbone"]
            config_lidar_backbone["image_size"] = config["lidar_image_size"]
            config_lidar_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_lidar_backbone["tf_d_model"] = config["tf_d_model"]
            self.lidar_backbone = ImgEncoder(config_lidar_backbone)
            self.lidar_scene_embeds = nn.Parameter(torch.randn(1, self.num_lidar, self._config.num_scene_tokens, self.image_backbone.num_features)*1e-6, requires_grad=True)

        # ego status encoder
        if self._config.full_history_status:
            self.hist_encoding = nn.Linear(11*4, config.tf_d_model)
        else:
            self.hist_encoding = nn.Linear(11, config.tf_d_model)

        # trajectory embdedding
        if self._config.one_token_per_traj:
            self.init_feature = nn.Embedding(config.proposal_num, config.tf_d_model)
            traj_head_output_size = self.poses_num*self.state_size
        else:
            self.init_feature = nn.Embedding(self.poses_num * config.proposal_num, config.tf_d_model)
            traj_head_output_size =self.state_size

        # trajectory decoder
        self.trajectory_decoder = TransformerDecoder(proj_drop=0.1, drop_path=0.2, config=config)

        # scorer decoder
        self.scorer_attention = TransformerDecoderScorer(num_layers=config.scorer_ref_num, d_model=config.tf_d_model, proj_drop=0.1, drop_path=0.2, config=config)

        self.pos_embed = nn.Sequential(
                nn.Linear(self.poses_num * 3, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, config.tf_d_model),
            )


        # get the trajectory decoders
        self.poses_num=config.num_poses
        self.state_size=3
        ref_num=config.ref_num
        self.traj_head = nn.ModuleList([MLP(config.tf_d_model, config.tf_d_ffn,  traj_head_output_size) for _ in range(ref_num+1)])

        # scorer
        self.scorer = Scorer(config)

        # Keep every original DrivoR module's RNG initialization unchanged by
        # constructing the new preference-only head after the original model.
        self.policy_head = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.GELU(),
            nn.Linear(config.tf_d_ffn, 1),
        )

        self.b2d=config.b2d

    def _trajectory_prior(self, raw: torch.Tensor, ego_status: torch.Tensor) -> torch.Tensor:
        """Decode dynamic residuals around ego-local kinematic anchors.

        ``trajectory_prior_weight=0`` exactly recovers the original DrivoR
        absolute-coordinate head.  The anchor mode only changes the coordinate
        origin of that same learnable head; it does not add trainable modules.
        """
        weight = float(getattr(self._config, "trajectory_prior_weight", 0.0))
        residual_scale = float(getattr(self._config, "trajectory_residual_scale", 0.5))
        if weight == 0.0:
            return raw

        mode = str(getattr(self._config, "trajectory_prior_mode", "constant_velocity"))
        # ego status layout: pose(3), velocity(3), acceleration(3), command.
        velocity = ego_status[:, 3:5].to(dtype=raw.dtype)
        interval = float(getattr(self._config.trajectory_sampling, "interval_length", 0.5))
        times = interval * torch.arange(1, self.poses_num + 1, device=raw.device, dtype=raw.dtype)
        prior = torch.zeros_like(raw)
        if mode == "constant_velocity":
            prior[..., 0] = velocity[:, None, 0, None] * times[None, None]
            prior[..., 1] = velocity[:, None, 1, None] * times[None, None]
        elif mode == "multimodal_anchors":
            # Cross product of speed and lateral-offset anchors.  This borrows
            # DriveDPO's useful multi-modal support while every slot remains a
            # scene-conditioned DrivoR proposal with a learnable residual.
            num_proposals = raw.shape[1]
            num_lateral = max(1, int(round(num_proposals ** 0.5)))
            num_speed = (num_proposals + num_lateral - 1) // num_lateral
            speed_min = float(getattr(self._config, "trajectory_anchor_speed_min", 0.65))
            speed_max = float(getattr(self._config, "trajectory_anchor_speed_max", 1.35))
            lateral_max = float(getattr(self._config, "trajectory_anchor_lateral_max", 4.0))
            speed = torch.linspace(speed_min, speed_max, num_speed, device=raw.device, dtype=raw.dtype)
            lateral = torch.linspace(-lateral_max, lateral_max, num_lateral, device=raw.device, dtype=raw.dtype)
            speed = speed.repeat_interleave(num_lateral)[:num_proposals]
            lateral = lateral.repeat(num_speed)[:num_proposals]
            progress = times / times[-1]

            vx = velocity[:, 0, None, None]
            vy = velocity[:, 1, None, None]
            prior[..., 0] = vx * speed[None, :, None] * times[None, None, :]
            prior[..., 1] = vy * times[None, None, :] + lateral[None, :, None] * progress[None, None, :].square()
            lateral_velocity = 2.0 * lateral[None, :, None] * progress[None, None, :] / times[-1]
            prior[..., 2] = torch.atan2(
                vy + lateral_velocity,
                (vx * speed[None, :, None]).expand(-1, -1, self.poses_num).clamp_min(1e-3),
            )
        else:
            raise ValueError(f"Unknown trajectory_prior_mode: {mode}")

        residual = residual_scale * torch.tanh(raw)
        return weight * prior + residual + (1.0 - weight) * raw


    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        # ego status and initial traj tokens
        if self._config.full_history_status:
            ego_status: torch.Tensor = features["ego_status"].flatten(-2)
        else:
            ego_status: torch.Tensor = features["ego_status"][:, -1]
        
        ego_token = self.hist_encoding(ego_status)[:, None]
        log.debug(f"Ego features - {ego_token.shape}")
        traj_tokens = ego_token + self.init_feature.weight[None]
        log.debug(f"Traj tokens initial - {traj_tokens.shape}")


        batch_size = ego_status.shape[0]



        scene_features = []
        # image features
        if self.num_cams > 0:
            
            if "image" in features :
                img = features["image"]
            elif "camera_feature" in features:
                img = features["camera_feature"]
            else:
                raise ValueError

            scene_tokens = self.scene_embeds.repeat(batch_size, 1, 1, 1)
            image_scene_tokens = self.image_backbone(img, scene_tokens)

            log.debug(f"Backbone image - {image_scene_tokens.shape}")
            scene_features.append(image_scene_tokens)

        # lidar features
        if self.num_lidar > 0:
            img = features["lidar_feature"]
            scene_tokens = self.lidar_scene_embeds.repeat(batch_size, 1, 1, 1)
            lidar_scene_tokens = self.lidar_backbone(img, scene_tokens)
            log.debug(f"Backbone lidar - {lidar_scene_tokens.shape}")
            scene_features.append(lidar_scene_tokens)

        scene_features = torch.cat(scene_features, dim=1)
        log.debug(f"Scene features - {scene_features.shape}")

        # initial trajectories
        proposals = self._trajectory_prior(
            self.traj_head[0](traj_tokens).reshape(traj_tokens.shape[0], -1, self.poses_num, self.state_size),
            ego_status,
        )
        proposal_list = [proposals]
        log.debug(f"Proposals initial - {proposals.shape}")

        # decode the trajectories at each step of the decoder
        token_list = self.trajectory_decoder(traj_tokens, scene_features)
        log.debug(f"Trajectory decoder - {len(token_list)}")
        for i in range(self._config.ref_num):
            tokens = token_list[i]
            proposals = self._trajectory_prior(
                self.traj_head[i+1](tokens).reshape(tokens.shape[0], -1, self.poses_num, self.state_size),
                ego_status,
            )
            proposal_list.append(proposals)
        
        traj_tokens = token_list[-1]
        proposals=proposal_list[-1]
        

        output={}
        output["proposals"] = proposals
        output["proposal_list"] = proposal_list
        policy_logits = self.policy_head(traj_tokens).squeeze(-1)
        output["policy_logits"] = policy_logits
        output["policy_log_probs"] = torch.log_softmax(policy_logits, dim=1)

        # scoring
        B,N,_,_=proposals.shape

        embedded_traj = self.pos_embed(proposals.reshape(B, N, -1).detach())  # (B, N, d_model)
        tr_out = self.scorer_attention(embedded_traj, scene_features)  # (B, N, d_model)
        tr_out = tr_out+ego_token
        pred_logit,pred_logit2, pred_agents_states, pred_area_logit ,bev_semantic_map,agent_states,agent_labels= self.scorer(proposals, tr_out)

        output["pred_logit"]=pred_logit
        output["pred_logit2"]=pred_logit2
        output["pred_agents_states"]=pred_agents_states
        output["pred_area_logit"]=pred_area_logit
        output["bev_semantic_map"]=bev_semantic_map
        output["agent_states"]=agent_states
        output["agent_labels"]=agent_labels

        pdm_score = (
        self._config.noc * pred_logit['no_at_fault_collisions'].sigmoid().log() +
        self._config.dac * pred_logit['drivable_area_compliance'].sigmoid().log() +
        self._config.ddc * pred_logit['driving_direction_compliance'].sigmoid().log() +    
        (self._config.ttc * pred_logit['time_to_collision_within_bound'].sigmoid() +
        self._config.ep * pred_logit['ego_progress'].sigmoid()  
        + self._config.comfort * pred_logit['comfort'].sigmoid()).log()
        )

        # Scorer-primary inference with an optional, conservative policy prior.
        # Setting policy_selection_weight=0 recovers the official DrivoR rule.
        shortlist_size = min(int(getattr(self._config, "selection_top_m", 8)), N)
        top_idx = torch.topk(pdm_score, k=shortlist_size, dim=1).indices
        weight = float(getattr(self._config, "policy_selection_weight", 0.0))
        if weight:
            q = pdm_score.gather(1, top_idx)
            p = output["policy_log_probs"].gather(1, top_idx)
            q = (q - q.mean(1, keepdim=True)) / q.std(1, keepdim=True).clamp_min(1e-6)
            p = (p - p.mean(1, keepdim=True)) / p.std(1, keepdim=True).clamp_min(1e-6)
            token = top_idx[torch.arange(batch_size, device=top_idx.device), (q + weight * p).argmax(1)]
        else:
            token = torch.argmax(pdm_score, dim=1)
        trajectory = proposals[torch.arange(batch_size), token]

        output["trajectory"] = trajectory
        output["pdm_score"] = pdm_score

        if os.environ.get("DRIVOR_DIAGNOSTIC_SHARED_HEAD", "0") == "1":
            shared_head_proposals = []
            shared_head_scores = []
            for stage_tokens in token_list:
                stage_proposals = self.traj_head[-1](stage_tokens).reshape(
                    stage_tokens.shape[0], -1, self.poses_num, self.state_size
                )
                stage_embedded = self.pos_embed(stage_proposals.reshape(B, N, -1).detach())
                stage_features = self.scorer_attention(stage_embedded, scene_features) + ego_token
                stage_logits = self.scorer(stage_proposals, stage_features)[0]
                stage_score = (
                    self._config.noc * stage_logits['no_at_fault_collisions'].sigmoid().log()
                    + self._config.dac * stage_logits['drivable_area_compliance'].sigmoid().log()
                    + self._config.ddc * stage_logits['driving_direction_compliance'].sigmoid().log()
                    + (
                        self._config.ttc * stage_logits['time_to_collision_within_bound'].sigmoid()
                        + self._config.ep * stage_logits['ego_progress'].sigmoid()
                        + self._config.comfort * stage_logits['comfort'].sigmoid()
                    ).log()
                )
                shared_head_proposals.append(stage_proposals)
                shared_head_scores.append(stage_score)
            output["shared_head_proposals"] = shared_head_proposals
            output["shared_head_scores"] = shared_head_scores

        return output
