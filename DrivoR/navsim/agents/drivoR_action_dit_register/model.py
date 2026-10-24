"""Joint Action/Register denoising, optional register-conditioned DrivoR scoring."""

import torch
from torch import nn

from navsim.agents.drivoR.multi_gt import multi_gt_enabled, valid_multi_gt, balanced_gt_inputs, multi_gt_velocity_loss, matched_proposal_count
from navsim.agents.drivoR_action_dit.flow import normalize_trajectory, denormalize_trajectory
from navsim.agents.drivoR_action_dit.model import DrivoRActionDiTModel

from .dit import ActionRegisterDiT
from .losses import assigned_register_loss, matched_register_loss
from .teacher import build_future_teacher


class DrivoRActionRegisterModel(DrivoRActionDiTModel):
    def __init__(self, config):
        super().__init__(config)
        if config.image_backbone.get('family') != 'dinov3':
            raise ValueError('Future registers are supervised by the current DINOv3 encoder')
        options = config.future_register
        self.future_stage = str(options.stage)
        if self.future_stage not in ('joint','future_only'):
            raise ValueError('future_register.stage must be joint or future_only')
        self.register_frames = 8
        self.register_loss_type = str(options.loss_type)
        self.use_register_scorer = bool(options.scorer.enabled)
        self.register_fusion = str(options.scorer.fusion)
        if config.multi_gt.get('input_gt_assignment','original') != 'original':
            raise ValueError('These register variants keep original-human Action and future-register noise anchors')
        self._validate_auxiliary_config(config)
        if options.get('freeze_current_vit', False):
            self.image_backbone.native_vit.requires_grad_(False)
        self.ema_teacher = build_future_teacher(self.image_backbone.native_vit, options)
        self.registers_per_frame = self.ema_teacher.register_count
        self.register_dim = self.ema_teacher.dim
        self.register_count = self.register_frames*self.registers_per_frame
        self.action_dit = ActionRegisterDiT(self.action_dit,self.register_dim,self.poses_num,
            self.register_frames,self.registers_per_frame,
            token_order=options.get('token_order', 'action_first'),
            attention_mode=options.get('attention_mode', 'register_reads_action'))
        if self.use_register_scorer:
            width = self.register_count*self.register_dim
            if self.register_fusion == 'flatten':
                self.pos_embed = nn.Sequential(nn.Linear(self.poses_num*3+width,config.tf_d_ffn),
                                               nn.ReLU(),nn.Linear(config.tf_d_ffn,config.tf_d_model))
            elif self.register_fusion == 'project':
                self.register_score_projection = nn.Sequential(nn.Linear(width,config.tf_d_model),nn.ReLU())
                self.score_fusion = nn.Sequential(nn.Linear(2*config.tf_d_model,config.tf_d_ffn),nn.ReLU(),
                                                  nn.Linear(config.tf_d_ffn,config.tf_d_model))
            else:
                raise ValueError('Register scorer fusion must be flatten or project')
        if self.future_stage == 'future_only':
            # GT actions are context; their final output head and the entire
            # scoring branch have no role in Stage1's future denoising loss.
            self.action_dit.head.requires_grad_(False)
            for name in ('pos_embed','scorer_attention','scorer','register_score_projection','score_fusion'):
                module = getattr(self,name,None)
                if module is not None:
                    module.requires_grad_(False)

    def _validate_auxiliary_config(self, config):
        experts = config.get('pseudo_expert', {})
        if experts.get('enabled', False) and experts.get('loss_type') != 'velocity':
            raise ValueError('Register pseudo-experts use pseudo_expert.loss_type=velocity')

    def score_proposals(self, proposals, scene, ego, predicted_registers=None):
        if not self.use_register_scorer:
            return super().score_proposals(proposals,scene,ego)
        if predicted_registers is None:
            raise ValueError('Register-conditioned scorer needs predicted future registers')
        action = proposals.detach().float().flatten(-2)
        registers = predicted_registers.detach().float().flatten(2)
        if self.register_fusion == 'flatten':
            embedding = self.pos_embed(torch.cat((action,registers),dim=-1))
        else:
            embedding = self.score_fusion(torch.cat((self.pos_embed(action),
                                                      self.register_score_projection(registers)),dim=-1))
        return self._score_embedded_proposals(proposals,embedding,scene,ego)

    def forward_train(self, features, targets):
        scene,ego = self.encode_scene(features)
        human = normalize_trajectory(targets['trajectory'].detach().float())
        human_valid = targets['human_future_register_valid'].bool()
        if multi_gt_enabled(self._config):
            gt,valid = valid_multi_gt(targets)
            normalized_gt = normalize_trajectory(gt)
            future_valid = targets['future_register_valid'].bool() & valid[...,None]
            pixels = torch.cat((targets['human_future_register_images'][:,None],targets['future_register_images']),dim=1)
            image_valid = torch.cat((human_valid[:,None],future_valid),dim=1)
            teacher_all = self.ema_teacher(pixels,image_valid)
            human_registers,teacher = teacher_all[:,0],teacher_all[:,1:]
        else:
            # A single logged-human trajectory supplies both the action target
            # and paired future targets. Encode its 8 frames once, reusing the
            # clean registers as the noise anchor and the regression target.
            normalized_gt = human[:,None]
            valid = torch.ones((len(scene),1),dtype=torch.bool,device=scene.device)
            future_valid = human_valid[:,None]
            teacher = self.ema_teacher(targets['human_future_register_images'][:,None],future_valid)
            human_registers = teacher[:,0]
        # All 64 register inputs use the ORIGINAL logged-human future frames.
        # Cosmos GT identity affects only output supervision, not the anchors.
        anchor_registers = human_registers.flatten(1,2)[:,None].expand(-1,self.proposal_num,-1,-1)
        times = self.schedule.sample_timesteps(len(scene),self.proposal_num,scene.device)
        sigma = self.schedule.sigma(times)[...,None,None]
        future_only = self.future_stage == 'future_only'
        if future_only:
            paired_actions,assignment = balanced_gt_inputs(normalized_gt,valid,self.proposal_num,human)
            action_input = paired_actions
        else:
            anchors = human[:,None].expand(-1,self.proposal_num,-1,-1)
            action_noise = torch.randn_like(anchors)
            action_input = self.schedule.add_noise(anchors,action_noise,times)
        register_noise = torch.randn_like(anchor_registers)
        noisy_registers = (1-sigma)*anchor_registers+sigma*register_noise
        predicted_action,predicted_registers = self.action_dit(
            action_input,noisy_registers,times,scene,ego,clean_action=future_only)
        predicted_registers = predicted_registers.reshape(len(scene),self.proposal_num,
                                                          self.register_frames,self.registers_per_frame,self.register_dim)
        if future_only:
            register_loss = assigned_register_loss(predicted_registers,teacher,future_valid,assignment,
                                                     times,self.schedule,self.register_loss_type)
            # These are input GTs for logging, not generated or scored plans.
            output = {'proposals':denormalize_trajectory(paired_actions).detach(),
                      '_flow_loss':register_loss.detach()*0.}
        else:
            action_loss,matched = multi_gt_velocity_loss(predicted_action,normalized_gt,valid,times,
                                                           self.schedule,matching='nearest')
            register_loss = matched_register_loss(predicted_registers,teacher,future_valid,matched,
                                                    times,self.schedule,self.register_loss_type)
            proposals = denormalize_trajectory(predicted_action.detach().float())
            output = self.score_proposals(proposals,scene,ego,predicted_registers)
            output['_flow_loss'] = action_loss
            output['_register_matched_indices'] = matched
        output['_register_loss'] = register_loss
        output['predicted_registers'] = predicted_registers
        diagnostics = {'flow_time_mean':times.detach().mean(),
                       'multi_gt_count':valid.sum(-1).float().mean(),
                       'future_register_gt_count':future_valid.any(-1).sum(-1).float().mean(),
                       'future_register_frame_fraction':future_valid.float().mean(),
                       'ema_updates':self.ema_teacher.updates.detach().float().clone()}
        if not future_only:
            diagnostics['flow_multi_gt_loss'] = action_loss.detach()
            diagnostics['multi_gt_matched_proposals'] = matched_proposal_count(matched,matched>=0,self.proposal_num)
            # CLOVER experts have trajectory targets but no paired future
            # videos. Reuse this same X prediction/time for their action loss;
            # keep future matching restricted to the original video-backed GTs.
            self._add_pseudo_expert_velocity_loss(targets,predicted_action,times,diagnostics)
            if '_pseudo_expert_velocity_loss' in diagnostics:
                output['_pseudo_expert_velocity_loss'] = diagnostics.pop('_pseudo_expert_velocity_loss')
        if 'multi_gt_fallback' in targets:
            diagnostics['multi_gt_fallback_fraction'] = targets['multi_gt_fallback'].float().mean()
        output['_flow_diagnostics'] = diagnostics
        return output

    def sample_joint(self, scene, ego, action_noise=None, register_noise=None):
        b = len(scene)
        action = (torch.randn(b,self.proposal_num,self.poses_num,3,device=scene.device)
                  if action_noise is None else action_noise.float())
        registers = (torch.randn(b,self.proposal_num,self.register_count,self.register_dim,device=scene.device)
                     if register_noise is None else register_noise.float())
        times,deltas = self.schedule.inference_schedule(self.num_inference_steps,scene.device)
        for index,(time,delta) in enumerate(zip(times,deltas)):
            timestep = time.expand(b,self.proposal_num)
            action_x,register_x = self.action_dit(action,registers,timestep,scene,ego)
            action_x,register_x = action_x.float(),register_x.float()
            if index == self.num_inference_steps-1:
                return action_x,register_x
            sigma = self.schedule.sigma(timestep)[...,None,None]
            # Inference uses the actual sigma; the loss floor is training-only.
            action = action+(action-action_x)/sigma*delta
            registers = registers+(registers-register_x)/sigma*delta
        return action,registers

    def forward(self, features, targets=None):
        if targets is not None:
            return self.forward_train(features,targets)
        if self.future_stage == 'future_only':
            raise ValueError('Stage1 requires GT action context; use Stage2 for planning inference')
        scene,ego = self.encode_scene(features)
        with torch.no_grad(),torch.autocast(device_type=scene.device.type,
                enabled=torch.is_autocast_enabled(scene.device.type),dtype=torch.get_autocast_dtype(scene.device.type),
                cache_enabled=False):
            action,registers = self.sample_joint(scene.detach(),ego.detach())
        registers = registers.reshape(len(scene),self.proposal_num,self.register_frames,self.registers_per_frame,self.register_dim)
        output = self.score_proposals(denormalize_trajectory(action),scene,ego,registers)
        output['predicted_registers'] = registers
        return output
