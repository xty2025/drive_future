"""Pretrained Action-DiT blocks with directed Action/Register self-attention."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from navsim.agents.drivoR_action_dit.action_dit import apply_rope, sinusoidal_embedding_1d


def directed_self_attention(attention, x, mask):
    b, candidates, tokens, dim = x.shape
    flat = x.reshape(b*candidates,tokens,dim)
    q = apply_rope(attention.norm_q(attention.q(flat)),attention.num_heads)
    k = apply_rope(attention.norm_k(attention.k(flat)),attention.num_heads)
    v = attention.v(flat)
    def split(value):
        return value.reshape(b*candidates,tokens,attention.num_heads,attention.attn_head_dim).transpose(1,2)
    value = F.scaled_dot_product_attention(split(q),split(k),split(v),attn_mask=mask)
    value = value.transpose(1,2).reshape(b,candidates,tokens,-1)
    return attention.o(value)


def joint_block(block, tokens, scene, modulation, mask):
    shift_sa,scale_sa,gate_sa,shift_ff,scale_ff,gate_ff = (
        modulation + block.modulation.to(modulation)).unbind(dim=-2)
    tokens = tokens + gate_sa * directed_self_attention(
        block.self_attn,block.norm1(tokens)*(1+scale_sa)+shift_sa,mask)
    tokens = tokens + block.cross_attn(block.norm3(tokens),scene)
    return tokens + gate_ff * block.ffn(block.norm2(tokens)*(1+scale_ff)+shift_ff)


class ActionRegisterDiT(nn.Module):
    def __init__(self, action_dit, register_dim, action_tokens=8, future_frames=8, registers_per_frame=4,
                 token_order='action_first', attention_mode='register_reads_action'):
        super().__init__()
        # Reuse pretrained modules and their parameter names without creating
        # or loading a second Action-DiT. New register adapters start randomly.
        for name in ('action_encoder','time_embedding','time_projection','blocks','head',
                     'scene_projection','ego_projection'):
            setattr(self,name,getattr(action_dit,name))
        self.hidden_dim = action_dit.hidden_dim
        self.freq_dim = action_dit.freq_dim
        self.use_gradient_checkpointing = action_dit.use_gradient_checkpointing
        self.load_report = action_dit.load_report
        self.action_tokens = int(action_tokens)
        self.future_frames = int(future_frames)
        self.registers_per_frame = int(registers_per_frame)
        self.register_tokens = self.future_frames*self.registers_per_frame
        self.register_dim = int(register_dim)
        if token_order not in ('action_first', 'register_first'):
            raise ValueError('token_order must be action_first or register_first')
        if attention_mode not in ('register_reads_action', 'bidirectional'):
            raise ValueError('attention_mode must be register_reads_action or bidirectional')
        self.token_order = token_order
        self.attention_mode = attention_mode
        self.action_slice = (slice(0, self.action_tokens) if token_order == 'action_first'
                             else slice(self.register_tokens, self.register_tokens + self.action_tokens))
        self.register_slice = (slice(self.action_tokens, None) if token_order == 'action_first'
                               else slice(0, self.register_tokens))
        self.register_encoder = nn.Linear(register_dim,self.hidden_dim)
        self.register_head = nn.Linear(self.hidden_dim,register_dim)
        self.future_time_embedding = nn.Parameter(torch.randn(self.future_frames,self.hidden_dim)*.02)
        self.register_slot_embedding = nn.Parameter(torch.randn(self.registers_per_frame,self.hidden_dim)*.02)
        mask = torch.ones(self.action_tokens+self.register_tokens,self.action_tokens+self.register_tokens,dtype=torch.bool)
        if attention_mode == 'register_reads_action':
            mask[self.action_slice,self.register_slice] = False
        self.register_buffer('attention_mask',mask,persistent=False)

    def modulation(self, timesteps, ego):
        b,m = timesteps.shape
        time = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim,timesteps.float().reshape(-1)))
        condition = time.reshape(b,m,self.hidden_dim)+ego[:,None]
        return self.time_projection(condition).reshape(b,m,6,self.hidden_dim)

    def forward(self, noisy_action, noisy_registers, timesteps, scene, ego, *, clean_action=False):
        b,m = noisy_action.shape[:2]
        scene, ego = self.scene_projection(scene), self.ego_projection(ego)
        action = self.action_encoder(noisy_action)
        registers = self.register_encoder(noisy_registers)
        position = (self.future_time_embedding[:,None]+self.register_slot_embedding[None]).reshape(1,1,self.register_tokens,-1)
        registers = registers+position.to(registers.dtype)
        parts = (action, registers) if self.token_order == 'action_first' else (registers, action)
        tokens = torch.cat(parts,dim=2)
        reg_mod = self.modulation(timesteps,ego)
        action_mod = self.modulation(torch.zeros_like(timesteps),ego) if clean_action else reg_mod
        mods = (action_mod[:,:,None].expand(-1,-1,self.action_tokens,-1,-1),
                reg_mod[:,:,None].expand(-1,-1,self.register_tokens,-1,-1))
        modulation = torch.cat(mods if self.token_order == 'action_first' else mods[::-1],dim=2)
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(joint_block,block,tokens,scene,modulation,self.attention_mask,use_reentrant=False)
            else:
                tokens = joint_block(block,tokens,scene,modulation,self.attention_mask)
        action_prediction = noisy_action if clean_action else self.head(tokens[:,:,self.action_slice])
        register_prediction = self.register_head(tokens[:,:,self.register_slice])
        return action_prediction,register_prediction
