"""FastWAM-style per-expert DiT blocks: Register reads Action, never the reverse.

Each expert owns its projections, norms, FFN and modulation. Q/K/V share an
attention width, are concatenated for one masked SDPA, then split back to the
experts. Current DINO features supply both cross-attention contexts. No video
VAE, text encoder or condition-video tokens are part of this generator.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from navsim.agents.drivoR_action_dit.action_dit import ActionDiT, apply_rope, sinusoidal_embedding_1d


def expert_attention_io(block, tokens, modulation, *, rope_offset=0):
    """Project an expert's hidden width to its shared H*Dh attention width."""
    b, m, length, width = tokens.shape
    mods = tuple(value[:, :, None] for value in
                 (modulation + block.modulation.to(modulation)).unbind(dim=-2))
    shift, scale, gate, shift_ff, scale_ff, gate_ff = mods
    normalized = (block.norm1(tokens) * (1 + scale) + shift).reshape(b * m, length, width)
    attention = block.self_attn
    q = apply_rope(attention.norm_q(attention.q(normalized)), attention.num_heads, offset=rope_offset)
    k = apply_rope(attention.norm_k(attention.k(normalized)), attention.num_heads, offset=rope_offset)
    v = attention.v(normalized)
    def split(value):
        return value.reshape(b * m, length, attention.num_heads, attention.attn_head_dim).transpose(1, 2)
    return (split(q), split(k), split(v)), (gate, shift_ff, scale_ff, gate_ff)


def expert_post_block(block, tokens, mixed, context, gates):
    gate, shift_ff, scale_ff, gate_ff = gates
    tokens = tokens + gate * block.self_attn.o(mixed)
    tokens = tokens + block.cross_attn(block.norm3(tokens), context)
    return tokens + gate_ff * block.ffn(block.norm2(tokens) * (1 + scale_ff) + shift_ff)


def mot_block(video_block, action_block, video, action, video_context, action_context,
              video_modulation, action_modulation, mask):
    # Keep A28's rotary positions: A starts at 0, R starts after the A tokens.
    # Packing experts as [R | A] does not change these position identities.
    video_qkv, video_gates = expert_attention_io(video_block, video, video_modulation,
                                                rope_offset=action.shape[2])
    action_qkv, action_gates = expert_attention_io(action_block, action, action_modulation)
    # Logical order [R | A]. Candidates remain separate in the flattened batch.
    q, k, v = (torch.cat((v_part, a_part), dim=-2) for v_part, a_part in zip(video_qkv, action_qkv))
    mixed = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    b, m = action.shape[:2]
    mixed = mixed.transpose(1, 2).reshape(b, m, video.shape[2] + action.shape[2], -1)
    r_part, a_part = mixed.split((video.shape[2], action.shape[2]), dim=2)
    return (expert_post_block(video_block, video, r_part, video_context, video_gates),
            expert_post_block(action_block, action, a_part, action_context, action_gates))


class ActionRegisterMoT(nn.Module):
    def __init__(self, action_dit, video_options, register_dim, context_dim=256,
                 action_tokens=8, future_frames=8, registers_per_frame=4):
        super().__init__()
        expected = action_dit.checkpoint_meta
        for key in ('num_layers', 'num_heads', 'attn_head_dim'):
            if int(video_options[key]) != expected[key]:
                raise ValueError(f'MoT experts must agree on {key}')
        # The native video backbone now has register-valued I/O. These two
        # linears are new; every compatible native block/time tensor is loaded.
        video = ActionDiT(hidden_dim=video_options.hidden_dim, action_dim=register_dim,
            ffn_dim=video_options.ffn_dim, freq_dim=video_options.freq_dim,
            num_heads=video_options.num_heads, attn_head_dim=video_options.attn_head_dim,
            num_layers=video_options.num_layers, eps=video_options.eps,
            use_gradient_checkpointing=video_options.gradient_checkpointing, context_dim=context_dim)
        if video_options.pretrained_path:
            video.load_wan_pretrained(video_options.pretrained_path,
                checkpoint_format=video_options.get('pretrained_format', 'wan22_native'))
        self.mixtures = nn.ModuleDict({'video': video, 'action': action_dit})
        self.expert_order = ('video', 'action')
        self.action_tokens = int(action_tokens)
        self.future_frames = int(future_frames)
        self.registers_per_frame = int(registers_per_frame)
        self.register_tokens = self.future_frames * self.registers_per_frame
        self.register_dim = int(register_dim)
        self.use_gradient_checkpointing = (action_dit.use_gradient_checkpointing
                                           or video.use_gradient_checkpointing)
        self.future_time_embedding = nn.Parameter(torch.randn(self.future_frames, video.hidden_dim) * .02)
        self.register_slot_embedding = nn.Parameter(torch.randn(self.registers_per_frame, video.hidden_dim) * .02)
        mask = torch.zeros(self.register_tokens + self.action_tokens,
                           self.register_tokens + self.action_tokens, dtype=torch.bool)
        mask[:self.register_tokens, :] = True
        mask[self.register_tokens:, self.register_tokens:] = True
        self.register_buffer('attention_mask', mask, persistent=False)
        self.load_report = dict(format='mot_register_reads_action', action=action_dit.load_report, video=video.load_report)

    @property
    def head(self):
        return self.mixtures['action'].head

    @property
    def register_head(self):
        return self.mixtures['video'].head

    @staticmethod
    def modulation(expert, timesteps, ego):
        b, m = timesteps.shape
        time = expert.time_embedding(sinusoidal_embedding_1d(expert.freq_dim, timesteps.float().reshape(-1)))
        condition = time.reshape(b, m, expert.hidden_dim) + expert.ego_projection(ego)[:, None]
        return expert.time_projection(condition).reshape(b, m, 6, expert.hidden_dim)

    def forward(self, noisy_action, noisy_registers, timesteps, scene, ego, *, clean_action=False):
        video_expert, action_expert = self.mixtures['video'], self.mixtures['action']
        video = video_expert.action_encoder(noisy_registers)
        position = (self.future_time_embedding[:, None] + self.register_slot_embedding[None]).reshape(
            1, 1, self.register_tokens, video_expert.hidden_dim)
        video = video + position.to(video.dtype)
        action = action_expert.action_encoder(noisy_action)
        video_context = video_expert.scene_projection(scene)
        action_context = action_expert.scene_projection(scene)
        video_mod = self.modulation(video_expert, timesteps, ego)
        action_mod = self.modulation(action_expert, torch.zeros_like(timesteps) if clean_action else timesteps, ego)
        for video_block, action_block in zip(video_expert.blocks, action_expert.blocks):
            args = (video_block, action_block, video, action, video_context, action_context,
                    video_mod, action_mod, self.attention_mask)
            if self.use_gradient_checkpointing and self.training and torch.is_grad_enabled():
                video, action = checkpoint(mot_block, *args, use_reentrant=False)
            else:
                video, action = mot_block(*args)
        return (noisy_action if clean_action else action_expert.head(action), video_expert.head(video))
