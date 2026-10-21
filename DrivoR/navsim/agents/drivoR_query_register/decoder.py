"""DrivoR blocks with a candidate-aware Action/Future attention mask."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from navsim.agents.drivoR.transformer_decoder import Block


def query_attention_mask(candidates, registers, mode='action_to_register'):
    """MHA mask: True blocks a read; layout is [all A | R0 | ... | R(M-1)].

    Action queries read all Action queries, as in DrivoR. Each candidate's
    future queries read its own Action and its own future queries. Optional
    bidirectional mode additionally lets an Action read its own future group.
    """
    if mode not in ('action_to_register', 'bidirectional'):
        raise ValueError('query_register.attention_mode must be action_to_register or bidirectional')
    count = candidates * (1 + registers)
    blocked = torch.ones(count, count, dtype=torch.bool)
    blocked[:candidates, :candidates] = False
    for candidate in range(candidates):
        start = candidates + candidate * registers
        end = start + registers
        blocked[start:end, start:end] = False
        blocked[start:end, candidate] = False
        if mode == 'bidirectional':
            blocked[candidate, start:end] = False
    return blocked


class QueryRegisterBlock(Block):
    """Original SA -> CA -> FFN, extending only the self-attention mask API."""

    def forward(self, x, x_cross, blocked):
        normalized = self.self_attn_norm(x)
        attention = self.self_attn.attn(
            normalized, normalized, normalized, attn_mask=blocked, need_weights=False)[0]
        x = x + self.self_attn_drop_path(self.self_attn_ls(self.self_attn.proj_drop(attention)))
        x = x + self.cross_attn_drop_path(self.cross_attn_ls(self.cross_attn(
            self.cross_attn_norm_q(x), self.cross_attn_norm_kv(x_cross))))
        return x + self.mlp_drop_path(self.mlp_ls(self.mlp(self.mlp_norm(x))))


class QueryRegisterDecoder(nn.Module):
    def __init__(self, config, registers):
        super().__init__()
        options = config.query_register
        self.candidates = int(config.proposal_num)
        self.registers = int(registers)
        self.use_gradient_checkpointing = bool(options.gradient_checkpointing)
        self.layers = nn.ModuleList([
            QueryRegisterBlock(dim=config.tf_d_model, num_heads=config.refiner_num_heads,
                               init_values=config.refiner_ls_values, proj_drop=0.1, drop_path=0.2)
            for _ in range(config.ref_num)])
        self.register_buffer('attention_mask', query_attention_mask(
            self.candidates, self.registers, options.attention_mode), persistent=False)

    def forward(self, action, future, scene):
        batch, candidates, registers, width = future.shape
        if candidates != self.candidates or registers != self.registers:
            raise ValueError('Query tensor layout differs from the configured candidates/registers')
        tokens = torch.cat((action, future.flatten(1, 2)), dim=1)
        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(layer, tokens, scene, self.attention_mask, use_reentrant=False)
            else:
                tokens = layer(tokens, scene, self.attention_mask)
        return tokens[:, :candidates], tokens[:, candidates:].reshape(batch, candidates, registers, width)
