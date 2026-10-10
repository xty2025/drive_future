"""Small PEFT layer used by the v6 student; the wrapped base stays frozen."""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 16.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base(inputs)
        update = self.dropout(inputs) @ self.lora_a.t() @ self.lora_b.t()
        return base + self.scaling * update


def attach_lora(module: nn.Module, names: tuple[str, ...] = ("qkv", "proj"), rank: int = 16, alpha: float = 16.0) -> list[str]:
    attached: list[str] = []
    for parent_name, parent in module.named_modules():
        for child_name, child in list(parent.named_children()):
            if child_name not in names or not isinstance(child, nn.Linear):
                continue
            setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha))
            attached.append(f"{parent_name}.{child_name}".strip("."))
    return attached


def freeze_base_keep_lora(module: nn.Module) -> None:
    for name, parameter in module.named_parameters():
        parameter.requires_grad_("lora_a" in name or "lora_b" in name)
