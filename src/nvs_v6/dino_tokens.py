"""Explicit DINOv3 token-family handling.

The generator and production scratch scorer use the NVS DINOv3
CLS/register/query path.  The optional native DrivoR comparison tool has its
own DINOv2 tower and must never pass through this adapter.
"""

from __future__ import annotations

import torch


def split_dinov3_tokens(
    features: torch.Tensor | dict[str, torch.Tensor], prefix_count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return rank-3 ``(CLS, registers, patches)`` without dropping a family.

    Timm exposes register-free and register-bearing DINOv3 variants through the
    same API.  A register-free model therefore returns an empty rank-3 register
    tensor rather than being mistaken for patch-only output.
    """

    if prefix_count < 1:
        raise ValueError("DINOv3 must expose a CLS prefix token")
    if isinstance(features, dict):
        cls = features.get("x_norm_clstoken")
        registers = features.get("x_norm_regtokens")
        patches = features.get("x_norm_patchtokens")
        if cls is not None and patches is not None:
            if cls.ndim == 2:
                cls = cls.unsqueeze(1)
            if registers is None and prefix_count == 1:
                registers = cls[:, :0]
            if registers is not None:
                return _validate(cls, registers, patches, prefix_count)
        features = features.get("x")
    if not isinstance(features, torch.Tensor) or features.ndim != 3:
        raise ValueError("DINOv3 output must be a token tensor or named dictionary")
    return _validate(
        features[:, :1],
        features[:, 1:prefix_count],
        features[:, prefix_count:],
        prefix_count,
    )


def _validate(
    cls: torch.Tensor,
    registers: torch.Tensor,
    patches: torch.Tensor,
    prefix_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if any(item.ndim != 3 for item in (cls, registers, patches)):
        raise ValueError("DINOv3 token families must have shape [batch,tokens,channels]")
    if cls.shape[1] != 1 or registers.shape[1] != prefix_count - 1 or patches.shape[1] < 1:
        raise ValueError("missing or mismatched DINOv3 CLS/register/patch tokens")
    if len({(item.shape[0], item.shape[2]) for item in (cls, registers, patches)}) != 1:
        raise ValueError("DINOv3 token-family batch/channel dimensions differ")
    return cls, registers, patches
