"""DrivoR-exact scene perception: prepended scene queries over a DINOv3 backbone.

DrivoR injects ``scene_embeds`` in front of the patch sequence inside
``timm_ViT._pos_embed`` and reads the first ``num_scene_tokens`` output tokens
back through a ``neck`` linear.  timm 1.0.27 implements DINOv3 as
``timm.models.eva.Eva``, which has **no** absolute ``pos_embed``: position comes
from ``RotaryEmbeddingDinoV3`` and ``_pos_embed`` returns
``(tokens, rot_pos_embed)``.  Consequences for this port:

* Any resolution works without interpolating a position table.  Passing
  ``img_size=(H, W)`` makes ``rope.get_embed(shape=(H', W'))`` build the rotary
  table analytically, so DrivoR's large input size costs nothing here.
* ``EvaAttention.forward`` applies RoPE to ``q[:, :, num_prefix_tokens:]``.  The
  prepended scene queries are prefix-style tokens, so that boundary must be
  widened by their count for the rotary table to keep lining up with the patches.
  The count never changes, so it is widened once at construction rather than per
  forward call.

LoRA reproduces DrivoR's ``_LoRA_qkv_timm``: low-rank residuals on q and v only,
k untouched, ``w_A`` kaiming(a=sqrt(5)), ``w_B`` zeros, base frozen.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

DINOV3_MODEL_NAME = "vit_large_patch16_dinov3.lvd1689m"
# DrivoR's official `image_size: [1148, 672]` (W, H) was chosen for patch14
# DINOv2, where 1148 = 14 * 82 exactly.  DINOv3 is patch16 and timm's PatchEmbed
# rejects that width outright:
#   AssertionError: Input width (1148) should be divisible by patch size (16).
# So DrivoR's resized pixels are kept byte-for-byte at 1148 and the 12 px
# remainder is closed with four ZERO columns.  Zero padding synthesizes no pixel
# value, which is the difference from interpolating the width up to 1152.  The
# pad is applied here rather than through timm's `dynamic_img_pad`, because that
# flag leaves `patch_embed.grid_size` at floor(1148/16)=71 while the padded
# forward emits 72 columns, and the resulting mismatch crashes `forward_features`.
DRIVOR_OFFICIAL_IMAGE_HW = (672, 1148)
DRIVOR_PATCH_SIZE = 16
DRIVOR_RIGHT_ZERO_PAD_PX = 4
# What the backbone is actually built for: DrivoR's size after the zero pad.  Pass
# DRIVOR_OFFICIAL_IMAGE_HW to loaders and to the Student; this one only describes
# the padded tensor the patch embedding receives.
DRIVOR_IMAGE_HW = (672, 1152)
DRIVOR_NUM_SCENE_TOKENS = 16
DRIVOR_LORA_RANK = 32


def pad_to_patch_multiple(images: torch.Tensor, patch: int = DRIVOR_PATCH_SIZE) -> torch.Tensor:
    """Zero-pad the right/bottom edge so a DrivoR-sized frame reaches patch16.

    ``F.pad`` appends zeros; it never resamples, so the 1148 columns produced by
    DrivoR's own resize pass through unchanged and the four extra columns carry no
    fabricated image content.
    """

    if images.ndim != 4:
        raise ValueError(f"expected [B,3,H,W] images, got shape {tuple(images.shape)}")
    _, _, height, width = images.shape
    pad_w = (-int(width)) % int(patch)
    pad_h = (-int(height)) % int(patch)
    if pad_w == 0 and pad_h == 0:
        return images
    return torch.nn.functional.pad(images, (0, pad_w, 0, pad_h))


class LoRAQKVtimm(nn.Module):
    """DrivoR ``_LoRA_qkv_timm``: rank-r additive residuals on the q and v slabs."""

    def __init__(self, qkv: nn.Linear, rank: int) -> None:
        super().__init__()
        if not isinstance(qkv, nn.Linear):
            raise TypeError(f"LoRA wraps a fused qkv Linear, got {type(qkv)}")
        if qkv.out_features != 3 * qkv.in_features:
            raise ValueError("LoRA expects a fused qkv projection with out_features == 3 * in_features")
        self.qkv = qkv
        self.dim = int(qkv.in_features)
        self.rank = int(rank)
        self.linear_a_q = nn.Linear(self.dim, self.rank, bias=False)
        self.linear_b_q = nn.Linear(self.rank, self.dim, bias=False)
        self.linear_a_v = nn.Linear(self.dim, self.rank, bias=False)
        self.linear_b_v = nn.Linear(self.rank, self.dim, bias=False)
        nn.init.kaiming_uniform_(self.linear_a_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.linear_a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.linear_b_q.weight)
        nn.init.zeros_(self.linear_b_v.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        dim = self.dim
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        # Out-of-place equivalent of DrivoR's in-place `qkv[:, :, :d] += new_q`,
        # `qkv[:, :, -d:] += new_v`; safe under gradient checkpointing.
        return torch.cat(
            (qkv[..., :dim] + new_q, qkv[..., dim : 2 * dim], qkv[..., 2 * dim :] + new_v),
            dim=-1,
        )


def attach_drivor_lora(backbone: nn.Module, rank: int) -> list[str]:
    """Wrap every block's fused ``attn.qkv`` exactly as ``LoRA_ViT_timm`` does."""

    if rank <= 0:
        raise ValueError("DrivoR LoRA requires a positive rank; use rank=0 semantics by not calling this")
    attached: list[str] = []
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise ValueError("backbone has no `blocks` to attach LoRA to")
    for index, block in enumerate(blocks):
        attention = getattr(block, "attn", None)
        qkv = getattr(attention, "qkv", None) if attention is not None else None
        if qkv is None:
            raise ValueError(f"block {index} has no attn.qkv Linear for LoRA attachment")
        attention.qkv = LoRAQKVtimm(qkv, rank)
        attached.append(f"blocks.{index}.attn.qkv")
    return attached


def freeze_backbone_keep_lora(backbone: nn.Module) -> None:
    for name, parameter in backbone.named_parameters():
        parameter.requires_grad_("linear_a_" in name or "linear_b_" in name)


class DrivoRDinoV3Student(nn.Module):
    """DINOv3 + DrivoR scene queries, emitting ``N * num_scene_tokens`` tokens.

    ``images`` is ``[B, N, 3, H, W]`` where N is DrivoR's camera axis; NVS uses it
    for the five history frames of the single front camera.  The output is
    ``[B, N * num_scene_tokens, output_dim]``, ordered frame-major, matching
    DrivoR's ``rearrange('(b n) t c -> b (n t) c')``.
    """

    initialization = "timm_dinov3_checkpoint_plus_scratch_scene_queries"

    def __init__(
        self,
        checkpoint: str,
        model_name: str = DINOV3_MODEL_NAME,
        image_hw: Sequence[int] = DRIVOR_OFFICIAL_IMAGE_HW,
        num_scene_tokens: int = DRIVOR_NUM_SCENE_TOKENS,
        num_frames: int = 5,
        lora_rank: int = DRIVOR_LORA_RANK,
        output_dim: int = 256,
        grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as error:  # pragma: no cover - environment guard
            raise ImportError("timm >= 1.0 is required for the DINOv3 scene encoder") from error
        if "dinov3" not in model_name:
            raise ValueError("the DrivoR scene port only accepts DINOv3 backbones")
        height, width = int(image_hw[0]), int(image_hw[1])
        self.drivor_image_hw = (height, width)
        # DrivoR's official 1148 width is 14*82 for its patch14 DINOv2 but leaves a
        # 12 px remainder under patch16.  Close it with zero columns rather than by
        # resampling the width up to 1152: DrivoR's own pixels then reach the
        # backbone byte-for-byte and the extra four columns invent nothing.
        self.right_zero_pad_px = (-width) % DRIVOR_PATCH_SIZE
        self.bottom_zero_pad_px = (-height) % DRIVOR_PATCH_SIZE
        self.image_hw = (height + self.bottom_zero_pad_px, width + self.right_zero_pad_px)
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            checkpoint_path=str(checkpoint),
            num_classes=0,
            img_size=self.image_hw,
        )
        if getattr(self.backbone, "pos_embed", None) is not None:
            raise ValueError(
                "expected a RoPE-only DINOv3 backbone; an absolute pos_embed would need resampling"
            )
        self.backbone_dim = int(self.backbone.num_features)
        self.num_prefix_tokens = int(self.backbone.num_prefix_tokens)
        self.num_scene_tokens = int(num_scene_tokens)
        self.num_frames = int(num_frames)
        self.output_dim = int(output_dim)
        default_cfg = getattr(self.backbone, "default_cfg", {})
        self.register_buffer(
            "mean",
            torch.tensor(default_cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 1, 1, 3),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(default_cfg.get("std", (0.229, 0.224, 0.225))).view(1, 1, 1, 3),
            persistent=False,
        )
        # DrivoR: `nn.Parameter(randn(1, num_cams, num_scene_tokens, num_features) * 1e-6)`.
        self.scene_embeds = nn.Parameter(
            torch.randn(1, self.num_frames, self.num_scene_tokens, self.backbone_dim) * 1e-6
        )
        self.neck = nn.Linear(self.backbone_dim, self.output_dim)
        if lora_rank > 0:
            attach_drivor_lora(self.backbone, lora_rank)
            self.lora_rank = int(lora_rank)
        else:
            self.lora_rank = 0
        freeze_backbone_keep_lora(self.backbone)
        self._widen_rope_prefix()
        if grad_checkpointing:
            self.backbone.set_grad_checkpointing(True)
        self.train(True)

    def _widen_rope_prefix(self) -> None:
        """Teach every ``EvaAttention`` that the scene queries are prefix tokens.

        The query count is fixed for the lifetime of this module, so the offset is
        set once here instead of per call.  A per-call swap would be restored
        before gradient checkpointing recomputes the block, silently applying RoPE
        at the wrong offset in the backward pass.
        """

        prefix = self.num_prefix_tokens + self.num_scene_tokens
        for block in self.backbone.blocks:
            block.attn.num_prefix_tokens = prefix
        self.rope_prefix_tokens = prefix

    def train(self, mode: bool = True) -> "DrivoRDinoV3Student":
        super().train(mode)
        # The frozen DINOv3 base must stay in eval so its dropout-free numerics
        # match the checkpoint, while LoRA/neck/scene_embeds follow `mode`.
        self.backbone.eval()
        return self

    def token_layout(self) -> dict[str, int]:
        grid = self.backbone.patch_embed.grid_size
        patch_tokens = int(grid[0]) * int(grid[1])
        return {
            "drivor_image_hw": list(self.drivor_image_hw),
            "image_hw": list(self.image_hw),
            "right_zero_pad_px": self.right_zero_pad_px,
            "bottom_zero_pad_px": self.bottom_zero_pad_px,
            "patch_size": DRIVOR_PATCH_SIZE,
            "patch_grid": [int(grid[0]), int(grid[1])],
            "native_prefix_tokens": self.num_prefix_tokens,
            "image_patches": patch_tokens,
            "scene_queries_per_frame": self.num_scene_tokens,
            "frames": self.num_frames,
            "backbone_qkv_tokens": self.num_prefix_tokens + self.num_scene_tokens + patch_tokens,
            "output_tokens": self.num_frames * self.num_scene_tokens,
            "position_encoding": "dinov3_rope_analytic_no_interpolation",
            "lora_rank": self.lora_rank,
        }

    def _prepare(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize DrivoR-style, then zero-pad the edge to a patch16 multiple.

        DrivoR resizes 1920x1080 to its official 1148x672 and stops there.  Padding
        happens *after* normalization so the added columns are exactly 0.0 in
        normalized space -- the dataset mean colour -- instead of a hard black edge
        that would read as out-of-distribution content.  Resampling is refused
        outright: silently interpolating to the padded size is how fabricated pixels
        end up in a scene encoder.
        """

        height, width = self.drivor_image_hw
        if tuple(int(value) for value in images.shape[-2:]) != (height, width):
            raise ValueError(
                f"the scene encoder does not resample: expected DrivoR's {height}x{width} frames, "
                f"got {int(images.shape[-2])}x{int(images.shape[-1])}. Apply DrivoR's own "
                "`im.resize((1148, 672))` in the loader instead."
            )
        channels_last = images.permute(0, 2, 3, 1)
        normalized = (channels_last - self.mean) / self.std
        normalized = normalized.permute(0, 3, 1, 2)
        return pad_to_patch_multiple(normalized).contiguous()

    def _forward_features(self, flat: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone
        tokens = backbone.patch_embed(flat)
        tokens, rope = backbone._pos_embed(tokens)
        tokens = backbone.norm_pre(tokens)
        token_count = queries.shape[1]
        if token_count != self.num_scene_tokens:
            raise ValueError(
                f"RoPE prefix was widened for {self.num_scene_tokens} scene queries, got {token_count}"
            )
        tokens = torch.cat((queries, tokens), dim=1)
        for block in backbone.blocks:
            tokens = block(tokens, rope=rope)
        return backbone.norm(tokens)[:, :token_count]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``[B, N, 3, H, W]`` images in [0, 1] -> ``[B, N * T, output_dim]``."""

        if images.ndim != 5:
            raise ValueError(f"scene images must be [B,N,3,H,W], got {tuple(images.shape)}")
        batch, frames = images.shape[0], images.shape[1]
        if frames != self.num_frames:
            raise ValueError(
                f"scene encoder was built for {self.num_frames} frames, got {frames}"
            )
        flat = self._prepare(images.reshape(batch * frames, *images.shape[2:]))
        queries = self.scene_embeds.repeat(batch, 1, 1, 1).reshape(
            batch * frames, self.num_scene_tokens, self.backbone_dim
        )
        tokens = self._forward_features(flat.to(self.scene_embeds.dtype), queries)
        tokens = self.neck(tokens)
        return tokens.reshape(batch, frames * self.num_scene_tokens, self.output_dim)
