"""Fixed Teacher + ordered future Register target compression."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .vision import pad_to_patch_multiple


def _concatenate_features(chunks: list[object]) -> object:
    """Re-join ``forward_features`` chunks along the image axis."""

    if isinstance(chunks[0], dict):
        return {key: torch.cat([chunk[key] for chunk in chunks], dim=0) for key in chunks[0]}
    if isinstance(chunks[0], torch.Tensor):
        return torch.cat(chunks, dim=0)
    raise ValueError(f"unsupported DINOv3 feature type {type(chunks[0])}")


# The generated future videos are natively 464x832 (H, W), and both are already
# multiples of 16, so patch16 tiles them exactly: 464 = 16*29, 832 = 16*52.  The
# Teacher therefore encodes them at their own resolution with no resize at all.
NATIVE_FUTURE_IMAGE_HW = (464, 832)


class TimmDinoV3Teacher(nn.Module):
    """Frozen raw DINOv3 patch-token Teacher used only during target construction."""

    def __init__(
        self,
        checkpoint: str | Path,
        model_name: str = "vit_large_patch16_dinov3.lvd1689m",
        image_hw: tuple[int, int] = NATIVE_FUTURE_IMAGE_HW,
        frame_chunk: int = 2,
    ) -> None:
        super().__init__()
        import timm

        height, width = int(image_hw[0]), int(image_hw[1])
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            checkpoint_path=str(checkpoint),
            num_classes=0,
            img_size=(height, width),
        )
        self.image_hw = (height, width)
        self.grid_hw = tuple(int(value) for value in self.backbone.patch_embed.grid_size)
        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
        # Attention is quadratic in the patch grid, so the eight future frames are
        # encoded a few at a time rather than as one batch of eight.
        self.frame_chunk = max(1, int(frame_chunk))
        cfg = getattr(self.backbone, "default_cfg", {})
        self.register_buffer("mean", torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 1, 1, 3), persistent=False)
        self.register_buffer("std", torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 1, 1, 3), persistent=False)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 6:
            batch, frames, views, channels, height, width = images.shape
            flat = images.reshape(batch * frames * views, channels, height, width)
        elif images.ndim == 5:
            batch, frames, channels, height, width = images.shape
            views = 1
            flat = images.reshape(batch * frames, channels, height, width)
        else:
            raise ValueError("Teacher images must be [B,8,V,3,H,W] or [B,8,3,H,W]")
        # No resampling.  The future frames are encoded at the resolution they were
        # generated in, so every patch token summarises real pixels rather than an
        # upsampling filter's guess at pixels that were never rendered.  Positional
        # semantics still line up with the Student's larger grid because DINOv3's
        # RoPE normalises coordinates per axis (``make_coords_dinov3`` divides by H
        # and W separately and maps to [-1, 1]), so a token's rotation depends on
        # where it sits in the frame, not on how many tokens the frame was cut into.
        if (int(height), int(width)) != self.image_hw:
            raise ValueError(
                f"Teacher refuses to interpolate: got {int(height)}x{int(width)} frames but was "
                f"built for {self.image_hw[0]}x{self.image_hw[1]}. Pass the native resolution."
            )
        flat = pad_to_patch_multiple(flat)
        flat = (flat.permute(0, 2, 3, 1) - self.mean) / self.std
        flat = flat.permute(0, 3, 1, 2).contiguous()
        encoded = [self.backbone.forward_features(chunk) for chunk in flat.split(self.frame_chunk)]
        features = encoded[0] if len(encoded) == 1 else _concatenate_features(encoded)
        if isinstance(features, dict):
            cls_token = features.get("x_norm_clstoken")
            register_tokens = features.get("x_norm_regtokens")
            patch_tokens = features.get("x_norm_patchtokens")
            if cls_token is None or register_tokens is None or patch_tokens is None:
                raw = features.get("x")
                if raw is None:
                    raise ValueError("DINOv3 output lacks CLS/register/patch token families")
                cls_token = raw[:, :1]
                register_tokens = raw[:, 1 : self.num_prefix_tokens]
                patch_tokens = raw[:, self.num_prefix_tokens :]
        else:
            cls_token = features[:, :1]
            register_tokens = features[:, 1 : self.num_prefix_tokens]
            patch_tokens = features[:, self.num_prefix_tokens :]
        # Keep the target's spatial contract while making the global token
        # families contribute: their mean is broadcast into every patch.  The
        # DrivoR/DINOv2 path is separate and remains patch-only.
        special_mean = torch.cat((cls_token, register_tokens), dim=1).mean(dim=1, keepdim=True)
        spatial = patch_tokens + special_mean
        grid_h, grid_w = self.grid_hw
        if spatial.shape[1] != grid_h * grid_w:
            raise ValueError(f"Teacher patch tokens {spatial.shape[1]} do not match grid {grid_h}x{grid_w}")
        spatial = spatial.reshape(batch, frames, views, grid_h, grid_w, -1).mean(dim=2)
        return spatial


class FixedTeacher(nn.Module):
    def __init__(self, backbone: nn.Module, checkpoint: str | Path | None = None) -> None:
        super().__init__()
        if checkpoint is None:
            raise ValueError("a fixed DINOv3 Teacher checkpoint is required for real target preprocessing")
        self.backbone = backbone
        self.checkpoint = str(checkpoint)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.backbone(images)


@dataclass
class OrderedCompressorState:
    mean: torch.Tensor
    projection: torch.Tensor
    input_dim: int
    output_dim: int
    fit_split: str
    source_hash: str


def ordered_quadrant_vectors(teacher_grid: torch.Tensor) -> torch.Tensor:
    """Pool ``[B,8,H,W,D]`` into ``[B,4,4D]``: four time chunks of four quadrants.

    Quadrant borders come from integer division, so an odd grid is fine and no
    token is dropped -- at the future videos' native 29x52 the rows split 14/15
    and the columns 26/26.  Each quadrant is therefore a *fraction* of the frame,
    which is the same normalised-coordinate convention DINOv3's RoPE uses, and is
    what keeps the target's spatial layout meaningful when the Teacher's grid is
    coarser than the Student's.
    """

    if teacher_grid.ndim != 5:
        raise ValueError("teacher_grid must be [B,8,H,W,D]")
    frames, height, width = int(teacher_grid.shape[1]), int(teacher_grid.shape[2]), int(teacher_grid.shape[3])
    if frames != 8:
        raise ValueError("v6 target compression expects eight future frames")
    if height < 2 or width < 2:
        raise ValueError(f"a 2x2 quadrant split needs at least a 2x2 grid, got {height}x{width}")
    chunks = []
    for chunk in range(4):
        regions = []
        for frame_index in (2 * chunk, 2 * chunk + 1):
            frame = teacher_grid[:, frame_index]
            for row in range(2):
                for col in range(2):
                    region = frame[:, row * height // 2 : (row + 1) * height // 2, col * width // 2 : (col + 1) * width // 2]
                    regions.append(region.mean(dim=(1, 2)))
        chunks.append(torch.cat(regions, dim=-1))
    return torch.stack(chunks, dim=1)


class Ordered2x2Compressor:
    def __init__(self, output_dim: int = 256) -> None:
        self.output_dim = output_dim
        self.state: OrderedCompressorState | None = None

    def fit(self, train_features: torch.Tensor, source_hash: str, split: str = "train") -> None:
        if split != "train":
            raise ValueError("target compressor may only be fit on train split")
        if train_features.ndim < 2:
            raise ValueError("train features must have a feature dimension")
        flat = train_features.reshape(-1, train_features.shape[-1]).float()
        mean = flat.mean(dim=0)
        centered = flat - mean
        _, _, v = torch.linalg.svd(centered, full_matrices=False)
        output_dim = self.output_dim
        rank = min(output_dim, v.shape[0], v.shape[1])
        projection = torch.zeros(flat.shape[-1], output_dim, dtype=flat.dtype)
        projection[:, :rank] = v[:rank].t()
        self.state = OrderedCompressorState(mean, projection, flat.shape[-1], output_dim, split, source_hash)

    def transform(self, teacher_grid: torch.Tensor) -> torch.Tensor:
        return self.transform_ordered(ordered_quadrant_vectors(teacher_grid))

    def transform_ordered(self, ordered: torch.Tensor) -> torch.Tensor:
        """Project already quadrant-pooled vectors, so the patch grids need never be held.

        A single grid is large next to its pooled form -- [8,29,52,1024] float32 is
        50 MB at the future videos' native resolution and [8,42,72,1024] is 103 MB
        at the Student's, while the ordered vectors are 131 KB either way.  A
        full-rank fit needs 16 scenes x 4 candidates x 4 chunks = 256 rows, so
        reducing first is the difference between fitting and exhausting RAM.
        """

        if self.state is None:
            raise RuntimeError("fit compressor on train targets before transform")
        if ordered.ndim != 3:
            raise ValueError("ordered vectors must be [B,4,input_dim]")
        if ordered.shape[-1] != self.state.input_dim:
            raise ValueError("compressor input dimension does not match ordered 2x2 Teacher target")
        return (ordered - self.state.mean) @ self.state.projection

    def fingerprint(self) -> str:
        if self.state is None:
            raise RuntimeError("compressor is not fit")
        payload = self.state.mean.detach().cpu().numpy().tobytes() + self.state.projection.detach().cpu().numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()
