"""Fixed Teacher + ordered future Register target compression."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


class TimmDinoV3Teacher(nn.Module):
    """Frozen raw DINOv3 patch-token Teacher used only during target construction."""

    def __init__(self, checkpoint: str | Path, model_name: str = "vit_large_patch16_dinov3.lvd1689m", image_size: int = 224) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(model_name, pretrained=False, checkpoint_path=str(checkpoint), num_classes=0)
        self.image_size = int(image_size)
        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
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
        flat = torch.nn.functional.interpolate(flat, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        flat = (flat.permute(0, 2, 3, 1) - self.mean) / self.std
        features = self.backbone.forward_features(flat.permute(0, 3, 1, 2).contiguous())
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
        side = int(spatial.shape[1] ** 0.5)
        if side * side != spatial.shape[1]:
            raise ValueError(f"Teacher patch tokens are not square: {spatial.shape[1]}")
        spatial = spatial.reshape(batch, frames, views, side, side, -1).mean(dim=2)
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
        if self.state is None:
            raise RuntimeError("fit compressor on train targets before transform")
        if teacher_grid.ndim != 5:
            raise ValueError("teacher_grid must be [B,8,H,W,D]")
        b, frames, height, width, dim = teacher_grid.shape
        if frames != 8:
            raise ValueError("v6 target compression expects eight future frames")
        if height % 2 or width % 2:
            raise ValueError("ordered 2x2 compression requires even spatial dimensions")
        chunks = []
        for chunk in range(4):
            regions = []
            for frame_index in (2 * chunk, 2 * chunk + 1):
                frame = teacher_grid[:, frame_index]
                for row in range(2):
                    for col in range(2):
                        region = frame[:, row * height // 2 : (row + 1) * height // 2, col * width // 2 : (col + 1) * width // 2]
                        regions.append(region.mean(dim=(1, 2)))
            ordered = torch.cat(regions, dim=-1)
            if ordered.shape[-1] != self.state.input_dim:
                raise ValueError("compressor input dimension does not match ordered 2x2 Teacher target")
            chunks.append((ordered - self.state.mean) @ self.state.projection)
        return torch.stack(chunks, dim=1)

    def fingerprint(self) -> str:
        if self.state is None:
            raise RuntimeError("compressor is not fit")
        payload = self.state.mean.detach().cpu().numpy().tobytes() + self.state.projection.detach().cpu().numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()
