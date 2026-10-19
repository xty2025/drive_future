"""Native patches or the original DrivoR per-camera Scene-query features."""

from copy import deepcopy
from pathlib import Path

import timm
import torch
from torch import nn

from navsim.agents.drivoR.layers.image_encoder.dinov2_lora import ImgEncoder, LoRA_ViT_timm
from navsim.agents.drivoR.layers.image_encoder.grid_mask import GridMask


class NativePatchImageEncoder(nn.Module):
    def __init__(self, config, image_size, output_dim=256):
        super().__init__()
        weight_file = Path(config.model_weights)
        if not weight_file.is_file():
            raise FileNotFoundError(f"DINOv2 local weights not found: {weight_file}")
        model_name = str(config.model_name).removeprefix("timm/")
        vit = timm.create_model(model_name, pretrained=True,
                                pretrained_cfg_overlay={"file": str(weight_file)},
                                img_size=(image_size[1], image_size[0]), num_classes=0)
        self.num_prefix_tokens = vit.num_prefix_tokens
        self.num_features = vit.num_features
        self.patch_size = vit.patch_embed.patch_size
        self.image_size = tuple(image_size)
        self.use_lora = bool(config.use_lora)
        if self.use_lora:
            # Reuse the original Q/V LoRA surgery, but never replace native ViT
            # methods with the DrivoR Scene-query injection implementation.
            self.model = LoRA_ViT_timm(vit, r=config.lora_rank)
        else:
            vit.requires_grad_(bool(config.finetune))
            self.model = vit
        vit.set_grad_checkpointing(bool(config.get("gradient_checkpointing", True)))
        self.neck = nn.Linear(vit.num_features, output_dim)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = bool(config.get("use_grid_mask", True))

    @property
    def native_vit(self):
        return self.model.lora_vit if self.use_lora else self.model

    def forward(self, images):
        b, cameras, channels, height, width = images.shape
        if (width, height) != self.image_size:
            raise ValueError(f"Expected image [W,H]={self.image_size}, got {(width, height)}")
        flat = images.reshape(b * cameras, channels, height, width)
        if self.use_grid_mask:
            flat = self.grid_mask(flat)
        # Native CLS + register tokens participate in every backbone attention
        # block. Only the final patch tokens become downstream visual K/V.
        tokens = self.native_vit.forward_features(flat)
        patches = tokens[:, self.num_prefix_tokens:]
        return self.neck(patches).reshape(b, cameras * patches.shape[1], -1)


class DrivoRSceneQueryImageEncoder(nn.Module):
    """Reuse DrivoR's query injection, prefix selection and shared neck exactly."""

    def __init__(self, config, image_size, output_dim, num_cameras, num_scene_tokens):
        super().__init__()
        if num_scene_tokens <= 0:
            raise ValueError("DrivoR Scene-query features require num_scene_tokens > 0")
        encoder_config = deepcopy(config)
        encoder_config.image_size = list(image_size)
        encoder_config.num_scene_tokens = int(num_scene_tokens)
        encoder_config.tf_d_model = int(output_dim)
        self.encoder = ImgEncoder(encoder_config)
        self.num_features = self.encoder.num_features
        # Same per-camera query shape and initialization as DrivoRModel.
        self.scene_embeds = nn.Parameter(
            torch.randn(1, num_cameras, num_scene_tokens, self.num_features) * 1e-6)
        self.encoder.use_grid_mask = bool(config.get("use_grid_mask", True))
        # Preserve the Action-DiT variant's activation-checkpointing setting.
        self.native_vit.set_grad_checkpointing(bool(config.get("gradient_checkpointing", True)))

    @property
    def native_vit(self):
        return self.encoder.model.lora_vit if self.encoder.use_lora else self.encoder.model

    def forward(self, images):
        queries = self.scene_embeds.repeat(images.shape[0], 1, 1, 1)
        return self.encoder(images, queries)
