"""V-JEPA 2 ViT-L as a frozen visual backbone with Q/V LoRA and optional queries."""

import importlib
import logging
import sys
from functools import lru_cache
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from navsim.agents.drivoR.layers.image_encoder.grid_mask import GridMask
from .dinov3_image_encoder import DinoV3LoRAQV as LoRAQV


log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def official_vjepa2_modules():
    source = Path(__file__).resolve().parents[3] / 'third_party/vjepa2'
    if not (source / 'src/models/vision_transformer.py').is_file():
        raise FileNotFoundError(f'Official V-JEPA 2 source not found: {source}')
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return (importlib.import_module('src.models.vision_transformer'),
            importlib.import_module('src.models.utils.modules'))


def build_official_vjepa2(model_name):
    if model_name != 'vjepa2_vit_large':
        raise ValueError(f'Unsupported V-JEPA 2 backbone: {model_name}')
    vit, _ = official_vjepa2_modules()
    # Same encoder settings as vjepa2_vit_large in official src/hub/backbones.py.
    # The predictor is deliberately never instantiated.
    return vit.vit_large(img_size=(256, 256), patch_size=16, num_frames=64,
                         tubelet_size=2, use_sdpa=True, use_silu=False,
                         wide_silu=True, uniform_power=False, use_rope=True,
                         handle_nonsquare_inputs=True, use_activation_checkpointing=False)


def query_attention(attention, x, query_count, temporal, height, width):
    """Preserve native patch RoPE while keeping added Scene queries unrotated."""
    _, native = official_vjepa2_modules()
    b, length, dim = x.shape
    qkv = attention.qkv(x).unflatten(-1, (3, attention.num_heads, -1)).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    positions = torch.arange(temporal * height * width, device=x.device)
    axes = attention.separate_positions(positions, height, width)
    q_patch, k_patch = q[:, :, query_count:], k[:, :, query_count:]
    q_parts, k_parts, offset = [], [], 0
    for axis_dim, axis in zip((attention.d_dim, attention.h_dim, attention.w_dim), axes):
        # Reuse the upstream rotation, including its pretrained frequency layout.
        q_parts.append(native.rotate_queries_or_keys(q_patch[..., offset:offset + axis_dim], axis))
        k_parts.append(native.rotate_queries_or_keys(k_patch[..., offset:offset + axis_dim], axis))
        offset += axis_dim
    if offset < attention.head_dim:
        q_parts.append(q_patch[..., offset:])
        k_parts.append(k_patch[..., offset:])
    q = torch.cat((q[:, :, :query_count], torch.cat(q_parts, -1)), dim=-2)
    k = torch.cat((k[:, :, :query_count], torch.cat(k_parts, -1)), dim=-2)
    with torch.backends.cuda.sdp_kernel():
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=attention.proj_drop_prob,
                                             is_causal=attention.is_causal)
    out = out.transpose(1, 2).reshape(b, length, dim)
    return attention.proj_drop(attention.proj(out))


def query_block(block, x, query_count, temporal, height, width):
    y = query_attention(block.attn, block.norm1(x), query_count, temporal, height, width)
    x = x + block.drop_path(y)
    return x + block.drop_path(block.mlp(block.norm2(x)))


class VJepa2ImageEncoder(nn.Module):
    def __init__(self, config, image_size, output_dim, num_cameras, num_scene_tokens,
                 visual_feature_mode='native_patches'):
        super().__init__()
        if visual_feature_mode not in ('native_patches', 'drivor_queries'):
            raise ValueError(f'Unknown visual_feature_mode: {visual_feature_mode}')
        self.use_scene_queries = visual_feature_mode == 'drivor_queries'
        self.num_scene_tokens = int(num_scene_tokens) if self.use_scene_queries else 0
        if self.use_scene_queries and self.num_scene_tokens <= 0:
            raise ValueError('drivor_queries requires num_scene_tokens > 0')
        weights = Path(config.model_weights)
        if not weights.is_file():
            raise FileNotFoundError(f'V-JEPA 2 encoder weights not found: {weights}')
        self.model = build_official_vjepa2(str(config.model_name))
        state = load_file(str(weights), device='cpu')
        self.model.load_state_dict(state, strict=True, assign=True)
        self.load_report = {'path': str(weights.resolve()), 'loaded_tensors': len(state),
                            'source_key': 'target_encoder', 'parameters': sum(p.numel() for p in state.values())}
        self.model.requires_grad_(False)
        self.num_features = self.model.embed_dim
        self.num_prefix_tokens = 0  # V-JEPA 2 has no native CLS/register tokens.
        self.patch_size = (self.model.patch_size, self.model.patch_size)
        self.tubelet_size = int(self.model.tubelet_size)
        self.image_size = tuple(image_size)
        self.num_cameras = int(num_cameras)
        self.use_lora = bool(config.use_lora) and int(config.lora_rank) > 0
        if self.use_lora:
            for block in self.model.blocks:
                block.attn.qkv = LoRAQV(block.attn.qkv, int(config.lora_rank))
        if self.use_scene_queries:
            self.scene_embeds = nn.Parameter(torch.randn(
                1, self.num_cameras, self.num_scene_tokens, self.num_features) * 1e-6)
        else:
            self.register_parameter('scene_embeds', None)
        self.neck = nn.Linear(self.num_features, output_dim)
        self.gradient_checkpointing = bool(config.get('gradient_checkpointing', True))
        self.use_grid_mask = bool(config.get('use_grid_mask', True))
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.train(True)
        log.info('V-JEPA 2 ViT-L: loaded %s encoder tensors; current image repeated %s times; visual_mode=%s; Q/V LoRA rank=%s',
                 len(state), self.tubelet_size, visual_feature_mode, config.lora_rank if self.use_lora else 0)

    @property
    def native_vit(self):
        return self.model

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, images):
        b, cameras, channels, height, width = images.shape
        if (width, height) != self.image_size or cameras != self.num_cameras:
            raise ValueError(f'Expected {self.num_cameras} images [W,H]={self.image_size}')
        flat = images.reshape(b * cameras, channels, height, width)
        if self.use_grid_mask:
            flat = self.grid_mask(flat)
        # Official image-evaluation adaptation: duplicate the SAME current image
        # into one temporal tubelet. Cameras stay separate; no history is read.
        video = flat.unsqueeze(2).repeat(1, 1, self.tubelet_size, 1, 1)
        tokens = self.model.patch_embed(video)
        grid_h, grid_w = height // self.patch_size[0], width // self.patch_size[1]
        if self.use_scene_queries:
            queries = self.scene_embeds.expand(b, -1, -1, -1).reshape(
                b * cameras, self.num_scene_tokens, self.num_features)
            tokens = torch.cat((queries.to(tokens.dtype), tokens), dim=1)
        for block in self.model.blocks:
            if self.use_scene_queries:
                fn, args, kwargs = query_block, (block, tokens, self.num_scene_tokens, 1, grid_h, grid_w), {}
            else:
                fn, args = block, (tokens,)
                kwargs = dict(T=1, H_patches=grid_h, W_patches=grid_w)
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(fn, *args, **kwargs, use_reentrant=False)
            else:
                tokens = fn(*args, **kwargs)
        tokens = self.model.norm(tokens)
        if self.use_scene_queries:
            tokens = tokens[:, :self.num_scene_tokens]
        return self.neck(tokens).reshape(b, cameras * tokens.shape[1], -1)
