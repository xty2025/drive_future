"""Official DINOv3 backbones with frozen weights, Q/V LoRA and optional scene queries."""

import logging
import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torch.utils.checkpoint import checkpoint

from navsim.agents.drivoR.layers.image_encoder.grid_mask import GridMask


log = logging.getLogger(__name__)


def build_official_dinov3(model_name):
    # Load only the cloned official backbone package. DINOv2 continues to use
    # the installed timm version, without importing this dependency at runtime.
    source = Path(__file__).resolve().parents[3] / "third_party/dinov3"
    if not (source / "dinov3/hub/backbones.py").is_file():
        raise FileNotFoundError(f"Official DINOv3 source not found: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from dinov3.hub.backbones import dinov3_vits16, dinov3_vitl16

    constructors = {"dinov3_vits16": dinov3_vits16, "dinov3_vitl16": dinov3_vitl16}
    if model_name not in constructors:
        raise ValueError(f"Unsupported DINOv3 backbone: {model_name}")
    return constructors[model_name](pretrained=False)


def load_modelscope_dinov3(model, path):
    """Map the public HF-format ModelScope tensors to the official backbone.

    HF splits Q/K/V and omits the masked K bias. The official implementation
    fuses Q/K/V and carries deterministic K-bias masks and RoPE periods.
    All source tensors are consumed; strict loading checks native key/shape
    compatibility. No file-hash checks or remote code loading are involved.
    """
    source = load_file(str(path), device="cpu")
    used = set()

    def take(name):
        used.add(name)
        return source[name]

    state = {
        "cls_token": take("embeddings.cls_token"),
        "storage_tokens": take("embeddings.register_tokens"),
        "mask_token": take("embeddings.mask_token").reshape_as(model.mask_token),
        "patch_embed.proj.weight": take("embeddings.patch_embeddings.weight"),
        "patch_embed.proj.bias": take("embeddings.patch_embeddings.bias"),
        "norm.weight": take("norm.weight"),
        "norm.bias": take("norm.bias"),
        "rope_embed.periods": model.rope_embed.periods,
    }
    for index, block in enumerate(model.blocks):
        src, dst = f"layer.{index}.", f"blocks.{index}."
        for norm in ("norm1", "norm2"):
            for kind in ("weight", "bias"):
                state[dst + norm + "." + kind] = take(src + norm + "." + kind)
        for target, origin in (("attn.proj", "attention.o_proj"),
                               ("mlp.fc1", "mlp.up_proj"), ("mlp.fc2", "mlp.down_proj")):
            for kind in ("weight", "bias"):
                state[dst + target + "." + kind] = take(src + origin + "." + kind)
        state[dst + "attn.qkv.weight"] = torch.cat(
            [take(src + f"attention.{part}_proj.weight") for part in ("q", "k", "v")], dim=0)
        q_bias = take(src + "attention.q_proj.bias")
        state[dst + "attn.qkv.bias"] = torch.cat(
            (q_bias, torch.zeros_like(q_bias), take(src + "attention.v_proj.bias")))
        state[dst + "attn.qkv.bias_mask"] = block.attn.qkv.bias_mask
        state[dst + "ls1.gamma"] = take(src + "layer_scale1.lambda1")
        state[dst + "ls2.gamma"] = take(src + "layer_scale2.lambda1")
    unused = set(source) - used
    if unused:
        raise ValueError(f"Unexpected DINOv3 source tensors: {sorted(unused)}")
    model.load_state_dict(state, strict=True, assign=True)
    return {"path": str(Path(path).resolve()), "source_tensors": len(source),
            "consumed_tensors": len(used), "native_state_tensors": len(state),
            "source_parameters": sum(value.numel() for value in source.values()),
            "generated_buffers": "fixed RoPE periods and K-bias masks; masked K bias is zero"}


class DinoV3LoRAQV(nn.Module):
    """The same unscaled Q/V low-rank update as DrivoR; preserve native K masking."""

    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features
        self.q_a = nn.Linear(self.in_features, rank, bias=False)
        self.q_b = nn.Linear(rank, self.in_features, bias=False)
        self.v_a = nn.Linear(self.in_features, rank, bias=False)
        self.v_b = nn.Linear(rank, self.in_features, bias=False)
        for layer in (self.q_a, self.v_a):
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        for layer in (self.q_b, self.v_b):
            nn.init.zeros_(layer.weight)

    def forward(self, x):
        q, k, v = self.base(x).chunk(3, dim=-1)
        return torch.cat((q + self.q_b(self.q_a(x)), k, v + self.v_b(self.v_a(x))), dim=-1)


class DinoV3ImageEncoder(nn.Module):
    def __init__(self, config, image_size, output_dim, num_cameras, num_scene_tokens,
                 visual_feature_mode="native_patches"):
        super().__init__()
        if visual_feature_mode not in ("native_patches", "drivor_queries"):
            raise ValueError(f"Unknown visual_feature_mode: {visual_feature_mode}")
        self.use_scene_queries = visual_feature_mode == "drivor_queries"
        self.num_scene_tokens = int(num_scene_tokens) if self.use_scene_queries else 0
        if self.use_scene_queries and self.num_scene_tokens <= 0:
            raise ValueError("drivor_queries requires num_scene_tokens > 0")
        weight_file = Path(config.model_weights)
        if not weight_file.is_file():
            raise FileNotFoundError(f"DINOv3 local weights not found: {weight_file}")
        self.model = build_official_dinov3(str(config.model_name))
        self.load_report = load_modelscope_dinov3(self.model, weight_file)
        self.num_features = self.model.num_features
        self.num_prefix_tokens = 1 + self.model.n_storage_tokens
        self.patch_size = tuple(self.model.patch_embed.patch_size)
        self.image_size = tuple(image_size)  # Existing config order: W,H.
        self.num_cameras = int(num_cameras)
        self.model.requires_grad_(False)
        self.use_lora = bool(config.use_lora) and int(config.lora_rank) > 0
        if self.use_lora:
            for block in self.model.blocks:
                block.attn.qkv = DinoV3LoRAQV(block.attn.qkv, int(config.lora_rank))
        if self.use_scene_queries:
            self.scene_embeds = nn.Parameter(torch.randn(
                1, self.num_cameras, self.num_scene_tokens, self.num_features) * 1e-6)
        else:
            self.register_parameter("scene_embeds", None)
        self.neck = nn.Linear(self.num_features, output_dim)
        self.gradient_checkpointing = bool(config.get("gradient_checkpointing", True))
        self.use_grid_mask = bool(config.get("use_grid_mask", True))
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.train(True)
        log.info("DINOv3 %s: loaded %s/%s source tensors; dim=%s; visual_mode=%s; Q/V LoRA rank=%s",
                 config.model_name, self.load_report["consumed_tensors"], self.load_report["source_tensors"],
                 self.num_features, visual_feature_mode, config.lora_rank if self.use_lora else 0)

    @property
    def native_vit(self):
        return self.model

    def train(self, mode=True):
        super().train(mode)
        # Freeze base behaviour as well as parameters. In particular this keeps
        # DINOv3's pretraining-only random RoPE rescaling disabled. LoRA/queries
        # still receive gradients, and the outer GridMask keeps its train mode.
        self.model.eval()
        return self

    def forward(self, images):
        b, cameras, channels, height, width = images.shape
        if (width, height) != self.image_size or cameras != self.num_cameras:
            raise ValueError(f"Expected {self.num_cameras} images [W,H]={self.image_size}")
        flat = images.reshape(b * cameras, channels, height, width)
        if self.use_grid_mask:
            flat = self.grid_mask(flat)
        # The data builder already resizes to config.image_size; use it directly.
        tokens, (grid_h, grid_w) = self.model.prepare_tokens_with_masks(flat)
        if self.use_scene_queries:
            queries = self.scene_embeds.expand(b, -1, -1, -1).reshape(
                b * cameras, self.num_scene_tokens, self.num_features)
            tokens = torch.cat((queries.to(tokens.dtype), tokens), dim=1)
        # Native attention applies RoPE to the final H*W patches only, leaving
        # [optional scene queries | CLS | storage tokens] as prefix tokens.
        rope = self.model.rope_embed(H=grid_h, W=grid_w)
        for block in self.model.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, rope, use_reentrant=False)
            else:
                tokens = block(tokens, rope)
        tokens = self.model.norm(tokens)
        if self.use_scene_queries:
            visual = tokens[:, :self.num_scene_tokens]
        else:
            visual = tokens[:, self.num_prefix_tokens:]
        return self.neck(visual).reshape(b, cameras * visual.shape[1], -1)
