"""NAVSIM v6-wan variant: same directed Scene/Action/Register generator, but the
Action DiT blocks follow the Wan2.2 ``DiTBlock`` design (self-attn with QK
RMSNorm + RoPE, cross-attn over ``[Scene | stopgrad(Register)]``, six-parameter
adaLN modulation, GELU(tanh) FFN, modulated output Head).  All Wan-style modules
are randomly initialized; no Wan checkpoint is ever loaded.  The v6 contracts
(token order, directed reads, A0 feature addition, single-step residual-velocity
endpoint readout) are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .data import Observation, decode_trajectory
from .dino_tokens import split_dinov3_tokens
from .lora import LoRALinear, attach_lora, freeze_base_keep_lora

GROUP_ALLOW = ((1, 0, 0), (1, 1, 1), (1, 0, 1))


def build_group_mask(ns: int, na: int, nr: int, device: torch.device | None = None) -> torch.Tensor:
    sizes = (ns, na, nr)
    mask = torch.zeros(sum(sizes), sum(sizes), dtype=torch.bool, device=device)
    row_start = 0
    for row, row_size in enumerate(sizes):
        col_start = 0
        for col, col_size in enumerate(sizes):
            if GROUP_ALLOW[row][col]:
                mask[row_start : row_start + row_size, col_start : col_start + col_size] = True
            col_start += col_size
        row_start += row_size
    return mask


@dataclass
class PairOutputs:
    scene_context: torch.Tensor
    endpoint: torch.Tensor
    trajectory: torch.Tensor
    predicted_register: torch.Tensor


class SpatialStudent(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, d_model, kernel_size=4, stride=4)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        for parameter in self.patch.parameters():
            parameter.requires_grad_(False)
        for parameter in self.qkv.parameters():
            parameter.requires_grad_(False)
        for parameter in self.proj.parameters():
            parameter.requires_grad_(False)
        for parameter in self.norm.parameters():
            parameter.requires_grad_(False)
        self.qkv = LoRALinear(self.qkv, rank=min(16, d_model), alpha=min(16, d_model))
        self.proj = LoRALinear(self.proj, rank=min(16, d_model), alpha=min(16, d_model))

    def forward(self, images: torch.Tensor, output_tokens: int) -> torch.Tensor:
        b, t, v, c, h, w = images.shape
        images = images.reshape(b * t * v, c, h, w)
        tokens = self.patch(images).flatten(2).transpose(1, 2)
        if tokens.shape[1] < output_tokens:
            tokens = torch.nn.functional.pad(tokens, (0, 0, 0, output_tokens - tokens.shape[1]))
        tokens = tokens[:, :output_tokens]
        qkv = self.qkv(tokens)
        q, k, value = qkv.chunk(3, dim=-1)
        attention = torch.softmax(q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5), dim=-1)
        tokens = self.norm(tokens + self.proj(attention @ value))
        return tokens.reshape(b, t, v, output_tokens, -1).mean(dim=2)


class TimmDinoV3Student(nn.Module):
    def __init__(
        self,
        checkpoint: str,
        model_name: str = "vit_large_patch16_dinov3.lvd1689m",
        output_dim: int = 256,
        image_size: int = 224,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        max_output_tokens: int = 32,
        use_all_native_tokens: bool = False,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise ImportError("timm is required for the real DINOv3 student") from error
        if "dinov3" not in model_name:
            raise ValueError("the NVS student token adapter only accepts DINOv3 backbones")
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            checkpoint_path=checkpoint,
            num_classes=0,
        )
        self.backbone_dim = int(getattr(self.backbone, "num_features", 1024))
        self.image_size = int(image_size)
        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
        if self.num_prefix_tokens < 1:
            raise ValueError("DINOv3 backbone must expose a CLS token")
        default_cfg = getattr(self.backbone, "default_cfg", {})
        mean = default_cfg.get("mean", (0.485, 0.456, 0.406))
        std = default_cfg.get("std", (0.229, 0.224, 0.225))
        self.register_buffer("mean", torch.tensor(mean).view(1, 1, 1, 3), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 1, 1, 3), persistent=False)
        attached = attach_lora(
            self.backbone,
            names=("qkv", "proj"),
            rank=lora_rank,
            alpha=lora_alpha,
        )
        if not attached:
            raise RuntimeError("DINOv3 backbone has no qkv/proj Linear modules for LoRA attachment")
        freeze_base_keep_lora(self.backbone)
        # DrivoR prepends trainable scene queries to the ViT sequence.  NVS
        # keeps DINOv3's native CLS/register tokens too, so only the remaining
        # output slots are injected as queries.  Image patches stay in every
        # block's Q/K/V and are summarized into these tokens.
        scene_query_init = (
            torch.randn(max_output_tokens - self.num_prefix_tokens, self.backbone_dim) * 1e-6
        )
        if use_all_native_tokens:
            # All-native mode emits every DINOv3 token and never injects scene
            # queries.  Keep the tensor as a non-trainable buffer so it stays out
            # of parameters()/the optimizer and DDP does not expect a gradient
            # for it (which would otherwise force find_unused_parameters=True).
            self.register_buffer("scene_queries", scene_query_init, persistent=False)
        else:
            self.scene_queries = nn.Parameter(scene_query_init)
        self.output_projection = nn.Linear(self.backbone_dim, output_dim)
        self.output_projection._v6_role = "eta"  # type: ignore[attr-defined]
        self.train(True)

    def train(self, mode: bool = True) -> "TimmDinoV3Student":
        super().train(mode)
        self.backbone.eval()
        return self

    def token_layout(self, output_tokens: int) -> dict[str, int]:
        if output_tokens <= self.num_prefix_tokens:
            raise ValueError("scene output must retain CLS, all registers, and at least one scene query")
        patch_size = getattr(self.backbone.patch_embed, "patch_size", None)
        if patch_size is None:
            raise ValueError("DINOv3 patch embedding does not expose patch_size")
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if self.image_size % int(patch_size[0]) or self.image_size % int(patch_size[1]):
            raise ValueError("configured image size must be divisible by the DINOv3 patch size")
        patch_tokens = (self.image_size // int(patch_size[0])) * (self.image_size // int(patch_size[1]))
        scene_queries = output_tokens - self.num_prefix_tokens
        return {
            "cls": 1,
            "registers": self.num_prefix_tokens - 1,
            "scene_queries": scene_queries,
            "image_patches": patch_tokens,
            "native_tokens": self.num_prefix_tokens + patch_tokens,
            "backbone_qkv_tokens": self.num_prefix_tokens + scene_queries + patch_tokens,
            "output_tokens": output_tokens,
        }

    def _forward_features_with_scene_queries(
        self, images: torch.Tensor, scene_query_count: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        backbone = self.backbone
        tokens = backbone.patch_embed(images)
        positional = backbone._pos_embed(tokens)
        if isinstance(positional, tuple):
            tokens, rope = positional
        else:
            tokens, rope = positional, None
        queries = self.scene_queries[:scene_query_count].to(tokens.dtype)
        queries = queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        # [native CLS/register | injected scene queries | image patches].
        tokens = torch.cat(
            (
                tokens[:, : self.num_prefix_tokens],
                queries,
                tokens[:, self.num_prefix_tokens :],
            ),
            dim=1,
        )
        tokens = backbone.norm_pre(tokens)
        # EVA/DINOv3 applies RoPE only after ``num_prefix_tokens``.  Our scene
        # queries are additional prefix-style tokens, so temporarily extend
        # that boundary while retaining RoPE for every image patch.
        attention_prefixes: list[tuple[nn.Module, int]] = []
        for block in backbone.blocks:
            attention = getattr(block, "attn", None)
            if attention is not None and hasattr(attention, "num_prefix_tokens"):
                attention_prefixes.append((attention, int(attention.num_prefix_tokens)))
                attention.num_prefix_tokens = self.num_prefix_tokens + scene_query_count
        try:
            if getattr(backbone, "rope_mixed", False) and rope is not None:
                for index, block in enumerate(backbone.blocks):
                    tokens = block(tokens, rope=rope[index])
            else:
                for block in backbone.blocks:
                    try:
                        tokens = block(tokens, rope=rope)
                    except TypeError:
                        tokens = block(tokens)
        finally:
            for attention, prefix_count in attention_prefixes:
                attention.num_prefix_tokens = prefix_count
        tokens = backbone.norm(tokens)
        cls = tokens[:, :1]
        registers = tokens[:, 1 : self.num_prefix_tokens]
        scene_queries = tokens[
            :, self.num_prefix_tokens : self.num_prefix_tokens + scene_query_count
        ]
        # Validate the special-token contract separately from the injected
        # queries.  This also prevents accidentally treating DINOv2 patch-only
        # output as DINOv3.
        patch_start = self.num_prefix_tokens + scene_query_count
        split_dinov3_tokens(
            torch.cat((cls, registers, tokens[:, patch_start:]), dim=1),
            self.num_prefix_tokens,
        )
        return cls, registers, scene_queries

    def _forward_all_native(self, images: torch.Tensor) -> torch.Tensor:
        """Run backbone without query injection; return all native tokens (CLS+reg+patches)."""
        backbone = self.backbone
        tokens = backbone.patch_embed(images)
        positional = backbone._pos_embed(tokens)
        if isinstance(positional, tuple):
            tokens, rope = positional
        else:
            tokens, rope = positional, None
        tokens = backbone.norm_pre(tokens)
        if getattr(backbone, "rope_mixed", False) and rope is not None:
            for index, block in enumerate(backbone.blocks):
                tokens = block(tokens, rope=rope[index])
        else:
            for block in backbone.blocks:
                try:
                    tokens = block(tokens, rope=rope)
                except TypeError:
                    tokens = block(tokens)
        tokens = backbone.norm(tokens)
        split_dinov3_tokens(tokens, self.num_prefix_tokens)
        return tokens

    def forward(self, images: torch.Tensor, output_tokens: int | None = None) -> torch.Tensor:
        b, t, v, c, _, _ = images.shape
        flat = images.reshape(b * t * v, c, images.shape[-2], images.shape[-1])
        flat = torch.nn.functional.interpolate(flat, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        flat = flat.permute(0, 2, 3, 1)
        flat = (flat - self.mean) / self.std
        flat = flat.permute(0, 3, 1, 2).contiguous()
        if output_tokens is None:
            tokens = self._forward_all_native(flat)
        else:
            layout = self.token_layout(output_tokens)
            cls_token, register_tokens, scene_queries = self._forward_features_with_scene_queries(
                flat, layout["scene_queries"]
            )
            tokens = torch.cat((cls_token, register_tokens, scene_queries), dim=1)
        tokens = self.output_projection(tokens)
        tokens = tokens.reshape(b, t, v, tokens.shape[1], -1)
        return tokens.mean(dim=2)


class SceneEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        current_tokens: int = 32,
        history_tokens: int = 8,
        ego_dim: int = 8,
        dino_checkpoint: str | None = None,
        dino_model_name: str = "vit_large_patch16_dinov3.lvd1689m",
        dino_image_size: int = 224,
        dino_lora_rank: int = 16,
        dino_lora_alpha: float = 16.0,
        dino_lora_dropout: float = 0.0,
        use_all_native_tokens: bool = False,
    ) -> None:
        super().__init__()
        self.current_tokens = current_tokens
        self.history_tokens = history_tokens
        self.use_all_native_tokens = use_all_native_tokens
        self.student = (
            TimmDinoV3Student(
                checkpoint=dino_checkpoint,
                model_name=dino_model_name,
                output_dim=d_model,
                image_size=dino_image_size,
                lora_rank=dino_lora_rank,
                lora_alpha=dino_lora_alpha,
                lora_dropout=dino_lora_dropout,
                use_all_native_tokens=use_all_native_tokens,
            )
            if dino_checkpoint
            else SpatialStudent(d_model)
        )
        if use_all_native_tokens and not isinstance(self.student, TimmDinoV3Student):
            raise ValueError("use_all_native_tokens requires a real DINOv3 checkpoint")
        self.time_embedding = nn.Parameter(torch.zeros(5, d_model))
        self.ego_mlp = nn.Sequential(nn.Linear(ego_dim + 4, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.ego_query = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.ego_attention = nn.MultiheadAttention(d_model, num_heads=8 if d_model % 8 == 0 else 1, batch_first=True)

    def encode_clean_ego(self, observation: Observation) -> torch.Tensor:
        """Encode DrivoR-ordered current ego state and route command once.

        The observable vector is ``pose3, velocity2, acceleration2, command4``.
        A trailing zero keeps the existing 12-wide NVS checkpoint interface.
        """
        observation.validate()
        current = observation.ego_history[:, -1]
        ego_features = torch.cat((current[:, :7], observation.route_command, current[:, 7:]), dim=-1)
        return self.ego_mlp(ego_features)

    def forward(
        self,
        observation: Observation,
        clean_ego: torch.Tensor | None = None,
    ) -> torch.Tensor:
        observation.validate()
        if self.use_all_native_tokens:
            all_frames = self.student(observation.rgb_history)
            all_frames = all_frames + self.time_embedding.view(1, 5, 1, -1)
            visual = all_frames.flatten(1, 2)
        else:
            visual_current = self.student(observation.rgb_history[:, -1:], self.current_tokens)[:, 0]
            visual_history = self.student(observation.rgb_history[:, :-1], self.history_tokens)
            visual_history = visual_history + self.time_embedding[:-1].view(1, 4, 1, -1)
            visual_current = visual_current + self.time_embedding[-1].view(1, 1, -1)
            visual = torch.cat((visual_current, visual_history.flatten(1, 2)), dim=1)
        if clean_ego is None:
            clean_ego = self.encode_clean_ego(observation)
        if clean_ego.ndim != 2 or clean_ego.shape != (visual.shape[0], visual.shape[-1]):
            raise ValueError("clean ego encoding must be [B,d_model]")
        queries = self.ego_query.unsqueeze(0).expand(visual.shape[0], -1, -1) + clean_ego.unsqueeze(1)
        fused, _ = self.ego_attention(queries, visual, visual)
        return torch.cat((visual, fused), dim=1)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class SceneBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim)

    def forward(self, scene: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_attn(scene)
        scene = scene + self.attn(normalized, normalized, normalized, need_weights=False)[0]
        return scene + self.ffn(self.norm_ffn(scene))


class RegisterBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim)

    def forward(self, register: torch.Tensor, scene: torch.Tensor) -> torch.Tensor:
        b, m, nr, d = register.shape
        scene = scene.unsqueeze(1).expand(-1, m, -1, -1).reshape(b * m, scene.shape[1], d)
        register = register.reshape(b * m, nr, d)
        query = self.norm_attn(register)
        key_value = torch.cat((scene, register), dim=1)
        register = register + self.attn(query, key_value, key_value, need_weights=False)[0]
        register = register + self.ffn(self.norm_ffn(register))
        return register.reshape(b, m, nr, d)


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        normalized = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(dtype) * self.weight


def wan_modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def build_rope_tables(head_dim: int, max_seq: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2:
        raise ValueError("Wan RoPE requires an even head dim")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(max_seq, dtype=torch.float32), inv_freq)
    return angles.cos(), angles.sin()


class WanSelfAttention(nn.Module):
    """Wan2.2-style self-attention: per-head QK RMSNorm + 1D RoPE over the
    action-token time axis, then SDPA.  Runs per candidate on [B*M, na, d]."""

    def __init__(self, hidden_dim: int, num_heads: int, rope_positions: int, eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        if self.head_dim % 2:
            raise ValueError("head dim must be even for RoPE")
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.o = nn.Linear(hidden_dim, hidden_dim)
        self.norm_q = WanRMSNorm(self.head_dim, eps=eps)
        self.norm_k = WanRMSNorm(self.head_dim, eps=eps)
        rope_cos, rope_sin = build_rope_tables(self.head_dim, rope_positions)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B,S,H,head_dim]; tables are [S,1,head_dim/2] to broadcast over heads.
        cos = self.rope_cos[: x.shape[1]].unsqueeze(1).to(x.dtype)
        sin = self.rope_sin[: x.shape[1]].unsqueeze(1).to(x.dtype)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        if s > self.rope_cos.shape[0]:
            raise ValueError("sequence longer than precomputed RoPE table")
        q = self.norm_q(self.q(x).view(b, s, self.num_heads, self.head_dim))
        k = self.norm_k(self.k(x).view(b, s, self.num_heads, self.head_dim))
        v = self.v(x).view(b, s, self.num_heads, self.head_dim)
        q = self._apply_rope(q).transpose(1, 2)
        k = self._apply_rope(k).transpose(1, 2)
        v = v.transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.o(out.transpose(1, 2).reshape(b, s, -1))


class WanCrossAttention(nn.Module):
    """Wan2.2-style cross-attention (no RoPE): Action queries read the directed
    context ``[Scene | stopgrad(Register)]``."""

    def __init__(self, hidden_dim: int, num_heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.o = nn.Linear(hidden_dim, hidden_dim)
        self.norm_q = WanRMSNorm(self.head_dim, eps=eps)
        self.norm_k = WanRMSNorm(self.head_dim, eps=eps)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        length = context.shape[1]
        q = self.norm_q(self.q(x).view(b, s, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.norm_k(self.k(context).view(b, length, self.num_heads, self.head_dim)).transpose(1, 2)
        v = self.v(context).view(b, length, self.num_heads, self.head_dim).transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.o(out.transpose(1, 2).reshape(b, s, -1))


class WanActionBlock(nn.Module):
    """Wan2.2 ``DiTBlock`` adapted to the directed NVS Action reads.

    Per candidate m: self-attn keeps ``A<-A``; cross-attn reads
    ``[S | sg(R)]``; together they preserve ``A<-[S,A,sg(R)]`` in Wan's
    self+cross form instead of one shared softmax.  ``t_mod`` carries the
    adaLN conditioning (mode code + projected clean ego) in place of Wan's
    diffusion timestep; NVS runs a single forward with no sampling loops.
    """

    def __init__(self, d_model: int, heads: int, ffn_dim: int, action_tokens: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.self_attn = WanSelfAttention(d_model, heads, rope_positions=action_tokens, eps=eps)
        self.cross_attn = WanCrossAttention(d_model, heads, eps=eps)
        self.norm1 = nn.LayerNorm(d_model, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(d_model, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, d_model)
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, d_model) / d_model**0.5)

    def forward(
        self,
        action: torch.Tensor,
        scene: torch.Tensor,
        register_memory: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> torch.Tensor:
        b, m, na, d = action.shape
        ns = scene.shape[1]
        action = action.reshape(b * m, na, d)
        scene = scene.unsqueeze(1).expand(b, m, ns, d).reshape(b * m, ns, d)
        context = torch.cat((scene, register_memory.reshape(b * m, register_memory.shape[2], d)), dim=1)
        if t_mod.shape != (b * m, 6, d):
            raise ValueError("t_mod must be [B*M,6,d]")
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation + t_mod).chunk(6, dim=1)
        action = action + gate_msa * self.self_attn(wan_modulate(self.norm1(action), shift_msa, scale_msa))
        action = action + self.cross_attn(self.norm3(action), context)
        action = action + gate_mlp * self.ffn(wan_modulate(self.norm2(action), shift_mlp, scale_mlp))
        return action.reshape(b, m, na, d)


class WanActionHead(nn.Module):
    """Wan2.2 ``Head`` for the single-step residual endpoint readout."""

    def __init__(self, d_model: int, out_dim: int = 4, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(d_model, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, d_model) / d_model**0.5)

    def forward(self, action: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, m, na, d = action.shape
        action = action.reshape(b * m, na, d)
        t = cond.reshape(b * m, d).unsqueeze(1)
        shift, scale = (self.modulation + t).chunk(2, dim=1)
        return self.head(self.norm(action) * (1 + scale) + shift).reshape(b, m, na, -1)


class DirectedPairDiT(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        heads: int = 8,
        ffn_dim: int = 1024,
        depth: int = 8,
        action_tokens: int = 8,
        register_tokens: int = 4,
        scene_tokens: int = 66,
        xy_scale: float = 20.0,
        dino_checkpoint: str | None = None,
        dino_model_name: str = "vit_large_patch16_dinov3.lvd1689m",
        dino_image_size: int = 224,
        dino_lora_rank: int = 16,
        dino_lora_alpha: float = 16.0,
        dino_lora_dropout: float = 0.0,
        ego_dim: int = 8,
        use_all_native_tokens: bool = False,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.d_model = d_model
        self.action_tokens = action_tokens
        self.register_tokens = register_tokens
        self.xy_scale = xy_scale
        self.use_all_native_tokens = use_all_native_tokens
        if use_all_native_tokens:
            scene_tokens = 5 * 201 + 2
        self.scene_tokens = scene_tokens
        self.eta = SceneEncoder(
            d_model,
            ego_dim=ego_dim,
            dino_checkpoint=dino_checkpoint,
            dino_model_name=dino_model_name,
            dino_image_size=dino_image_size,
            dino_lora_rank=dino_lora_rank,
            dino_lora_alpha=dino_lora_alpha,
            dino_lora_dropout=dino_lora_dropout,
            use_all_native_tokens=use_all_native_tokens,
        )
        self.phi = nn.ModuleDict(
            {
                "mode_projection": nn.Linear(2, d_model),
                "blocks": nn.ModuleList([RegisterBlock(d_model, heads, ffn_dim) for _ in range(depth)]),
            }
        )
        self.phi_time_queries = nn.Parameter(torch.randn(register_tokens, d_model) * 0.02)
        self.theta = nn.ModuleDict(
            {
                "source_projection": nn.Linear(4, d_model),
                "mode_projection": nn.Linear(2, d_model),
                # Action owns this projection, while eta.ego_mlp is the shared
                # clean observable encoder also used by Scene conditioning.
                "ego_to_action_projection": nn.Linear(d_model, d_model),
                # Wan time_projection analogue: mode + projected clean ego play
                # the role of Wan's diffusion timestep (NVS has no flow time).
                "cond_projection": nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model * 6)),
                "blocks": nn.ModuleList(
                    [WanActionBlock(d_model, heads, ffn_dim, action_tokens) for _ in range(depth)]
                ),
                "trajectory_head": WanActionHead(d_model),
            }
        )
        self.theta_time_embedding = nn.Parameter(torch.randn(action_tokens, d_model) * 0.02)
        self.scene_blocks = nn.ModuleList([SceneBlock(d_model, heads, ffn_dim) for _ in range(depth)])
        self.register_readout = nn.Identity()
        self.register_tokens = register_tokens
        self.depth = depth
        self.register_parameter("_unused_scene_token_anchor", nn.Parameter(torch.zeros(scene_tokens, d_model), requires_grad=False))

    def _initialize_register(self, modes: torch.Tensor) -> torch.Tensor:
        b, m, _ = modes.shape
        query = self.phi_time_queries.view(1, 1, self.register_tokens, self.d_model)
        mode = self.phi["mode_projection"](modes).unsqueeze(2)
        return query.expand(b, m, -1, -1) + mode

    def _initialize_action(
        self,
        source: torch.Tensor,
        modes: torch.Tensor,
        clean_ego: torch.Tensor,
    ) -> torch.Tensor:
        """Build Ego-conditioned A0 by same-width feature addition.

        ``source`` remains the only noisy input.  ``clean_ego`` is eta's clean
        observable encoding, never raw state added to trajectory coordinates.
        """
        if source.ndim != 4 or source.shape[-1] != 4:
            raise ValueError("source must be [B,M,action_tokens,4]")
        if modes.ndim != 3 or modes.shape[-1] != 2 or source.shape[:2] != modes.shape[:2]:
            raise ValueError("modes must be [B,M,2] and align with source")
        if source.shape[2] != self.action_tokens:
            raise ValueError(f"source action axis must contain {self.action_tokens} tokens")
        if clean_ego.ndim == 3:
            clean_ego = clean_ego.mean(dim=1)
        if clean_ego.ndim != 2 or clean_ego.shape != (source.shape[0], self.d_model):
            raise ValueError("clean ego encoding must be [B,d_model] or [B,L,d_model]")
        action = self.theta["source_projection"](source)
        mode = self.theta["mode_projection"](modes).unsqueeze(2)
        time = self.theta_time_embedding.view(1, 1, self.action_tokens, self.d_model)
        ego = self.theta["ego_to_action_projection"](clean_ego).view(source.shape[0], 1, 1, self.d_model)
        return action + ego + time + mode

    def _action_conditioning(self, modes: torch.Tensor, clean_ego: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the Wan adaLN condition from mode code + projected clean ego.

        ``cond`` [B,M,d] feeds the modulated Head; ``t_mod`` [B*M,6,d] feeds
        every block's six-parameter shift/scale/gate modulation, replacing
        Wan's timestep embedding.  Both reuse theta-owned projections that also
        form A0, so conditioning enters by feature addition only.
        """
        if clean_ego.ndim == 3:
            clean_ego = clean_ego.mean(dim=1)
        cond = self.theta["mode_projection"](modes) + self.theta["ego_to_action_projection"](clean_ego).unsqueeze(1)
        b, m, d = cond.shape
        t_mod = self.theta["cond_projection"](cond).view(b * m, 6, d)
        return cond, t_mod

    def _directed_pass(
        self,
        scene: torch.Tensor,
        action: torch.Tensor | None,
        register: torch.Tensor,
        t_mod: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        for scene_block, register_block, action_block in zip(self.scene_blocks, self.phi["blocks"], self.theta["blocks"]):
            scene = scene_block(scene)
            register = register_block(register, scene)
            if action is not None:
                action = action_block(action, scene, self.register_readout(register).detach(), t_mod)
        return scene, register, action

    def forward_register_only(self, observation: Observation, modes: torch.Tensor) -> torch.Tensor:
        clean_ego = self.eta.encode_clean_ego(observation)
        scene = self.eta(observation, clean_ego=clean_ego)
        register = self._initialize_register(modes)
        _, register, _ = self._directed_pass(scene, None, register)
        return self.register_readout(register)

    def forward_pairs(self, observation: Observation, source: torch.Tensor, modes: torch.Tensor) -> PairOutputs:
        clean_ego = self.eta.encode_clean_ego(observation)
        scene = self.eta(observation, clean_ego=clean_ego)
        action = self._initialize_action(source, modes, clean_ego)
        cond, t_mod = self._action_conditioning(modes, clean_ego)
        register = self._initialize_register(modes)
        scene, register, action = self._directed_pass(scene, action, register, t_mod)
        endpoint = source + self.theta["trajectory_head"](action, cond)
        source_yaw = torch.atan2(source[..., 2], source[..., 3])
        trajectory = decode_trajectory(endpoint, self.xy_scale, source_yaw=source_yaw)
        return PairOutputs(scene, endpoint, trajectory, self.register_readout(register))


def parameter_groups(model: DirectedPairDiT) -> dict[str, list[nn.Parameter]]:
    groups = {"eta": [], "theta": [], "phi": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("eta.") or name.startswith("scene_blocks."):
            groups["eta"].append(parameter)
        elif name.startswith("theta.") or name == "theta_time_embedding":
            groups["theta"].append(parameter)
        elif name.startswith("phi.") or name == "phi_time_queries":
            groups["phi"].append(parameter)
    return groups
