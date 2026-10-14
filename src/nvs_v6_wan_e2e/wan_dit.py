"""Faithful port of the Wan2.2 ``DiTBlock`` stack used by SimWAM's ActionDiT.

Everything transferable keeps SimWAM's exact submodule names and shapes so a
payload built by :mod:`nvs_v6_wan_e2e.wan_init` loads with ``strict=True``:
``self_attn``/``cross_attn`` (``q,k,v,o`` + ``norm_q``/``norm_k`` over the whole
``attn_hidden_dim``), ``norm1``/``norm2`` (no affine), ``norm3``, ``ffn`` with
GELU(tanh), the six-parameter ``modulation``, plus ``text_embedding``,
``time_embedding``, ``time_projection``.

Two NVS deviations, both required by the v6 contract and both outside the
transferable key set:

* Wan conditions on a diffusion timestep alone.  NVS also conditions on the
  mode+ego vector, so ``time_embedding`` sees the flow timestep's
  ``sinusoidal_embedding_1d`` *plus* that vector.  ``freq_dim`` therefore equals
  the condition width.
* Wan builds the cross-attention context once.  NVS's Action stream reads the
  Register at every layer, so the context is re-embedded per layer.

Scale contract
--------------
Wan's transferred weights (``blocks.*``, ``text_embedding``, ``time_embedding``,
``time_projection``) were trained on VAE latents whose per-element RMS is ~1 and
on ``sinusoidal_embedding_1d`` conditions whose per-element RMS is ``1/sqrt(2)``.
NVS feeds *metric* trajectories instead, whose raw coordinates reach ~80 m, so
``action_encoder(source)`` alone produced token norms of 20-200 -- up to 8x the
conditioning terms.  The action therefore enters through SimWAM's ``norm_odo``
map onto ``[-1, 1]`` (see :mod:`nvs_v6_wan_e2e.flow`), and the learned ego term
is RMS-clamped to Wan's condition RMS.  Together those keep the A0 residual
stream at ~1.0 without a data-dependent rescale of the flow state itself -- the
per-token ``rms_unit`` that would do that is deliberately *not* applied here,
because the magnitude of ``x_sigma`` is the quantity the rectified flow encodes.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .flow import ODO_METRIC_JACOBIAN_MAX, norm_odo

#: Physical span of the 8x8 anchor grid, used to bound the mode code before
#: ``mode_projection``: velocity in m/s and yaw rate in rad/s differ by 13x.
MODE_SCALE = (4.0, 0.30)
#: Per-element RMS of Wan's ``sinusoidal_embedding_1d`` output (cos/sin pairs).
WAN_CONDITION_RMS = 1.0 / math.sqrt(2.0)
#: Per-element RMS of Wan's VAE latent tokens.
WAN_TOKEN_RMS = 1.0


def rms_unit(x: torch.Tensor, eps: float = 1e-6, scale: float = 1.0) -> torch.Tensor:
    """Rescale ``x`` to a fixed per-element RMS without touching its direction.

    Parameter-free and differentiable, so it can sit inside the condition path
    without adding state that :func:`wan_init.load_action_dit_backbone` would have
    to reconcile against the Wan payload.
    """
    return x * (scale * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)).to(x.dtype)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_apply(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    b, s, _ = x.shape
    x = x.view(b, s, num_heads, -1)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(b, s, num_heads, -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(dtype) * self.weight


class SelfAttention(nn.Module):
    """Wan self-attention: RoPE over the token axis, QK RMSNorm before RoPE."""

    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = num_heads * attn_head_dim
        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, self.num_heads, length, self.attn_head_dim)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, self_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = rope_apply(self.norm_q(self.q(x)), freqs, self.num_heads)
        k = rope_apply(self.norm_k(self.k(x)), freqs, self.num_heads)
        v = self.v(x)
        out = F.scaled_dot_product_attention(self._split(q), self._split(k), self._split(v), attn_mask=self_attn_mask)
        batch, length, _ = x.shape
        return self.o(out.transpose(1, 2).reshape(batch, length, self.attn_hidden_dim))


class CrossAttention(nn.Module):
    """Wan cross-attention: no RoPE, Action queries read the NVS context."""

    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = num_heads * attn_head_dim
        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, self.num_heads, length, self.attn_head_dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        out = F.scaled_dot_product_attention(self._split(q), self._split(k), self._split(v), attn_mask=ctx_mask)
        batch, length, _ = x.shape
        return self.o(out.transpose(1, 2).reshape(batch, length, self.attn_hidden_dim))


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = CrossAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, hidden_dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)
        x = x + gate_msa * self.self_attn(modulate(self.norm1(x), shift_msa, scale_msa), freqs, self_attn_mask=self_attn_mask)
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        return x + gate_mlp * self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp))


class Head(nn.Module):
    """Wan's modulated output head, here emitting the rectified-flow velocity.

    ``proj`` is initialized near zero so ``v_hat ~= 0`` at step 0.  The single
    forward then inverts to ``x_sigma - sigma * v_hat == x_sigma``, and at the
    deployed ``sigma = 1`` that is exactly the anchor: the 64-candidate
    velocity/yaw-rate grid already brackets the GT well (oracle-best PDMS ~0.99
    on navtest), so training starts from that coverage and can only refine it,
    instead of starting from a random O(1) velocity that destroys every anchor
    before any gradient signal exists.
    """

    #: ``ActionFlow.recover`` converts this head's normalized-space velocity into
    #: metres with the constant Jacobian ``range / 2`` of ``denorm_odo``, whose
    #: largest entry is 33.37 m.  Dividing the near-zero init by it keeps the
    #: deployed ``sigma = 1`` forward within ~0.04 m of the anchor at step 0.  It
    #: stays nonzero (rather than DiT's exact zero) so every upstream block still
    #: receives gradient on the very first step.
    HEAD_INIT_STD = 1e-3 / ODO_METRIC_JACOBIAN_MAX

    def __init__(self, hidden_dim: int, out_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)
        with torch.no_grad():
            self.proj.weight.normal_(0.0, self.HEAD_INIT_STD)
            self.proj.bias.zero_()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() != 2:
            raise ValueError(f"head condition must be [B,hidden_dim], got {tuple(t.shape)}")
        shift, scale = (self.modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1)).chunk(2, dim=1)
        return self.proj(modulate(self.norm(x), shift, scale))


class NvsActionDiT(nn.Module):
    """SimWAM's ActionDiT at ``hidden_dim=256`` with NVS's action/mode/ego inputs.

    ``BACKBONE_SKIP_PREFIXES`` mirrors SimWAM's ``ACTION_BACKBONE_SKIP_PREFIXES``
    (``action_encoder.``, ``head.``) and adds the NVS-only condition projections,
    so those stay randomly initialized while the rest loads from Wan.
    """

    BACKBONE_SKIP_PREFIXES = (
        "action_encoder.",
        "head.",
        "mode_projection.",
        "ego_to_action_projection.",
        "time_queries",
    )
    META_KEYS = ("hidden_dim", "ffn_dim", "num_layers", "num_heads", "attn_head_dim", "text_dim", "freq_dim", "eps")

    def __init__(
        self,
        hidden_dim: int = 256,
        action_dim: int = 3,
        ffn_dim: int = 1024,
        text_dim: int = 256,
        freq_dim: int = 256,
        eps: float = 1e-6,
        num_heads: int = 4,
        attn_head_dim: int = 64,
        num_layers: int = 8,
        action_tokens: int = 8,
        mode_dim: int = 2,
        ego_dim: int = 256,
    ) -> None:
        super().__init__()
        if num_heads <= 0 or attn_head_dim <= 0:
            raise ValueError("num_heads and attn_head_dim must be positive")
        if attn_head_dim % 2:
            raise ValueError("attn_head_dim must be even for RoPE")
        if text_dim != hidden_dim:
            raise ValueError("the NVS cross-attention context is the d_model Scene/Register stream")
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.eps = eps
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.num_layers = num_layers
        self.action_tokens = action_tokens

        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.mode_projection = nn.Linear(mode_dim, hidden_dim)
        self.ego_to_action_projection = nn.Linear(ego_dim, hidden_dim)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.GELU(approximate="tanh"), nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps) for _ in range(num_layers)]
        )
        self.head = Head(hidden_dim, action_dim, eps)
        self.time_queries = nn.Parameter(torch.randn(action_tokens, hidden_dim) * 0.02)
        self.register_buffer("freqs", precompute_freqs_cis(attn_head_dim, end=1024), persistent=False)
        # Non-persistent on purpose: a persistent buffer would land in
        # ``state_dict()``, be counted as a backbone key by ``backbone_key_set``
        # (it matches no skip prefix), and break the payload's strict key check.
        self.register_buffer("mode_scale", torch.tensor(MODE_SCALE), persistent=False)

    def meta(self) -> dict[str, float]:
        return {
            "hidden_dim": self.hidden_dim,
            "ffn_dim": self.ffn_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "attn_head_dim": self.attn_head_dim,
            "text_dim": self.text_dim,
            "freq_dim": self.freq_dim,
            "eps": self.eps,
        }

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {key for key in keys if not any(key.startswith(prefix) for prefix in cls.BACKBONE_SKIP_PREFIXES)}

    def rope_for(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len > self.freqs.shape[0]:
            raise ValueError(f"action token length {seq_len} exceeds the RoPE cache {self.freqs.shape[0]}")
        return self.freqs[:seq_len].view(seq_len, 1, -1).to(device)

    def _mode_features(self, modes: torch.Tensor) -> torch.Tensor:
        """Project the anchor's ``(velocity, yaw_rate)`` code at a bounded scale."""
        return self.mode_projection(modes / self.mode_scale.to(device=modes.device, dtype=modes.dtype))

    def _ego_features(self, clean_ego: torch.Tensor) -> torch.Tensor:
        """eta's encoded ego, RMS-clamped so it cannot swamp the flow state."""
        return rms_unit(self.ego_to_action_projection(clean_ego), self.eps, WAN_CONDITION_RMS)

    def encode_condition(
        self,
        modes: torch.Tensor,
        clean_ego: torch.Tensor,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Wan's ``t``: a ``[B,M,freq_dim]`` condition embedded by ``time_embedding``.

        ``timestep`` is the rectified-flow ``t`` in ``[0, num_train_timesteps]``,
        shaped ``[B]`` or ``[B,M]``.  It enters through Wan's own
        ``sinusoidal_embedding_1d``, which is what the transferred
        ``time_embedding``/``time_projection`` were trained on, so the adaLN
        modulation finally receives a real diffusion time.  The mode+ego vector is
        added to it and the sum is rescaled to Wan's condition RMS; without
        ``timestep`` this reduces to the mode+ego condition alone.
        """

        if modes.dim() != 3 or modes.shape[-1] != 2:
            raise ValueError(f"modes must be [B,M,2], got {tuple(modes.shape)}")
        if clean_ego.dim() != 2 or clean_ego.shape[0] != modes.shape[0]:
            raise ValueError("clean ego encoding must be [B,ego_dim] aligned with modes")
        condition = self._mode_features(modes) + self._ego_features(clean_ego).unsqueeze(1)
        if timestep is not None:
            batch, candidates = modes.shape[:2]
            if tuple(timestep.shape) not in ((batch,), (batch, candidates)):
                raise ValueError(f"timestep must be [B] or [B,M], got {tuple(timestep.shape)}")
            per_candidate = timestep.reshape(batch, -1)
            per_candidate = per_candidate.expand(batch, candidates) if per_candidate.shape[1] == 1 else per_candidate
            sinusoid = sinusoidal_embedding_1d(self.freq_dim, per_candidate.reshape(-1))
            condition = condition.reshape(-1, self.hidden_dim) + sinusoid.to(condition.dtype)
            condition = condition.reshape(batch, candidates, self.hidden_dim)
        return self.time_embedding(rms_unit(condition, self.eps, WAN_CONDITION_RMS))

    def encode_action(self, source: torch.Tensor, modes: torch.Tensor, clean_ego: torch.Tensor) -> torch.Tensor:
        """NVS A0: the flow state ``x_sigma`` plus same-width mode/ego/time features.

        ``source`` is the metric action the flow produced -- the clean anchor at
        ``sigma = 1``, the clean GT at ``sigma = 0``, an interpolation between.  It
        stays the only trajectory-valued input; ``clean_ego`` is eta's encoded
        observable, never raw state added to metric coordinates.

        The metric action goes through SimWAM's ``norm_odo`` before
        ``action_encoder``, so metres and radians contribute comparably and the
        term lands at Wan's latent magnitude without a data-dependent rescale.
        The sum is intentionally *not* RMS-normalized afterwards: ``x_sigma``'s
        magnitude is the flow state, and normalizing per token would erase both
        the distance travelled and the distinction between noise levels.
        """
        if source.shape[-1] != self.action_dim:
            raise ValueError(f"source must be [B,M,{self.action_tokens},{self.action_dim}]")
        if source.shape[2] != self.action_tokens:
            raise ValueError(f"source action axis must hold {self.action_tokens} tokens")
        if modes.shape[:2] != source.shape[:2] or modes.shape[-1] != 2:
            raise ValueError("modes must be [B,M,2] aligned with source")
        if clean_ego.dim() != 2 or clean_ego.shape != (source.shape[0], self.hidden_dim):
            raise ValueError("clean ego encoding must be [B,hidden_dim]")
        trajectory = self.action_encoder(norm_odo(source))
        mode = self._mode_features(modes).unsqueeze(2)
        ego = self._ego_features(clean_ego).view(source.shape[0], 1, 1, self.hidden_dim)
        return trajectory + ego + self.time_queries.view(1, 1, self.action_tokens, self.hidden_dim) + mode

    def forward(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
        contexts,
    ) -> torch.Tensor:
        """Run the Wan block stack over per-layer contexts.

        ``tokens``/``condition`` are flattened per candidate (``[B*M, ...]``) and
        ``contexts`` yields one ``[B*M, L, hidden_dim]`` context per layer.
        """
        t_mod = self.time_projection(condition).unflatten(1, (6, self.hidden_dim))
        freqs = self.rope_for(tokens.shape[1], tokens.device)
        x = tokens
        for block, context in zip(self.blocks, contexts):
            x = block(x, context, t_mod, freqs)
        return self.head(x, condition)
