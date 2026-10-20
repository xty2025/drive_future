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
* Wan denoises latents only, and its sequence holds nothing but them.  NVS runs
  ``[scene | action | register]``: the Scene tokens are prepended exactly as
  FastWAM prepends its visual tokens, and ``future_tokens`` learnable future
  queries are appended behind the action tokens with a second head read off them,
  so one forward through the same block stack yields both the flow velocity and
  the predicted future representation.  That is V-JEPA's mask-token pattern -- a
  learnable query appended to the context sequence and supervised against a
  frozen encoder's own output -- with the action and the scene as the extra
  conditioning V-JEPA's appearance-only predictor does not have.  The action
  tokens keep Wan's exact input path, Wan's exact head and Wan's exact RoPE
  positions ``0..7``; scene and register are new segments around them, told apart
  by three type embeddings because their RoPE ranges overlap on purpose.

  Unlike Wan's latents-only sequence, this one is **masked**, and there is exactly
  one mask, shared by every stage and every entry point -- see
  :attr:`NvsActionDiT.SEGMENT_VISIBILITY`.  It is lower triangular: Scene reads only
  itself, Action reads Scene and itself, Register reads all three.  The one
  structural invariant it encodes is that **Action never reads Register**, so the
  velocity head can never see the future it is being asked to anticipate.  The
  register segment is therefore a pure auxiliary readout of the shared trunk: its
  loss shapes the blocks the action stream also runs through, but at no point does
  a predicted future enter the tokens the action is generated from.

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

from .data import REGISTER_TOKENS
from .flow import ODO_METRIC_JACOBIAN_MAX, norm_odo

#: Register-segment length: one query token per (future frame, DINOv3 prefix query),
#: frame-major.  Aliased rather than recomputed so the DiT's default and the target
#: cache's token axis are literally the same constant.
DEFAULT_FUTURE_TOKENS = REGISTER_TOKENS

#: Physical span of the 8x8 anchor grid, used to bound the mode code before
#: ``mode_projection``: velocity in m/s and yaw rate in rad/s differ by 13x.
MODE_SCALE = (4.0, 0.30)
#: Per-element RMS of Wan's ``sinusoidal_embedding_1d`` output (cos/sin pairs).
WAN_CONDITION_RMS = 1.0 / math.sqrt(2.0)
#: Per-element RMS of Wan's VAE latent tokens.
WAN_TOKEN_RMS = 1.0
#: Physical scale of the *detrended* trajectory residual, per channel (x, y, heading).
#:
#: ``norm_odo`` divides by the corpus-absolute span ``(66.74, 42.00, 3.53)``, which is
#: the right choice for the flow state -- ``denorm_odo`` has to invert it and
#: ``velocity_to_metric_scale`` reads ``range / 2`` off it -- but the wrong one for
#: *conditioning*.  Two Multi-GT rows of the same scene differ by 0.03-2.0 m over the
#: 4 s horizon, i.e. 0.2-8.4% of that span, while their register targets differ by
#: ~29-60% of the detrended scale.  A 121-token self-attention whose action tokens move
#: by a few percent cannot produce a register that moves by half its own magnitude, and
#: the measured consequence was a Stage-1 register whose response to the trajectory sat
#: at 1e-4..1e-3 against a target spread of 0.5-0.76.
#:
#: Calibrated so each detrended channel lands at the *overall* RMS ``norm_odo`` reaches
#: on real GT trajectories (0.413), measured over the smoke cache's 32 rows:
#: detrended RMS is 0.541 m in x, 0.171 m in y and 0.0335 rad in heading, giving
#: 0.541/0.413, 0.171/0.413, 0.0335/0.413.  Per-channel rather than one shared divisor
#: because ``norm_odo`` leaves y at 0.064 and heading at 0.081 against x's 0.708 -- the
#: lateral and yaw components that actually distinguish a turn from a straight line are
#: the ones it suppresses hardest.  A fixed constant, not a batch statistic: see
#: :func:`detrend_trajectory`.
SHAPE_RANGE = (1.31, 0.414, 0.0811)
#: Init scale of ``future_queries``.  The register tokens are ``queries + ego``, and
#: ``ego`` is RMS-clamped to ``WAN_CONDITION_RMS`` and *identical* across all of them, so
#: at the previous 0.02 the tokens differed from each other by 0.02 / 0.707 = 2.8% and
#: were told apart almost entirely by their RoPE phase.  0.5 makes the per-token identity
#: comparable to the conditioning it is added to, and keeps the residual stream near
#: ``WAN_TOKEN_RMS`` (sqrt(0.5**2 + 0.707**2) = 0.87).
#:
#: The scale is load-bearing rather than merely generous since the segment went to
#: ``REGISTER_QUERIES_PER_FRAME`` tokens per frame: those queries deliberately *share*
#: one RoPE phase (see :meth:`NvsActionDiT.register_rope`), so this parameter is the only
#: thing distinguishing the four reads of a single future frame at initialisation.  At
#: 0.02 they would start as four near-identical tokens and have to be separated by
#: gradient alone.
FUTURE_QUERY_SCALE = 0.5


def rms_unit(x: torch.Tensor, eps: float = 1e-6, scale: float = 1.0) -> torch.Tensor:
    """Rescale ``x`` to a fixed per-element RMS without touching its direction.

    Parameter-free and differentiable, so it can sit inside the condition path
    without adding state that :func:`wan_init.load_action_dit_backbone` would have
    to reconcile against the Wan payload.
    """
    return x * (scale * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)).to(x.dtype)


def detrend_trajectory(trajectory: torch.Tensor) -> torch.Tensor:
    """Subtract each trajectory's own least-squares linear trend over its waypoints.

    A pure function of a *single* ``[..., tokens, 3]`` trajectory: it reads no other
    candidate, no batch statistic and no dataset constant.  That is deliberate.  Multi-GT
    is K independent ``(scene, trajectory) -> future`` samples, not a set to be
    normalised against itself, so the encoding of one row must not depend on which
    siblings happen to share its scene -- otherwise the same trajectory would mean
    something different in Stage 1 (K ground truths) than in Stage 2 (64 sampled
    candidates), and a Stage-1 checkpoint would not transfer.

    The trend is the closed-form fit of ``a + b * t`` with ``t`` symmetric over
    ``[-1, 1]``, so ``t.mean() == 0`` and the two coefficients decouple: ``a`` is the
    waypoint mean, ``b`` is the projection onto ``t`` divided by ``mean(t**2)``.  No
    ``lstsq``, no inverse, nothing to go ill-conditioned, and the returned residual is
    exactly orthogonal to both basis vectors -- which is why feeding it *alongside*
    ``norm_odo(source)`` adds information instead of restating it.
    """

    tokens = trajectory.shape[-2]
    if tokens < 2:
        raise ValueError(f"detrending needs at least two waypoints, got {tokens}")
    t = torch.linspace(-1.0, 1.0, tokens, device=trajectory.device, dtype=trajectory.dtype).view(tokens, 1)
    mean = trajectory.mean(dim=-2, keepdim=True)
    slope = (trajectory * t).mean(dim=-2, keepdim=True) / t.pow(2).mean()
    return trajectory - (mean + slope * t)


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
        # ``x`` is ``[B, L, H*D]`` in (token, head, dim) order, so the head axis has to
        # be *transposed* into position 1, not reshaped into it: ``view(B, H, L, D)``
        # reinterprets the flat buffer and hands SDPA a strided mix of every token,
        # which silently defeats ``attn_mask`` and misplaces the RoPE phases
        # ``rope_apply`` just wrote.
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.attn_head_dim).transpose(1, 2)

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
        # ``x`` is ``[B, L, H*D]`` in (token, head, dim) order, so the head axis has to
        # be *transposed* into position 1, not reshaped into it: ``view(B, H, L, D)``
        # reinterprets the flat buffer and hands SDPA a strided mix of every token,
        # which silently defeats ``attn_mask`` and misplaces the RoPE phases
        # ``rope_apply`` just wrote.
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.attn_head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        out = F.scaled_dot_product_attention(self._split(q), self._split(k), self._split(v), attn_mask=ctx_mask)
        batch, length, _ = x.shape
        return self.o(out.transpose(1, 2).reshape(batch, length, self.attn_hidden_dim))


class TrajectoryGate(nn.Module):
    """One-way cross-attention giving the register a guaranteed route to the action segment.

    Measured on the production geometry with the real DINOv3 scene stream, register rows
    put only **8-10%** of their self-attention mass on the eight action keys (Scene takes
    61-66% with 81 keys, the register's own segment 24-31%), while the register's *input*
    is trajectory-independent by construction (``future_queries`` + ego).  The trajectory
    therefore reaches the future readout only through a ~9%-per-hop channel that compounds
    across eight blocks into the ~583x attenuation seen in
    ``trajectory_conditioning_fidelity``.  Softmax competition with 81 scene keys is the
    mechanism, and no reweighting of the objective can remove it: a contrast term that
    supervises the across-GT difference *exactly* still left ``residual_relative_l1``
    pinned at 1.0 after 50 overfitting steps.

    The gate is one-way -- register queries, action keys/values -- so it cannot reopen the
    forbidden ``Action <- Register`` direction in reverse: action tokens read neither the
    register here nor anywhere else, and :data:`NvsActionDiT.SEGMENT_VISIBILITY` is
    untouched.  ``out`` is zero-initialised, so step 0 is bitwise identical to the ungated
    stack: the gate can only add trajectory signal, never displace a pretrained one.
    """

    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        action_tokens: int,
        queries_per_frame: int,
        eps: float = 1e-6,
    ) -> None:
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
        # Frame-aligned read: register row ``q + Q*f`` attends to action token ``f`` and
        # nothing else.  Without this the gate's softmax spreads over all eight action
        # rows, so every register row reads the *mean* trajectory and the rows stay
        # bitwise identical -- measured as ``residual_relative_l1`` pinned at 1.0 even
        # with the gate live.  The alignment is the one the RoPE phases already intend
        # (register token ``q + Q*f`` is the future of the instant action token ``f``
        # predicts); the mask makes it exact instead of merely suggested.
        rows = queries_per_frame * action_tokens
        mask = torch.zeros(rows, action_tokens, dtype=torch.bool)
        for frame in range(action_tokens):
            for query in range(queries_per_frame):
                mask[query + queries_per_frame * frame, frame] = True
        self.register_buffer("frame_mask", mask, persistent=False)
        # ``o`` keeps nn.Linear's default init.  Two rejected alternatives, both measured:
        # zero-init left ``residual_relative_l1`` pinned at 1.0 through the whole warmup
        # (``o`` only reaches ~4e-3 in 20 AdamW steps at lr 2e-4), and std 1e-2 -- 6x below
        # the default 1/sqrt(256) -- still left the gate ~100x too weak to move the
        # across-GT row spread at init, and 80 probe steps cannot grow it the remaining
        # distance.  There is no pretrained register stream to protect, so a
        # default-strength route from step 0 is the honest choice; the bias stays zero.
        nn.init.zeros_(self.o.bias)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.attn_head_dim).transpose(1, 2)

    def forward(self, register: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q = self._split(self.norm_q(self.q(register)))
        k = self._split(self.norm_k(self.k(action)))
        v = self._split(self.v(action))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=self.frame_mask)
        batch, length = register.shape[0], register.shape[1]
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
    """SimWAM's ActionDiT at ``hidden_dim=256``, extended with a future-query head.

    The self-attention sequence is ``[scene | action | register]``, which is
    FastWAM's layout -- it prepends its visual tokens to the action tokens and keeps
    "Action RoPE 0..T-1 even though visual tokens are prepended" -- extended with the
    register segment Wan has no counterpart for.

    ``BACKBONE_SKIP_PREFIXES`` mirrors SimWAM's ``ACTION_BACKBONE_SKIP_PREFIXES``
    (``action_encoder.``, ``head.``) and adds the NVS-only condition projections, the
    whole future-query path and the three segment type embeddings, so those stay
    randomly initialized while the rest loads from Wan.
    """

    BACKBONE_SKIP_PREFIXES = (
        "action_encoder.",
        "head.",
        "mode_projection.",
        "ego_to_action_projection.",
        "time_queries",
        "future_queries",
        "future_head.",
        "trajectory_gates.",
        "scene_type_embedding",
        "action_type_embedding",
        "register_type_embedding",
    )
    META_KEYS = ("hidden_dim", "ffn_dim", "num_layers", "num_heads", "attn_head_dim", "text_dim", "freq_dim", "eps")

    #: Segment order of the self-attention sequence, and the row/column order of the
    #: visibility table below: rows are **queries**, columns are **keys**.
    SEGMENT_ORDER = ("scene", "action", "register")
    #: The one and only mask, lower triangular: Scene reads only itself, Action reads
    #: Scene and itself, Register reads all three.
    #:
    #: Read every arrow below as **data flow** -- ``A -> B`` means "B reads A", i.e.
    #: ``VISIBILITY[row B][col A]`` -- which is the reverse of the row/column order
    #: declared above.  The indices are spelled out next to each arrow because the two
    #: readings disagree, and inverting one of them silently opens the leak this table
    #: exists to close.
    #:
    #: ``Register -> Action`` (``VISIBILITY[1][2] = 0``) is closed unconditionally.
    #: That is the invariant: the velocity head never sees a predicted future, in any
    #: stage and through any entry point.  Earlier revisions opened that entry for
    #: Stage 2 (and closed ``Action -> Register`` there instead) so the future could
    #: condition planning; the two stages then ran different information flows, which
    #: meant Stage 2's ``loss_future`` measured a different function from the one
    #: Stage 1 trained.  One table removes that discrepancy at the source.
    #:
    #: ``Action -> Register`` (``VISIBILITY[2][1] = 1``) staying open is what makes the
    #: register non-trivial: :meth:`encode_future` supplies no trajectory and no mode
    #: code, so if the register could not read the action segment it would receive only
    #: ``(Scene, ego)`` and be **bitwise identical across the K Multi-GT rows of a
    #: scene** -- pinning ``paired_future_l1`` at the ``scene_mean_target_l1`` floor and
    #: forcing ``gain_over_trajectory_blind`` to exactly zero for every step.  Measured
    #: on a depth-2 trunk, one scene, four GT trajectories and a zero mode code: closing
    #: that entry gives cross-GT ``max|dRegister| = 0.0`` exactly, versus ``7.5e-3``
    #: with it open.
    #:
    #: Scene reads only itself (``VISIBILITY[0][1] = VISIBILITY[0][2] = 0``), so neither
    #: the action nor the future ever writes to the tokens DrivoR's scorer takes as K/V.
    SEGMENT_VISIBILITY = ((1, 0, 0), (1, 1, 0), (1, 1, 1))

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
        future_tokens: int = DEFAULT_FUTURE_TOKENS,
        target_dim: int = 256,
        mode_dim: int = 2,
        ego_dim: int = 256,
    ) -> None:
        super().__init__()
        if num_heads <= 0 or attn_head_dim <= 0:
            raise ValueError("num_heads and attn_head_dim must be positive")
        if attn_head_dim % 2:
            raise ValueError("attn_head_dim must be even for RoPE")
        if text_dim != hidden_dim:
            raise ValueError("the NVS cross-attention context is the d_model Scene stream")
        if future_tokens % action_tokens:
            raise ValueError(
                f"the register segment must hold a whole number of queries per future frame; "
                f"{future_tokens} tokens over {action_tokens} frames does not"
            )
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
        self.future_tokens = future_tokens
        self.target_dim = target_dim

        #: Doubled width: ``[norm_odo(source) | detrend(source) / SHAPE_RANGE]``.  The
        #: absolute channel keeps the flow state in the coordinates Wan was trained on,
        #: the detrended channel carries the manoeuvre at O(1).  ``action_encoder`` is in
        #: ``skip_prefixes``, so it is randomly initialised and this shape change costs
        #: no transferred weight; ``head`` stays ``action_dim``-wide because the velocity
        #: the flow predicts is still metric ``(x, y, heading)``.
        self.action_encoder = nn.Linear(action_dim * 2, hidden_dim)
        self.mode_projection = nn.Linear(mode_dim, hidden_dim)
        self.ego_to_action_projection = nn.Linear(ego_dim, hidden_dim)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.GELU(approximate="tanh"), nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps) for _ in range(num_layers)]
        )
        #: One guaranteed trajectory route per block; see :class:`TrajectoryGate`.  Only
        #: applied when the register segment is present, and frozen (so DDP's reducer
        #: ignores it) when ``predict_future`` is off and there is no register to gate.
        self.trajectory_gates = nn.ModuleList(
            [
                TrajectoryGate(hidden_dim, attn_head_dim, num_heads, action_tokens,
                               future_tokens // action_tokens, eps)
                for _ in range(num_layers)
            ]
        )
        self.head = Head(hidden_dim, action_dim, eps)
        self.time_queries = nn.Parameter(torch.randn(action_tokens, hidden_dim) * 0.02)
        #: One query per (future frame, DINOv3 prefix query), frame-major: token
        #: ``q + R * f`` is prefix query ``q`` reading the frame at ``(f + 1) * 0.5 s``,
        #: and its cached target is the cosine-attention readout of that frame's patch
        #: grid by that same prefix query.  Perturbed rather than left at exactly zero
        #: (V-JEPA zeroes its mask tokens) so the tokens are distinguishable in the
        #: residual stream from step 0 -- and, since the four queries of one frame share
        #: a RoPE phase on purpose, ``future_queries`` is the *only* thing separating
        #: them at init.  Unlike ``time_queries`` this is not small relative to the ego
        #: term it is added to; see :data:`FUTURE_QUERY_SCALE`.
        self.future_queries = nn.Parameter(torch.randn(future_tokens, hidden_dim) * FUTURE_QUERY_SCALE)
        #: LayerNorm then a projection back to the target width, which is what
        #: V-JEPA's predictor does before its L1.  The norm matters here because
        #: the residual stream's scale drifts across the eight modulated blocks,
        #: while the standardized target sits at unit variance per component.
        self.future_head = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False),
            nn.Linear(hidden_dim, target_dim),
        )
        #: Segment identities for the single self-attention sequence
        #: ``[scene | action | register]``.  Scene and Action deliberately share
        #: RoPE position ranges (FastWAM keeps "Action RoPE 0..T-1 even though
        #: visual tokens are prepended"), so without these three vectors a block
        #: could not tell scene token 5 from action token 5.  The scale matches
        #: ``time_queries``/``future_queries`` and stays far below the residual
        #: stream, so the Wan-transferred blocks still see essentially the tokens
        #: they were pretrained on.
        self.scene_type_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.action_type_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.register_type_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.register_buffer("freqs", precompute_freqs_cis(attn_head_dim, end=1024), persistent=False)
        # Non-persistent on purpose: a persistent buffer would land in
        # ``state_dict()``, be counted as a backbone key by ``backbone_key_set``
        # (it matches no skip prefix), and break the payload's strict key check.
        self.register_buffer("mode_scale", torch.tensor(MODE_SCALE), persistent=False)
        self.register_buffer("shape_range", torch.tensor(SHAPE_RANGE), persistent=False)

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

    def register_rope(self, future_tokens: int, device: torch.device) -> torch.Tensor:
        """Frame-repeated RoPE: one phase per future frame, shared by that frame's queries.

        The register segment holds ``REGISTER_QUERIES_PER_FRAME`` tokens per frame, so
        the phase has to key on the *frame* and not on the token: register token
        ``q + R * f`` is query ``q`` of the frame at ``(f + 1) * 0.5 s``, which is the
        instant action token ``f`` predicts.  Counting ``0..R*T-1`` instead would place
        one frame's four queries at four different times and stretch the segment past
        the action's own ``0..T-1``, breaking the one-to-one temporal correspondence the
        shared phase exists to hand attention for free.  The queries of a frame are told
        apart by ``future_queries`` and ``register_type_embedding``, which is what those
        are for; RoPE carries time and nothing else.
        """
        if future_tokens % self.action_tokens:
            raise ValueError(
                f"the register segment holds {future_tokens} tokens across {self.action_tokens} "
                "future frames; the per-frame query count must divide it exactly"
            )
        queries = future_tokens // self.action_tokens
        return self.rope_for(self.action_tokens, device).repeat_interleave(queries, dim=0)

    def rope_for_segments(self, lengths: list[int], device: torch.device) -> torch.Tensor:
        """Per-segment RoPE: every segment of the sequence restarts at position 0.

        Action keeps Wan's own ``0..7`` -- ``head``, ``time_queries`` and all eight
        transferred blocks were pretrained with the action tokens at exactly those
        positions, so shifting them by the scene length would hand Wan a positional
        distribution it never saw.  Scene takes ``0..ns-1`` as a spatial grid whose
        ordering carries no meaning shared with the trajectory.  Register takes the
        same ``0..7`` *frames*, repeated once per query within a frame -- see
        :meth:`register_rope`.
        """
        if len(lengths) not in (2, 3):
            raise ValueError(f"expected 2 or 3 segment lengths, got {lengths}")
        if len(lengths) == 3 and lengths[1] != self.action_tokens:
            raise ValueError(
                f"the register phase is aligned to the action segment actually in the "
                f"sequence; lengths[1] must be {self.action_tokens}, got {lengths[1]}"
            )
        segments = [self.rope_for(lengths[0], device), self.rope_for(lengths[1], device)]
        if len(lengths) == 3:
            segments.append(self.register_rope(lengths[2], device))
        return torch.cat(segments, dim=0)

    def segment_attention_mask(self, lengths: list[int], device: torch.device) -> torch.Tensor:
        """Expand the 3x3 segment table into a boolean ``[L, L]`` token mask.

        ``True`` means "this query may read this key", which is exactly the convention
        ``F.scaled_dot_product_attention`` uses for a boolean ``attn_mask``.  The
        result is 2-D so it broadcasts over batch and heads for free.

        The table has a fully visible diagonal block, so no query row is ever
        all-False -- a fully masked row is what makes softmax return NaN, and that is
        the one way to get a silent ``NaN`` out of adding a mask.

        ``lengths`` has two entries when ``predict_future`` is off; the top-left 2x2
        of the table is ``[[1,0],[1,1]]``, so the truncated mask is well defined
        without a special case.
        """
        if len(lengths) not in (2, 3) or any(length <= 0 for length in lengths):
            raise ValueError(f"segment lengths must be 2 or 3 positive ints, got {lengths}")
        table = torch.tensor(self.SEGMENT_VISIBILITY, dtype=torch.bool, device=device)
        table = table[: len(lengths), : len(lengths)]
        segment_of_token = torch.cat([torch.full((length,), index, dtype=torch.long, device=device)
                                      for index, length in enumerate(lengths)])
        return table[segment_of_token[:, None], segment_of_token[None, :]]

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

        ``action_encoder`` sees two channels per waypoint.  The first is SimWAM's
        ``norm_odo`` of the metric action, which keeps metres and radians comparable and
        lands the term at Wan's latent magnitude without a data-dependent rescale.  The
        second is the same action with its own linear trend removed and divided by
        :data:`SHAPE_RANGE`, and it is the one that distinguishes a Multi-GT row from its
        siblings: the K trajectories of a scene share almost all of their absolute extent
        -- 0.9% of the ``norm_odo`` span -- and differ mainly in curvature, which is
        precisely what detrending leaves behind.  See :func:`detrend_trajectory` for why
        that has to be a per-sample function and not an across-candidate one.

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
        shape = detrend_trajectory(source) / self.shape_range.to(device=source.device, dtype=source.dtype)
        trajectory = self.action_encoder(torch.cat([norm_odo(source), shape], dim=-1))
        mode = self._mode_features(modes).unsqueeze(2)
        ego = self._ego_features(clean_ego).view(source.shape[0], 1, 1, self.hidden_dim)
        return trajectory + ego + self.time_queries.view(1, 1, self.action_tokens, self.hidden_dim) + mode

    def encode_future(self, action: torch.Tensor, clean_ego: torch.Tensor) -> torch.Tensor:
        """NVS F0: the learnable future queries plus the same ego conditioning.

        The trajectory is deliberately *not* an input here.  The future queries sit
        behind the action tokens in the same self-attention, so they read the flow
        state through it -- which is the only conditioning that cannot be
        short-cut from a per-row code, and under the single segment mask it is the
        only thing that makes the register differ between the K Multi-GT rows of one
        scene.  An earlier revision fed the register a projection of the metric
        trajectory plus an optional mode code, and the mode code had to be explicitly
        suppressed in Stage 1 because a constant per-row code let the branch answer
        "which GT am I" without reading the trajectory at all.  Dropping both removes
        the failure mode instead of guarding against it.
        """

        ego = self._ego_features(clean_ego).view(clean_ego.shape[0], 1, 1, self.hidden_dim)
        queries = self.future_queries.view(1, 1, self.future_tokens, self.hidden_dim)
        return (queries + ego).expand(action.shape[0], action.shape[1], -1, -1)

    def forward(
        self,
        action: torch.Tensor,
        future: torch.Tensor | None,
        condition: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the Wan block stack over ``[scene | action | register]``, read both heads.

        ``action``/``future``/``condition`` are flattened per candidate
        (``[B*M, ...]``).  ``context`` is the Scene stream at its own 81-token
        length, already expanded per candidate.  It is embedded once by
        ``text_embedding`` as Wan's ``pre_dit`` does, and that single embedding is
        then used twice, which is deliberate rather than redundant:

        * **prepended into self-attention**, so the scene tokens take part in the
          same block stack and are readable by the action and register queries;
        * **kept as the cross-attention K/V**, so Wan's pretrained
          ``cross_attn.q/k/v/o`` and ``norm_q``/``norm_k`` still receive the
          read-only context they were trained on and the ``strict=True`` payload
          load keeps its meaning.

        There is no stage argument: :data:`SEGMENT_VISIBILITY` is one table for
        every caller, and the invariant it enforces -- Action never reads Register --
        holds identically for Stage 1's register-only forward, Stage 2's flow
        forward and the deployed inference path.  A stage selector here would exist
        only to let one of them silently run a different information flow.  Scene
        reads only itself, so the ActionDiT never writes to the tokens the scorer
        consumes as K/V.  No GT ever enters the sequence: the register slots start
        as ``future_queries`` plus the RMS-clamped ego encoding, and the cached
        Teacher target is only ever compared against ``future_head``'s output.

        ``future=None`` runs ``[scene | action]`` and returns ``None`` for the
        predicted register.  The caller decides this once at construction
        (``predict_future``), and the joint trainer turns it off only when there is
        *neither* a Teacher target cache *nor* a Stage 1 checkpoint to restore:
        appending the unsupervised register tokens at Wan's condition RMS would only
        perturb a pretrained action stream.  A run that does restore Stage 1 keeps
        the tokens and freezes the branch instead, because the trunk was shaped
        around them -- and freezing is by itself enough to keep ``future_head`` out
        of the DDP reducer, which only expects gradients from parameters with
        ``requires_grad=True``.
        """

        embedded = self.text_embedding(context)
        scene_tokens = embedded.shape[1]
        segments = [embedded + self.scene_type_embedding, action + self.action_type_embedding]
        lengths = [scene_tokens, self.action_tokens]
        if future is not None:
            segments.append(future + self.register_type_embedding)
            lengths.append(self.future_tokens)
        tokens = torch.cat(segments, dim=1)
        t_mod = self.time_projection(condition).unflatten(1, (6, self.hidden_dim))
        freqs = self.rope_for_segments(lengths, tokens.device)
        mask = self.segment_attention_mask(lengths, tokens.device)
        x = tokens
        action_start = scene_tokens
        register_start = scene_tokens + self.action_tokens
        for block, gate in zip(self.blocks, self.trajectory_gates):
            x = block(x, embedded, t_mod, freqs, self_attn_mask=mask)
            if future is not None:
                # The gate writes only the register slice: scene and action tokens pass
                # through untouched, so the scorer's K/V and the flow velocity see exactly
                # the sequence the ungated stack produced, plus a register that finally
                # hears the trajectory at full volume instead of ~9% per hop.
                register = x[:, register_start:]
                action = x[:, action_start:register_start]
                x = torch.cat([x[:, :register_start], register + gate(register, action)], dim=1)
        readout = x[:, scene_tokens:]
        velocity = self.head(readout[:, : self.action_tokens], condition)
        if future is None:
            return velocity, None
        return velocity, self.future_head(readout[:, self.action_tokens :])
