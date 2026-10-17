"""Fixed Teacher + the query-extracted future Register target."""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as torch_functional

from .data import FUTURE_FRAMES, REGISTER_QUERIES_PER_FRAME, REGISTER_TOKENS
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
    """Frozen raw DINOv3 Teacher used only during target construction."""

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
        #: The patch/CLS width, which is also the register target's width: the query
        #: extractor returns a convex combination of patch tokens and standardization
        #: is diagonal, so nothing between the backbone and the cache changes it.
        #: Callers build the Student's future head from this rather than hard-coding
        #: 1024, which is a property of the checkpoint and not of the pipeline.
        self.feature_dim = int(self.backbone.embed_dim)
        # Attention is quadratic in the patch grid, so the eight future frames are
        # encoded a few at a time rather than as one batch of eight.
        self.frame_chunk = max(1, int(frame_chunk))
        cfg = getattr(self.backbone, "default_cfg", {})
        self.register_buffer("mean", torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 1, 1, 3), persistent=False)
        self.register_buffer("std", torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 1, 1, 3), persistent=False)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def _check_shape(self, images: torch.Tensor) -> tuple[int, int, int]:
        if images.ndim == 6:
            batch, frames, views = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
        elif images.ndim == 5:
            batch, frames, views = int(images.shape[0]), int(images.shape[1]), 1
        else:
            raise ValueError("Teacher images must be [B,8,V,3,H,W] or [B,8,3,H,W]")
        height, width = int(images.shape[-2]), int(images.shape[-1])
        # No resampling.  The future frames are encoded at the resolution they were
        # generated in, so every patch token summarises real pixels rather than an
        # upsampling filter's guess at pixels that were never rendered.  Positional
        # semantics still line up with the Student's larger grid because DINOv3's
        # RoPE normalises coordinates per axis (``make_coords_dinov3`` divides by H
        # and W separately and maps to [-1, 1]), so a token's rotation depends on
        # where it sits in the frame, not on how many tokens the frame was cut into.
        if (height, width) != self.image_hw:
            raise ValueError(
                f"Teacher refuses to interpolate: got {height}x{width} frames but was "
                f"built for {self.image_hw[0]}x{self.image_hw[1]}. Pass the native resolution."
            )
        return batch, frames, views

    def _families(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, frames, views = self._check_shape(images)
        flat = pad_to_patch_multiple(images.reshape(-1, *images.shape[-3:]))
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
        grid_h, grid_w = self.grid_hw
        if patch_tokens.shape[1] != grid_h * grid_w:
            raise ValueError(f"Teacher patch tokens {patch_tokens.shape[1]} do not match grid {grid_h}x{grid_w}")

        def average_views(tensor: torch.Tensor, count: int) -> torch.Tensor:
            return tensor.reshape(batch, frames, views, count, -1).mean(dim=2)

        return (
            average_views(cls_token, 1),
            average_views(register_tokens, int(register_tokens.shape[1])),
            average_views(patch_tokens, grid_h * grid_w),
        )

    @torch.no_grad()
    def encode_tokens(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(cls, patches)`` as ``[B,F,1,D]`` and ``[B,F,P,D]``, views averaged.

        The token families are handed back *separately*, which is the whole point.
        An earlier revision averaged CLS together with the four DINOv3 register
        tokens and added the result to every patch as a broadcast offset.  That
        cannot survive any per-region reduction: the offset is the same at every
        position, so ``mean(region + c) == mean(region) + c`` contributes an
        identical constant to every region and distinguishes nothing.  It spent
        DINOv3's own pretrained query readout -- measured at 0.2116 on the
        scale-free separation ratio against the quadrant mean's 0.1730, the best
        cheap representation available -- on an additive constant, then threw the
        result away.
        """

        cls_token, _, patch_tokens = self._families(images)
        return cls_token, patch_tokens

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """The raw patch grid ``[B,F,H,W,D]``, for diagnostics that want every token."""

        _, _, patch_tokens = self._families(images)
        batch, frames, _, dim = patch_tokens.shape
        grid_h, grid_w = self.grid_hw
        return patch_tokens.reshape(batch, frames, grid_h, grid_w, dim)


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


#: Cosine-attention temperature of the query readout, and its head count.
#:
#: Neither is a stylistic choice; both were measured on eight real scenes, after the
#: diagonal standardisation production applies, against two numbers: the scale-free
#: separation ratio (within-scene GT distance / between-scene GT distance, higher =
#: more of the target's spread is trajectory rather than scene) and the correlation
#: between target distance and trajectory ADE (higher = the target is actually a
#: function of the trajectory, so Stage 1 can fit it).
#:
#: ======  ========  ==========  ==========  =========  ==========================
#: tau     entropy   separation  traj. corr  traj. rank reading
#: ======  ========  ==========  ==========  =========  ==========================
#: 0.10    0.936     --          --          --         softmax still 93.6% of
#:                                                      uniform: a mean pool, and
#:                                                      it scored *below* the
#:                                                      pooling it replaced
#: 0.02    0.436     0.3593      0.7184      0.7418
#: 0.01    0.320     0.4498      0.7032      0.6626
#: 0.005   0.098     0.5091      0.6890      0.6423     chosen: 2.9x the quadrant
#:                                                      mean it replaced (0.1730)
#:                                                      at 1/1508 of full_patch's
#:                                                      width, keeping 91% of its
#:                                                      correlation (0.7578)
#: 0.003   0.055     0.5357      0.6816      0.6335     separation still climbing,
#:                                                      correlation now falling
#:                                                      toward the random-query
#:                                                      control's 0.4121
#: ======  ========  ==========  ==========  =========  ==========================
#:
#: Eight heads beat one at every tau tested (0.4654 vs 0.4214 at tau=0.005 before
#: standardisation), so different heads really are selecting different regions.
#: Standardisation improved *both* columns at every tau, which is why it is not
#: optional -- see :class:`TargetStandardizer`.
QUERY_TEMPERATURE = 0.005
QUERY_HEADS = 8

#: Ceiling on the query readout's normalized attention entropy, as a fraction of
#: ``log2(P)``, above which the target cache is refused.
#:
#: This is the gate that replaces ``compressor_full_rank``.  That one policed a
#: failure the PCA target could have -- dimensions past the fitted rank were
#: bitwise zero, so "emit zeros" beat learning.  A diagonal standardizer has no
#: rank and cannot fail that way, but the query readout has its own silent failure,
#: and it is the same shape: if the attention comes out uniform the extractor *is*
#: the mean pool it replaced, every downstream number stays finite, and 81% of the
#: GT-discriminating variance is gone without anything raising.
#:
#: 0.75 sits in the gap the tau sweep opened.  The selective operating points all
#: land at or below 0.623 -- 0.098 at the chosen tau=0.005 with one query, 0.320 at
#: 0.01, 0.436 at 0.02, and 0.623 for the four DINOv3 register queries that read each
#: frame since :data:`REGISTER_QUERIES_PER_FRAME` went to 4, which is inherently
#: flatter per query because each of the four covers one region instead of all of
#: them -- while every degenerate one measured at or above 0.936: tau=0.1, the
#: raw-dot-product variant at 0.9987, and the random-query control at 0.999997.
#: Nothing real sits in between, so the threshold is not tuned fine and a cache
#: that trips it is telling you the readout stopped reading.  The earlier 0.5 was
#: calibrated on the single-CLS-query readout alone and would have rejected the
#: four-query readout that measurably tracks trajectory distance better.
MAX_QUERY_ATTENTION_ENTROPY = 0.75


def normalized_attention_entropy(weights: torch.Tensor) -> torch.Tensor:
    """``[..., P]`` attention probabilities -> ``[...]`` entropy over ``log2(P)``.

    Dividing by the uniform maximum is what makes the number comparable across
    patch counts: 1.0 is exactly a mean pool and 0.0 is exactly one patch, so the
    result can be read without knowing ``P`` and audited against
    :data:`MAX_QUERY_ATTENTION_ENTROPY`.
    """

    if weights.shape[-1] < 2:
        raise ValueError("attention entropy needs at least two patches to be meaningful")
    entropy = -(weights.clamp_min(1e-12).log2() * weights).sum(dim=-1)
    return entropy / math.log2(int(weights.shape[-1]))


class FutureQueryExtractor(nn.Module):
    """Reduce each future frame's patch grid to a few tokens by *attention*, never by averaging.

    This replaces ``ordered_quadrant_vectors``, which cut the 29x52 grid into four
    quadrants and mean-pooled each.  A two-way ANOVA on real Teacher features says
    why that cannot work: of the variance distinguishing one GT's future from
    another, **81.0% lives in the GT-specific spatial pattern** and only 19.0% in
    the per-GT average level, and a mean keeps exactly the 19.0% by construction
    (per-scene: 75.1 / 87.1 / 80.1 / 82.3%).  The ordered 2x2 apparatus then bought
    only ~3% relative separation over a single global mean per frame, at four times
    the width.

    **The queries are the frame's own DINOv3 prefix tokens, and the extractor has no
    parameters.**  Both halves of that are forced:

    * *Frozen*, because this defines the training target.  A learned extractor
      would make the target move while the student chases it, and a moving target
      is precisely how a predict/target pair collapses to a constant.  Baking a
      parameter-free readout into the cache makes the target a fixed tensor, so
      "predict the mean" stops being a competitive strategy -- it is only optimal
      if the target itself has no spread.
    * *A backbone prefix token rather than a learned query*, because a randomly
      initialised one is measurably useless.  Its attention entropy was 0.999997 of
      ``log2(P)`` -- exactly uniform -- so it is a mean pool wearing an attention
      costume, and at matched temperature its target distance correlated 0.4121 with
      trajectory ADE against a prefix query's 0.6759.  It reaches a similar
      *separation ratio* (0.4957 vs 0.4934) purely by being scene-blind noise, which
      is why separation alone is not a sufficient statistic and the correlation is
      the number to read.  DINOv3's prefix tokens are already learned queries,
      trained through 24 layers on a billion-scale corpus; nothing fittable on a few
      hundred scenes would beat them.

    Which prefix tokens is measured, not assumed -- ``scripts/probe_future_readout.py``
    on 12 scenes / 48 GTs / 384 frames, standardized, so these are the numbers the
    Stage-1 L1 actually lives in:

    ==========================  ========  =======  ==========================
    readout                     sep       pear    notes
    ==========================  ========  =======  ==========================
    ``patch_mean``              0.1574    --      the pooling this replaced
    ``dinov3_registers``        0.1942    --      the 4 registers read verbatim
    ``cls_plus_registers``      0.1986    --      concatenating them adds nothing
    ``cls_only``                0.2114    0.6881  **CLS alone is not enough**
    ``cls_query_attn``          0.4559    0.6836  CLS as the query, 1 token/frame
    ``register_query_attn``     **0.4581** **0.7709** spearman 0.8402, lift 1.427
    ``cls_query_quadrants``     0.4485    0.7581  4 tokens/frame, ordered 2x2
    ==========================  ========  =======  ==========================

    Reading the four registers *verbatim* is the worst of the token-shaped options,
    but using those same four as queries over the grid is the best: a register token
    is a poor summary of a frame and a good index into it.  They also attend to
    complementary regions -- CLS and the registers pick different patches, which is
    why ``cls_plus_registers`` verbatim does not beat ``cls_only`` while four queries
    do.  So production is :meth:`forward_registers` at
    :data:`data.REGISTER_QUERIES_PER_FRAME` queries per frame, and :meth:`forward`
    stays as the CLS variant the probe measured and as the reference the registers
    path is tested against.

    **The logits are cosine, not dot product.**  DINOv3 features are strongly
    anisotropic, so a shared mean direction dominates every query/patch product and
    a raw dot-product softmax comes out flat: measured entropy 0.9987 of ``log2(P)``
    and separation 0.1414, *worse* than the mean pooling it was meant to replace.
    L2-normalising query and key per head removes that shared component and
    :data:`QUERY_TEMPERATURE` then sets how selective the readout is.  Values stay
    un-normalised -- rescaling every patch to unit length would discard the
    magnitude the target is supposed to carry.

    Heads are split with ``view(B,L,H,D).transpose(1,2)``, the same convention as
    ``wan_dit.CrossAttention._split``: these tensors are ``[B, L, H*D]`` in
    (token, head, dim) order, so a plain ``view(B,H,L,D)`` would reinterpret the
    buffer and hand each head a strided mix of every other head's dimensions.

    The register target's 1:1 correspondence with the eight trajectory poses is now
    per *frame* rather than per token: the queries of one frame are contiguous and
    frame-major, so token ``R * f + q`` is query ``q`` of the frame at
    ``(f + 1) * 0.5 s`` -- the instant pose ``f`` predicts -- and the DiT's register
    RoPE repeats one phase per frame instead of counting
    :data:`data.REGISTER_TOKENS` distinct positions.  ``PairTargets.validate``
    enforces the resulting axis length.
    """

    def __init__(
        self,
        feature_dim: int,
        temperature: float = QUERY_TEMPERATURE,
        heads: int = QUERY_HEADS,
    ) -> None:
        super().__init__()
        if feature_dim % heads:
            raise ValueError(f"feature_dim {feature_dim} is not divisible by {heads} query heads")
        if temperature <= 0.0:
            raise ValueError("the query temperature must be positive")
        self.feature_dim = int(feature_dim)
        self.temperature = float(temperature)
        self.heads = int(heads)
        self.head_dim = self.feature_dim // self.heads

    def _split(self, tensor: torch.Tensor, length: int) -> torch.Tensor:
        batch, _, _ = tensor.shape
        return tensor.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

    @torch.no_grad()
    def attention_weights(self, cls_token: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        """``[B*F, heads, P]``: the distribution each frame's query puts over its patches.

        Exposed separately because the extractor's one failure mode is silent.  If
        the weights come out uniform the readout has degenerated into a mean pool
        and every downstream number still looks finite; the normalised entropy of
        this tensor is what catches it, and it is audited into the target cache.
        """

        query, patches = self._validate(cls_token, patch_tokens)
        logits = self._logits(query, patches)
        return torch_functional.softmax(logits, dim=-1).squeeze(2)

    @torch.no_grad()
    def attention_entropy(self, cls_token: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        """``[B*F, heads]`` normalized entropy, the :data:`MAX_QUERY_ATTENTION_ENTROPY` check.

        Returned per frame and per head rather than pre-averaged: a readout that is
        selective on average can still have dead heads, and the builder audits both
        the mean and the max so a uniformly flat readout and a partly flat one are
        distinguishable in the cache record.
        """

        return normalized_attention_entropy(self.attention_weights(cls_token, patch_tokens))

    def _validate(self, cls_token: torch.Tensor, patch_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if cls_token.ndim != 4 or cls_token.shape[2] != 1:
            raise ValueError(f"cls_token must be [B,F,1,D], got {tuple(cls_token.shape)}")
        if patch_tokens.ndim != 4:
            raise ValueError(f"patch_tokens must be [B,F,P,D], got {tuple(patch_tokens.shape)}")
        if patch_tokens.shape[:2] != cls_token.shape[:2] or patch_tokens.shape[-1] != self.feature_dim:
            raise ValueError(
                f"patch_tokens {tuple(patch_tokens.shape)} do not align with cls_token "
                f"{tuple(cls_token.shape)} at feature_dim {self.feature_dim}"
            )
        rows = cls_token.shape[0] * cls_token.shape[1]
        return (
            cls_token.reshape(rows, 1, self.feature_dim),
            patch_tokens.reshape(rows, patch_tokens.shape[2], self.feature_dim),
        )

    def _logits(self, query: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        key = self._split(patches, patches.shape[1])
        return (
            torch_functional.normalize(self._split(query, query.shape[1]), dim=-1)
            @ torch_functional.normalize(key, dim=-1).transpose(-1, -2)
            / self.temperature
        )

    @torch.no_grad()
    def forward(self, cls_token: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        """``cls [B,F,1,D]`` and ``patches [B,F,P,D]`` -> one ``[B,F,D]`` token per frame."""

        query, patches = self._validate(cls_token, patch_tokens)
        rows, patch_count, _ = patches.shape
        weights = torch_functional.softmax(self._logits(query, patches), dim=-1)
        out = weights @ self._split(patches, patch_count)
        return out.transpose(1, 2).reshape(rows, self.feature_dim).reshape(*cls_token.shape[:2], self.feature_dim)

    @torch.no_grad()
    def forward_registers(
        self, register_tokens: torch.Tensor, patch_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``registers [B,F,R,D]`` and ``patches [B,F,P,D]`` -> ``([B,F,R,D], [B*F*R,heads])``.

        The same cosine attention as :meth:`forward`, with DINOv3's own four prefix
        queries in place of its CLS.  They are the frozen-Teacher equivalent of
        DrivoR's learnable register tokens -- prefix queries read out of a ViT trunk
        -- except these were trained on a billion-scale corpus instead of a few
        hundred driving scenes, and they are used at the Teacher's native 29x52 grid
        rather than DrivoR's resolution.  The second return is each query's attention
        entropy, per query and per head, so the cache audit sees a dead query instead
        of averaging it away.
        """

        # A rank check before the unpack: without it a [B,F,D] tensor -- the CLS
        # path's shape, and the easy mistake when the two readouts sit side by side --
        # fails as "not enough values to unpack", which names neither the argument
        # nor the axis that is missing.
        if register_tokens.ndim != 4:
            raise ValueError(f"register_tokens must be [B,F,R,D], got {tuple(register_tokens.shape)}")
        batch, frames, queries, width = register_tokens.shape
        if queries < 1:
            raise ValueError("the register readout needs at least one query per frame")
        if patch_tokens.ndim != 4 or patch_tokens.shape[:2] != (batch, frames) or (
            patch_tokens.shape[-1] != width or width != self.feature_dim
        ):
            raise ValueError(
                f"patch_tokens {tuple(patch_tokens.shape)} do not align with register_tokens "
                f"{tuple(register_tokens.shape)} at feature_dim {self.feature_dim}"
            )
        rows, patch_count = batch * frames, patch_tokens.shape[2]
        query = register_tokens.reshape(rows, queries, self.feature_dim)
        patches = patch_tokens.reshape(rows, patch_count, self.feature_dim)
        weights = torch_functional.softmax(self._logits(query, patches), dim=-1)
        out = weights @ self._split(patches, patch_count)
        tokens = out.transpose(1, 2).reshape(rows, queries, self.feature_dim)
        entropy = normalized_attention_entropy(weights).permute(0, 2, 1).reshape(rows * queries, self.heads)
        return tokens.reshape(batch, frames, queries, self.feature_dim), entropy


#: A DINOv3 dimension that barely varies across the fit set would be divided by
#: ~0 and turned into pure noise, so each std is floored at this fraction of the
#: largest one.  The count that hit the floor is audited, never silent.
TARGET_STD_FLOOR = 1e-2


@dataclass
class TargetStandardizerState:
    mean: torch.Tensor
    #: Per-dimension std of the fit set, floored.  Diagonal on purpose -- see
    #: :class:`TargetStandardizer`.
    scale: torch.Tensor
    feature_dim: int
    fit_split: str
    source_hash: str
    temperature: float
    heads: int
    clamped_components: int = 0


class TargetStandardizer:
    """Diagonal standardisation of the query-extracted register target.

    Replaces the whitened-PCA ``Ordered2x2Compressor``, and the replacement is the
    point rather than a renaming.  PCA projected the target onto a subspace spanned
    by the *fit rows*, which made the target's rank ``min(target_dim, fit_rows)``
    and every dimension past that rank **bitwise zero**: on a 4-scene fit, rank 128
    of 256 left half the target at exactly 0, so "emit zeros" scored better than
    trying, and the whole ``compressor_full_rank`` gate plus
    ``--allow-rank-deficient-target`` existed only to police that.  A diagonal
    standardizer has no subspace and therefore **no rank at all** -- every one of
    the ``feature_dim`` dimensions keeps its own value, none is ever zeroed, and the
    number of fit scenes affects only how well the mean and std are estimated, not
    whether the target is degenerate.

    Some rescaling is still needed.  An L1 over raw DINOv3 coordinates is almost
    entirely the highest-variance dimensions, and those are scene-level rather than
    trajectory-level, so the loss would spend the model's capacity on the part the
    Scene tokens already provide -- which is exactly the measured Stage 1 failure,
    where a trajectory-blind per-scene-mean memorizer reached 0.5839 against the
    trained register's 0.7515.  Equalising per dimension makes the loss weigh the
    trajectory-discriminating directions comparably.  It is a loss-weighting choice,
    not an information-preserving one, so both the separation ratio and the
    trajectory-distance correlation are re-measured after it and audited.
    """

    def __init__(self, feature_dim: int, temperature: float = QUERY_TEMPERATURE, heads: int = QUERY_HEADS) -> None:
        self.feature_dim = int(feature_dim)
        self.temperature = float(temperature)
        self.heads = int(heads)
        self.state: TargetStandardizerState | None = None

    def fit(self, targets: torch.Tensor, source_hash: str, split: str = "train") -> None:
        """``targets`` is ``[scenes, K, F, D]`` or ``[K, F, D]``; every frame is one row."""

        if split != "train":
            raise ValueError("the target standardizer may only be fit on the train split")
        if targets.ndim < 2:
            raise ValueError("targets must carry a feature dimension")
        if targets.shape[-1] != self.feature_dim:
            raise ValueError(f"targets are {targets.shape[-1]}-wide, standardizer expects {self.feature_dim}")
        flat = targets.reshape(-1, self.feature_dim).float()
        mean = flat.mean(dim=0)
        std = flat.std(dim=0)
        if float(std.max()) <= 0.0:
            raise ValueError("the fit rows carry no variance in any dimension, so no target scale exists")
        floor = TARGET_STD_FLOOR * float(std.max())
        clamped = std < floor
        self.state = TargetStandardizerState(
            mean=mean,
            scale=std.clamp_min(floor),
            feature_dim=self.feature_dim,
            fit_split=split,
            source_hash=source_hash,
            temperature=self.temperature,
            heads=self.heads,
            clamped_components=int(clamped.sum()),
        )

    def transform(self, targets: torch.Tensor) -> torch.Tensor:
        if self.state is None:
            raise RuntimeError("fit the standardizer on train targets before transform")
        if targets.shape[-1] != self.state.feature_dim:
            raise ValueError(
                f"target width {targets.shape[-1]} does not match the fitted {self.state.feature_dim}"
            )
        return (targets - self.state.mean.to(targets.device)) / self.state.scale.to(targets.device)

    def fingerprint(self) -> str:
        """Identity of the frozen target space: two caches agreeing on this are comparable."""

        if self.state is None:
            raise RuntimeError("standardizer is not fit")
        payload = (
            f"{self.state.feature_dim}|{self.state.temperature}|{self.state.heads}|".encode()
            + self.state.mean.detach().cpu().numpy().tobytes()
            + self.state.scale.detach().cpu().numpy().tobytes()
        )
        return hashlib.sha256(payload).hexdigest()


@torch.no_grad()
def query_future_register(
    teacher: TimmDinoV3Teacher,
    extractor: FutureQueryExtractor,
    future_rgb: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One scene's generated futures -> ``([K,32,D] targets, [K*32,heads] entropies)``.

    ``future_rgb`` is the loader's ``[1,K,8,V,3,H,W]``: batch 1, one entry per GT
    trajectory, eight frames at +0.5..+4.0s, ``V`` camera views.  Each GT is encoded
    and reduced on its own and the grid is dropped immediately, because a
    ``[1,8,1508,1024]`` patch grid is 49 MB while its thirty-two query tokens are
    128 KB -- the difference between a 512-scene fit set that stays in RAM and one
    that does not.

    The readout is :meth:`FutureQueryExtractor.forward_registers`: DINOv3's four
    prefix queries as cosine-attention queries, frame-major, so target token
    ``f * R + q`` is query ``q`` of frame ``f`` -- the layout the DiT's register
    segment and its RoPE phase expect.  The entropies come back per query rather
    than pre-averaged for the reason :func:`normalized_attention_entropy` exists:
    the builder audits the mean *and* the max, so a readout with a few dead queries
    is not hidden behind a good average.
    """

    if future_rgb.ndim != 7 or int(future_rgb.shape[2]) != FUTURE_FRAMES:
        raise ValueError(
            f"future_rgb must be [1,K,{FUTURE_FRAMES},V,3,H,W], got {tuple(future_rgb.shape)}"
        )
    blocks: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    for gt_index in range(int(future_rgb.shape[1])):
        images = future_rgb[:, gt_index].to(device)
        _, register_tokens, patch_tokens = teacher._families(images)
        tokens, entropy = extractor.forward_registers(register_tokens, patch_tokens)
        batch, frames, queries, width = tokens.shape
        # The backbone's own prefix count, not a requested one: a checkpoint with a
        # different number of registers would otherwise produce a target whose token
        # axis silently disagrees with REGISTER_TOKENS and with the DiT's query count.
        if frames * queries != REGISTER_TOKENS:
            raise ValueError(
                f"the Teacher returned {queries} registers over {frames} frames; the target "
                f"axis must be {REGISTER_TOKENS} tokens"
            )
        blocks.append(tokens.reshape(batch, REGISTER_TOKENS, width).float().cpu())
        entropies.append(entropy.float().cpu())
        del images, register_tokens, patch_tokens, tokens, entropy
    return torch.cat(blocks, dim=0), torch.cat(entropies, dim=0)


def _mean_l1(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """The ``mean_paired_future_l1`` convention: mean over features, then over tokens."""

    return (left - right).abs().mean(dim=-1).mean(dim=-1)


def _rank(values: torch.Tensor) -> torch.Tensor:
    """Ordinal ranks.  Ties are not averaged because the inputs are continuous."""

    return values.argsort().argsort().float()


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left, right = left - left.mean(), right - right.mean()
    denominator = float(left.norm() * right.norm())
    return float(left @ right) / denominator if denominator > 0 else float("nan")


def measure_target_quality(
    blocks: Sequence[torch.Tensor],
    trajectories: Sequence[torch.Tensor],
) -> dict[str, Any]:
    """How much of a target's spread is trajectory rather than scene.

    ``blocks`` is one ``[K,32,D]`` tensor per scene and ``trajectories`` the matching
    ``[K,8,3]`` metric GT, both in the space the loss lives in -- for production that
    means *after* :class:`TargetStandardizer`, since a per-dimension affine map moves
    every number below.  Distances average over the whole register axis, so a GT pair
    whose futures differ in one region out of four scores a quarter of what one that
    differs everywhere does; that is the intended weighting, since the loss reduces the
    same way.

    Two numbers, because one is not a sufficient statistic.  The **separation ratio**
    (within-scene across-GT distance over across-scene distance) says how much of the
    target's variance distinguishes two futures of the same scene from the scene
    itself.  It can be made large by throwing the scene away, which is exactly what
    the random-query control did: it reached the CLS query's ratio (0.4957 against
    0.4934) while its distance correlated only 0.4121 with trajectory ADE against
    0.6647.  So the **correlation** between target distance and trajectory ADE is read
    alongside it -- together they say "keeps the trajectory signal", where either
    alone only says "discards something".

    Audited, never gated.  The rank gate this replaced had a computable failure
    threshold: past the fitted rank the PCA target was bitwise zero, so "emit zeros"
    had a definite score and beat it.  Neither number here has one.  A separation
    ratio estimated from a few hundred scenes moves by more than any threshold worth
    writing, and a low correlation has no trivial-predictor baseline to compare
    against.  Turning them into refusals would mostly fire on noise, so they are
    recorded where a bad run can be diagnosed from the cache record instead.
    Attention entropy is the exception, and :data:`MAX_QUERY_ATTENTION_ENTROPY` says
    why.
    """

    within: list[float] = []
    across: list[float] = []
    trajectory_ade: list[float] = []
    for block, gt in zip(blocks, trajectories):
        for i, j in itertools.combinations(range(int(block.shape[0])), 2):
            within.append(float(_mean_l1(block[i], block[j])))
            trajectory_ade.append(float((gt[i, :, :2] - gt[j, :, :2]).norm(dim=-1).mean()))
    for left_block, right_block in itertools.combinations([block for block in blocks], 2):
        for index in range(min(int(left_block.shape[0]), int(right_block.shape[0]))):
            across.append(float(_mean_l1(left_block[index], right_block[index])))
    if not within or not across:
        return {
            "within_scene_gt_distance": float("nan"),
            "across_scene_gt_distance": float("nan"),
            "separation_ratio": float("nan"),
            "trajectory_distance_pearson": float("nan"),
            "trajectory_distance_spearman": float("nan"),
            "quality_pairs": len(within),
            "quality_scene_pairs": len(across),
        }
    within_tensor = torch.tensor(within)
    across_mean = float(torch.tensor(across).mean())
    ade = torch.tensor(trajectory_ade)
    return {
        "within_scene_gt_distance": float(within_tensor.mean()),
        "across_scene_gt_distance": across_mean,
        "separation_ratio": float(within_tensor.mean()) / across_mean if across_mean else float("nan"),
        "trajectory_distance_pearson": _pearson(within_tensor, ade),
        "trajectory_distance_spearman": _pearson(_rank(within_tensor), _rank(ade)),
        "quality_pairs": len(within),
        "quality_scene_pairs": len(across),
    }
