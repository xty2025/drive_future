"""Which future-frame readout should Stage 1 predict: CLS, or several register tokens?

Answers one question with numbers rather than with taste: is a single token per future
frame enough to carry the future, and if not, how many tokens bought the DrivoR way --
learned prefix queries prepended to the backbone's own sequence and sliced back out
after the trunk -- are worth it?

DrivoR's ``scene_embeds`` are ``nn.Parameter`` queries trained with the model
(``vendor/DrivoR/.../drivor_model.py:56``, concatenated in ``dinov2_lora.py:61``,
sliced back out at ``:401``).  A *target* extractor cannot have those: a learned query
makes the target move while the student chases it, and a moving target is how a
predict/target pair collapses to a constant.  What DINOv3 offers instead is the same
mechanism already trained, on a billion-scale corpus: ``x_norm_regtokens`` is four
prefix queries that ran through the same 24 blocks as the patches.
``TimmDinoV3Teacher._families`` computes them and ``encode_tokens`` throws them away, so
candidates (4) and (5) below cost one extra matmul, not one extra backbone pass.

Candidates, all at the Teacher's own width ``D = feature_dim`` per token:

===  =======================  ======  =====================================================
 #   name                     tokens  what it is
===  =======================  ======  =====================================================
 1   ``patch_mean``              1    control: the mean pool the query readout replaced
 2   ``cls_only``                1    ``x_norm_clstoken`` verbatim -- "is CLS enough"
 3   ``cls_query_attn``          1    production: CLS as query, cosine attention over 1508
 4   ``dinov3_registers``        4    ``x_norm_regtokens`` verbatim
 5   ``register_query_attn``     4    each DINOv3 register used as a query over the patches
 6   ``cls_query_quadrants``     4    the CLS-query attention restricted to each quadrant
 7   ``cls_plus_registers``      5    (2) and (4) side by side
===  =======================  ======  =====================================================

Three measurements, because no single number separates "carries the future" from
"carries noise":

* **target quality** -- :func:`measure_target_quality`'s separation ratio and
  trajectory-distance correlation, on the standardized target, which is the space the
  Stage-1 L1 lives in;
* **degeneracy** -- constant dimensions and numeric rank, so a candidate that wins by
  discarding information is visible;
* **where it looks** -- the attention map correlated against the *inter-GT motion map*
  of the same frame, i.e. the pixels that actually differ between two futures of one
  scene.  This is the direct answer to "do the attended regions contain future
  information": a readout whose mass sits on motion-free pixels is reading the scene,
  which the Scene tokens already provide.

Bounded on purpose.  Default six scenes, one GPU, a few minutes; it encodes each
scene's futures once and reuses the tokens for every candidate.

    source /usr/local/PPU_SDK/envsetup.sh ppu
    export PYTHONPATH=<repo>/src:<repo>/vendor/DrivoR:<repo>/vendor/DrivoR/nuplan-devkit
    python scripts/probe_future_readout.py --scenes 6
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as torch_functional

from nvs_v6_wan_e2e.multiview_data import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_LOG_ROOT,
    DEFAULT_MULTIVIEW_ROOT,
    DEFAULT_SENSOR_ROOT,
    iter_ready_scenes,
    load_multiview_sample,
    scene_labels,
)
from nvs_v6_wan_e2e.teacher import (
    MAX_QUERY_ATTENTION_ENTROPY,
    NATIVE_FUTURE_IMAGE_HW,
    QUERY_HEADS,
    QUERY_TEMPERATURE,
    FutureQueryExtractor,
    TargetStandardizer,
    TimmDinoV3Teacher,
    measure_target_quality,
    normalized_attention_entropy,
)

DEFAULT_DINO_CHECKPOINT = "/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors"
DEFAULT_OUT = "artifacts/future_readout_probe"

#: Frames rendered as figures: +0.5 s, +2.0 s, +4.0 s -- the near, middle and far end of
#: the horizon, where the inter-GT motion is smallest, middling and largest.
FIGURE_FRAMES = (0, 3, 7)

#: The production readout is the reference every other candidate is judged against.
BASELINE = "cls_query_attn"

#: With one GT there is no "which future was taken" signal and every cross-GT number is 0/0.
MIN_GT_PER_SCENE = 2


# --------------------------------------------------------------------------- #
# readouts
# --------------------------------------------------------------------------- #


def _quadrant_index(grid_hw: tuple[int, int], device: torch.device) -> torch.Tensor:
    """``[P]`` in ``{0,1,2,3}``: which quadrant of the patch grid each token sits in."""

    grid_h, grid_w = grid_hw
    rows = torch.arange(grid_h, device=device) >= (grid_h + 1) // 2
    cols = torch.arange(grid_w, device=device) >= (grid_w + 1) // 2
    return (rows[:, None].long() * 2 + cols[None, :].long()).reshape(-1)


def _weighted_values(extractor: FutureQueryExtractor, weights: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
    """``[rows, heads, P]`` probabilities over ``[rows, P, D]`` patches -> ``[rows, D]``.

    Values stay un-normalised and the head outputs are concatenated rather than summed,
    which is what ``FutureQueryExtractor.forward`` does; reproducing it here keeps the
    quadrant and register variants on the same footing as production.
    """

    rows, patch_count, _ = patches.shape
    values = extractor._split(patches, patch_count)
    # ``unsqueeze(2)`` restores the length-1 query axis production's softmax keeps.
    # Without it matmul reads ``[rows, heads]`` as a batch pair and broadcasts, which
    # returns ``[rows, heads, D]`` instead of ``[rows, D]`` -- finite, wrong, silent.
    return (weights.unsqueeze(2) @ values).transpose(1, 2).reshape(rows, extractor.feature_dim)


def _attention(
    extractor: FutureQueryExtractor,
    query: torch.Tensor,
    patches: torch.Tensor,
    keep: torch.Tensor | None = None,
) -> torch.Tensor:
    """Softmax weights ``[rows, heads, P]`` for an arbitrary query, optionally masked.

    ``keep`` restricts the softmax to a subset of the grid.  That is the whole
    difference between candidate (3) and candidate (6): the same query, the same
    temperature, but each quadrant renormalised on its own so four different regions of
    one frame produce four different tokens.
    """

    rows = query.shape[0] * query.shape[1]
    logits = extractor._logits(query.reshape(rows, 1, -1), patches.reshape(rows, patches.shape[2], -1))
    if keep is not None:
        logits = logits.masked_fill(~keep.view(1, 1, 1, -1), float("-inf"))
    return torch_functional.softmax(logits, dim=-1).squeeze(2)


def token_saliency(query_tokens: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
    """``[K,8,Q,P]`` cosine similarity of each prefix token against every patch.

    CLS and the four registers are read out of the trunk, not out of an attention layer
    this script can instrument, so "where does token q look" has no weight matrix to
    read.  The cosine map is the observable proxy: the patches a prefix token is closest
    to in feature space.  It is what the figures' second row renders.
    """

    query = torch_functional.normalize(query_tokens, dim=-1)
    key = torch_functional.normalize(patch_tokens, dim=-1)
    return torch.einsum("kfqd,kfpd->kfqp", query, key)


def build_candidates(
    extractor: FutureQueryExtractor,
    cls_token: torch.Tensor,
    register_tokens: torch.Tensor,
    patch_tokens: torch.Tensor,
    grid_hw: tuple[int, int],
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor | None]], torch.Tensor]:
    """One GT's Teacher tokens -> ``({name: ([1,8,T,D] target, [1,8,P] map)}, [1,8,R,P])``.

    The attention map is returned only for candidates whose tokens really are an
    attention readout of the grid; ``(2)``, ``(4)`` and ``(7)`` have no such map, and
    their "where does it look" answer is :func:`token_saliency` instead.  Maps are
    averaged over heads, which is what production's entropy audit is computed on.  The
    second element keeps the four register-query maps *un*averaged, because the figure
    has to show them one per panel and every metric wants them collapsed.
    """

    gt_count, frames = cls_token.shape[0], cls_token.shape[1]
    flat_patches = patch_tokens.reshape(-1, *patch_tokens.shape[2:])
    quadrant = _quadrant_index(grid_hw, patch_tokens.device)

    def flatten(tokens: torch.Tensor) -> torch.Tensor:
        """``[K,8,T,D]`` -> ``[K,8,T*D]``, the shape :func:`measure_target_quality` wants.

        Concatenating rather than summing is deliberate: the Stage-1 L1 over this is a
        *mean* over elements, so widening the target must not silently multiply the loss
        by ``T``.  Any trainer fed these targets has to keep that reduction.
        """

        return tokens.reshape(gt_count, frames, -1)

    cls_query_weights = _attention(extractor, cls_token, patch_tokens)
    cls_query = _weighted_values(extractor, cls_query_weights, flat_patches).reshape(gt_count, frames, 1, -1)
    production = extractor(cls_token, patch_tokens)
    if not torch.allclose(cls_query.squeeze(2), production, atol=1e-5, rtol=1e-4):
        raise AssertionError(
            "the re-implemented readout drifted from FutureQueryExtractor.forward, so every "
            f"candidate below would be measured against a different baseline "
            f"(max|diff|={float((cls_query.squeeze(2) - production).abs().max()):.3e})"
        )

    register_count = register_tokens.shape[2]
    # ``_attention`` has already folded ``(GT, frame)`` into one row axis, so the
    # register axis goes in at dim 1 and every later reshape runs through ``rows``.
    register_weights = torch.stack(
        [_attention(extractor, register_tokens[:, :, index : index + 1], patch_tokens) for index in range(register_count)],
        dim=1,
    )  # [rows, R, heads, P]
    register_query = torch.stack(
        [
            _weighted_values(extractor, register_weights[:, index], flat_patches).reshape(gt_count, frames, -1)
            for index in range(register_count)
        ],
        dim=2,
    )  # [K,8,R,D]

    quadrant_tokens, quadrant_maps = [], []
    for index in range(4):
        weights = _attention(extractor, cls_token, patch_tokens, quadrant == index)
        quadrant_tokens.append(_weighted_values(extractor, weights, flat_patches).reshape(gt_count, frames, -1))
        quadrant_maps.append(weights.mean(dim=1).reshape(gt_count, frames, -1))
    # Each quadrant map is a distribution over its own quadrant, so dividing by four
    # makes their union a distribution over the whole grid again and comparable with the
    # single-query maps.
    quadrant_map = torch.stack(quadrant_maps, dim=2).sum(dim=2) / 4.0

    candidates = {
        "patch_mean": (flatten(patch_tokens.mean(dim=2, keepdim=True)), None),
        "cls_only": (flatten(cls_token), None),
        BASELINE: (flatten(cls_query), cls_query_weights.mean(dim=1).reshape(gt_count, frames, -1)),
        "dinov3_registers": (flatten(register_tokens), None),
        "register_query_attn": (flatten(register_query), register_weights.mean(dim=(1, 2)).reshape(gt_count, frames, -1)),
        "cls_query_quadrants": (flatten(torch.stack(quadrant_tokens, dim=2)), quadrant_map),
        "cls_plus_registers": (flatten(torch.cat([cls_token, register_tokens], dim=2)), None),
    }
    # Every reshape above is a silent-wrong-answer opportunity: a mis-ordered stack or a
    # dropped head axis still produces finite numbers, just about a different readout.
    patch_count, width = patch_tokens.shape[2], extractor.feature_dim
    rows = gt_count * frames
    for name, (target, attention) in candidates.items():
        tokens = target.shape[-1] // width
        if target.shape[:2] != (gt_count, frames) or target.shape[-1] != tokens * width or tokens < 1:
            raise AssertionError(
                f"{name}: target is {tuple(target.shape)}, expected "
                f"[{gt_count},{frames},T*{width}] for some T >= 1"
            )
        if attention is not None and tuple(attention.shape) != (gt_count, frames, patch_count):
            raise AssertionError(
                f"{name}: attention map is {tuple(attention.shape)}, expected "
                f"{(gt_count, frames, patch_count)}"
            )
    register_maps = register_weights.mean(dim=2)
    if tuple(register_maps.shape) != (rows, register_count, patch_count):
        raise AssertionError(
            f"per-register maps are {tuple(register_maps.shape)}, expected "
            f"{(rows, register_count, patch_count)}"
        )
    return candidates, register_maps


# --------------------------------------------------------------------------- #
# degeneracy and spatial diagnostics
# --------------------------------------------------------------------------- #


def degeneracy_report(blocks: list[torch.Tensor]) -> dict[str, Any]:
    """Constant dimensions and numeric rank of a candidate's standardized target.

    Both exist to catch the failure the retired PCA target had, where dimensions past
    the fitted rank were *bitwise zero* and "emit zeros" beat learning.  A diagonal
    standardizer cannot produce that, but a readout can still collapse: if all four
    register tokens of a frame came out equal, ``dinov3_registers`` would be a 1-token
    target wearing a 4-token shape, and only the rank would say so.
    """

    rows = torch.cat([block.reshape(-1, block.shape[-1]) for block in blocks], dim=0).double()
    centred = rows - rows.mean(dim=0, keepdim=True)
    std = centred.std(dim=0)
    singular = torch.linalg.svdvals(centred)
    largest = float(singular[0])
    rank = int((singular > 1e-6 * largest).sum()) if largest > 0 else 0
    return {
        "rows": int(rows.shape[0]),
        "width": int(rows.shape[1]),
        "constant_dims": int((std == 0).sum()),
        "numeric_rank": rank,
        "rank_over_min_dim": round(rank / min(rows.shape), 4),
        "std_ratio_min_over_max": round(float(std.min() / std.max()), 6) if float(std.max()) > 0 else 0.0,
    }


def motion_maps(future_rgb: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    """``[K,8,P]``: how much each patch of a frame differs from the *other* GTs' frames.

    ``future_rgb`` is the loader's ``[1,K,8,V,3,H,W]`` in ``[0,1]``.  The map for GT
    ``k`` averages ``|I_k - I_k'|`` over ``k' != k``, views and channels, then pools
    16x16 down to the Teacher's grid, so one entry is "how much does this patch's future
    depend on which trajectory was taken".  It is the ground truth the attention maps are
    scored against: a readout that puts its mass where this is large is reading the
    future, and one that does not is reading the scene.
    """

    images = future_rgb[0].float()
    gt_count, frames, views = images.shape[0], images.shape[1], images.shape[2]
    grid_h, grid_w = grid_hw
    kernel = (images.shape[-2] // grid_h, images.shape[-1] // grid_w)
    flat = images.reshape(gt_count, frames, views, 3, images.shape[-2], images.shape[-1])
    rows = []
    for index in range(gt_count):
        others = [slot for slot in range(gt_count) if slot != index]
        if not others:
            rows.append(torch.zeros(frames, grid_h * grid_w))
            continue
        stacked = torch.stack([flat[index] - flat[other] for other in others], dim=0)
        magnitude = stacked.abs().mean(dim=(0, 2, 3))  # [others,frames,views,3,H,W] -> [frames,H,W]
        pooled = torch_functional.avg_pool2d(magnitude.unsqueeze(1), kernel_size=kernel)
        rows.append(pooled.squeeze(1).reshape(frames, grid_h * grid_w))
    return torch.stack(rows, dim=0)


def spatial_report(attention: torch.Tensor, motion: torch.Tensor, top_fraction: float = 0.1) -> dict[str, float]:
    """Does the readout look where the future actually differs between trajectories?

    ``attention`` and ``motion`` are both ``[K,8,P]`` non-negative.  Two numbers:

    * ``motion_pearson`` -- correlation of the two maps.  Zero means the readout's
      preference is unrelated to which trajectory was taken.
    * ``mass_on_moving_top_lift`` -- attention mass inside the top ``top_fraction`` of
      patches by motion, divided by ``top_fraction``.  Read as a lift: 1.0 is chance.
    """

    flat_attention = attention.reshape(-1, attention.shape[-1]).double()
    flat_motion = motion.reshape(-1, motion.shape[-1]).double()
    correlations, lifts = [], []
    for row_attention, row_motion in zip(flat_attention, flat_motion):
        if float(row_attention.std()) <= 0 or float(row_motion.std()) <= 0:
            continue
        centred_attention = row_attention - row_attention.mean()
        centred_motion = row_motion - row_motion.mean()
        denominator = float(centred_attention.norm() * centred_motion.norm())
        if denominator <= 0:
            continue
        correlations.append(float(centred_attention @ centred_motion) / denominator)
        kept = max(1, int(round(top_fraction * row_motion.numel())))
        moving = torch.zeros_like(row_motion, dtype=torch.bool)
        moving[row_motion.topk(kept).indices] = True
        total = float(row_attention.sum())
        if total > 0:
            lifts.append(float(row_attention[moving].sum()) / total / top_fraction)
    if not correlations:
        return {"motion_pearson": float("nan"), "mass_on_moving_top_lift": float("nan"), "spatial_rows": 0}
    return {
        "motion_pearson": round(sum(correlations) / len(correlations), 4),
        "mass_on_moving_top_lift": round(sum(lifts) / len(lifts), 4) if lifts else float("nan"),
        "spatial_rows": len(correlations),
    }


def attention_concentration(attention: torch.Tensor) -> dict[str, float]:
    """Entropy and top-32 mass of an attention map, over every frame and head.

    The entropy is the same quantity :data:`MAX_QUERY_ATTENTION_ENTROPY` gates the
    production cache on, so a candidate that scores well here but trips that ceiling has
    quietly become the mean pool it was meant to replace.
    """

    flat = attention.reshape(-1, attention.shape[-1]).float()
    entropy = normalized_attention_entropy(flat)
    kept = min(32, flat.shape[-1])
    top = flat.topk(kept, dim=-1).values.sum(dim=-1) / flat.sum(dim=-1).clamp_min(1e-12)
    return {
        "attention_entropy_mean": round(float(entropy.mean()), 4),
        "attention_entropy_max": round(float(entropy.max()), 4),
        "entropy_ceiling": MAX_QUERY_ATTENTION_ENTROPY,
        "top32_mass_mean": round(float(top.mean()), 4),
    }


def cross_gt_map_report(attention: torch.Tensor) -> dict[str, float]:
    """How far the attention map itself moves between two GTs of one scene.

    A readout can have a sharp, low-entropy map and still be scene-static -- sharp on
    the *same* pixels for every trajectory.  The total-variation distance between the
    maps of GT ``k`` and ``k'`` of one frame separates the two: near zero means the
    readout looks at the scene, and its tokens then differ only through whatever the
    values at those fixed pixels carry.
    """

    distances = []
    for index, other in itertools.combinations(range(attention.shape[0]), 2):
        left = attention[index] / attention[index].sum(dim=-1, keepdim=True).clamp_min(1e-12)
        right = attention[other] / attention[other].sum(dim=-1, keepdim=True).clamp_min(1e-12)
        distances.append(0.5 * (left - right).abs().sum(dim=-1))
    if not distances:
        return {"attention_tv_across_gt": float("nan")}
    return {"attention_tv_across_gt": round(float(torch.stack(distances).mean()), 4)}


def mean_record(records: list[dict[str, float]]) -> dict[str, float]:
    """Per-scene diagnostics -> one row, averaging only the finite entries."""

    out: dict[str, float] = {}
    for key in records[0]:
        values = [float(record[key]) for record in records if record[key] == record[key]]
        out[key] = round(sum(values) / len(values), 4) if values else float("nan")
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def _overlay(axis, image: torch.Tensor, heat: torch.Tensor, grid_hw: tuple[int, int], title: str) -> None:
    """One frame with one ``[P]`` map upsampled over it."""

    axis.imshow(image.permute(1, 2, 0).clamp(0.0, 1.0).numpy())
    grid_h, grid_w = grid_hw
    resized = torch_functional.interpolate(
        heat.reshape(1, 1, grid_h, grid_w).float(),
        size=(image.shape[-2], image.shape[-1]),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    resized = (resized - resized.min()) / (resized.max() - resized.min()).clamp_min(1e-12)
    axis.imshow(resized.numpy(), cmap="inferno", alpha=0.55)
    axis.set_title(title, fontsize=7)
    axis.set_xticks([]), axis.set_yticks([])


def save_readout_figure(
    path: Path,
    scene_id: str,
    frame_image: torch.Tensor,
    cls_attention: torch.Tensor,
    register_attention: torch.Tensor,
    saliency: torch.Tensor,
    grid_hw: tuple[int, int],
    frame_index: int,
) -> None:
    """Two rows for one frame: the attention readouts, then the prefix tokens' saliency.

    Row 1 is ``frame | CLS-query attention | 4 register-query attention maps`` -- what
    production selects against what DINOv3's own four prefix queries select, at the same
    temperature.  Row 2 is ``frame | CLS saliency | 4 register saliencies``, i.e. the
    cosine of each prefix token against the whole 29x52 grid, which is the only view
    available for tokens read out of the trunk rather than out of an attention layer.
    """

    panels = 2 + register_attention.shape[0]
    widths = {cls_attention.shape[-1], register_attention.shape[-1], saliency.shape[-1]}
    if cls_attention.ndim != 1 or register_attention.ndim != 2 or saliency.shape[0] != panels - 1 or len(widths) != 1:
        raise ValueError(
            f"the readout figure needs [P] CLS maps, [R,P] register maps and [R+1,P] saliencies, got "
            f"{tuple(cls_attention.shape)}, {tuple(register_attention.shape)} and {tuple(saliency.shape)}"
        )
    figure, axes = plt.subplots(2, panels, figsize=(2.3 * panels, 5.0))
    frame_title = f"future +{(frame_index + 1) * 0.5:.1f}s"
    for row in (0, 1):
        axes[row, 0].imshow(frame_image.permute(1, 2, 0).clamp(0.0, 1.0).numpy())
        axes[row, 0].set_title(frame_title, fontsize=7)
        axes[row, 0].set_xticks([]), axes[row, 0].set_yticks([])
    _overlay(axes[0, 1], frame_image, cls_attention, grid_hw, "CLS-query attention (production)")
    _overlay(axes[1, 1], frame_image, saliency[0], grid_hw, "CLS cosine saliency")
    for index in range(register_attention.shape[0]):
        _overlay(axes[0, 2 + index], frame_image, register_attention[index], grid_hw, f"register {index} query attention")
        _overlay(axes[1, 2 + index], frame_image, saliency[index + 1], grid_hw, f"register {index} cosine saliency")
    figure.suptitle(f"{scene_id}  GT0   top row: attention readouts   bottom row: prefix-token saliency", fontsize=8)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(path, dpi=105)
    plt.close(figure)


def save_cross_gt_figure(
    path: Path,
    scene_id: str,
    frame_images: torch.Tensor,
    attention: torch.Tensor,
    motion: torch.Tensor,
    grid_hw: tuple[int, int],
    frame_index: int,
) -> None:
    """The CLS-query map under each GT of one scene, plus the inter-GT motion map.

    Read left to right: if the hot region moves across the first ``K`` panels the readout
    is tracking the trajectory; if it is pinned to the same pixels in all of them while
    the last panel is bright elsewhere, it is reading the scene and the token differences
    come only from the values at those fixed pixels.
    """

    count = attention.shape[0]
    figure, axes = plt.subplots(1, count + 1, figsize=(2.3 * (count + 1), 2.6))
    for index in range(count):
        _overlay(axes[index], frame_images[index], attention[index], grid_hw, f"GT {index} CLS-query attention")
    _overlay(axes[count], frame_images[0], motion[0], grid_hw, "inter-GT motion (target)")
    figure.suptitle(f"{scene_id}  frame +{(frame_index + 1) * 0.5:.1f}s: does the attended region follow the trajectory?", fontsize=8)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    figure.savefig(path, dpi=105)
    plt.close(figure)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", type=int, default=6, help="scenes to encode; every candidate is measured on the same ones")
    parser.add_argument("--start", type=int, default=0, help="offset into the ready-scene index")
    parser.add_argument("--multiview-root", default=DEFAULT_MULTIVIEW_ROOT)
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    parser.add_argument("--sensor-root", default=DEFAULT_SENSOR_ROOT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--dino-checkpoint", default=DEFAULT_DINO_CHECKPOINT)
    parser.add_argument("--teacher-image-hw", type=int, nargs=2, default=list(NATIVE_FUTURE_IMAGE_HW))
    parser.add_argument("--query-temperature", type=float, default=QUERY_TEMPERATURE)
    parser.add_argument("--query-heads", type=int, default=QUERY_HEADS)
    parser.add_argument("--figures", type=int, default=2, help="scenes to render; 0 writes the JSON only")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.out = str(Path(args.out))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    return args


def load_sample(args: argparse.Namespace, scene_dir: Path, log_cache: dict[str, list[dict]]):
    """The production loader path, including its navsim log pickle."""

    metadata = json.loads((scene_dir / "scene_metadata.json").read_text())
    log_name = str(metadata["log_name"])
    if log_name not in log_cache:
        with (Path(args.log_root) / f"{log_name}.pkl").open("rb") as handle:
            log_cache[log_name] = pickle.load(handle)
        while len(log_cache) > 2:
            log_cache.pop(next(iter(log_cache)))
    return load_multiview_sample(
        scene_dir,
        log_root=args.log_root,
        sensor_root=args.sensor_root,
        cache_root=args.cache_root,
        future_image_hw=tuple(args.teacher_image_hw),
        log_frames=log_cache[log_name],
    )


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    teacher = TimmDinoV3Teacher(checkpoint=args.dino_checkpoint, image_hw=tuple(args.teacher_image_hw)).to(device)
    extractor = FutureQueryExtractor(teacher.feature_dim, args.query_temperature, args.query_heads).to(device)
    grid_hw = teacher.grid_hw
    print(json.dumps({
        "event": "start",
        "teacher_feature_dim": teacher.feature_dim,
        "teacher_num_prefix_tokens": teacher.num_prefix_tokens,
        "grid_hw": list(grid_hw),
        "patches": grid_hw[0] * grid_hw[1],
        "query_temperature": args.query_temperature,
        "query_heads": args.query_heads,
        "scenes_requested": args.scenes,
        "device": str(device),
    }), flush=True)

    scene_dirs: list[Path] = []
    skipped_thin = 0
    for scene_dir in iter_ready_scenes(
        args.multiview_root, args.cache_root, limit=None, start=args.start, require_distinct_labels=True
    ):
        if len(scene_labels(scene_dir)) < MIN_GT_PER_SCENE:
            skipped_thin += 1
            continue
        scene_dirs.append(scene_dir)
        if len(scene_dirs) >= args.scenes:
            break
    if not scene_dirs:
        raise RuntimeError(f"no ready scene with >= {MIN_GT_PER_SCENE} distinct GTs under {args.multiview_root}")
    print(json.dumps({
        "event": "scenes_selected",
        "selected": len(scene_dirs),
        "skipped_fewer_than_min_gt": skipped_thin,
        "min_gt_per_scene": MIN_GT_PER_SCENE,
    }), flush=True)

    log_cache: dict[str, list[dict]] = {}
    per_candidate: dict[str, list[torch.Tensor]] = {}
    spatial_rows: dict[str, list[dict[str, float]]] = {}
    trajectories: list[torch.Tensor] = []
    scene_ids: list[str] = []
    figure_scenes: list[dict[str, Any]] = []
    started = time.time()

    for position, scene_dir in enumerate(scene_dirs):
        sample = load_sample(args, scene_dir, log_cache)
        future_rgb = sample.future_rgb
        if future_rgb is None:
            raise RuntimeError(f"{sample.scene_id} was loaded without future RGB")
        gt_count = int(future_rgb.shape[1])
        if int(future_rgb.shape[3]) != 1:
            # The Teacher averages views, so with more than one there is no image the
            # reported attention map is actually a map of.
            raise RuntimeError(f"{sample.scene_id} has {future_rgb.shape[3]} future views, expected 1")
        # Computed from pixels before anything moves to the GPU, so the decoded frames
        # can be dropped as soon as the tokens exist.
        motion = motion_maps(future_rgb, grid_hw)
        tokens_by_gt: list[dict[str, torch.Tensor]] = []
        maps_by_gt: list[dict[str, torch.Tensor | None]] = []
        saliency_by_gt: list[torch.Tensor] = []
        register_maps_gt0: torch.Tensor | None = None
        for gt_index in range(gt_count):
            images = future_rgb[:, gt_index].to(device)
            cls_token, register_tokens, patch_tokens = teacher._families(images)
            built, register_maps = build_candidates(extractor, cls_token, register_tokens, patch_tokens, grid_hw)
            if gt_index == 0:
                register_maps_gt0 = register_maps.float().cpu()
            tokens_by_gt.append({name: value[0].float().cpu() for name, value in built.items()})
            maps_by_gt.append({name: (None if value[1] is None else value[1].float().cpu()) for name, value in built.items()})
            # CLS first, then the four registers, so saliency[:, :, 0] is the CLS map and
            # [:, :, 1:] are the register maps -- the two rows of the figure line up.
            saliency_by_gt.append(token_saliency(torch.cat([cls_token, register_tokens], dim=2), patch_tokens).float().cpu())
            del images, cls_token, register_tokens, patch_tokens, built, register_maps

        names = sorted(tokens_by_gt[0])
        for name in names:
            per_candidate.setdefault(name, []).append(
                torch.cat([tokens_by_gt[gt_index][name] for gt_index in range(gt_count)], dim=0)
            )
            joined = [maps_by_gt[gt_index][name] for gt_index in range(gt_count)]
            if joined[0] is not None:
                attention = torch.cat(joined, dim=0)
                record = spatial_report(attention, motion)
                record.update(cross_gt_map_report(attention))
                record.update(attention_concentration(attention))
                spatial_rows.setdefault(name, []).append(record)
        trajectories.append(sample.targets.gt_trajectory[0].float().cpu())
        scene_ids.append(sample.scene_id)
        if position < args.figures:
            slots = [index for index in FIGURE_FRAMES if index < int(future_rgb.shape[2])]
            figure_scenes.append({
                "scene_id": sample.scene_id,
                "slots": slots,
                "frames": future_rgb[0][:, slots, 0].cpu(),
                "motion": motion[:, slots],
                "register_maps": register_maps_gt0[slots],
                "attention": {
                    name: torch.cat([maps_by_gt[gt_index][name] for gt_index in range(gt_count)], dim=0)[:, slots]
                    for name in names if maps_by_gt[0][name] is not None
                },
                "saliency": saliency_by_gt[0][:, slots],
            })
        print(json.dumps({
            "event": "scene", "scene": sample.scene_id, "gt": gt_count,
            "scene_index": position + 1, "scenes": len(scene_dirs),
            "elapsed_s": round(time.time() - started, 1),
        }), flush=True)
        del sample, future_rgb, tokens_by_gt, maps_by_gt, saliency_by_gt, motion, register_maps_gt0

    summary: dict[str, Any] = {}
    for name in sorted(per_candidate):
        blocks = per_candidate[name]
        width = int(blocks[0].shape[-1])
        standardizer = TargetStandardizer(width, args.query_temperature, args.query_heads)
        standardizer.fit(
            torch.cat([block.reshape(-1, width) for block in blocks], dim=0),
            source_hash="+".join(scene_ids),
            split="train",
        )
        standardized_blocks = [standardizer.transform(block) for block in blocks]
        entry: dict[str, Any] = {
            "tokens_per_frame": width // int(teacher.feature_dim),
            "target_width": width,
            "raw": measure_target_quality(blocks, trajectories),
            "standardized": measure_target_quality(standardized_blocks, trajectories),
            "degeneracy": degeneracy_report(standardized_blocks),
            "standardizer_clamped_components": int(standardizer.state.clamped_components),
            "spatial": mean_record(spatial_rows[name]) if name in spatial_rows else None,
        }
        entry["quality_delta"] = {
            key: round(float(entry["standardized"][key]) - float(entry["raw"][key]), 4)
            for key in ("separation_ratio", "trajectory_distance_pearson")
        }
        summary[name] = entry

    baseline = summary[BASELINE]["standardized"]
    for name, entry in summary.items():
        standardized = entry["standardized"]
        entry["vs_baseline"] = {
            "separation_ratio": round(float(standardized["separation_ratio"]) / float(baseline["separation_ratio"]), 4),
            "trajectory_distance_pearson": round(
                float(standardized["trajectory_distance_pearson"]) / float(baseline["trajectory_distance_pearson"]), 4
            ),
        }

    ordering = sorted(summary, key=lambda name: -float(summary[name]["standardized"]["separation_ratio"]))
    report = {
        "event": "summary",
        "scenes": len(scene_ids),
        "scene_ids": scene_ids,
        "frames_per_scene": int(per_candidate[BASELINE][0].shape[1]),
        "teacher_feature_dim": int(teacher.feature_dim),
        "teacher_num_prefix_tokens": int(teacher.num_prefix_tokens),
        "grid_hw": list(grid_hw),
        "query_temperature": args.query_temperature,
        "query_heads": args.query_heads,
        "baseline": BASELINE,
        "best_by_separation_ratio": ordering[0],
        "best_by_trajectory_pearson": max(
            summary, key=lambda name: float(summary[name]["standardized"]["trajectory_distance_pearson"])
        ),
        "elapsed_s": round(time.time() - started, 1),
        "candidates": summary,
    }
    (Path(args.out) / "readout_report.json").write_text(json.dumps(report, indent=2))

    for entry in figure_scenes:
        for slot_index, frame_index in enumerate(entry["slots"]):
            stem = f"{entry['scene_id']}_f{frame_index}"
            save_readout_figure(
                Path(args.out) / f"readout_{stem}.png",
                entry["scene_id"],
                entry["frames"][0, slot_index],
                entry["attention"][BASELINE][0, slot_index],
                entry["register_maps"][slot_index],
                entry["saliency"][0, slot_index],
                grid_hw,
                frame_index,
            )
            save_cross_gt_figure(
                Path(args.out) / f"crossgt_{stem}.png",
                entry["scene_id"],
                entry["frames"][:, slot_index],
                entry["attention"][BASELINE][:, slot_index],
                entry["motion"][:, slot_index],
                grid_hw,
                frame_index,
            )
    print(json.dumps({
        "event": "figures",
        "count": sum(2 * len(entry["slots"]) for entry in figure_scenes),
        "out": args.out,
    }), flush=True)

    print(f"\n{'candidate':24} {'tok':>4} {'width':>6} {'sep':>8} {'traj.r':>8} {'rank':>6} "
          f"{'const':>6} {'motion.r':>9} {'tvGT':>7} {'entropy':>8}")
    for name in ordering:
        entry = summary[name]
        standardized = entry["standardized"]
        spatial = entry["spatial"] or {}
        print(f"{name:24} {entry['tokens_per_frame']:>4} {entry['target_width']:>6} "
              f"{standardized['separation_ratio']:>8.4f} {standardized['trajectory_distance_pearson']:>8.4f} "
              f"{entry['degeneracy']['numeric_rank']:>6} {entry['degeneracy']['constant_dims']:>6} "
              f"{spatial.get('motion_pearson', float('nan')):>9.4f} "
              f"{spatial.get('attention_tv_across_gt', float('nan')):>7.4f} "
              f"{spatial.get('attention_entropy_mean', float('nan')):>8.4f}")
    print(f"\nbest by separation ratio   : {report['best_by_separation_ratio']}")
    print(f"best by trajectory pearson : {report['best_by_trajectory_pearson']}")
    print(f"report -> {Path(args.out) / 'readout_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
