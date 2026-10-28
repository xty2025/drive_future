"""Offline Future-Register target cache and the streaming Stage-1 dataset.

Stage 1 used to hold every scene in RAM: 46.3 MB of ``rgb_history`` plus 148.2 MB
of decoded ``future_rgb`` per scene, 194.5 MB total, which caps the corpus at
~970 resident scenes against the 103075 that exist.  This module splits that into
two halves with very different costs.

The *target* half is expensive and constant.  Decoding four ``future.mp4`` files
and running the frozen Teacher over 32 frames is the same work at step 1 and at
step 50000, and the query extractor and standardizer are both in ``stage1.freeze``,
so the whole thing is precomputed once into
``{target_cache_root}/targets/{token[:2]}/{token}.pt`` -- ``[K,32,D]`` float32 at
the Teacher's own feature width (512 KB at ``K=4``, ``D=1024``) plus the metric
trajectories.  The token axis is ``REGISTER_TOKENS``: four DINOv3 prefix-query
readouts per future frame, frame-major, so entry ``4 * f + q`` is query ``q`` of the
frame at ``(f + 1) * 0.5 s``.

The *observation* half is cheap and must never be resident.  ``rgb_history`` is
therefore left on disk as JPEGs and decoded per ``__getitem__``, kept as ``uint8``
until it reaches the GPU: 11.6 MB instead of 46.3 MB per scene, so the in-flight
worker buffers stay an order of magnitude smaller.  The ``uint8 -> float32 / 255``
conversion is deferred rather than skipped, which keeps it bit-identical to
``multiview_data._resize_rgb``.

The ego side (``ego_history``, ``route_command``, ``relative_ego_poses``,
``history_valid``, and the five resolved JPEG paths) is cached too.  It is a few
hundred bytes, and caching it is what stops every worker from unpickling a whole
NAVSIM log to read five fields out of it.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .data import Observation, PairTargets, REGISTER_TOKENS
from .teacher import TargetStandardizer, TargetStandardizerState
from .vision import DRIVOR_OFFICIAL_IMAGE_HW

READY_SCENES_NAME = "_ready_scenes.json"
STANDARDIZER_NAME = "standardizer.pt"
MANIFEST_NAME = "_manifest.json"
TARGETS_DIRNAME = "targets"
HISTORY_FRAMES = 5
MAX_GT = 4
#: Bumped from 1 by the query-extracted target.  A format-1 entry holds a
#: whitened-PCA vector in a 256-d subspace; a format-2 entry holds a standardized
#: query token at the Teacher's own width.  The two are not comparable and the
#: fingerprint check alone would report that as an opaque mismatch, so the format
#: field names it first.
PAYLOAD_FORMAT = 2

#: Two Multi-GT rows whose trajectories are closer than this over the 4 s horizon are
#: the same manoeuvre, whatever their labels say.  ``degenerate_multi_gt`` is
#: ``len(set(labels)) < len(labels)``, so it only catches *duplicate labels* and a
#: scene carrying K distinct labels whose trajectories differ by centimetres sails
#: through it.  Such a scene is worse than useless to Stage 1: its K targets differ
#: while its K inputs are identical to within float noise, so the only way to reduce
#: the loss is to stop reading the action tokens.
#:
#: The floor is set from the smoke cache's eight scenes, whose mean pairwise spreads
#: were 2.032 / 0.943 / 0.882 / 0.813 / 0.528 / 0.503 / 0.216 / 0.028 m.  Across those
#: scenes ``target_cross_gt_rms`` tracks the spread at pearson +0.874, so the target is
#: largely trajectory-explainable and a low spread is not by itself a problem -- the one
#: scene that *is* a problem is the 0.028 m one, whose target spread per metre of
#: trajectory spread is 11.7 against 0.36-2.2 for the other seven.  0.1 m isolates it
#: with a 3.5x margin on both sides and drops 1 of 8; 0.5 m would also drop the 0.216 m
#: scene, whose ratio of 2.2 sits inside the pack, i.e. it would discard usable signal
#: to avoid a scene that is not actually contradictory.
#:
#: Evaluated at load time from ``gt_trajectory``, which the payload already carries,
#: so tightening it does not invalidate a built cache -- re-running Stage 0 over the
#: 103075-scene corpus costs ~163 GPU-hours.
MIN_GT_SPREAD_M = 0.1


def multi_gt_spread_metres(gt_trajectory: torch.Tensor, gt_valid: torch.Tensor) -> float:
    """Mean pairwise xy distance between the valid Multi-GT rows, in metres.

    Accepts either the payload's ``[K,8,3]`` / ``[K]`` or the collated ``PairTargets``'
    ``[1,K,8,3]`` / ``[1,K]``, because the two callers hold different ones: the cache
    filters run on the stored payload before any batching, while the inline loader
    already has the batch axis.  Silently indexing ``[0]`` would read the first *GT* of
    an unbatched payload as if it were the batch, and a spread computed from one row is
    ``nan`` -- which every threshold comparison reads as "not degenerate", so the gate
    would fail open rather than loudly.

    Averaged over the eight waypoints rather than read at the endpoint, because the
    register predicts all eight frames and two rows that diverge only at 4 s are still
    one manoeuvre for most of the horizon.

    Returns ``nan`` when fewer than two rows are valid: the spread of a single
    trajectory is undefined, not zero, and reporting zero would make a legitimate
    single-GT scene look degenerate to every caller that compares against a threshold.
    """

    if gt_trajectory.dim() == 4:
        if gt_trajectory.shape[0] != 1:
            raise ValueError(f"expected one scene per call, got batch {gt_trajectory.shape[0]}")
        gt_trajectory = gt_trajectory[0]
    if gt_valid.dim() == 2:
        gt_valid = gt_valid[0]
    if gt_trajectory.dim() != 3 or gt_trajectory.shape[-2:] != (8, 3):
        raise ValueError(f"gt_trajectory must be [K,8,3] or [1,K,8,3], got {tuple(gt_trajectory.shape)}")
    if gt_valid.shape != gt_trajectory.shape[:1]:
        raise ValueError(f"gt_valid {tuple(gt_valid.shape)} must match the K axis of {tuple(gt_trajectory.shape)}")

    kept = gt_trajectory[gt_valid.bool()]
    if kept.shape[0] < 2:
        return float("nan")
    delta = kept[:, None, :, :2] - kept[None, :, :, :2]
    pairwise = delta.pow(2).sum(dim=-1).sqrt().mean(dim=-1)
    upper = torch.triu(torch.ones_like(pairwise, dtype=torch.bool), diagonal=1)
    return float(pairwise[upper].mean())


def is_trajectory_degenerate(
    gt_trajectory: torch.Tensor,
    gt_valid: torch.Tensor,
    min_spread_m: float = MIN_GT_SPREAD_M,
) -> bool:
    """True when the K GT rows are distinct labels but the same manoeuvre.

    ``nan`` compares False against every threshold, so a scene with fewer than two
    valid rows passes -- there is no pair to contrast and nothing contradictory about
    one (input, target) example.
    """

    return multi_gt_spread_metres(gt_trajectory, gt_valid) < min_spread_m


# --------------------------------------------------------------------------- #
# Paths and atomic writes
# --------------------------------------------------------------------------- #
def scene_target_path(target_cache_root: str | Path, scene_id: str) -> Path:
    """``{root}/targets/{token[:2]}/{token}.pt``.

    The two-character fan-out keeps 103075 files out of a single directory; at 256
    buckets that is ~400 files each, which every filesystem here handles without
    the readdir cost a flat layout would add to every rank's startup scan.
    """

    if not scene_id or "/" in scene_id or scene_id.startswith("."):
        raise ValueError(f"refusing to build a cache path from scene id {scene_id!r}")
    return Path(target_cache_root) / TARGETS_DIRNAME / scene_id[:2] / f"{scene_id}.pt"


def standardizer_path(target_cache_root: str | Path) -> Path:
    return Path(target_cache_root) / STANDARDIZER_NAME


def manifest_path(target_cache_root: str | Path) -> Path:
    return Path(target_cache_root) / MANIFEST_NAME


def ready_scenes_path(target_cache_root: str | Path) -> Path:
    return Path(target_cache_root) / READY_SCENES_NAME


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    """Write to a sibling temp file then ``os.replace`` it into place.

    A builder killed mid-write must not leave a truncated ``.pt`` that the trainer
    later trusts: ``os.replace`` is atomic within a filesystem, so a reader sees
    either nothing or a complete file.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_write(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(payload))
    os.replace(temporary, path)


def list_cached_scenes(target_cache_root: str | Path) -> list[str]:
    """Every scene token whose target ``.pt`` is present, sorted."""

    root = Path(target_cache_root) / TARGETS_DIRNAME
    if not root.is_dir():
        return []
    tokens: list[str] = []
    for bucket in root.iterdir():
        if not bucket.is_dir():
            continue
        for entry in os.scandir(bucket):
            if entry.is_file() and entry.name.endswith(".pt"):
                tokens.append(entry.name[: -len(".pt")])
    return sorted(tokens)


# --------------------------------------------------------------------------- #
# Train/eval split -- shared by the builder and the trainer
# --------------------------------------------------------------------------- #
def split_scene_tokens(tokens: Sequence[str], eval_scenes: int, seed: int) -> tuple[list[str], list[str]]:
    """Deterministic disjoint split, identical in the builder and the trainer.

    Both sides must agree, because the standardizer may only be fit on train
    (``TargetStandardizer.fit`` raises on any other split) and an eval scene that
    leaked into the fit would make the held-out metrics meaningless.  A seeded
    permutation is used rather than a head/tail cut so the eval scenes are spread
    over logs instead of clustering in one drive.
    """

    ordered = sorted(tokens)
    eval_scenes = int(eval_scenes)
    if eval_scenes < 0:
        raise ValueError("eval_scenes must be non-negative")
    if eval_scenes >= len(ordered):
        raise ValueError(f"eval_scenes={eval_scenes} leaves no training scene out of {len(ordered)}")
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    evaluation = sorted(shuffled[:eval_scenes])
    chosen = set(evaluation)
    training = [token for token in ordered if token not in chosen]
    return training, evaluation


# --------------------------------------------------------------------------- #
# Standardizer persistence
# --------------------------------------------------------------------------- #
def save_standardizer(target_cache_root: str | Path, standardizer: TargetStandardizer, audit: dict[str, Any]) -> None:
    if standardizer.state is None:
        raise RuntimeError("refusing to persist an unfit standardizer")
    state = standardizer.state
    atomic_torch_save(
        {
            "mean": state.mean.detach().cpu().float(),
            "scale": state.scale.detach().cpu().float(),
            "feature_dim": int(state.feature_dim),
            "fit_split": str(state.fit_split),
            "source_hash": str(state.source_hash),
            "temperature": float(state.temperature),
            "heads": int(state.heads),
            "clamped_components": int(state.clamped_components),
            "fingerprint": standardizer.fingerprint(),
            "audit": audit,
        },
        standardizer_path(target_cache_root),
    )


def load_standardizer(target_cache_root: str | Path) -> tuple[TargetStandardizer, dict[str, Any]]:
    """Rebuild the exact frozen target scaling the builder used.

    Every rank loads this instead of fitting its own.  A diagonal mean/std has no
    basis ambiguity -- unlike the SVD projector it replaced, two fits on different
    subsets differ only by estimation noise rather than by an arbitrary column
    order and sign -- but they still differ, and the difference is a per-dimension
    rescaling of every cached target.  Rank 3's targets would then not be
    comparable to rank 7's and the all-reduced gradient would average them anyway.

    The recorded ``temperature``/``heads`` are the query readout's identity, which
    is *not* recoverable from ``mean`` and ``scale``: a cache built at another
    temperature holds different tensors of the same width and would pass every
    shape check here.  They ride in the fingerprint, so a mismatch is caught.
    """

    path = standardizer_path(target_cache_root)
    if not path.exists():
        raise FileNotFoundError(
            f"no fitted target standardizer at {path}; run scripts/build_future_target_cache.py first"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "feature_dim" not in payload:
        raise ValueError(
            f"{path} predates the query-extracted target and holds a PCA projector; "
            "rebuild it with scripts/build_future_target_cache.py"
        )
    standardizer = TargetStandardizer(
        feature_dim=int(payload["feature_dim"]),
        temperature=float(payload["temperature"]),
        heads=int(payload["heads"]),
    )
    standardizer.state = TargetStandardizerState(
        mean=payload["mean"],
        scale=payload["scale"],
        feature_dim=int(payload["feature_dim"]),
        fit_split=str(payload["fit_split"]),
        source_hash=str(payload["source_hash"]),
        temperature=float(payload["temperature"]),
        heads=int(payload["heads"]),
        clamped_components=int(payload.get("clamped_components", 0)),
    )
    recorded = str(payload.get("fingerprint", ""))
    if recorded and recorded != standardizer.fingerprint():
        raise ValueError(
            f"persisted standardizer fingerprint {recorded} does not match the loaded {standardizer.fingerprint()}"
        )
    return standardizer, dict(payload.get("audit") or {})


# --------------------------------------------------------------------------- #
# Lazy history decode
# --------------------------------------------------------------------------- #
def load_history_uint8(paths: Sequence[str | Path], image_hw: tuple[int, int] = DRIVOR_OFFICIAL_IMAGE_HW) -> torch.Tensor:
    """DrivoR's ``im.resize()`` bicubic downscale, kept as ``uint8``.

    ``multiview_data._resize_rgb`` produces ``float32(uint8) / 255``.  Deferring
    the divide to the GPU is exact rather than approximate: the resize happens
    here on the same PIL BICUBIC filter at the same ``(672, 1148)``, so
    ``load_history_uint8(...).float().div(255)`` is bit-identical to the eager
    float path it replaces, at a quarter of the host memory.
    """

    height, width = int(image_hw[0]), int(image_hw[1])
    if len(paths) != HISTORY_FRAMES:
        raise ValueError(f"v6 needs {HISTORY_FRAMES} history frames, got {len(paths)}")
    frames = []
    for path in paths:
        with Image.open(path) as image:
            resized = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
            array = np.asarray(resized)
        frames.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
    stacked = torch.stack(frames)
    if stacked.dtype != torch.uint8 or tuple(stacked.shape) != (HISTORY_FRAMES, 3, height, width):
        raise ValueError(f"unexpected history decode result {tuple(stacked.shape)} {stacked.dtype}")
    return stacked


def observation_to_device(observation: Observation, device: torch.device) -> Observation:
    """Move to ``device``, finishing the deferred ``uint8 -> float32 / 255``."""

    rgb_history = observation.rgb_history.to(device, non_blocking=True)
    if rgb_history.dtype == torch.uint8:
        rgb_history = rgb_history.float().div(255.0)
    return replace(
        observation,
        rgb_history=rgb_history,
        history_valid=observation.history_valid.to(device, non_blocking=True),
        ego_history=observation.ego_history.to(device, non_blocking=True),
        route_command=observation.route_command.to(device, non_blocking=True),
        relative_ego_poses=observation.relative_ego_poses.to(device, non_blocking=True),
    )


# --------------------------------------------------------------------------- #
# Per-scene cache payload
# --------------------------------------------------------------------------- #
CACHED_TENSOR_KEYS = (
    "future_register",
    "future_chunk_valid",
    "future_quality",
    "gt_trajectory",
    "gt_valid",
    "ego_history",
    "route_command",
    "relative_ego_poses",
    "history_valid",
)


def build_scene_payload(
    future_register: torch.Tensor,
    future_chunk_valid: torch.Tensor,
    future_quality: torch.Tensor,
    gt_trajectory: torch.Tensor,
    gt_valid: torch.Tensor,
    observation: Observation,
    pair_ids: list[dict[str, Any]],
    metadata: dict[str, Any],
    target_fingerprint: str,
) -> dict[str, Any]:
    """Flatten one scene to the batch-axis-free tensors the Dataset re-stacks.

    Only tensors and JSON primitives go in, so ``torch.load(weights_only=True)``
    can read the cache back without an unrestricted unpickle of 103075 files.

    ``history_paths`` comes from ``observation.camera_metadata`` and *not* from
    ``metadata``, which is where every other scalar here lives -- the loader resolves
    the five JPEGs while building the observation and records them there.  Reading
    them from ``metadata`` instead raised ``KeyError: 'history_paths'`` on every
    scene, which the builder's per-scene ``except`` collected as 8/8 failures while
    still exiting 0.  ``history_policy`` is in both places; ``metadata`` is the
    canonical one and is what the rest of the payload is read from.
    """

    if future_register.ndim != 3:
        raise ValueError(f"future_register must be [K,{REGISTER_TOKENS},D], got {tuple(future_register.shape)}")
    if observation.rgb_history.shape[0] != 1:
        raise ValueError("build_scene_payload expects a single-scene observation")
    gt_count = int(gt_trajectory.shape[0])
    if not 1 <= gt_count <= MAX_GT:
        raise ValueError(f"Multi-GT allows 1..{MAX_GT} trajectories, got {gt_count}")
    if future_register.shape[0] != gt_count:
        raise ValueError("future_register and gt_trajectory must share the GT axis")
    camera_metadata = observation.camera_metadata if isinstance(observation.camera_metadata, dict) else {}
    history_paths = camera_metadata.get("history_paths")
    if not isinstance(history_paths, (list, tuple)) or len(history_paths) != HISTORY_FRAMES:
        raise ValueError(
            f"observation.camera_metadata['history_paths'] must hold {HISTORY_FRAMES} resolved JPEG paths, "
            f"got {history_paths!r}; only load_multiview_sample populates it, so the observation did not "
            "come from the Multi-GT loader"
        )
    return {
        "format": PAYLOAD_FORMAT,
        "scene_id": str(metadata["scene_id"]),
        "log_name": str(metadata["log_name"]),
        "frame_index": int(metadata["frame_index"]),
        "gt_count": gt_count,
        "labels": [str(label) for label in metadata["labels"]],
        "degenerate_multi_gt": bool(metadata["degenerate_multi_gt"]),
        "paired_pdms": [float(value) for value in metadata["paired_pdms"]],
        "history_policy": str(metadata["history_policy"]),
        "history_paths": [str(path) for path in history_paths],
        "image_hw": [int(value) for value in metadata["image_hw"]],
        "pair_ids": pair_ids,
        "target_fingerprint": target_fingerprint,
        "future_register": future_register.detach().cpu().float(),
        "future_chunk_valid": future_chunk_valid.detach().cpu().bool(),
        "future_quality": future_quality.detach().cpu().float(),
        "gt_trajectory": gt_trajectory.detach().cpu().float(),
        "gt_valid": gt_valid.detach().cpu().bool(),
        "ego_history": observation.ego_history[0].detach().cpu().float(),
        "route_command": observation.route_command[0].detach().cpu().float(),
        "relative_ego_poses": observation.relative_ego_poses[0].detach().cpu().float(),
        "history_valid": observation.history_valid[0, :, 0].detach().cpu().bool(),
    }


def read_scene_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    found = int(payload.get("format", 0))
    if found != PAYLOAD_FORMAT:
        raise ValueError(
            f"{path} is a format-{found} Future Register target cache entry, this loader reads format-{PAYLOAD_FORMAT}"
            + ("; format 1 held a whitened-PCA target, rebuild the cache" if found == 1 else "")
        )
    return payload


def targets_from_payload(payload: dict[str, Any], register_target_dim: int) -> PairTargets:
    gt_count = int(payload["gt_count"])
    register = payload["future_register"]
    if register.shape != (gt_count, REGISTER_TOKENS, register_target_dim):
        raise ValueError(
            f"cached future_register is {tuple(register.shape)}, "
            f"expected {(gt_count, REGISTER_TOKENS, register_target_dim)}"
        )
    return PairTargets(
        gt_trajectory=payload["gt_trajectory"].unsqueeze(0),
        gt_valid=payload["gt_valid"].unsqueeze(0),
        future_register=register.unsqueeze(0),
        future_chunk_valid=payload["future_chunk_valid"].unsqueeze(0),
        future_quality=payload["future_quality"].unsqueeze(0),
        pair_ids=[list(payload["pair_ids"])],
    )


# --------------------------------------------------------------------------- #
# Streaming dataset
# --------------------------------------------------------------------------- #
class Stage1SceneDataset(Dataset):
    """Cached targets plus a lazy ``uint8`` history decode, one scene per item.

    ``__getitem__`` never touches a ``future.mp4`` or the Teacher: the register
    target comes from the cache.  It reads five JPEGs, which is the only part of a
    scene too large to keep resident.

    A failed decode must not change the number of batches a rank produces -- that
    desyncs DDP and hangs the all-reduce -- so on failure the next token is tried,
    matching ``NavtrainSceneDataset`` in ``run_navtrain_wan_e2e_ddp.py``.
    """

    #: Row-axis tensors a mined-negative payload contributes when it is merged into
    #: its scene.  The observation side is deliberately absent: a negative row is the
    #: same scene at the same ``t0``, so the positive payload's history is the only
    #: history, and carrying a second copy would only create a chance to disagree.
    NEGATIVE_ROW_KEYS = (
        "gt_trajectory",
        "gt_valid",
        "future_register",
        "future_chunk_valid",
        "future_quality",
    )

    def __init__(
        self,
        scene_ids: Sequence[str],
        target_cache_root: str | Path,
        image_hw: tuple[int, int] = DRIVOR_OFFICIAL_IMAGE_HW,
        skip_degenerate: bool = False,
        target_fingerprint: str | None = None,
        max_retry: int = 12,
        min_gt_spread_m: float = MIN_GT_SPREAD_M,
        negative_target_cache_root: str | Path | None = None,
    ) -> None:
        self.scene_ids = [str(token) for token in scene_ids]
        if not self.scene_ids:
            raise ValueError("Stage1SceneDataset needs at least one scene")
        self.target_cache_root = Path(target_cache_root)
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.skip_degenerate = bool(skip_degenerate)
        self.target_fingerprint = target_fingerprint
        self.max_retry = int(max_retry)
        self.min_gt_spread_m = float(min_gt_spread_m)
        self.negative_target_cache_root = (
            None if negative_target_cache_root is None else Path(negative_target_cache_root)
        )
        if self.negative_target_cache_root is not None:
            # Checked once here rather than per scene: ``__getitem__`` retries the next
            # token on any exception, so a target-space mismatch found per scene would
            # be swallowed into "could not load any scene near index" instead of naming
            # the actual problem -- two caches standardized into different spaces.
            negative_standardizer, _ = load_standardizer(self.negative_target_cache_root)
            negative_fingerprint = negative_standardizer.fingerprint()
            if target_fingerprint is not None and negative_fingerprint != target_fingerprint:
                raise ValueError(
                    f"negative cache {self.negative_target_cache_root} is standardized as "
                    f"{negative_fingerprint} but the positive cache is {target_fingerprint}; "
                    f"rebuild the negative cache re-using the positive cache's standardizer.pt"
                )

    def __len__(self) -> int:
        return len(self.scene_ids)

    def path_for(self, scene_id: str) -> Path:
        return scene_target_path(self.target_cache_root, scene_id)

    def load_payload(self, scene_id: str) -> dict[str, Any]:
        payload = read_scene_payload(self.path_for(scene_id))
        if self.target_fingerprint is not None and payload["target_fingerprint"] != self.target_fingerprint:
            raise ValueError(
                f"scene {scene_id} was extracted with target space {payload['target_fingerprint']}, "
                f"but the cache standardizer is {self.target_fingerprint}"
            )
        if self.skip_degenerate and bool(payload["degenerate_multi_gt"]):
            raise ValueError(f"scene {scene_id} has duplicate Multi-GT trajectory labels")
        if self.skip_degenerate and is_trajectory_degenerate(
            payload["gt_trajectory"], payload["gt_valid"], self.min_gt_spread_m
        ):
            spread = multi_gt_spread_metres(payload["gt_trajectory"], payload["gt_valid"])
            raise ValueError(
                f"scene {scene_id} has {int(payload['gt_valid'].sum())} distinct Multi-GT labels "
                f"whose trajectories are {spread:.3f} m apart on average, under the "
                f"{self.min_gt_spread_m} m floor"
            )
        if [int(value) for value in payload["image_hw"]] != [self.image_hw[0], self.image_hw[1]]:
            raise ValueError(
                f"scene {scene_id} was cached at {payload['image_hw']}, loader wants {list(self.image_hw)}"
            )
        return payload

    def load_negative_rows(self, scene_id: str) -> dict[str, Any] | None:
        """Mined ``poor_negative`` rows for this scene, or ``None`` when it has none.

        ``None`` is the normal case, not a failure: 26,006 of the 103,075 navtrain
        scenes were mined to zero negatives and the published negative video corpus
        covers 77,067 scenes, so a positive token with no negative counterpart is
        expected and must not drop the scene -- dropping it would silently train on a
        different corpus than the positive-only run this one is compared against.

        A *present but mismatched* payload is the error.  ``target_fingerprint``
        hashes the standardizer's fitted mean/scale, so a difference means the two
        caches were standardized into different target spaces; concatenating their
        rows would supervise one head against a mixture no single scaling describes,
        and the numbers would still look finite.  The negative cache is built by
        re-using the positive cache's ``standardizer.pt`` rather than fitting its own
        precisely so this cannot happen by accident.

        ``skip_degenerate`` is *not* applied here.  On the positive side it protects
        the Multi-GT contrast from scenes whose four labels collapse onto one
        trajectory; a negative set with little internal spread still adds the
        positive-versus-negative contrast, which is the point of merging it.
        """

        if self.negative_target_cache_root is None:
            return None
        path = scene_target_path(self.negative_target_cache_root, scene_id)
        if not path.exists():
            return None
        payload = read_scene_payload(path)
        if self.target_fingerprint is not None and payload["target_fingerprint"] != self.target_fingerprint:
            raise ValueError(
                f"negative scene {scene_id} was extracted with target space {payload['target_fingerprint']}, "
                f"but the cache standardizer is {self.target_fingerprint}"
            )
        if [int(value) for value in payload["image_hw"]] != [self.image_hw[0], self.image_hw[1]]:
            raise ValueError(
                f"negative scene {scene_id} was cached at {payload['image_hw']}, loader wants {list(self.image_hw)}"
            )
        return payload

    def __getitem__(self, index: int) -> dict[str, Any]:
        total = len(self.scene_ids)
        last_error: Exception | None = None
        for attempt in range(self.max_retry):
            scene_id = self.scene_ids[(index + attempt) % total]
            try:
                payload = self.load_payload(scene_id)
                rgb_history = load_history_uint8(payload["history_paths"], self.image_hw)
            except Exception as error:  # one unreadable scene must not stall DDP
                last_error = error
                continue
            item = {
                "scene_id": scene_id,
                "rgb_history": rgb_history,
                "history_valid": payload["history_valid"],
                "ego_history": payload["ego_history"],
                "route_command": payload["route_command"],
                "relative_ego_poses": payload["relative_ego_poses"],
                "gt_trajectory": payload["gt_trajectory"],
                "gt_valid": payload["gt_valid"],
                "future_register": payload["future_register"],
                "future_chunk_valid": payload["future_chunk_valid"],
                "future_quality": payload["future_quality"],
                "labels": list(payload["labels"]),
                "degenerate_multi_gt": bool(payload["degenerate_multi_gt"]),
                "negative_rows": 0,
            }
            negative = self.load_negative_rows(scene_id)
            if negative is not None:
                # Positives stay first, so row k of the positive-only batch is still
                # row k here -- negatives are appended, never interleaved.  Nothing
                # downstream may key off that order (the mode code is zeroed exactly
                # so a row cannot be identified by its position), but keeping it
                # stable makes a pos-only and a pos+neg run diffable row by row.
                for key in self.NEGATIVE_ROW_KEYS:
                    item[key] = torch.cat([item[key], negative[key]], dim=0)
                item["labels"] = [*item["labels"], *[f"neg:{label}" for label in negative["labels"]]]
                item["negative_rows"] = int(negative["gt_valid"].shape[0])
            return item
        raise RuntimeError(f"could not load any scene near index {index}: {last_error!r}")


def collate_stage1_batch(batch: list[dict[str, Any]]) -> tuple[Observation, PairTargets, list[str]]:
    """Pad the GT axis only; every scene keeps all of its own Multi-GT pairs."""

    size = len(batch)
    max_gt = max(int(item["gt_valid"].shape[0]) for item in batch)

    def pad(key: str, fill: Any) -> torch.Tensor:
        prototype = batch[0][key]
        out = torch.full((size, max_gt, *prototype.shape[1:]), fill, dtype=prototype.dtype)
        for row, item in enumerate(batch):
            value = item[key]
            out[row, : value.shape[0]] = value
        return out

    observation = Observation(
        rgb_history=torch.stack([item["rgb_history"] for item in batch]).unsqueeze(2),
        history_valid=torch.stack([item["history_valid"] for item in batch]).unsqueeze(2),
        ego_history=torch.stack([item["ego_history"] for item in batch]),
        route_command=torch.stack([item["route_command"] for item in batch]),
        relative_ego_poses=torch.stack([item["relative_ego_poses"] for item in batch]),
        camera_metadata=[
            {
                "scene_id": item["scene_id"],
                "labels": item["labels"],
                # Logging only: it rides in metadata, never in a tensor, so the model
                # cannot see how many of its rows were mined negatives.
                "negative_rows": int(item.get("negative_rows", 0)),
            }
            for item in batch
        ],
    )
    targets = PairTargets(
        gt_trajectory=pad("gt_trajectory", 0.0),
        gt_valid=pad("gt_valid", False),
        future_register=pad("future_register", 0.0),
        future_chunk_valid=pad("future_chunk_valid", False),
        future_quality=pad("future_quality", 0.0),
        pair_ids=[[{"scene_id": item["scene_id"], "labels": item["labels"]}] for item in batch],
    )
    observation.validate()
    targets.validate(max_gt=max_gt, registers=REGISTER_TOKENS)
    return observation, targets, [item["scene_id"] for item in batch]


def targets_to_device(targets: PairTargets, device: torch.device) -> PairTargets:
    return replace(
        targets,
        **{
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in targets.__dict__.items()
        },
    )


def filter_usable_tokens(
    tokens: Iterable[str],
    target_cache_root: str | Path,
    skip_degenerate: bool = False,
    target_fingerprint: str | None = None,
    min_gt_spread_m: float = MIN_GT_SPREAD_M,
) -> tuple[list[str], list[dict[str, str]]]:
    """Drop tokens with no cached target, without decoding anything.

    ``list_cached_scenes`` already did one directory scan, so this is a set
    intersection plus, only when a filter is asked for, a small header read per
    surviving token.

    The degeneracy tests here must be exactly ``Stage1SceneDataset.load_payload``'s.
    When they differ, a token this function approves is one the dataset rejects, and
    ``__getitem__`` walks forward to the next token -- so a scene the train/eval split
    never approved gets trained on or scored, and because the walk is silent the only
    symptom is a metric attributed to the wrong scene.
    """

    cached = set(list_cached_scenes(target_cache_root))
    usable: list[str] = []
    rejected: list[dict[str, str]] = []
    for token in tokens:
        if token not in cached:
            rejected.append({"scene": token, "reason": "no_cached_target"})
            continue
        usable.append(token)
    if not skip_degenerate and target_fingerprint is None:
        return usable, rejected
    kept: list[str] = []
    for token in usable:
        try:
            payload = read_scene_payload(scene_target_path(target_cache_root, token))
        except Exception as error:
            rejected.append({"scene": token, "reason": f"unreadable_target:{error}"})
            continue
        if skip_degenerate and bool(payload["degenerate_multi_gt"]):
            rejected.append({"scene": token, "reason": "degenerate_multi_gt"})
            continue
        if skip_degenerate and is_trajectory_degenerate(
            payload["gt_trajectory"], payload["gt_valid"], min_gt_spread_m
        ):
            spread = multi_gt_spread_metres(payload["gt_trajectory"], payload["gt_valid"])
            rejected.append({"scene": token, "reason": f"gt_spread_{spread:.3f}_m"})
            continue
        if target_fingerprint is not None and payload["target_fingerprint"] != target_fingerprint:
            rejected.append({"scene": token, "reason": "target_fingerprint_mismatch"})
            continue
        kept.append(token)
    return kept, rejected
