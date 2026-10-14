"""Offline Future-Register target cache and the streaming Stage-1 dataset.

Stage 1 used to hold every scene in RAM: 46.3 MB of ``rgb_history`` plus 148.2 MB
of decoded ``future_rgb`` per scene, 194.5 MB total, which caps the corpus at
~970 resident scenes against the 103075 that exist.  This module splits that into
two halves with very different costs.

The *target* half is expensive and constant.  Decoding four ``future.mp4`` files
and running the frozen Teacher over 32 frames is the same work at step 1 and at
step 50000, and the ordered-2x2 compressor is in ``stage1.freeze``, so the whole
thing is precomputed once into ``{target_cache_root}/targets/{token[:2]}/{token}.pt``
-- ``[K,4,256]`` float32 plus the metric trajectories, ~66 KB per scene.

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

from .data import Observation, PairTargets
from .teacher import Ordered2x2Compressor, OrderedCompressorState
from .vision import DRIVOR_OFFICIAL_IMAGE_HW

READY_SCENES_NAME = "_ready_scenes.json"
COMPRESSOR_NAME = "compressor.pt"
MANIFEST_NAME = "_manifest.json"
TARGETS_DIRNAME = "targets"
HISTORY_FRAMES = 5
MAX_GT = 4
FUTURE_CHUNKS = 4


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


def compressor_path(target_cache_root: str | Path) -> Path:
    return Path(target_cache_root) / COMPRESSOR_NAME


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

    Both sides must agree, because the compressor may only be fit on train
    (``Ordered2x2Compressor.fit`` raises on any other split) and an eval scene
    that leaked into the fit would make the held-out metrics meaningless.  A
    seeded permutation is used rather than a head/tail cut so the eval scenes are
    spread over logs instead of clustering in one drive.
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
# Compressor persistence
# --------------------------------------------------------------------------- #
def save_compressor(target_cache_root: str | Path, compressor: Ordered2x2Compressor, audit: dict[str, Any]) -> None:
    if compressor.state is None:
        raise RuntimeError("refusing to persist an unfit compressor")
    state = compressor.state
    atomic_torch_save(
        {
            "mean": state.mean.detach().cpu().float(),
            "projection": state.projection.detach().cpu().float(),
            "input_dim": int(state.input_dim),
            "output_dim": int(state.output_dim),
            "fit_split": str(state.fit_split),
            "source_hash": str(state.source_hash),
            "fingerprint": compressor.fingerprint(),
            "audit": audit,
        },
        compressor_path(target_cache_root),
    )


def load_compressor(target_cache_root: str | Path) -> tuple[Ordered2x2Compressor, dict[str, Any]]:
    """Rebuild the exact frozen projector the builder used.

    Every rank loads this instead of fitting its own: an SVD over a different
    scene subset yields a different orthonormal basis (sign and column order are
    both arbitrary), so 32 independently-fit compressors would produce 32
    mutually incomparable target spaces and the all-reduced gradient would be
    meaningless.
    """

    path = compressor_path(target_cache_root)
    if not path.exists():
        raise FileNotFoundError(
            f"no fitted target compressor at {path}; run scripts/build_future_target_cache.py first"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    compressor = Ordered2x2Compressor(output_dim=int(payload["output_dim"]))
    compressor.state = OrderedCompressorState(
        mean=payload["mean"],
        projection=payload["projection"],
        input_dim=int(payload["input_dim"]),
        output_dim=int(payload["output_dim"]),
        fit_split=str(payload["fit_split"]),
        source_hash=str(payload["source_hash"]),
    )
    recorded = str(payload.get("fingerprint", ""))
    if recorded and recorded != compressor.fingerprint():
        raise ValueError(f"persisted compressor fingerprint {recorded} does not match the loaded {compressor.fingerprint()}")
    return compressor, dict(payload.get("audit") or {})


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
    compressor_fingerprint: str,
) -> dict[str, Any]:
    """Flatten one scene to the batch-axis-free tensors the Dataset re-stacks.

    Only tensors and JSON primitives go in, so ``torch.load(weights_only=True)``
    can read the cache back without an unrestricted unpickle of 103075 files.
    """

    if future_register.ndim != 3:
        raise ValueError(f"future_register must be [K,4,D], got {tuple(future_register.shape)}")
    if observation.rgb_history.shape[0] != 1:
        raise ValueError("build_scene_payload expects a single-scene observation")
    gt_count = int(gt_trajectory.shape[0])
    if not 1 <= gt_count <= MAX_GT:
        raise ValueError(f"Multi-GT allows 1..{MAX_GT} trajectories, got {gt_count}")
    if future_register.shape[0] != gt_count:
        raise ValueError("future_register and gt_trajectory must share the GT axis")
    return {
        "format": 1,
        "scene_id": str(metadata["scene_id"]),
        "log_name": str(metadata["log_name"]),
        "frame_index": int(metadata["frame_index"]),
        "gt_count": gt_count,
        "labels": [str(label) for label in metadata["labels"]],
        "degenerate_multi_gt": bool(metadata["degenerate_multi_gt"]),
        "paired_pdms": [float(value) for value in metadata["paired_pdms"]],
        "history_policy": str(metadata["history_policy"]),
        "history_paths": [str(path) for path in metadata["history_paths"]],
        "image_hw": [int(value) for value in metadata["image_hw"]],
        "pair_ids": pair_ids,
        "compressor_fingerprint": compressor_fingerprint,
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
    if int(payload.get("format", 0)) != 1:
        raise ValueError(f"{path} is not a format-1 Future Register target cache entry")
    return payload


def targets_from_payload(payload: dict[str, Any], target_dim: int) -> PairTargets:
    gt_count = int(payload["gt_count"])
    register = payload["future_register"]
    if register.shape != (gt_count, FUTURE_CHUNKS, target_dim):
        raise ValueError(f"cached future_register is {tuple(register.shape)}, expected {(gt_count, FUTURE_CHUNKS, target_dim)}")
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

    def __init__(
        self,
        scene_ids: Sequence[str],
        target_cache_root: str | Path,
        target_dim: int = 256,
        image_hw: tuple[int, int] = DRIVOR_OFFICIAL_IMAGE_HW,
        skip_degenerate: bool = False,
        compressor_fingerprint: str | None = None,
        max_retry: int = 12,
    ) -> None:
        self.scene_ids = [str(token) for token in scene_ids]
        if not self.scene_ids:
            raise ValueError("Stage1SceneDataset needs at least one scene")
        self.target_cache_root = Path(target_cache_root)
        self.target_dim = int(target_dim)
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.skip_degenerate = bool(skip_degenerate)
        self.compressor_fingerprint = compressor_fingerprint
        self.max_retry = int(max_retry)

    def __len__(self) -> int:
        return len(self.scene_ids)

    def path_for(self, scene_id: str) -> Path:
        return scene_target_path(self.target_cache_root, scene_id)

    def load_payload(self, scene_id: str) -> dict[str, Any]:
        payload = read_scene_payload(self.path_for(scene_id))
        if self.compressor_fingerprint is not None and payload["compressor_fingerprint"] != self.compressor_fingerprint:
            raise ValueError(
                f"scene {scene_id} was compressed with {payload['compressor_fingerprint']}, "
                f"but the cache compressor is {self.compressor_fingerprint}"
            )
        if self.skip_degenerate and bool(payload["degenerate_multi_gt"]):
            raise ValueError(f"scene {scene_id} has duplicate Multi-GT trajectory labels")
        if [int(value) for value in payload["image_hw"]] != [self.image_hw[0], self.image_hw[1]]:
            raise ValueError(
                f"scene {scene_id} was cached at {payload['image_hw']}, loader wants {list(self.image_hw)}"
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
            return {
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
            }
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
        camera_metadata=[{"scene_id": item["scene_id"], "labels": item["labels"]} for item in batch],
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
    targets.validate(max_gt=max_gt, registers=FUTURE_CHUNKS)
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
    compressor_fingerprint: str | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    """Drop tokens with no cached target, without decoding anything.

    ``list_cached_scenes`` already did one directory scan, so this is a set
    intersection plus, only when a filter is asked for, a small header read per
    surviving token.
    """

    cached = set(list_cached_scenes(target_cache_root))
    usable: list[str] = []
    rejected: list[dict[str, str]] = []
    for token in tokens:
        if token not in cached:
            rejected.append({"scene": token, "reason": "no_cached_target"})
            continue
        usable.append(token)
    if not skip_degenerate and compressor_fingerprint is None:
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
        if compressor_fingerprint is not None and payload["compressor_fingerprint"] != compressor_fingerprint:
            rejected.append({"scene": token, "reason": "compressor_fingerprint_mismatch"})
            continue
        kept.append(token)
    return kept, rejected
