"""Loader for the Cosmos Multi-GT future-view dataset under /mnt/data/multi_view.

One directory per NAVSIM frame token.  Each holds ``scene_metadata.json``
(log, frame index, t0 timestamp), ``initial_front.jpg`` and up to four
``trajectory_k`` folders; every folder carries the dense 0.1 s ego pose that
conditioned the generation plus the generated ``future.mp4``.

Observations stay DrivoR-shaped: five real front-camera history frames read from
the NAVSIM sensor blobs, DrivoR's 7-wide ego status per frame, the four-way
driving command, and metric ``[x, y, heading]`` trajectories sampled at
+0.5 s ... +4.0 s.  The generated future frames are target-only: they feed the
fixed Teacher that builds the Future Register target and are never given to the
student or the scorer.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image

from .config import EXPECTED_REGISTER_TARGET_DIM
from .data import (
    Observation,
    PairTargets,
    REGISTER_QUERIES_PER_FRAME,
    REGISTER_TOKENS,
    drivor_observable_ego_feature,
    drivor_route_command,
)
from .vision import DRIVOR_OFFICIAL_IMAGE_HW

# The full generated Multi-GT corpus.  /mnt/data/multi_view/... resolves here
# through a symlink, so both spellings describe the same scenes; this one is the
# real directory.
DEFAULT_MULTIVIEW_ROOT = (
    "/mnt/data/multi_future_view/navtrain_full_allcosmos_cuda_v1_gpu64/"
    "navtrain_full_allcosmos_cuda_v1_gpu64"
)
DEFAULT_LOG_ROOT = "/mnt/data/navsim/trainval_navsim_logs/trainval"
DEFAULT_SENSOR_ROOT = "/mnt/data/navsim/trainval_sensor_blobs/trainval"
DEFAULT_CACHE_ROOT = (
    "/mnt/zhaozc_workspace/project/DrivoR_multigt_kvcache_final_20261004/data_runtime/train_metric_cache"
)
DENSE_FRAMES = 41
FUTURE_POSE_INDICES = tuple(range(5, DENSE_FRAMES, 5))
# Every decoded future frame is read by ``REGISTER_QUERIES_PER_FRAME`` of DINOv3's
# prefix queries, so the Teacher's frame list times that count must equal the target's
# register axis.  Checking it here rather than in ``validate`` is what stops a
# ``DENSE_FRAMES`` edit from silently producing targets that no longer line up with the
# eight poses.
if len(FUTURE_POSE_INDICES) * REGISTER_QUERIES_PER_FRAME != REGISTER_TOKENS:
    raise ValueError(
        f"{len(FUTURE_POSE_INDICES)} future frames x {REGISTER_QUERIES_PER_FRAME} queries "
        f"cannot fill {REGISTER_TOKENS} register targets"
    )
HISTORY_FRAMES = 5
MAX_GT = 4


@dataclass
class MultiViewSample:
    observation: Observation
    targets: PairTargets
    scene_id: str
    log_name: str
    frame_index: int
    cache_path: Path
    future_rgb: torch.Tensor | None
    metadata: dict[str, Any]


def _load_image(path: Path, image_hw: tuple[int, int]) -> torch.Tensor:
    """DrivoR's CPU-side preprocessing: PIL resize to (W,H) then /255.

    ``drivor_features.py`` calls ``im.resize(image_size)`` with no ``resample``,
    so it inherits Pillow's BICUBIC default; naming it here keeps the two paths
    bit-identical instead of silently diverging on a Pillow default change.
    """

    with Image.open(path) as image:
        return _resize_rgb(np.asarray(image.convert("RGB")), image_hw)


def _relative_pose(reference: np.ndarray, frame: np.ndarray) -> np.ndarray:
    relative = np.linalg.inv(reference) @ frame
    return np.asarray(
        [relative[0, 3], relative[1, 3], np.arctan2(relative[1, 0], relative[0, 0])],
        dtype=np.float32,
    )


def _resize_rgb(array: np.ndarray, image_hw: tuple[int, int]) -> torch.Tensor:
    """DrivoR's ``im.resize(image_size)``: PIL bicubic down to the official size.

    The real ``CAM_F0`` frames are 1920x1080, so this is a 0.6x *downscale* -- it
    discards pixels, it does not invent any.  Pillow's default resample filter is
    bicubic and ``drivor_features.py:107`` passes no ``resample``, so naming it
    here keeps the two paths bit-identical instead of silently diverging if that
    default ever changes.  The generated future frames never come through this
    function: 464x832 is already a whole number of patch16 tiles, so the Teacher
    encodes them as rendered.
    """

    height, width = int(image_hw[0]), int(image_hw[1])
    with Image.fromarray(array) as image:
        image = image.resize((width, height), Image.Resampling.BICUBIC)
        resized = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(resized).permute(2, 0, 1).contiguous()


def _read_video_frames(path: Path, frame_indices: tuple[int, ...], image_hw: tuple[int, int] | None) -> torch.Tensor:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"unable to open generated future video: {path}")
    frames = []
    try:
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"unable to decode frame {frame_index} from {path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if image_hw is None:
                frames.append(torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous())
            else:
                frames.append(_resize_rgb(frame, image_hw))
    finally:
        capture.release()
    return torch.stack(frames)


def scene_is_ready(scene_dir: Path, cache_root: Path) -> bool:
    """True when a scene can produce a full Multi-GT training sample."""

    completion = scene_dir / "completion.json"
    if not completion.exists() or not (scene_dir / "scene_metadata.json").exists():
        return False
    try:
        status = json.loads(completion.read_text())
    except json.JSONDecodeError:
        return False
    if status.get("status") != "COMPLETE":
        return False
    trajectory_dirs = [path for path in scene_dir.glob("trajectory_*") if path.is_dir()]
    if not 1 <= len(trajectory_dirs) <= MAX_GT:
        return False
    if any(not (path / "future.mp4").exists() or not (path / "ego_pose.npy").exists() for path in trajectory_dirs):
        return False
    return metric_cache_path(scene_dir, cache_root).exists()


def metric_cache_path(scene_dir: Path, cache_root: Path) -> Path:
    metadata = json.loads((scene_dir / "scene_metadata.json").read_text())
    return (
        Path(cache_root)
        / str(metadata["log_name"])
        / "unknown"
        / str(metadata["token"])
        / "metric_cache.pkl"
    )


def scene_labels(scene_dir: str | Path) -> list[str]:
    """Multi-GT trajectory labels, read without decoding any video."""

    scene_dir = Path(scene_dir)
    return [
        str(json.loads((path / "metadata.json").read_text())["trajectory_label"])
        for path in sorted(scene_dir.glob("trajectory_*"))
        if path.is_dir()
    ]


def iter_ready_scenes(
    multiview_root: str | Path = DEFAULT_MULTIVIEW_ROOT,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    limit: int | None = None,
    start: int = 0,
    require_distinct_labels: bool = False,
) -> Iterator[Path]:
    root = Path(multiview_root)
    cache_root = Path(cache_root)
    emitted = 0
    for index, scene_dir in enumerate(sorted(path for path in root.iterdir() if path.is_dir())):
        if index < start:
            continue
        if not scene_is_ready(scene_dir, cache_root):
            continue
        if require_distinct_labels:
            labels = scene_labels(scene_dir)
            if len(set(labels)) < len(labels):
                continue
        yield scene_dir
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def load_multiview_sample(
    scene_dir: str | Path,
    log_root: str | Path = DEFAULT_LOG_ROOT,
    sensor_root: str | Path = DEFAULT_SENSOR_ROOT,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    image_hw: tuple[int, int] = DRIVOR_OFFICIAL_IMAGE_HW,
    future_image_hw: tuple[int, int] | None = None,
    camera: str = "CAM_F0",
    register_target_dim: int = EXPECTED_REGISTER_TARGET_DIM,
    load_future_rgb: bool = True,
    log_frames: list[dict[str, Any]] | None = None,
) -> MultiViewSample:
    """Build one Multi-GT sample: DrivoR observation + metric trajectories + videos.

    ``image_hw`` is DrivoR's official ``(672, 1148)``: history JPEGs are bicubic
    *downscaled* from the real 1920x1080 ``CAM_F0`` blob, and the scene encoder
    later closes the patch16 remainder with zero columns.

    ``future_image_hw=None`` leaves the generated frames at their native 464x832,
    which patch16 tiles exactly (29x52), so the Teacher target is built from
    rendered pixels with no resampling in either direction.

    ``register_target_dim`` sizes only the ``future_register`` *placeholder*: the
    Teacher has not been constructed yet at this point, and the real target is
    attached later by :func:`with_future_register`.  Its width is the Teacher's
    ``feature_dim`` -- DINOv3-large's 1024 -- and is deliberately **not**
    ``d_model`` (256).  The default is the constant :func:`load_v6_config` checks
    the yaml's ``future_target.register_target_dim`` against, so the placeholder,
    the config and the future head cannot drift apart.

    ``log_frames`` lets a caller reuse an already unpickled NAVSIM log, which is
    what makes multi-sample batches cheap (the log is shared by a whole scene).
    """

    scene_dir = Path(scene_dir)
    scene_metadata = json.loads((scene_dir / "scene_metadata.json").read_text())
    scene_id = str(scene_metadata["token"])
    if scene_dir.name != scene_id:
        raise ValueError(f"scene directory {scene_dir.name} does not match metadata token {scene_id}")
    log_name = str(scene_metadata["log_name"])
    frame_index = int(scene_metadata["navsim_frame_index"])

    if log_frames is None:
        log_path = Path(log_root) / f"{log_name}.pkl"
        if not log_path.exists():
            raise FileNotFoundError(f"NAVSIM log missing: {log_path}")
        with log_path.open("rb") as handle:
            log_frames = pickle.load(handle)
    if frame_index < HISTORY_FRAMES - 1 or frame_index >= len(log_frames):
        raise ValueError(f"invalid NAVSIM frame index {frame_index} for log {log_name}")
    frame = log_frames[frame_index]
    if str(frame["token"]) != scene_id:
        raise ValueError(f"scene token mismatch: metadata={scene_id}, log={frame['token']}")
    if str(frame["scene_token"]) != str(scene_metadata["scene_token"]):
        raise ValueError(f"scene_token mismatch: metadata={scene_metadata['scene_token']}, log={frame['scene_token']}")
    if int(frame["timestamp"]) != int(scene_metadata["t0_timestamp_us"]):
        raise ValueError(f"t0 timestamp mismatch: metadata={scene_metadata['t0_timestamp_us']}, log={frame['timestamp']}")
    if "driving_command" not in frame:
        raise ValueError(f"frame {frame_index} has no NAVSIM driving_command")
    route = drivor_route_command(frame["driving_command"])

    context_indices = list(range(frame_index - (HISTORY_FRAMES - 1), frame_index + 1))
    reference = np.asarray(frame["ego2global"], dtype=np.float64)
    relative_poses = torch.from_numpy(
        np.stack([_relative_pose(reference, np.asarray(log_frames[i]["ego2global"], dtype=np.float64)) for i in context_indices])
    )
    ego_features = []
    for index, pose in zip(context_indices, relative_poses):
        if "ego_dynamic_state" not in log_frames[index]:
            raise ValueError(f"frame {index} has no observable ego_dynamic_state")
        dynamic = np.asarray(log_frames[index]["ego_dynamic_state"], dtype=np.float32).reshape(-1)
        ego_features.append(drivor_observable_ego_feature(pose, torch.from_numpy(dynamic)))
    ego_history = torch.stack(ego_features)

    fallback = scene_dir / "initial_front.jpg"
    if not fallback.exists():
        raise FileNotFoundError(f"scene {scene_id} has neither sensor blobs nor initial_front.jpg")
    history_paths: list[Path | None] = []
    for index in context_indices:
        relative = Path(log_frames[index]["cams"][camera]["data_path"])
        candidate = Path(sensor_root) / relative
        history_paths.append(candidate if candidate.exists() else None)
    history_valid = torch.tensor([[path is not None for path in history_paths]], dtype=torch.bool)
    if all(path is not None for path in history_paths):
        history_policy = f"real_{camera}_sensor_blob"
    elif any(path is not None for path in history_paths):
        history_policy = f"partial_{camera}_sensor_blob_plus_initial_front"
    else:
        history_policy = "initial_front_repeated_5x"
    rgb_history = torch.stack(
        [_load_image(path if path is not None else fallback, image_hw) for path in history_paths]
    )

    trajectory_dirs = sorted(
        (path for path in scene_dir.glob("trajectory_*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
    )
    if not 1 <= len(trajectory_dirs) <= MAX_GT:
        raise ValueError(f"{scene_dir} has {len(trajectory_dirs)} GT pairs; Multi-GT allows 1..{MAX_GT}")

    trajectories = []
    future_videos = []
    pair_ids = []
    for trajectory_dir in trajectory_dirs:
        trajectory_metadata = json.loads((trajectory_dir / "metadata.json").read_text())
        if trajectory_metadata.get("status") != "COMPLETE" or not trajectory_metadata.get("valid", False):
            raise ValueError(f"{trajectory_dir} is not a COMPLETE valid generation")
        if trajectory_metadata["pose_coordinate_frame"] != "t0 local rear; x forward/y left/yaw CCW; meters/radians":
            raise ValueError(f"{trajectory_dir} uses an unexpected pose frame")
        poses = np.load(trajectory_dir / "ego_pose.npy").astype(np.float32)
        if poses.shape != (DENSE_FRAMES, 3):
            raise ValueError(f"expected dense [{DENSE_FRAMES},3] poses at {trajectory_dir}, got {poses.shape}")
        if not np.allclose(poses[0], 0.0, atol=1e-6):
            raise ValueError(f"{trajectory_dir} poses are not anchored at the t0 origin")
        trajectories.append(torch.from_numpy(poses[list(FUTURE_POSE_INDICES)]))
        if load_future_rgb:
            future_videos.append(
                _read_video_frames(trajectory_dir / "future.mp4", FUTURE_POSE_INDICES, future_image_hw)
            )
        pair_ids.append(
            {
                "gt_id": str(trajectory_dir.name.rsplit("_", 1)[-1]),
                "original_gt_id": str(trajectory_metadata["source_selected_candidate_index"]),
                "source": str(trajectory_dir),
                "label": str(trajectory_metadata["trajectory_label"]),
                "behavior": str(trajectory_metadata.get("behavior")),
                "is_human_log": bool(trajectory_metadata.get("is_human_log", False)),
                "pdms": float(trajectory_metadata["PDMS"]),
                "video_sha256": trajectory_metadata["output_artifact_sha256"]["future.mp4"],
            }
        )

    gt_trajectory = torch.stack(trajectories)
    gt_count = gt_trajectory.shape[0]
    labels = [pair["label"] for pair in pair_ids]
    targets = PairTargets(
        gt_trajectory=gt_trajectory.unsqueeze(0),
        gt_valid=torch.ones(1, gt_count, dtype=torch.bool),
        future_register=torch.zeros(1, gt_count, REGISTER_TOKENS, register_target_dim),
        future_chunk_valid=torch.ones(1, gt_count, REGISTER_TOKENS, dtype=torch.bool),
        future_quality=torch.ones(1, gt_count, REGISTER_TOKENS),
        pair_ids=[pair_ids],
    )
    targets.validate(max_gt=MAX_GT, registers=REGISTER_TOKENS)

    cache_path = metric_cache_path(scene_dir, Path(cache_root))
    if not cache_path.exists():
        raise FileNotFoundError(f"DrivoR metric cache missing: {cache_path}")
    observation = Observation(
        rgb_history=rgb_history.unsqueeze(0).unsqueeze(2),
        history_valid=history_valid.unsqueeze(2),
        ego_history=ego_history.unsqueeze(0),
        route_command=route.unsqueeze(0),
        relative_ego_poses=relative_poses.unsqueeze(0),
        camera_metadata={
            "camera": camera,
            "log_name": log_name,
            "history_paths": [str(path) if path is not None else str(fallback) for path in history_paths],
            "history_policy": history_policy,
        },
    )
    observation.validate()
    future_rgb = None
    if load_future_rgb:
        future_rgb = torch.stack(future_videos).unsqueeze(0).unsqueeze(3)
    metadata = {
        "scene_dir": str(scene_dir),
        "scene_id": scene_id,
        "scene_token": str(scene_metadata["scene_token"]),
        "log_name": log_name,
        "frame_index": frame_index,
        "timestamp_us": int(scene_metadata["t0_timestamp_us"]),
        "map_location": str(scene_metadata.get("map_location")),
        "command": str(scene_metadata.get("command")),
        "history_policy": history_policy,
        "history_indices": context_indices,
        "future_pose_indices": list(FUTURE_POSE_INDICES),
        "image_hw": [int(image_hw[0]), int(image_hw[1])],
        "future_image_hw": None if future_image_hw is None else [int(future_image_hw[0]), int(future_image_hw[1])],
        "gt_count": gt_count,
        "labels": labels,
        "degenerate_multi_gt": len(set(labels)) < len(labels),
        "paired_pdms": [pair["pdms"] for pair in pair_ids],
        "cache_path": str(cache_path),
    }
    return MultiViewSample(
        observation=observation,
        targets=targets,
        scene_id=scene_id,
        log_name=log_name,
        frame_index=frame_index,
        cache_path=cache_path,
        future_rgb=future_rgb,
        metadata=metadata,
    )


def with_future_register(
    targets: PairTargets,
    future_register: torch.Tensor,
    future_quality: torch.Tensor | None = None,
) -> PairTargets:
    """Attach the query-extracted, standardized Teacher target, keeping every other field intact.

    The comparison covers the width too, so a placeholder allocated at the wrong
    ``register_target_dim`` fails here rather than as an unexplained broadcast inside
    the future loss; when it is only the width that differs the message says which
    side to fix, since both shapes are otherwise just numbers.
    """

    if future_register.shape != targets.future_register.shape:
        expected, got = tuple(targets.future_register.shape), tuple(future_register.shape)
        hint = (
            ""
            if expected[-1] == got[-1]
            else "; the last axis is the Teacher's feature_dim, so the placeholder was allocated by "
            f"load_multiview_sample(register_target_dim={expected[-1]})"
        )
        raise ValueError(f"future register target must be {expected}, got {got}{hint}")
    quality = targets.future_quality if future_quality is None else future_quality
    if quality.shape != targets.future_quality.shape:
        raise ValueError("future quality must match the register target shape")
    return replace(
        targets,
        future_register=future_register,
        future_quality=quality,
        future_chunk_valid=torch.ones_like(targets.future_chunk_valid),
    )
