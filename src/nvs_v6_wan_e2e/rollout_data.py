"""Adapters for the small paired Cosmos/NAVSIM rollout dataset.

The generated future videos are target-only inputs.  Student observations use the
scene's initial image, repeated across the five required history slots when the
original historical sensor blobs are unavailable.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from .data import (
    Observation,
    PairTargets,
    drivor_observable_ego_feature,
    drivor_route_command,
    encode_trajectory,
)


@dataclass
class PairedRolloutSample:
    observation: Observation
    targets: PairTargets
    scene_id: str
    log_name: str
    frame_index: int
    cache_path: Path
    future_rgb: torch.Tensor
    metadata: dict[str, Any]


def _load_image(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _relative_pose(reference: np.ndarray, frame: np.ndarray) -> np.ndarray:
    relative = np.linalg.inv(reference) @ frame
    return np.asarray(
        [relative[0, 3], relative[1, 3], np.arctan2(relative[1, 0], relative[0, 0])],
        dtype=np.float32,
    )


def _read_video_frames(path: Path, frame_indices: tuple[int, ...], image_size: int) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"unable to open paired future video: {path}")
    frames = []
    try:
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"unable to decode frame {frame_index} from {path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            frames.append(torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous())
    finally:
        capture.release()
    return torch.stack(frames)


def load_paired_rollout_sample(
    scene_dir: str | Path,
    cache_root: str | Path,
    image_size: int = 224,
    target_dim: int = 256,
) -> PairedRolloutSample:
    scene_dir = Path(scene_dir)
    cache_root = Path(cache_root)
    scene_metadata = json.loads((scene_dir / "scene_metadata.json").read_text())
    scene_id = str(scene_metadata["token"])
    log_name = str(scene_metadata["log_name"])
    frame_index = int(scene_metadata["navsim_frame_index"])
    log_path = Path(scene_metadata["source_navsim_log"]["path"])
    with log_path.open("rb") as handle:
        frames = pickle.load(handle)
    if frame_index < 4 or frame_index >= len(frames):
        raise ValueError(f"invalid NAVSIM frame index {frame_index} for {log_path}")
    frame = frames[frame_index]
    if str(frame["token"]) != scene_id:
        raise ValueError(f"scene token mismatch: metadata={scene_id}, log={frame['token']}")

    initial_path = scene_dir / "initial_front.jpg"
    with Image.open(initial_path) as source_image:
        source_image_hw = [int(source_image.height), int(source_image.width)]
    initial = _load_image(initial_path, image_size)
    rgb_history = initial.unsqueeze(0).repeat(5, 1, 1, 1)
    context_indices = list(range(frame_index - 4, frame_index + 1))
    reference = np.asarray(frame["ego2global"], dtype=np.float64)
    relative_poses = torch.from_numpy(
        np.stack([_relative_pose(reference, np.asarray(frames[index]["ego2global"], dtype=np.float64)) for index in context_indices])
    )
    ego_features = []
    for index, pose in zip(context_indices, relative_poses):
        if "ego_dynamic_state" not in frames[index]:
            raise ValueError(f"frame {index} has no observable ego_dynamic_state")
        dynamic = np.asarray(frames[index]["ego_dynamic_state"], dtype=np.float32).reshape(-1)
        ego_features.append(drivor_observable_ego_feature(pose, torch.from_numpy(dynamic)))
    ego_history = torch.stack(ego_features)
    if "driving_command" not in frame:
        raise ValueError(f"frame {frame_index} has no NAVSIM driving_command")
    route = drivor_route_command(frame["driving_command"])

    trajectory_dirs = sorted(
        (path for path in scene_dir.glob("trajectory_*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
    )
    if not trajectory_dirs:
        raise FileNotFoundError(f"no paired trajectory directories under {scene_dir}")
    if len(trajectory_dirs) > 4:
        raise ValueError(f"{scene_dir} has {len(trajectory_dirs)} GT pairs; the Multi-GT contract allows at most 4")
    trajectories = []
    future_videos = []
    pair_ids = []
    paired_pdms = []
    sample_indices = np.arange(5, 41, 5)
    for trajectory_dir in trajectory_dirs:
        trajectory_index = int(trajectory_dir.name.rsplit("_", 1)[-1])
        trajectory_metadata = json.loads((trajectory_dir / "metadata.json").read_text())
        poses = np.load(trajectory_dir / "ego_pose.npy").astype(np.float32)
        if poses.shape != (41, 3):
            raise ValueError(f"expected dense [41,3] trajectory at {trajectory_dir}, got {poses.shape}")
        trajectories.append(torch.from_numpy(poses[sample_indices]))
        future_videos.append(_read_video_frames(trajectory_dir / "future.mp4", tuple(sample_indices.tolist()), image_size))
        paired_pdms.append(float(trajectory_metadata["PDMS"]))
        pair_ids.append(
            {
                "gt_id": str(trajectory_index),
                "original_gt_id": str(trajectory_metadata.get("source_selected_candidate_index", trajectory_index)),
                "source": str(trajectory_dir),
                "label": str(trajectory_metadata.get("trajectory_label", trajectory_index)),
                "pdms": paired_pdms[-1],
            }
        )

    gt_trajectory = torch.stack(trajectories)
    gt_count = len(trajectory_dirs)
    gt_valid = torch.ones(gt_count, dtype=torch.bool)
    targets = PairTargets(
        gt_trajectory=gt_trajectory.unsqueeze(0),
        gt_encoded=encode_trajectory(gt_trajectory).unsqueeze(0),
        gt_valid=gt_valid.unsqueeze(0),
        future_register=torch.zeros(1, gt_count, 4, target_dim),
        future_chunk_valid=torch.ones(1, gt_count, 4, dtype=torch.bool),
        future_quality=torch.ones(1, gt_count, 4),
        pair_ids=[pair_ids],
    )
    cache_path = cache_root / log_name / "unknown" / scene_id / "metric_cache.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(f"DrivoR metric cache missing: {cache_path}")
    observation = Observation(
        rgb_history=rgb_history.unsqueeze(0).unsqueeze(2),
        history_valid=torch.ones(1, 5, 1, dtype=torch.bool),
        ego_history=ego_history.unsqueeze(0),
        route_command=route.unsqueeze(0),
        relative_ego_poses=relative_poses.unsqueeze(0),
        camera_metadata={"source": str(scene_dir / "initial_front.jpg"), "history_policy": "repeat-current-no-future"},
    )
    future_rgb = torch.stack(future_videos).unsqueeze(0).unsqueeze(3)
    metadata = {
        "scene_dir": str(scene_dir),
        "log_path": str(log_path),
        "log_name": log_name,
        "scene_id": scene_id,
        "scene_token": str(scene_metadata["scene_token"]),
        "frame_index": frame_index,
        "timestamp_us": int(scene_metadata["t0_timestamp_us"]),
        "history_policy": "initial_front_repeated_5x",
        "source_image_hw": source_image_hw,
        "loaded_observation_shape": [1, 5, 1, 3, image_size, image_size],
        "camera_count": 1,
        "history_frames": 4,
        "future_frame_indices": sample_indices.tolist(),
        "paired_pdms": paired_pdms,
        "gt_count": gt_count,
        "cache_path": str(cache_path),
    }
    return PairedRolloutSample(observation, targets, scene_id, log_name, frame_index, cache_path, future_rgb, metadata)
