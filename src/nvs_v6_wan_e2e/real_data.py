"""Real NAVSIM scene adapter for the v6 single-card smoke run.

Only observations and externally prepared targets cross into the student. Future RGB is
consumed by the fixed Teacher target builder and never by the student or scorer.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .data import (
    Observation,
    PairTargets,
    drivor_observable_ego_feature,
    drivor_route_command,
)
from .vision import DRIVOR_OFFICIAL_IMAGE_HW


@dataclass
class RealNavsimSample:
    observation: Observation
    targets: PairTargets
    scene_id: str
    log_name: str
    frame_index: int
    frame_token: str
    cache_path: Path
    future_rgb: torch.Tensor
    metadata: dict[str, Any]


def _load_image(path: Path, image_hw: tuple[int, int]) -> torch.Tensor:
    """DrivoR's preprocessing: PIL resize to (W,H), /255 only -- no normalization.

    ``DrivoRDinoV3Student._prepare`` applies the ImageNet mean/std, exactly where
    DrivoR's ``DrivoRFeatureBuilder`` applies it, so loading stays raw here.
    """

    height, width = int(image_hw[0]), int(image_hw[1])
    with Image.open(path) as image:
        image = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _relative_pose(reference: np.ndarray, frame: np.ndarray) -> np.ndarray:
    relative = np.linalg.inv(reference) @ frame
    return np.asarray(
        [relative[0, 3], relative[1, 3], np.arctan2(relative[1, 0], relative[0, 0])],
        dtype=np.float32,
    )


def _frame_image(frame: dict[str, Any], sensor_root: Path, camera: str, image_hw: tuple[int, int]) -> torch.Tensor:
    relative = Path(frame["cams"][camera]["data_path"])
    return _load_image(sensor_root / relative, image_hw)


def _find_multigt(multigt_root: Path, frame_token: str) -> Path:
    for split in ("navtrain", "navval"):
        candidate = multigt_root / split / frame_token
        if candidate.is_dir() and (candidate / "gt_trajectories.npy").exists():
            return candidate
    raise FileNotFoundError(f"no Multi-GT directory for frame token {frame_token} under {multigt_root}")


def load_real_navsim_sample(
    log_path: str | Path,
    sensor_root: str | Path,
    multigt_root: str | Path,
    frame_index: int = 755,
    camera: str = "CAM_F0",
    image_hw: tuple[int, int] = DRIVOR_OFFICIAL_IMAGE_HW,
    cache_root: str | Path = "/mnt/zhaozc_workspace/project/DrivoR_multigt_kvcache_final_20261004/data_runtime/train_metric_cache",
    target_dim: int = 256,
) -> RealNavsimSample:
    log_path = Path(log_path)
    sensor_root = Path(sensor_root)
    multigt_root = Path(multigt_root)
    cache_root = Path(cache_root)
    with log_path.open("rb") as handle:
        frames = pickle.load(handle)
    if frame_index < 4 or frame_index + 8 >= len(frames):
        raise ValueError(f"frame_index={frame_index} does not leave v6 history/future context")
    frame = frames[frame_index]
    log_name = str(frame["log_name"])
    frame_token = str(frame["token"])
    context_indices = list(range(frame_index - 4, frame_index + 1))
    future_indices = list(range(frame_index + 1, frame_index + 9))
    reference = np.asarray(frame["ego2global"], dtype=np.float64)

    source_image_paths = [sensor_root / Path(frames[i]["cams"][camera]["data_path"]) for i in context_indices]
    source_image_hw = []
    for image_path in source_image_paths:
        with Image.open(image_path) as source_image:
            source_image_hw.append([int(source_image.height), int(source_image.width)])
    rgb_history = torch.stack([_frame_image(frames[i], sensor_root, camera, image_hw) for i in context_indices])
    future_rgb = torch.stack([_frame_image(frames[i], sensor_root, camera, image_hw) for i in future_indices])
    relative_poses = torch.from_numpy(np.stack([_relative_pose(reference, np.asarray(frames[i]["ego2global"], dtype=np.float64)) for i in context_indices]))
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

    multi_gt_dir = _find_multigt(multigt_root, frame_token)
    trajectories = torch.from_numpy(np.load(multi_gt_dir / "gt_trajectories.npy").astype(np.float32))
    valid = torch.from_numpy(np.load(multi_gt_dir / "gt_valid_mask.npy").astype(bool))
    if trajectories.ndim != 3 or trajectories.shape[1:] != (8, 3):
        raise ValueError(f"unexpected Multi-GT shape {tuple(trajectories.shape)} at {multi_gt_dir}")
    gt_count = trajectories.shape[0]
    # This adapter has no Teacher target: the Future Register must be marked
    # INVALID rather than valid-with-zeros.  Zeros carrying chunk_valid=True
    # silently train the register toward the origin, which is why the joint
    # trainer had to freeze phi and hardcode lambda_future=0.  Real targets come
    # from scripts/build_future_target_cache.py and are read back by
    # target_cache.Stage1SceneDataset / scene_target_path.
    future_register = torch.zeros(gt_count, 4, target_dim, dtype=torch.float32)
    future_chunk_valid = torch.zeros(gt_count, 4, dtype=torch.bool)
    future_quality = torch.zeros(gt_count, 4, dtype=torch.float32)
    targets = PairTargets(
        gt_trajectory=trajectories.unsqueeze(0),
        gt_valid=valid.unsqueeze(0),
        future_register=future_register.unsqueeze(0),
        future_chunk_valid=future_chunk_valid.unsqueeze(0),
        future_quality=future_quality.unsqueeze(0),
        pair_ids=[[{"gt_id": str(index), "original_gt_id": str(index), "source": str(multi_gt_dir)} for index in range(gt_count)]],
    )
    cache_path = cache_root / log_name / "unknown" / frame_token / "metric_cache.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(f"DrivoR metric cache missing: {cache_path}")
    observation = Observation(
        rgb_history=rgb_history.unsqueeze(0).unsqueeze(2),
        history_valid=torch.ones(1, 5, 1, dtype=torch.bool),
        ego_history=ego_history.unsqueeze(0),
        route_command=route.unsqueeze(0),
        relative_ego_poses=relative_poses.unsqueeze(0),
        camera_metadata={"camera": camera, "log_path": str(log_path)},
    )
    metadata = {
        "log_path": str(log_path),
        "log_name": log_name,
        "frame_token": frame_token,
        "frame_timestamp_us": int(frame["timestamp"]),
        "multigt_dir": str(multi_gt_dir),
        "history_indices": context_indices,
        "future_indices": future_indices,
        "source_image_hw": source_image_hw,
        "loaded_observation_shape": [1, 5, 1, 3, int(image_hw[0]), int(image_hw[1])],
        "camera_count": 1,
        "history_frames": 4,
        "cache_path": str(cache_path),
    }
    return RealNavsimSample(observation, targets, frame_token, log_name, frame_index, frame_token, cache_path, future_rgb.unsqueeze(0).unsqueeze(2), metadata)
