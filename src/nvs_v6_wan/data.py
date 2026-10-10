"""v6 observation/target boundaries and Multi-GT manifest helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch


@dataclass
class Observation:
    rgb_history: torch.Tensor
    history_valid: torch.Tensor
    ego_history: torch.Tensor
    route_command: torch.Tensor
    relative_ego_poses: torch.Tensor
    camera_metadata: Any = None

    def validate(self) -> None:
        if self.rgb_history.ndim != 6:
            raise ValueError("rgb_history must be [B,5,V,3,H,W]")
        if self.history_valid.shape[:3] != self.rgb_history.shape[:3]:
            raise ValueError("history_valid must align with B,T,V")
        if self.rgb_history.shape[1] != 5:
            raise ValueError("v6 requires current plus four history frames")
        if self.ego_history.shape[:2] != self.rgb_history.shape[:2]:
            raise ValueError("ego_history must align with B,T")
        if self.ego_history.shape[-1] != 8:
            raise ValueError("ego_history must be [pose3, velocity2, acceleration2, reserved0]")
        if self.route_command.shape != (self.rgb_history.shape[0], 4):
            raise ValueError("route_command must be the NAVSIM/DrivoR four-way command [B,4]")


def drivor_observable_ego_feature(relative_pose: torch.Tensor, dynamic_state: torch.Tensor) -> torch.Tensor:
    """Build DrivoR-ordered observable ego state plus one checkpoint-compatible pad.

    The first seven values are exactly DrivoR's current pose3, velocity2 and
    acceleration2.  The final zero preserves the existing NVS ``ego_dim=8``
    checkpoint shape; ``SceneEncoder`` inserts the four-way command before it.
    """

    relative_pose = torch.as_tensor(relative_pose, dtype=torch.float32).flatten()
    dynamic_state = torch.as_tensor(dynamic_state, dtype=torch.float32).flatten()
    if relative_pose.numel() != 3 or dynamic_state.numel() != 4:
        raise ValueError("NAVSIM ego input must contain pose3 and velocity/acceleration4")
    return torch.cat((relative_pose, dynamic_state, relative_pose.new_zeros(1)))


def drivor_route_command(value: Any) -> torch.Tensor:
    command = torch.as_tensor(value, dtype=torch.float32).flatten()
    if command.numel() != 4:
        raise ValueError("NAVSIM driving_command must contain exactly four values")
    return command


@dataclass
class PairTargets:
    gt_trajectory: torch.Tensor
    gt_encoded: torch.Tensor
    gt_valid: torch.Tensor
    future_register: torch.Tensor
    future_chunk_valid: torch.Tensor
    future_quality: torch.Tensor
    pair_ids: Any

    def validate(self, max_gt: int = 4, registers: int = 4) -> None:
        if self.gt_trajectory.ndim != 4 or self.gt_trajectory.shape[-2:] != (8, 3):
            raise ValueError("gt_trajectory must be [B,K,8,3]")
        if self.gt_encoded.shape[-2:] != (8, 4):
            raise ValueError("gt_encoded must be [B,K,8,4]")
        if self.gt_trajectory.shape[1] > max_gt:
            raise ValueError(f"at most {max_gt} unique GT trajectories are supported")
        if self.future_register.shape[:2] != self.gt_trajectory.shape[:2]:
            raise ValueError("future_register and trajectory must share B,K")
        if self.future_register.shape[2] != registers:
            raise ValueError("future_register must contain four ordered chunks")
        if self.future_chunk_valid.shape != self.future_register.shape[:3]:
            raise ValueError("future_chunk_valid must be [B,K,4]")
        if self.future_quality.shape != self.future_register.shape[:3]:
            raise ValueError("future_quality must be [B,K,4]")


class FutureViewProvider(Protocol):
    def get(self, scene_id: str, gt_id: str, timestamps_s: list[float], camera_id: str = "front") -> dict[str, Any]:
        ...


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_multigt_npz(path: str | Path, target_dim: int = 256) -> PairTargets:
    """Load DriveZero's 0.5s Multi-GT artifact without exposing it to the model."""
    import numpy as np

    path = Path(path)
    data = np.load(path, allow_pickle=True)
    trajectories = torch.as_tensor(data["trajectories"], dtype=torch.float32)
    valid = torch.as_tensor(data["valid_mask"], dtype=torch.bool)
    if trajectories.ndim != 3 or trajectories.shape[-2:] != (4, 3):
        raise ValueError(f"{path} is not a v6 8-step trajectory artifact")
    trajectories = _resample_or_pad_trajectory(trajectories, 8)
    encoded = encode_trajectory(trajectories, xy_scale=20.0)
    k = trajectories.shape[0]
    future = torch.zeros(k, 4, target_dim, dtype=torch.float32)
    chunk_valid = torch.zeros(k, 4, dtype=torch.bool)
    quality = torch.ones(k, 4, dtype=torch.float32)
    pair_ids = [{"gt_id": str(i), "original_gt_id": str(i), "source": str(path)} for i in range(k)]
    return PairTargets(
        gt_trajectory=trajectories.unsqueeze(0),
        gt_encoded=encoded.unsqueeze(0),
        gt_valid=valid.unsqueeze(0),
        future_register=future.unsqueeze(0),
        future_chunk_valid=chunk_valid.unsqueeze(0),
        future_quality=quality.unsqueeze(0),
        pair_ids=[pair_ids],
    )


def _resample_or_pad_trajectory(trajectory: torch.Tensor, steps: int) -> torch.Tensor:
    if trajectory.shape[1] == steps:
        return trajectory
    source = torch.linspace(0.0, 1.0, trajectory.shape[1])
    target = torch.linspace(0.0, 1.0, steps)
    result = torch.empty(trajectory.shape[0], steps, trajectory.shape[2], dtype=trajectory.dtype)
    for index in range(trajectory.shape[0]):
        for channel in range(trajectory.shape[2]):
            result[index, :, channel] = torch.from_numpy(
                __import__("numpy").interp(target.numpy(), source.numpy(), trajectory[index, :, channel].numpy())
            )
    return result


def encode_trajectory(trajectory: torch.Tensor, xy_scale: float = 20.0) -> torch.Tensor:
    if trajectory.shape[-1] != 3:
        raise ValueError("trajectory must end in x,y,yaw")
    x, y, yaw = trajectory.unbind(-1)
    return torch.stack((x / xy_scale, y / xy_scale, yaw.sin(), yaw.cos()), dim=-1)


def decode_trajectory(encoded: torch.Tensor, xy_scale: float = 20.0, source_yaw: torch.Tensor | None = None) -> torch.Tensor:
    x = encoded[..., 0] * xy_scale
    y = encoded[..., 1] * xy_scale
    sin_yaw, cos_yaw = encoded[..., 2], encoded[..., 3]
    yaw = torch.atan2(sin_yaw, cos_yaw)
    if source_yaw is not None:
        norm = torch.sqrt(sin_yaw.square() + cos_yaw.square())
        yaw = torch.where(norm < 1e-4, source_yaw, yaw)
    return torch.stack((x, y, yaw), dim=-1)
