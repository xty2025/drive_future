"""v6 observation/target boundaries and the Multi-GT ``.npz`` loader."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

#: Future frames decoded per GT trajectory, at +0.5..+4.0s.  Nothing is pooled across
#: time: an earlier revision folded each *pair* of frames into one target, which put
#: +0.5s and +1.0s in the same linear subspace and left the token able to tell them
#: apart only through the shape of the PCA basis.  This is also the trajectory's pose
#: count, so the frame axis has a 1:1 correspondence with the eight waypoints.
FUTURE_FRAMES = 8

#: How many of DINOv3's own prefix queries read each future frame, and therefore how
#: many register tokens one frame contributes.
#:
#: CLS alone is not enough, and the four registers read out verbatim are no better than
#: it.  On 12 scenes (48 GTs, 384 frames), standardized, the within-scene/across-scene
#: separation ratio is 0.211 for ``cls_only`` and 0.194 for ``dinov3_registers``, against
#: 0.456 for the CLS used as a *query* in the cosine attention.  What works is using the
#: four registers as queries in that same attention: 0.458 separation and, decisively,
#: 0.771 correlation between target distance and trajectory distance against the CLS
#: query's 0.684 -- because the four land on complementary regions (left road edge, right
#: verge, intersection, traffic furniture) where one CLS dot lands on a handful of pixels.
#: Four is DINOv3's own prefix count at this resolution, so "more tokens" means the
#: Teacher's trained queries rather than an invented split of the grid.  Measured by
#: ``scripts/probe_future_readout.py``; figures in ``artifacts/future_readout_probe/``.
#:
#: Defined here rather than next to the readout because it sets the *target axis* every
#: consumer shares -- ``PairTargets``, the Stage-0 cache, the loader's placeholders and
#: the DiT's ``future_tokens`` -- and ``data`` is the module none of those can import
#: from in the other direction.
REGISTER_QUERIES_PER_FRAME = 4

#: Length of the register axis: ``[B, K, REGISTER_TOKENS, target_dim]``, frame-major so
#: token ``q + REGISTER_QUERIES_PER_FRAME * f`` is query ``q`` of frame ``f``.  The
#: per-frame multiplicity multiplies the token count and leaves the frame axis alone:
#: the DiT's RoPE phase and the trajectory alignment stay per frame (see
#: ``wan_dit.NvsActionDiT.register_rope``), and ``future_chunk_valid``/``future_quality``
#: repeat each frame's value across its queries so the weighted L1 keeps reducing over
#: tokens as an element-mean.
REGISTER_TOKENS = FUTURE_FRAMES * REGISTER_QUERIES_PER_FRAME


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
        if self.ego_history.shape[-1] != 7:
            raise ValueError("ego_history must be DrivoR's [pose3, velocity2, acceleration2]")
        if self.route_command.shape != (self.rgb_history.shape[0], 4):
            raise ValueError("route_command must be the NAVSIM/DrivoR four-way command [B,4]")


def drivor_observable_ego_feature(relative_pose: torch.Tensor, dynamic_state: torch.Tensor) -> torch.Tensor:
    """DrivoR's per-frame ego state: ``pose3, velocity2, acceleration2``.

    The four-way ``driving_command`` is carried separately and concatenated by
    ``SceneEncoder.encode_clean_ego``, matching DrivoR's feature builder order.
    """

    relative_pose = torch.as_tensor(relative_pose, dtype=torch.float32).flatten()
    dynamic_state = torch.as_tensor(dynamic_state, dtype=torch.float32).flatten()
    if relative_pose.numel() != 3 or dynamic_state.numel() != 4:
        raise ValueError("NAVSIM ego input must contain pose3 and velocity/acceleration4")
    return torch.cat((relative_pose, dynamic_state))


def drivor_route_command(value: Any) -> torch.Tensor:
    command = torch.as_tensor(value, dtype=torch.float32).flatten()
    if command.numel() != 4:
        raise ValueError("NAVSIM driving_command must contain exactly four values")
    return command


@dataclass
class PairTargets:
    """DrivoR-native metric targets; nothing here is rescaled or re-encoded."""

    gt_trajectory: torch.Tensor
    gt_valid: torch.Tensor
    future_register: torch.Tensor
    future_chunk_valid: torch.Tensor
    future_quality: torch.Tensor
    pair_ids: Any

    def validate(self, max_gt: int = 4, registers: int = REGISTER_TOKENS) -> None:
        if self.gt_trajectory.ndim != 4 or self.gt_trajectory.shape[-2:] != (8, 3):
            raise ValueError("gt_trajectory must be [B,K,8,3] metric x,y,heading")
        if self.gt_valid.shape != self.gt_trajectory.shape[:2]:
            raise ValueError("gt_valid must be [B,K]")
        if self.gt_trajectory.shape[1] > max_gt:
            raise ValueError(f"at most {max_gt} unique GT trajectories are supported")
        if self.future_register.shape[:2] != self.gt_trajectory.shape[:2]:
            raise ValueError("future_register and trajectory must share B,K")
        if self.future_register.shape[2] != registers:
            raise ValueError(f"future_register must contain {registers} ordered register targets")
        if self.future_chunk_valid.shape != self.future_register.shape[:3]:
            raise ValueError(f"future_chunk_valid must be [B,K,{registers}]")
        if self.future_quality.shape != self.future_register.shape[:3]:
            raise ValueError(f"future_quality must be [B,K,{registers}]")


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
    k = trajectories.shape[0]
    future = torch.zeros(k, REGISTER_TOKENS, target_dim, dtype=torch.float32)
    chunk_valid = torch.zeros(k, REGISTER_TOKENS, dtype=torch.bool)
    quality = torch.ones(k, REGISTER_TOKENS, dtype=torch.float32)
    pair_ids = [{"gt_id": str(i), "original_gt_id": str(i), "source": str(path)} for i in range(k)]
    return PairTargets(
        gt_trajectory=trajectories.unsqueeze(0),
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
