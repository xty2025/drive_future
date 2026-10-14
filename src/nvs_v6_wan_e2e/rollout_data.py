"""Adapter for the older paired Cosmos/NAVSIM rollout dumps.

Those dumps share the on-disk schema of ``/mnt/data/multi_view`` (a
``scene_metadata.json`` plus ``trajectory_k/{future.mp4,ego_pose.npy,
metadata.json}``), so this module is a thin wrapper over
:mod:`nvs_v6_wan_e2e.multiview_data` that keeps the historical call signature and
resolves the NAVSIM log from a local root instead of the recorded ``/cpfs`` path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .data import Observation, PairTargets
from .multiview_data import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_LOG_ROOT,
    DEFAULT_SENSOR_ROOT,
    load_multiview_sample,
)
from .vision import DRIVOR_OFFICIAL_IMAGE_HW


@dataclass
class PairedRolloutSample:
    observation: Observation
    targets: PairTargets
    scene_id: str
    log_name: str
    frame_index: int
    cache_path: Path
    future_rgb: torch.Tensor | None
    metadata: dict[str, Any]


def load_paired_rollout_sample(
    scene_dir: str | Path,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    image_hw: tuple[int, int] = DRIVOR_OFFICIAL_IMAGE_HW,
    future_image_hw: tuple[int, int] | None = None,
    target_dim: int = 256,
    log_root: str | Path = DEFAULT_LOG_ROOT,
    sensor_root: str | Path = DEFAULT_SENSOR_ROOT,
    load_future_rgb: bool = True,
) -> PairedRolloutSample:
    sample = load_multiview_sample(
        scene_dir,
        log_root=log_root,
        sensor_root=sensor_root,
        cache_root=cache_root,
        image_hw=image_hw,
        future_image_hw=future_image_hw,
        target_dim=target_dim,
        load_future_rgb=load_future_rgb,
    )
    return PairedRolloutSample(
        observation=sample.observation,
        targets=sample.targets,
        scene_id=sample.scene_id,
        log_name=sample.log_name,
        frame_index=sample.frame_index,
        cache_path=sample.cache_path,
        future_rgb=sample.future_rgb,
        metadata=sample.metadata,
    )
