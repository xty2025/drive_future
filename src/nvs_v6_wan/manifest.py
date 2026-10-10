"""Pair manifest construction, duplicate handling, and future validity metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PairRecord:
    scene_id: str
    split: str
    t0_timestamp: float
    gt_id: str
    original_gt_id: str
    trajectory_path: str
    trajectory_coordinate_frame: str
    trajectory_pose_hash: str
    future_rgb_paths: tuple[str, ...]
    future_times_s: tuple[float, ...]
    future_camera_id: str
    camera_calibration_hash: str
    render_world_version: str
    frame_valid: tuple[bool, ...]
    chunk_valid: tuple[bool, ...]
    quality: tuple[float, ...]
    teacher_checkpoint_hash: str
    compressor_hash: str
    target_cache_path: str


def deduplicate_pairs(records: Iterable[PairRecord]) -> list[PairRecord]:
    unique: dict[tuple[str, str], PairRecord] = {}
    for record in records:
        key = (record.scene_id, record.original_gt_id)
        if key not in unique:
            unique[key] = record
    return list(unique.values())


def write_manifest(records: Iterable[PairRecord], path: str | Path) -> str:
    rows = [asdict(record) for record in deduplicate_pairs(records)]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()
    Path(path).write_text(json.dumps({"records": rows, "manifest_hash": digest}, indent=2) + "\n")
    return digest


def read_manifest(path: str | Path) -> tuple[list[PairRecord], str]:
    data = json.loads(Path(path).read_text())
    records = [PairRecord(**row) for row in data["records"]]
    return records, str(data["manifest_hash"])
