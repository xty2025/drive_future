from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for source in (REPO_ROOT / "src", REPO_ROOT / "vendor" / "DrivoR", REPO_ROOT / "vendor" / "DrivoR" / "nuplan-devkit"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from nvs_v6_wan_e2e.target_cache import scene_target_path
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_navtrain_wan_e2e_ddp.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("stage2_navtrain", SCRIPT_PATH)
stage2 = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = stage2
SCRIPT_SPEC.loader.exec_module(stage2)


ROW_KEYS = (
    "gt_trajectory",
    "gt_valid",
    "future_register",
    "future_chunk_valid",
    "future_quality",
)


def _future_rows(count: int, offset: float = 0.0) -> dict[str, torch.Tensor]:
    return {
        "gt_trajectory": torch.full((count, 8, 3), offset + 1),
        "gt_valid": torch.ones(count, dtype=torch.bool),
        "future_register": torch.full((count, 32, 1024), offset + 2),
        "future_chunk_valid": torch.ones(count, 32, dtype=torch.bool),
        "future_quality": torch.full((count, 32), offset + 3),
    }


def _dataset(cache: Path | None = None, fingerprint: str = "shared") -> stage2.NavtrainSceneDataset:
    dataset = stage2.NavtrainSceneDataset(
        candidates=[], sensor_root="", multigt_root="", cache_root="", log_root="",
        target_cache_root=None, register_target_dim=1024, target_fingerprint=fingerprint,
        negative_target_cache_root=str(cache) if cache is not None else None,
    )
    return dataset


def test_negative_future_rows_load_from_cache(tmp_path):
    dataset = _dataset(tmp_path)
    path = scene_target_path(tmp_path, "scene-a")
    path.parent.mkdir(parents=True)
    expected = _future_rows(3)
    torch.save({"format": 2, **expected, "target_fingerprint": "shared"}, path)

    loaded = dataset._load_negative_rows("scene-a")

    assert tuple(dataset.NEGATIVE_ROW_KEYS) == ROW_KEYS
    assert set(loaded) == set(ROW_KEYS)
    for key in ROW_KEYS:
        assert torch.equal(loaded[key], expected[key])


def test_positive_and_negative_rows_are_merged_in_order(monkeypatch, tmp_path):
    positive = _future_rows(2, 10)
    negative = _future_rows(3, 20)
    observation = SimpleNamespace(
        rgb_history=torch.zeros(1, 5, 3, 2, 2), history_valid=torch.ones(1, 5, dtype=torch.bool),
        ego_history=torch.zeros(1, 5, 11), route_command=torch.zeros(1, 4),
        relative_ego_poses=torch.zeros(1, 5, 3),
    )
    targets = SimpleNamespace(gt_trajectory=positive["gt_trajectory"].unsqueeze(0),
                              gt_valid=positive["gt_valid"].unsqueeze(0))
    monkeypatch.setattr(stage2, "load_real_navsim_sample", lambda *args, **kwargs: SimpleNamespace(
        observation=observation, targets=targets,
    ))
    candidate = stage2.Candidate(token="scene-a", log_name="unused", cache_path="unused", frame_index=0)
    dataset = _dataset(tmp_path)
    dataset.candidates = [candidate]
    monkeypatch.setattr(dataset, "_cached_future", lambda *_: positive)
    monkeypatch.setattr(dataset, "_load_negative", lambda *_: {
        "neg_trajectory": torch.zeros(0, 8, 3), "neg_valid": torch.zeros(0, dtype=torch.bool),
        "neg_score": torch.zeros(0),
    })
    monkeypatch.setattr(dataset, "_load_negative_rows", lambda *_: negative)

    item = dataset[0]

    assert item["gt_valid"].shape[0] == 5
    assert item["negative_future_rows"] == 3
    for key in ROW_KEYS:
        assert torch.equal(item[key][:2], positive[key])
        assert torch.equal(item[key][2:], negative[key])


def test_negative_cache_fingerprint_mismatch_raises(tmp_path):
    dataset = _dataset(tmp_path, fingerprint="positive-space")
    path = scene_target_path(tmp_path, "scene-a")
    path.parent.mkdir(parents=True)
    torch.save({"format": 2, **_future_rows(1), "target_fingerprint": "other-space"}, path)

    with pytest.raises(ValueError, match="positive cache standardizer"):
        dataset._load_negative_rows("scene-a")


def test_missing_negative_scene_returns_none(tmp_path):
    assert _dataset(tmp_path)._load_negative_rows("missing") is None


def test_negative_future_disabled_preserves_supervision_bytes(monkeypatch):
    positive = _future_rows(2, 10)
    observation = SimpleNamespace(
        rgb_history=torch.zeros(1, 5, 3, 2, 2), history_valid=torch.ones(1, 5, dtype=torch.bool),
        ego_history=torch.zeros(1, 5, 11), route_command=torch.zeros(1, 4),
        relative_ego_poses=torch.zeros(1, 5, 3),
    )
    targets = SimpleNamespace(gt_trajectory=positive["gt_trajectory"].unsqueeze(0),
                              gt_valid=positive["gt_valid"].unsqueeze(0))
    monkeypatch.setattr(stage2, "load_real_navsim_sample", lambda *args, **kwargs: SimpleNamespace(
        observation=observation, targets=targets,
    ))
    candidate = stage2.Candidate(token="scene-a", log_name="unused", cache_path="unused", frame_index=0)
    dataset = _dataset(None)
    dataset.candidates = [candidate]
    monkeypatch.setattr(dataset, "_cached_future", lambda *_: positive)
    monkeypatch.setattr(dataset, "_load_negative", lambda *_: {
        "neg_trajectory": torch.zeros(0, 8, 3), "neg_valid": torch.zeros(0, dtype=torch.bool),
        "neg_score": torch.zeros(0),
    })
    item = dataset[0]

    for key in ROW_KEYS:
        assert item[key].contiguous().numpy().tobytes() == positive[key].contiguous().numpy().tobytes()
    assert "negative_future_rows" not in item
