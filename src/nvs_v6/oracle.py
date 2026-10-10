"""Oracle boundaries for DrivoR online supervision and FastWAM offline analysis."""

from __future__ import annotations

import hashlib
import lzma
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

RAW_ORDER = ("NC", "DAC", "EP", "TTC", "C", "DDC", "Final")
NAMED_ORDER = ("NC", "DAC", "DDC", "TTC", "EP", "C")
HEAD_INDICES = (0, 1, 5, 3, 2, 4)


@dataclass
class OracleLabels:
    raw7: torch.Tensor
    named_six: torch.Tensor
    head_valid: torch.Tensor
    final_score: torch.Tensor
    final_valid: torch.Tensor
    scene_hashes: list[str]
    trajectory_hashes: list[list[str]]
    evaluator_version: str
    failure_diagnostics: list[dict[str, Any]]


class OnlineOracle:
    def evaluate(self, scene_ids: list[str], trajectories: torch.Tensor, batch_id: str = "") -> OracleLabels:
        raise NotImplementedError


def load_drivor_metric_cache(path: str | Path) -> Any:
    """Load the local DrivoR cache, including the observed XZ-compressed `.pkl` format."""
    path = Path(path)
    try:
        with lzma.open(path, "rb") as handle:
            return pickle.load(handle)
    except (lzma.LZMAError, EOFError, pickle.UnpicklingError):
        with path.open("rb") as handle:
            return pickle.load(handle)


class DrivoRMetricCacheOracle(OnlineOracle):
    """Adapter around the audited DrivoR PDM scorer; never exposes cache to the model."""

    def __init__(self, evaluator: Callable[[str, torch.Tensor, Any], Any], metric_cache_loader: Callable[[str], Any], evaluator_version: str) -> None:
        self.evaluator = evaluator
        self.metric_cache_loader = metric_cache_loader
        self.evaluator_version = evaluator_version

    @torch.no_grad()
    def evaluate(self, scene_ids: list[str], trajectories: torch.Tensor, batch_id: str = "") -> OracleLabels:
        if trajectories.ndim != 4 or trajectories.shape[1] != 64 or trajectories.shape[2:] != (8, 3):
            raise ValueError("DrivoR oracle requires current [B,64,8,3] trajectories")
        raw_rows: list[torch.Tensor] = []
        scene_hashes: list[str] = []
        trajectory_hashes: list[list[str]] = []
        failures: list[dict[str, Any]] = []
        for batch, scene_id in enumerate(scene_ids):
            metric_cache = self.metric_cache_loader(scene_id)
            scene_hashes.append(_hash_object(metric_cache))
            per_scene: list[torch.Tensor] = []
            per_hashes: list[str] = []
            for candidate in range(trajectories.shape[1]):
                proposal = trajectories[batch, candidate].detach().cpu().to(torch.float64)
                per_hashes.append(_hash_tensor(proposal))
                try:
                    values = self.evaluator(scene_id, proposal, metric_cache)
                    row = torch.as_tensor(values, dtype=torch.float32).flatten()
                    if row.numel() != 7:
                        raise ValueError(f"DrivoR evaluator returned {row.numel()} values, expected raw7")
                except Exception as error:
                    row = torch.full((7,), float("nan"), dtype=torch.float32)
                    failures.append({"scene_id": scene_id, "candidate": candidate, "error": repr(error)})
                per_scene.append(row)
            raw_rows.append(torch.stack(per_scene))
            trajectory_hashes.append(per_hashes)
        raw = torch.stack(raw_rows)
        valid = torch.isfinite(raw)
        ttc_invalid = raw[..., 3].eq(2.0) | ~torch.isfinite(raw[..., 3])
        valid[..., 3] &= ~ttc_invalid
        named = raw[..., list(HEAD_INDICES)]
        named_valid = valid[..., list(HEAD_INDICES)]
        final = raw[..., 6]
        final_valid = valid[..., 6]
        return OracleLabels(raw, named, named_valid, final, final_valid, scene_hashes, trajectory_hashes, self.evaluator_version, failures)


class DrivoRBatchMetricCacheOracle(OnlineOracle):
    """Exact DrivoR training-label path: one call scores all 64 proposals."""

    def __init__(self, cache_paths: dict[str, str | Path], evaluator_version: str) -> None:
        try:
            from navsim.agents.drivoR.score_module.compute_navsim_score import get_sub_score
        except ImportError as error:
            raise ImportError("official DrivoR compute_navsim_score is required") from error
        self.cache_paths = {key: str(value) for key, value in cache_paths.items()}
        self.get_sub_score = get_sub_score
        self.evaluator_version = evaluator_version

    @torch.no_grad()
    def evaluate(self, scene_ids: list[str], trajectories: torch.Tensor, batch_id: str = "") -> OracleLabels:
        del batch_id
        if trajectories.ndim != 4 or trajectories.shape[1:] != (64, 8, 3):
            raise ValueError("DrivoR batch oracle requires [B,64,8,3] trajectories")
        raw_rows: list[torch.Tensor] = []
        scene_hashes: list[str] = []
        trajectory_hashes: list[list[str]] = []
        failures: list[dict[str, Any]] = []
        for batch, scene_id in enumerate(scene_ids):
            if scene_id not in self.cache_paths:
                raise ValueError(f"unknown oracle scene: {scene_id}")
            cache_path = self.cache_paths[scene_id]
            scene_hashes.append(_hash_object(cache_path))
            trajectory_hashes.append([_hash_tensor(item) for item in trajectories[batch].detach().cpu()])
            try:
                scores, *_ = self.get_sub_score(
                    cache_path,
                    trajectories[batch].detach().cpu().numpy().astype("float32"),
                    True,
                )
                raw_rows.append(torch.as_tensor(scores, dtype=torch.float32))
            except Exception as error:
                failures.append({"scene_id": scene_id, "error": repr(error)})
                raw_rows.append(torch.full((64, 7), float("nan"), dtype=torch.float32))
        raw = torch.stack(raw_rows)
        valid = torch.isfinite(raw)
        ttc_invalid = raw[..., 3].eq(2.0) | ~torch.isfinite(raw[..., 3])
        valid[..., 3] &= ~ttc_invalid
        named = raw[..., list(HEAD_INDICES)]
        named_valid = valid[..., list(HEAD_INDICES)]
        final = raw[..., 6]
        final_valid = valid[..., 6]
        return OracleLabels(raw, named, named_valid, final, final_valid, scene_hashes, trajectory_hashes, self.evaluator_version, failures)


class FastWAMOfflineMetricCache:
    """Offline-only reader; requires a trajectory-keyed cache and is not a training oracle."""

    def __init__(self, cache_root: str | Path) -> None:
        self.cache_root = Path(cache_root)

    def lookup(self, scene_id: str, trajectory: torch.Tensor) -> dict[str, Any]:
        candidates = [self.cache_root / scene_id / "metric_cache.pkl", self.cache_root / f"{scene_id}.pkl"]
        for path in candidates:
            if path.exists():
                with path.open("rb") as handle:
                    record = pickle.load(handle)
                if not isinstance(record, dict) or "trajectory" not in record:
                    raise ValueError(f"offline cache {path} has no trajectory key")
                expected = _hash_tensor(trajectory.detach().cpu())
                if record["trajectory"] not in (expected, trajectory.detach().cpu().tolist()):
                    raise ValueError("offline FastWAM cache trajectory key mismatch")
                return record
        raise FileNotFoundError(f"no trajectory-keyed FastWAM cache record for {scene_id}")


def _hash_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def _hash_object(value: Any) -> str:
    return hashlib.sha256(repr(value).encode()).hexdigest()
