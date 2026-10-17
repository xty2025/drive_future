"""Oracle boundaries for DrivoR online supervision and FastWAM offline analysis."""

from __future__ import annotations

import atexit
import hashlib
import lzma
import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
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


def _score_scene_payload(payload: tuple[str, np.ndarray]) -> tuple[np.ndarray, str | None]:
    """Score one scene's 64 proposals via DrivoR's ``get_sub_score``, in a worker.

    Top-level with the navsim import inside so a spawn ``ProcessPoolExecutor`` can
    pickle it by reference and each worker imports navsim once (reused across
    steps).  Returns ``(scores[64,7] float32, error_repr | None)``; a failed scene
    yields all-NaN scores, matching the sequential path's failure contract exactly.
    """
    path, poses = payload
    try:
        from navsim.agents.drivoR.score_module.compute_navsim_score import get_sub_score

        scores, *_ = get_sub_score(path, poses, True)
        return np.asarray(scores, dtype="float32"), None
    except Exception as error:  # mirror the sequential per-scene failure contract
        return np.full((64, 7), float("nan"), dtype="float32"), repr(error)


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

    def __init__(self, cache_paths: dict[str, str | Path], evaluator_version: str, parallel_workers: int = 0) -> None:
        try:
            from navsim.agents.drivoR.score_module.compute_navsim_score import get_sub_score
        except ImportError as error:
            raise ImportError("official DrivoR compute_navsim_score is required") from error
        self.cache_paths = {key: str(value) for key, value in cache_paths.items()}
        self.get_sub_score = get_sub_score
        self.evaluator_version = evaluator_version
        # DrivoR scores its per-scene ``get_sub_score`` calls in parallel (ray
        # ``worker_map``); the sequential Python loop left the accelerator idle for
        # the whole CPU oracle every step, which is the joint run's per-step
        # bottleneck at batch 12 x 16 ranks.  A persistent per-rank spawn pool
        # reproduces that parallelism under torchrun without a cross-rank ray
        # cluster.  ``parallel_workers=0`` keeps the exact sequential path
        # (bit-identical scores) and is also the fallback if a pool cannot start.
        self._parallel_workers = max(0, int(parallel_workers))
        self._pool: ProcessPoolExecutor | None = None
        if self._parallel_workers > 0:
            try:
                self._pool = ProcessPoolExecutor(
                    max_workers=self._parallel_workers, mp_context=mp.get_context("spawn")
                )
                atexit.register(self._pool.shutdown)
            except Exception:
                self._pool = None

    @torch.no_grad()
    def evaluate(self, scene_ids: list[str], trajectories: torch.Tensor, batch_id: str = "") -> OracleLabels:
        del batch_id
        if trajectories.ndim != 4 or trajectories.shape[1:] != (64, 8, 3):
            raise ValueError("DrivoR batch oracle requires [B,64,8,3] trajectories")
        # One GPU->CPU transfer for the whole batch instead of two per scene: the
        # oracle is CPU-only (get_sub_score reads numpy), so proposals leave the
        # accelerator exactly once per step.  float32 + ascontiguousarray keep the
        # hashed bytes and the get_sub_score input identical to the old per-scene
        # ``.detach().cpu()``, so scores and hashes are unchanged.
        cpu_poses = trajectories.detach().cpu().numpy().astype("float32")
        scene_hashes: list[str] = []
        trajectory_hashes: list[list[str]] = []
        payloads: list[tuple[str, np.ndarray]] = []
        for batch, scene_id in enumerate(scene_ids):
            if scene_id not in self.cache_paths:
                raise ValueError(f"unknown oracle scene: {scene_id}")
            cache_path = self.cache_paths[scene_id]
            scene_hashes.append(_hash_object(cache_path))
            trajectory_hashes.append(
                [hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in cpu_poses[batch]]
            )
            payloads.append((cache_path, cpu_poses[batch]))
        # Parallel per-scene scoring (DrivoR's ray worker_map equivalent).  Any
        # pool failure falls back to the exact sequential path, so correctness and
        # the returned labels never depend on the pool being alive.
        results: list[tuple[np.ndarray, str | None]] | None = None
        if self._pool is not None:
            try:
                results = list(self._pool.map(_score_scene_payload, payloads))
            except Exception:
                results = None
        if results is None:
            results = [_score_scene_payload(payload) for payload in payloads]
        raw_rows: list[torch.Tensor] = []
        failures: list[dict[str, Any]] = []
        for batch, scene_id in enumerate(scene_ids):
            scores, error = results[batch]
            if error is not None:
                failures.append({"scene_id": scene_id, "error": error})
            raw_rows.append(torch.as_tensor(np.asarray(scores), dtype=torch.float32))
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
