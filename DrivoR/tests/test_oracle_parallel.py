"""CPU candidate sharding preserves all Oracle labels and process isolation."""
from concurrent.futures import Future
import json
import lzma
from pathlib import Path
import pickle

import numpy as np
import pytest

from navsim.agents.drivoR.score_module import compute_navsim_score as oracle


def _poses(count=64):
    times = np.arange(1, 9) * 0.5
    speeds = np.resize(np.linspace(0, 20, 8), count)
    rates = np.resize(np.linspace(-0.4, 0.4, 8), count)
    result = np.zeros((count, 8, 3))
    angles = rates[:, None] * times
    result[..., 0] = speeds[:, None] * times * np.sinc(angles / np.pi)
    result[..., 1] = speeds[:, None] * (1 - np.cos(angles)) / rates[:, None]
    result[..., 2] = angles
    return result


def _cache_path():
    root = Path(__file__).resolve().parents[2]
    fixtures = root / "environment/drivor_two_gpu_smoke_scenes.json"
    if not fixtures.exists():
        pytest.skip("Real metric-cache fixture is unavailable")
    path = Path(json.loads(fixtures.read_text())["train"][0]["cache_path"])
    if not path.exists():
        pytest.skip("Real metric cache is unavailable")
    return path


def test_candidate_order_and_ttc_batch_fallback(monkeypatch):
    calls = []

    def serial(path, poses, test, fast, pruned):
        calls.append(len(poses))
        scores = np.zeros((len(poses), 7))
        scores[:, 0] = poses[:, 0, 0]
        if len(poses) < 5 and poses[0, 0, 0] == 0:
            scores[:, 3] = 2  # Existing TTC exception sentinel.
        return scores, np.zeros((len(poses), 2)), np.zeros((len(poses), 2), bool), np.zeros((len(poses), 2), bool)

    class Executor:
        def submit(self, function, *args):
            result = Future()
            result.set_result(function(*args))
            return result

    monkeypatch.setattr(oracle, "_get_sub_score_serial", serial)
    monkeypatch.setattr(oracle, "_get_oracle_executor", lambda workers: Executor())
    poses = np.zeros((5, 8, 3))
    poses[:, 0, 0] = np.arange(5)
    result = oracle.get_sub_score("unused", poses, False, workers=2)
    assert calls == [3, 2, 5]  # One failed chunk reruns the whole candidate set.
    np.testing.assert_array_equal(result[0][:, 0], np.arange(5))
    np.testing.assert_array_equal(result[0][:, 3], 0)
    calls.clear()
    poses[:, 0, 0] += 1
    result = oracle.get_sub_score("unused", poses, True, workers=2)
    assert calls == [3, 2]
    np.testing.assert_array_equal(result[0][:, 0], np.arange(1, 6))


@pytest.mark.parametrize("workers", [2, 4])
def test_real_process_pool_preserves_training_eval_and_input(workers):
    path, poses = _cache_path(), _poses()
    original_poses = poses.copy()
    try:
        for test in (False, True):
            expected = oracle.get_sub_score(path, poses, test, workers=1)
            actual = oracle.get_sub_score(path, poses, test, workers=workers)
            for part, reference in zip(actual, expected):
                np.testing.assert_allclose(part, reference, rtol=0, atol=1e-10)
        executor = oracle._oracle_executor
        assert executor._mp_context.get_start_method() == "spawn"
        repeat = oracle.get_sub_score(path, poses, False, workers=workers)
        assert oracle._oracle_executor is executor
        for part, reference in zip(repeat, oracle.get_sub_score(path, poses, False, workers=1)):
            np.testing.assert_allclose(part, reference, rtol=0, atol=1e-10)
        np.testing.assert_array_equal(poses, original_poses)
    finally:
        oracle.shutdown_oracle_executor()


def test_persistent_workers_invalidate_rewritten_cache_and_return_fresh_arrays(tmp_path):
    with lzma.open(_cache_path(), "rb") as stream:
        cache = pickle.load(stream)
    path = tmp_path / "metric_cache.pkl"
    poses = _poses(8)

    def write(progress):
        cache.pdm_progress = np.array([progress])
        with lzma.open(path, "wb") as stream:
            pickle.dump(cache, stream)

    try:
        write(1.0)
        before = oracle.get_sub_score(path, poses, False, workers=2)
        saved = tuple(item.copy() for item in before)
        before[0].fill(-99)
        repeat = oracle.get_sub_score(path, poses, False, workers=2)
        for actual, expected in zip(repeat, saved):
            np.testing.assert_array_equal(actual, expected)
        write(1e6)
        rewritten = oracle.get_sub_score(path, poses, False, workers=2)
        for actual, expected in zip(rewritten, oracle.get_sub_score(path, poses, False, workers=1)):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
        assert np.max(np.abs(saved[0] - rewritten[0])) > 0
    finally:
        oracle.shutdown_oracle_executor()
