import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "environment/performance/summarize_rank_performance.py"
SPEC = importlib.util.spec_from_file_location("rank_summary", SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def row(rank, batch=10, session="run", kind="batch", **extra):
    return dict(schema="drivor_rank_perf_v1", type=kind, session_id=session,
                rank=rank, world_size=2, epoch=0, batch_idx=batch,
                optimizer_step_before=batch, optimizer_step_after=batch + 1, **extra)


def test_straggler_alignment_and_nested_timings():
    records = [row(0, kind="metadata"), row(1, kind="metadata"),
               row(0, data_wait_s=0.1, oracle_cpu_s=0.8, oracle_s=0.9, loss_s=1.0, backward_s=0.1),
               row(1, data_wait_s=0.1, oracle_cpu_s=0.2, oracle_s=0.3, loss_s=0.4, backward_s=0.7),
               row(0, batch=11, oracle_cpu_s=0.5)]
    result = summary.summarize(records)
    assert result["complete_positions"] == 1 and result["incomplete_positions"] == 1
    assert result["largest_duration_rank_counts"]["oracle_cpu_s"] == {0: 1}
    assert result["largest_duration_rank_counts"]["backward_s"] == {1: 1}
    assert result["largest_duration_rank_counts"]["data_wait_s"] == {}
    assert result["aligned_max_minus_min_seconds"]["oracle_cpu_s"]["p50"] == pytest.approx(0.6)
    assert result["per_rank_seconds"]["0"]["loss_s"]["mean"] == 1.0


def test_excludes_old_sessions_warmup_and_missing_rank():
    result = summary.summarize([row(0, session="old", kind="metadata"), row(0, session="old"),
                                row(0, session="new", kind="metadata"), row(0, batch=0, session="new"),
                                row(0, session="new", oracle_cpu_s=0.2)])
    assert result["complete_positions"] == 0 and result["sampled_positions"] == 1
    assert result["missing_ranks"] == [1]
    assert result["sessions"] == {"0": "new"}


def test_rejects_mismatched_optimizer_positions():
    other = row(1)
    other["optimizer_step_before"] = 9
    with pytest.raises(ValueError, match="optimizer positions"):
        summary.summarize([row(0), other])


def test_resumed_session_excludes_its_own_spawn_warmup():
    rows = [row(rank, kind="metadata", optimizer_step_start=1000) for rank in (0, 1)]
    rows += [row(rank, batch=batch, oracle_cpu_s=15 if batch == 1000 else 0.2)
             for rank in (0, 1) for batch in (1000, 1005)]
    result = summary.summarize(rows)
    assert result["complete_positions"] == 1
    assert result["per_rank_seconds"]["0"]["oracle_cpu_s"]["max"] == 0.2


def test_different_rank_resume_generations_are_never_paired():
    rows = [row(0, kind="metadata", session="old"), row(1, kind="metadata", session="old"),
            row(1, session="old", oracle_cpu_s=0.2), row(0, kind="metadata", session="new"),
            row(0, session="new", oracle_cpu_s=15)]
    result = summary.summarize(rows)
    assert not result["alignment_allowed"] and result["complete_positions"] == 0
    assert result["cohort_issues"]
    assert result["aligned_max_minus_min_seconds"]["oracle_cpu_s"] is None


def test_distinct_dlc_attempts_with_equal_file_generations_are_rejected():
    rows = [row(rank, kind="metadata", distributed_attempt={"PAI_JOB_ID": str(rank)}) for rank in (0, 1)]
    rows += [row(rank, oracle_cpu_s=0.2) for rank in (0, 1)]
    assert not summary.summarize(rows)["alignment_allowed"]


def test_partial_active_write_is_ignored_but_corruption_is_not(tmp_path):
    path = tmp_path / "rank_00000.jsonl"
    path.write_text(json.dumps(row(0)) + '\n{"type":')
    records, partial = summary.read_records(tmp_path)
    assert len(records) == 1 and partial == [str(path)]
    path.write_text('{bad json}\n' + json.dumps(row(0)) + "\n")
    with pytest.raises(ValueError, match="Invalid JSON"):
        summary.read_records(tmp_path)
