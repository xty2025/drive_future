"""CPU timing contracts and real Lightning accumulation; no GPU/Oracle claims."""
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from navsim.planning.training import agent_lightning_module as runtime
from navsim.planning.training.rank_performance import RankPerformanceRecorder


class Clock:
    value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def recorder(tmp_path, clock, **kwargs):
    return RankPerformanceRecorder(tmp_path, rank=kwargs.pop("rank", 0), world_size=2,
                                   device=kwargs.pop("device", "cpu"), clock=clock, **kwargs)


def test_host_phases_and_nested_oracle_have_exact_nonoverlapping_boundaries(tmp_path):
    clock = Clock()
    trace = recorder(tmp_path, clock, sample_every=1, flush_every=4, optimizer_step_start=10)
    trace.reset_wait()
    clock.advance(3)
    trace.begin_transfer(epoch=2, optimizer_step=10, accumulation=1)
    clock.advance(1)
    trace.end_phase("transfer")
    trace.batch_start(10)
    with trace.phase("forward"):
        clock.advance(5)
    with trace.phase("loss"):
        clock.advance(1)
        with trace.phase("oracle"):
            clock.advance(1)
            with trace.phase("oracle_cpu"):
                clock.advance(6)
            clock.advance(1)
        clock.advance(1)
    with trace.phase("backward"):
        clock.advance(3)
    trace.optimizer_started()
    clock.advance(2)
    trace.end_phase("optimizer")
    clock.advance(1)
    trace.finish_batch(batch_idx=10, optimizer_step=11)
    assert len(records(trace.path)) == 1  # Metadata only until periodic flush/close.
    trace.close()
    metadata, row = records(trace.path)
    assert metadata["cpu_affinity_count"] is None or metadata["cpu_affinity_count"] > 0
    assert metadata["torch_num_threads"] > 0
    assert metadata["optimizer_step_start"] == 10
    assert set(metadata["distributed_attempt"]) == {"PAI_JOB_ID", "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT"}
    assert "NAVSIM_ORACLE_WORKERS" in metadata["environment"]
    assert row["data_wait_s"] == 3 and row["transfer_s"] == 1
    assert row["forward_s"] == 5 and row["loss_s"] == 10
    assert row["oracle_s"] == 8 and row["oracle_cpu_s"] == 6
    assert row["backward_s"] == 3 and row["optimizer_s"] == 2
    assert row["batch_s"] == 25 and row["optimizer_step_called"]
    assert row["optimizer_step_before"] == 10 and row["optimizer_step_after"] == 11
    assert row["cuda_status"] == "disabled" and row["cuda_ms"] == {}


def test_sampling_keeps_whole_accumulation_window_and_resets_validation_wait(tmp_path):
    clock = Clock()
    trace = recorder(tmp_path, clock, sample_every=2, flush_every=1)
    trace.reset_wait()
    for index, (before, after) in enumerate([(0, 0), (0, 1), (1, 1), (1, 2), (2, 3)]):
        clock.advance(2)
        trace.begin_transfer(epoch=0, optimizer_step=before, accumulation=2)
        trace.end_phase("transfer")
        if before != after:
            trace.optimizer_started()
            clock.advance(1)
            trace.end_phase("optimizer")
        trace.finish_batch(batch_idx=index, optimizer_step=after)
        if index == 3:
            trace.reset_wait(paused=True)
            clock.advance(1000)  # A validation interval must not become input wait.
            trace.reset_wait()
    trace.close()
    rows = records(trace.path)[1:]
    assert [row["batch_idx"] for row in rows] == [0, 1, 4]
    assert rows[0]["optimizer_s"] is None and not rows[0]["optimizer_step_called"]
    assert rows[1]["optimizer_s"] == rows[2]["optimizer_s"] == 1
    assert all(row["data_wait_s"] == 2 for row in rows)


def test_temporary_oracle_methods_restore_class_and_instance_attributes_after_failure(tmp_path):
    clock = Clock()
    trace = recorder(tmp_path, clock, sample_every=1)
    trace.begin_transfer(epoch=0, optimizer_step=0, accumulation=1)

    class Agent(nn.Module):
        def __init__(self):
            super().__init__()
            self.worker_map = lambda: clock.advance(3)

        def compute_score(self):
            self.worker_map()
            raise ValueError("oracle failed")

    agent = Agent()
    worker_map = agent.worker_map
    with pytest.raises(ValueError, match="oracle failed"):
        with trace.oracle_phases(agent):
            agent.compute_score()
    assert "compute_score" not in agent.__dict__
    assert agent.worker_map is worker_map
    assert trace.current["row"]["oracle_s"] == trace.current["row"]["oracle_cpu_s"] == 3
    assert trace.current["row"]["phase_calls"]["oracle"] == 1
    trace.close()


def test_optimizer_wall_time_excludes_the_training_closure(tmp_path):
    clock = Clock()
    module = runtime.AgentLightningModule(nn.Linear(1, 1), console_log_interval=0)
    trace = recorder(tmp_path, clock, sample_every=1)
    module._performance = trace
    trace.begin_transfer(epoch=0, optimizer_step=0, accumulation=1)

    class Optimizer:
        def step(self, closure):
            closure()
            clock.advance(2)

    calls = []

    def closure():
        calls.append("forward-loss-backward")
        clock.advance(100)
        return torch.tensor(1.0)

    module.optimizer_step(0, 0, Optimizer(), closure)
    assert calls == ["forward-loss-backward"]
    assert trace.current["row"]["optimizer_s"] == 2
    trace.close()


@pytest.mark.parametrize("ready_before_close", [False, True])
def test_cuda_events_are_queried_later_without_any_synchronization(tmp_path, monkeypatch, ready_before_close):
    clock = Clock()
    state = {"ready": False, "elapsed_calls": 0}

    class Event:
        def record(self):
            self.timestamp = clock()

        def query(self):
            return state["ready"]

        def elapsed_time(self, end):
            assert state["ready"]
            state["elapsed_calls"] += 1
            return 1000 * (end.timestamp - self.timestamp)

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: pytest.fail("unexpected CUDA synchronize"))
    trace = recorder(tmp_path, clock, device="cuda:0", cuda_events=True, event_factory=Event, sample_every=2)
    trace.reset_wait()
    trace.begin_transfer(epoch=0, optimizer_step=0, accumulation=1)
    clock.advance(1)
    trace.end_phase("transfer")
    trace.finish_batch(batch_idx=0, optimizer_step=1)
    assert state["elapsed_calls"] == 0 and len(trace.pending) == 1
    state["ready"] = ready_before_close
    trace.close()
    row = records(trace.path)[1]
    assert row["cuda_status"] == ("ready" if ready_before_close else "not_ready_at_close")
    assert row["cuda_ms"] == ({"transfer": 1000, "batch": 1000} if ready_before_close else {})


def test_rank_one_writes_own_file_without_printing_a_global_summary(tmp_path, caplog):
    clock = Clock()
    trace = recorder(tmp_path, clock, rank=1, flush_every=1)
    trace.begin_transfer(epoch=0, optimizer_step=0, accumulation=1)
    with caplog.at_level(logging.INFO):
        trace.finish_batch(batch_idx=0, optimizer_step=1)
        trace.close()
    assert trace.path.name == "rank_00001.jsonl"
    assert records(trace.path)[1]["rank"] == 1
    assert not any("[perf]" in record.getMessage() for record in caplog.records)


def test_disabled_diagnostics_do_not_construct_recorder_or_change_optimizer(monkeypatch):
    monkeypatch.setattr(runtime, "RankPerformanceRecorder", lambda *a, **k: pytest.fail("unexpected recorder"))
    module = runtime.AgentLightningModule(nn.Linear(1, 1), console_log_interval=0)
    module._trainer = SimpleNamespace(global_rank=0)
    module.on_train_start()
    assert module._performance is None


class TinyDataset(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        return {"input": torch.tensor([float(index + 1)])}, {"trajectory": torch.ones(8, 3)}


class TinyAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.worker_map = lambda: None

    def name(self):
        return "TinyAgent"

    def forward(self, features):
        return features["input"] * self.weight

    def compute_score(self):
        self.worker_map()

    def compute_loss(self, features, targets, pred):
        self.compute_score()
        loss = (pred - 1).square().mean()
        return {"loss": loss, "trajectory_loss": 0.75 * loss, "final_score_loss": 0.25 * loss}

    def get_optimizers(self):
        return [torch.optim.SGD(self.parameters(), lr=0.01)]


def test_real_lightning_hooks_preserve_updates_and_accumulation_with_validation(tmp_path):
    states = []
    for enabled in (False, True):
        pl.seed_everything(2, workers=True)
        module = runtime.AgentLightningModule(TinyAgent(), console_log_interval=0,
            log_local_steps=False, performance_diagnostics={"enabled": enabled, "output_dir": str(tmp_path / "perf"),
                                                          "sample_every": 1, "flush_every": 3})
        trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                             enable_progress_bar=False, enable_model_summary=False, max_epochs=1,
                             accumulate_grad_batches=2, num_sanity_val_steps=0, default_root_dir=str(tmp_path))
        loader = DataLoader(TinyDataset(), batch_size=1)
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
        states.append(module.agent.weight.detach().clone())
        assert "compute_score" not in module.agent.__dict__
        assert trainer.global_step == 3
    torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
    rows = records(tmp_path / "perf/rank_00000.jsonl")[1:]
    assert len(rows) == 5  # Validation must not produce sampled train rows.
    assert [row["optimizer_step_before"] for row in rows] == [0, 0, 1, 1, 2]
    assert [row["optimizer_step_after"] for row in rows] == [0, 1, 1, 2, 3]
    assert [row["optimizer_step_called"] for row in rows] == [False, True, False, True, True]
    for row in rows:
        assert row["phase_calls"]["forward"] == row["phase_calls"]["loss"] == row["phase_calls"]["backward"] == 1
        assert row["phase_calls"]["oracle"] == row["phase_calls"]["oracle_cpu"] == 1
        assert row["batch_s"] >= row["loss_s"] >= row["oracle_s"] >= row["oracle_cpu_s"] >= 0
