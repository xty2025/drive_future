"""Console training progress without per-batch distributed reductions."""
import logging
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from navsim.planning.training import agent_lightning_module as runtime


@pytest.mark.parametrize("batches,epochs,epoch,index,max_steps,step,accum,remaining", [
    (10, 2, 0, 0, -1, 0, 4, 19),
    (10, 2, 0, 3, 4, 1, 4, 10),
    (10, 2, 0, 9, 4, 3, 4, 4),
    (10, 2, 0, 8, 3, 2, 4, 1),
    (10, 2, 0, 3, 2, 1, 4, 4),
    (10, 1, 0, 3, 4, 1, 4, 6),
    (10, 2, 0, 0, 2, 1, 1, 1),
    (10, 2, 0, 7, 2, 2, 4, 0),
    (float("inf"), -1, 0, 0, 2, 0, 4, 7),
    (float("inf"), -1, 0, 0, -1, 0, 4, None),
])
def test_remaining_microbatches_respect_accumulation_partial_epochs_and_max_steps(
        batches, epochs, epoch, index, max_steps, step, accum, remaining):
    assert runtime.remaining_train_batches(batches, epochs, epoch, index, max_steps, step, accum) == remaining


@pytest.mark.parametrize("seconds,formatted", [(None, "unknown"), (float("nan"), "unknown"),
                                               (float("inf"), "unknown"), (0, "00:00:00"),
                                               (3661.1, "01:01:02")])
def test_eta_cold_start_and_nonfinite_inputs(seconds, formatted):
    assert runtime._format_eta(seconds) == formatted


class LossAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

    def forward(self, features):
        return self.weight * features["input"]

    def compute_loss(self, features, targets, prediction):
        return {"loss": prediction.mean(), "trajectory_loss": prediction.mean() * 0.75,
                "final_score_loss": prediction.mean() * 0.25}


def make_module(monkeypatch, rank=0, interval=25):
    module = runtime.AgentLightningModule(LossAgent(), console_log_interval=interval)
    module._trainer = SimpleNamespace(global_rank=rank, current_epoch=0, global_step=0,
                                      num_training_batches=10, max_epochs=2, max_steps=-1,
                                      accumulate_grad_batches=4,
                                      optimizers=[torch.optim.SGD(module.parameters(), lr=0.002)])
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)
    batch = ({"input": torch.ones(2, 1)}, {"trajectory": torch.zeros(2, 8, 3)})
    return module, batch


def test_first_progress_has_original_losses_lr_and_train_only_eta(monkeypatch, caplog):
    module, batch = make_module(monkeypatch)
    times = iter([0.0, 2.0])
    monkeypatch.setattr(runtime, "perf_counter", lambda: next(times))
    module.on_train_epoch_start()
    module.on_train_batch_start(batch, 0)
    loss = module.training_step(batch, 0)
    with caplog.at_level(logging.INFO, logger=runtime.__name__):
        module.on_train_batch_end({"loss": loss / 4}, batch, 0)
    message = caplog.records[-1].getMessage()
    assert "rank=0 epoch=1/2 batch=1/10 optimizer_step=0 accum=4" in message
    assert "ETA_train=00:00:38" in message and "remaining_train_batches=19" in message
    assert "loss=2" in message and "trajectory_loss=1.5" in message and "final_score_loss=0.5" in message
    assert "lr=0.002" in message and "batch_s=2.000" in message
    assert not module._console_losses["loss"].requires_grad


@pytest.mark.parametrize("rank,interval", [(1, 25), (0, 0)])
def test_nonzero_rank_and_disabled_interval_do_not_time_or_print(monkeypatch, caplog, rank, interval):
    module, batch = make_module(monkeypatch, rank=rank, interval=interval)
    monkeypatch.setattr(runtime, "perf_counter", lambda: pytest.fail("unexpected timer"))
    with caplog.at_level(logging.INFO, logger=runtime.__name__):
        module.on_train_epoch_start()
        module.on_train_batch_start(batch, 0)
        loss = module.training_step(batch, 0)
        module.on_train_batch_end(loss, batch, 0)
    assert not any("[train]" in record.getMessage() for record in caplog.records)


def test_validation_time_is_excluded_but_between_batch_wait_is_included(monkeypatch):
    module, batch = make_module(monkeypatch)
    times = iter([0.0, 2.0, 1002.0, 1005.0])
    monkeypatch.setattr(runtime, "perf_counter", lambda: next(times))
    module.on_train_epoch_start()
    module.on_train_batch_end(None, batch, 0)
    module.on_validation_start()
    module.on_validation_end()
    module.on_train_batch_end(None, batch, 1)
    assert list(module._console_batch_durations) == [2.0, 3.0]


def test_interval_and_final_batch_print_even_for_short_run(monkeypatch, caplog):
    module, batch = make_module(monkeypatch, interval=3)
    module._trainer.num_training_batches = 5
    times = iter(range(6))
    monkeypatch.setattr(runtime, "perf_counter", lambda: next(times))
    with caplog.at_level(logging.INFO, logger=runtime.__name__):
        module.on_train_epoch_start()
        for index in range(5):
            module.on_train_batch_end(None, batch, index)
    records = [record.getMessage() for record in caplog.records if "[train]" in record.getMessage()]
    assert len(records) == 3
    assert all(f"batch={index}/5" in line for index, line in zip((1, 3, 5), records))


@pytest.mark.parametrize("rank", [0, 1])
def test_epoch_metric_access_on_all_ranks_but_only_rank_zero_prints(monkeypatch, caplog, rank):
    module, _ = make_module(monkeypatch, rank=rank)

    class Trainer:
        global_rank = rank
        current_epoch = 0
        reads = 0

        @property
        def callback_metrics(self):
            self.reads += 1
            return {"train/loss": torch.tensor(2.0), "val/score_epoch": torch.tensor(0.8)}

    module._trainer = Trainer()
    with caplog.at_level(logging.INFO, logger=runtime.__name__):
        module.on_train_epoch_end()
    assert module._trainer.reads == 1
    records = [record.getMessage() for record in caplog.records if "[epoch]" in record.getMessage()]
    assert bool(records) is (rank == 0)
    if records:
        assert "scope=global" in records[0] and "train/loss=2" in records[0] and "val/score_epoch=0.8" in records[0]
