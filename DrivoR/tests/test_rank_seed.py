"""Separate rank-specific model RNG from shared model initialization and sampling."""

import os
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DistributedSampler

from navsim.agents.drivoR_action_dit.flow import ContinuousFlowSchedule
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.random_seed import seed_model_randomness


def draw():
    schedule = ContinuousFlowSchedule()
    return (random.random(), np.random.rand(4),
            schedule.sample_timesteps(1, 64, 'cpu'), torch.randn(1, 64, 8, 3))


def test_same_rank_replays_but_global_ranks_have_different_streams(monkeypatch):
    monkeypatch.setenv('PL_GLOBAL_SEED', '2')
    monkeypatch.setenv('PL_SEED_WORKERS', '1')
    samples = {}
    for rank in (0, 1, 8, 63):
        assert seed_model_randomness(2, rank) == 2 + rank
        samples[rank] = draw()
        seed_model_randomness(2, rank)
        replay = draw()
        assert replay[0] == samples[rank][0]
        np.testing.assert_array_equal(replay[1], samples[rank][1])
        for actual, expected in zip(replay[2:], samples[rank][2:]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Each scene's 64 candidates still share t but have distinct noise.
        torch.testing.assert_close(replay[2], replay[2][:, :1].expand(-1, 64))
        assert not torch.equal(replay[3][:, 0], replay[3][:, 1])
    for rank in (1, 8, 63):
        assert samples[rank][0] != samples[0][0]
        assert not np.array_equal(samples[rank][1], samples[0][1])
        assert not torch.equal(samples[rank][2], samples[0][2])
        assert not torch.equal(samples[rank][3], samples[0][3])
    assert os.environ['PL_GLOBAL_SEED'] == '2'
    assert os.environ['PL_SEED_WORKERS'] == '1'


def test_model_seeding_does_not_change_sampler_partition_or_loader_generator():
    dataset = list(range(16))
    generator = torch.Generator().manual_seed(2)
    state = generator.get_state().clone()
    samples = []
    for rank in range(2):
        sampler = DistributedSampler(dataset, num_replicas=2, rank=rank, seed=2)
        expected = list(sampler)
        seed_model_randomness(2, rank)
        assert list(sampler) == expected
        samples.append(expected)
    assert set(samples[0]).isdisjoint(samples[1])
    assert sorted(samples[0] + samples[1]) == dataset
    torch.testing.assert_close(generator.get_state(), state, rtol=0, atol=0)


def test_seed_is_applied_at_training_start_not_construction_or_every_batch():
    torch.manual_seed(29)
    agent = nn.Linear(3, 2)
    parameters = {name: value.clone() for name, value in agent.state_dict().items()}
    state = torch.get_rng_state().clone()
    module = AgentLightningModule(agent, training_seed=2, console_log_interval=0)
    torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
    module._trainer = SimpleNamespace(global_rank=8)
    module.on_train_start()
    assert module._model_rng_seed == 10
    for name, value in agent.state_dict().items():
        torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
    first = torch.randn(10)
    module.on_train_epoch_start()
    module.on_train_batch_start(None, 0)
    second = torch.randn(10)
    seed_model_randomness(2, 8)
    torch.testing.assert_close(first, torch.randn(10), rtol=0, atol=0)
    torch.testing.assert_close(second, torch.randn(10), rtol=0, atol=0)


def test_disabled_rank_seeding_leaves_rng_untouched():
    module = AgentLightningModule(nn.Linear(1, 1), console_log_interval=0)
    state = torch.get_rng_state().clone()
    module.on_train_start()
    torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
