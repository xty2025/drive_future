"""Check the port's data reuse, BF16 oracle boundary and launcher runtime behavior."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from navsim.agents.drivoR.drivor_agent import DrivoRAgent
from navsim.agents.drivoR_action_dit.agent import DrivoRActionDiTAgent
from navsim.planning.script.run_training import build_dataloader, resolve_train_checkpoint
from navsim.planning.training.dataset import Dataset


def test_scene_images_and_full_token_index_are_not_loaded_again():
    counts = dict(tokens=0, scenes=0, inputs=0)

    class Scene:
        def get_agent_input(self):
            counts['inputs'] += 1
            return torch.tensor([3.0])

    class Loader:
        @property
        def tokens(self):
            counts['tokens'] += 1
            return ['scene']

        def get_scene_from_token(self, token):
            assert token == 'scene'
            counts['scenes'] += 1
            return Scene()

        def get_agent_input_from_token(self, token):
            pytest.fail('A second image decode was requested')

    builder = SimpleNamespace(compute_features=lambda value: {'image': value})
    dataset = Dataset(Loader(), [builder], [], cache_path=None)
    for _ in range(2):
        features, _ = dataset[0]
        torch.testing.assert_close(features['image'], torch.tensor([3.0]))
    assert counts == dict(tokens=1, scenes=2, inputs=2)


def test_zero_worker_loader_disables_prefetch_and_persistence():
    loader = build_dataloader(TensorDataset(torch.arange(4)),
        dict(batch_size=2, num_workers=0, prefetch_factor=2, persistent_workers=True),
        shuffle=False, seed=2)
    assert not loader.persistent_workers and loader.prefetch_factor is None
    assert [batch[0].tolist() for batch in loader] == [[0, 1], [2, 3]]


def test_new_run_does_not_resume_another_run(tmp_path):
    old_run, new_run = tmp_path / 'old', tmp_path / 'new'
    old_run.mkdir()
    new_run.mkdir()
    (old_run / 'last.ckpt').touch()
    assert resolve_train_checkpoint(new_run) is None
    last = new_run / 'last.ckpt'
    last.touch()
    assert resolve_train_checkpoint(new_run) == str(last)
    assert resolve_train_checkpoint(new_run, auto_resume=False) is None
    assert resolve_train_checkpoint(new_run, str(old_run / 'last.ckpt'), False) == str(old_run / 'last.ckpt')


def test_bfloat16_candidates_cross_numpy_boundary_as_float32():
    agent = DrivoRAgent.__new__(DrivoRAgent)
    nn.Module.__init__(agent)
    agent.train_metric_cache_paths = {'scene': 'unused'}
    agent.train_metric_cache_paths_synthetic = {}
    agent.worker = None
    agent.get_scores = None

    def score(worker, function, points):
        assert points[0]['poses'].dtype == np.float32
        assert points[0]['poses'].shape == (64, 8, 3)
        return [(np.ones((64, 7)), np.zeros((64, 40, 2, 4, 2)),
                 np.zeros((64, 40, 2), bool), np.zeros((64, 40, 2), bool))]

    agent.worker_map = score
    output = agent.compute_score({'token': ['scene'], 'trajectory': torch.zeros(1, 8, 3)},
                                 torch.ones(1, 64, 8, 3, dtype=torch.bfloat16), test=False)
    assert output[2].shape == (1, 64, 7) and output[2].dtype == torch.float32


@pytest.mark.parametrize('agent_type', [DrivoRAgent, DrivoRActionDiTAgent])
def test_fused_option_keeps_cpu_optimizer_usable(agent_type):
    agent = agent_type.__new__(agent_type)
    nn.Module.__init__(agent)
    agent._drivor_model = nn.Linear(2, 1)
    agent._lr_args = dict(name='AdamW', base_lr=2e-4, base_batch_size=64, fused=True)
    agent.batch_size, agent.num_gpus, agent.accumulate_grad_batches = 1, 64, 1
    agent.scheduler_args = None
    optimizer = agent.get_optimizers()[0]
    assert optimizer.defaults['fused'] is False
    before = agent._drivor_model.weight.detach().clone()
    agent._drivor_model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    assert not torch.equal(before, agent._drivor_model.weight)
