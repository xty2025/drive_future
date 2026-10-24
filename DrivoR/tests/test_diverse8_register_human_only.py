"""Keep H/D8 trajectory gradients while restricting future targets to H."""
from types import SimpleNamespace
import json

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from navsim.agents.drivoR_action_dit.flow import normalize_trajectory
from navsim.agents.drivoR_action_dit_register import diverse_targets
from navsim.agents.drivoR_action_dit_register.diverse_targets import DiverseFutureRegisterTargetBuilder
from test_diverse8_register import make_agent, compose, agent_name, sample


BASE = 'action_dit_human_xl1_diverse8_l1_register'


@pytest.mark.parametrize('scorer', [False, True])
def test_human_future_gradient_and_diverse_trajectory_gradient(monkeypatch, sample, scorer):
    variant = BASE + ('_scorer' if scorer else '') + '_human_only'
    agent = make_agent(monkeypatch, variant).eval()
    model = agent._drivor_model
    features, targets = sample
    # Training must work without any D8 future-image tensors.
    targets.pop('future_register_images')
    targets.pop('future_register_valid')
    targets['trajectory'].zero_()
    targets['pseudo_experts'][:, :2].zero_()
    targets['pseudo_experts'][:, 0, :, 0] = 20.2
    targets['pseudo_experts'][:, 1, :, 0] = 20.4
    proposals = torch.zeros(1, 4, 8, 3)
    proposals[:, 0, :, 0] = 1
    proposals[:, 1, :, 2] = .1  # Human winner.
    proposals[:, 2, :, 0] = 20  # Both D8 winners.
    proposals[:, 3, :, 0] = 50
    proposals.requires_grad_()
    registers = torch.zeros(1, 4, 8, 4, 16, requires_grad=True)
    seen = []

    def teacher(pixels, valid):
        assert pixels.shape == (1, 1, 8, 3, 32, 32)
        torch.testing.assert_close(pixels[:, 0], targets['human_future_register_images'])
        assert valid.shape == (1, 1, 8)
        return torch.ones(1, 1, 8, 4, 16)

    def predict(action, future, times, scene, ego):
        seen.append((action.detach(), future.detach()))
        return normalize_trajectory(proposals), registers.flatten(2, 3)

    monkeypatch.setattr(model.ema_teacher, 'forward', teacher)
    monkeypatch.setattr(model.schedule, 'sample_timesteps',
                        lambda b, m, device: torch.full((b, m), 500., device=device))
    monkeypatch.setattr(model.action_dit, 'forward', predict)
    output = agent.forward_train(features, targets)
    assert len(seen) == 1
    assert output['_human_matched_indices'].tolist() == [[1]]
    assert output['_diverse_matched_indices'].tolist() == [[2, 2] + [-1] * 6]
    assert output['_register_matched_indices'].tolist() == [[1]]
    assert output['_flow_diagnostics']['future_register_gt_count'] == 1
    torch.testing.assert_close(output['_flow_loss'], torch.tensor(.25), atol=1e-5, rtol=1e-5)
    time = torch.tensor([[500.]])
    sigma = model.schedule.sigma(time).clamp_min(model.schedule.sigma_min)
    expected_future = (model.schedule.training_weight(time) / sigma.square()).mean()
    torch.testing.assert_close(output['_register_loss'], expected_future)
    output['_register_loss'].backward()
    assert (registers.grad.abs().sum((-3, -2, -1)) > 0).tolist() == [[False, True, False, False]]
    output['_flow_loss'].backward()
    assert (proposals.grad.abs().sum((-2, -1)) > 0).tolist() == [[False, True, True, False]]


@pytest.mark.parametrize('scorer', [False, True])
def test_outputs_checkpoint_structure_scoring_and_ema_stay_compatible(monkeypatch, sample, scorer):
    old_variant = BASE + ('_scorer' if scorer else '')
    old = make_agent(monkeypatch, old_variant).eval()
    new = make_agent(monkeypatch, old_variant + '_human_only').eval()
    new.load_state_dict(old.state_dict(), strict=True)
    features, targets = sample
    human_targets = {k: v for k, v in targets.items()
                     if k not in ('future_register_images', 'future_register_valid')}
    torch.manual_seed(53)
    previous = old.forward_train(features, targets)
    torch.manual_seed(53)
    current = new.forward_train(features, human_targets)
    for key in ('proposals', 'pdm_score', 'predicted_registers', '_flow_loss'):
        torch.testing.assert_close(current[key], previous[key])
    loss = new.compute_loss(features, human_targets, current)
    torch.testing.assert_close(loss['loss'], loss['trajectory_loss'] + loss['future_register_loss'] + loss['final_score_loss'])
    assert loss['future_register_gt_count'] == 1
    loss['final_score_loss'].backward()
    model = new._drivor_model
    assert all(p.grad is None for p in model.action_dit.parameters())
    assert model.image_backbone.neck.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    current = new.forward_train(features, human_targets)
    loss = new.compute_loss(features, human_targets, current)
    optimizer = new.get_optimizers()[0]
    loss['loss'].backward()
    optimizer.step()
    assert model.ema_teacher.updates == 1
    assert all(p.grad is None for p in model.ema_teacher.parameters())
    monkeypatch.setattr(model.ema_teacher, 'forward',
                        lambda *args: (_ for _ in ()).throw(AssertionError('Inference read future GT')))
    with torch.no_grad():
        predicted = new(features)
    assert predicted['proposals'].shape == (1, 4, 8, 3)


def test_target_builder_reads_human_frames_and_keeps_sparse_d8_without_videos(tmp_path, monkeypatch):
    token = '0123456789abcdef'
    folder = tmp_path / token
    folder.mkdir()
    poses = np.zeros((3, 8, 3), np.float32)
    poses[2] = 2
    np.save(folder / 'gt_trajectories.npy', poses)
    np.save(folder / 'gt_valid_mask.npy', [True, True, True])
    np.save(folder / 'gt_like_mask.npy', [True, False, False])
    (folder / 'complete.json').write_text('{}')
    (folder / 'trajectory.json').write_text(json.dumps(dict(
        gt_padded=[False, True, False], target_times=(np.arange(1, 9) * .5).tolist())))
    options = OmegaConf.create(dict(path=str(tmp_path), top_k=8, missing_policy='ignore'))
    future = OmegaConf.create(dict(supervision_scope='human_only', image_size=[16, 16], sensor_blobs_path='unused'))
    reader = DiverseFutureRegisterTargetBuilder(options, future)
    monkeypatch.setattr(diverse_targets, 'extract_diverse_scene_frames',
                        lambda *args: (_ for _ in ()).throw(AssertionError('Read a D8 video')))
    scene = SimpleNamespace(
        scene_metadata=SimpleNamespace(initial_token=token, num_history_frames=4),
        frames=[SimpleNamespace(cameras=SimpleNamespace(cam_f0=SimpleNamespace(
            image=np.full((16, 16, 3), index, np.uint8)))) for index in range(12)])
    targets = reader.compute_targets(scene)
    assert targets['pseudo_expert_mask'].tolist() == [True, False, True] + [False] * 5
    torch.testing.assert_close(targets['pseudo_experts'][2], torch.full((8, 3), 2.))
    assert targets['human_future_register_images'].shape == (8, 3, 16, 16)
    assert targets['human_future_register_images'][:, 0, 0, 0].tolist() == list(range(4, 12))
    assert targets['human_future_register_valid'].all()
    assert 'future_register_images' not in targets and 'future_register_valid' not in targets
    assert reader.get_unique_name() == 'drivor_diverse8_human_future_rgb_v1_k8_16x16'
    assert reader.fallback_roots == []


@pytest.mark.parametrize('scorer', [False, True])
def test_new_presets_only_change_future_supervision(scorer):
    variant = BASE + ('_scorer' if scorer else '')
    old = OmegaConf.to_container(compose(agent_name(variant)).config, resolve=False)
    new = OmegaConf.to_container(compose(agent_name(variant + '_human_only')).config, resolve=False)
    assert new['future_register'].pop('supervision_scope') == 'human_only'
    for key in ('frames_root', 'rollouts_root', 'build_missing_cache'):
        new['future_register'].pop(key)
        old['future_register'].pop(key)
    assert new == old
    assert new['proposal_num'] == 64 and new['pseudo_expert']['weight'] == .5
