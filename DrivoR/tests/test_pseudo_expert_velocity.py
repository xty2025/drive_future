"""Pseudo-expert X->v regression uses the original noisy input/time and GT masks."""

import torch
from torch import nn

from navsim.agents.drivoR_action_dit.flow import normalize_trajectory
from test_action_dit_auxiliary import agent, fixture


def test_velocity_expert_loss_matches_explicit_x_to_v_with_per_scene_time(fixture, monkeypatch):
    features, targets = fixture
    value, _ = agent('pseudo_expert_vmse')
    model = value._drivor_model
    times = torch.tensor([[1.], [500.]]).expand(2, 64)
    monkeypatch.setattr(model.schedule, 'sample_timesteps', lambda *args: times)
    targets['pseudo_expert_mask'][1, 1:] = False
    with torch.no_grad():
        targets['pseudo_experts'][1, 1:] = float('nan')
    captured = []
    model.action_dit.register_forward_hook(lambda module,args,out: captured.append((args[0],args[1],out)))
    prediction = value.forward_train(features, targets)
    assert len(captured) == 1
    noisy, actual_times, pred_x = captured[0]
    valid = targets['pseudo_expert_mask']
    gt = normalize_trajectory(targets['pseudo_experts'].detach()).masked_fill(~valid[...,None,None],0.)
    sigma = model.schedule.sigma(actual_times).clamp_min(.1)[...,None,None]
    v_pred = (noisy - pred_x.float()) / sigma
    v_gt = (noisy[:,None] - gt[:,:,None]) / sigma[:,None]
    per_expert = (v_pred[:,None] - v_gt).square().mean((-1,-2)).min(-1).values
    per_expert *= model.schedule.training_weight(actual_times)[:,0,None]
    expected = (per_expert.masked_fill(~valid,0.).sum(-1)/valid.sum(-1)).mean()
    result = value.compute_loss(features, targets, prediction)
    torch.testing.assert_close(result['pseudo_expert_loss'], expected, rtol=2e-5, atol=2e-5)
    assert '_pseudo_expert_velocity_loss' not in prediction['_flow_diagnostics']
    assert torch.isfinite(result['pseudo_expert_loss'])


def test_velocity_matching_uses_normalized_mse_instead_of_physical_l1(fixture):
    features, targets = fixture
    value, _ = agent('pseudo_expert_vmse')
    model = value._drivor_model
    raw = torch.full((2,64,8,3),100.)
    raw[:,0] = torch.tensor([.4,0.,.1])  # Smaller physical L1: .5.
    raw[:,1] = torch.tensor([1.,0.,0.])  # Smaller normalized MSE: heading is exact.
    class Predictions(nn.Module):
        def __init__(self):
            super().__init__()
            self.value = nn.Parameter(normalize_trajectory(raw))
        def forward(self, noisy, times, scene, ego):
            return self.value
    model.action_dit = Predictions()
    targets['pseudo_experts'] = torch.zeros(2,8,8,3,requires_grad=True)
    targets['pseudo_expert_mask'].zero_()
    targets['pseudo_expert_mask'][:,:2] = True  # Two experts can select the same proposal.
    prediction = value.forward_train(features, targets)
    result = value.compute_loss(features, targets, prediction)
    result['pseudo_expert_loss'].backward()
    gradient = model.action_dit.value.grad
    assert gradient[:,1].abs().sum() > 0
    assert not gradient[:,0].any() and not gradient[:,2:].any()
    assert result['pseudo_expert_matched_proposals'] == 1
    assert targets['pseudo_experts'].grad is None


def test_velocity_expert_loss_backprops_to_dit_and_vision_but_not_scorer(fixture):
    features, targets = fixture
    value, _ = agent('pseudo_expert_vmse')
    model = value._drivor_model
    prediction = value.forward_train(features, targets)
    assert '_pseudo_expert_proposals' not in prediction  # No denormalize/renormalize round-trip.
    result = value.compute_loss(features, targets, prediction)
    result['pseudo_expert_loss'].backward()
    assert model.action_dit.head.weight.grad.abs().sum() > 0
    assert model.image_backbone.neck.weight.grad.abs().sum() > 0
    assert model.hist_encoding.weight.grad.abs().sum() > 0
    assert targets['pseudo_experts'].grad is None
    assert all(p.grad is None for p in model.scorer.parameters())
    assert all(p.grad is None for p in model.scorer_attention.parameters())


def test_single_and_long_helpers_keep_auxiliary_context_out_of_diagnostics(fixture):
    features, targets = fixture
    value, _ = agent('pseudo_expert_vmse')
    model = value._drivor_model
    model._config.multi_gt.enabled = False
    model._config.long_trajectory_additional_poses = 2
    targets['trajectory_long'] = targets['trajectory'] + 1
    calls = []
    model.action_dit.register_forward_pre_hook(lambda module,args: calls.append(args[0].shape[1]))
    prediction = value.forward_train(features, targets)
    assert calls == [128]
    assert 'flow_long_loss' in prediction['_flow_diagnostics']
    assert '_pseudo_expert_velocity_loss' in prediction
    scene, ego = model.encode_scene(features)
    _, diagnostics = model.flow_training_loss(targets,scene,ego)
    assert '_pseudo_expert_velocity_loss' not in diagnostics
