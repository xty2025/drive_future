"""Physical X L1, independent human/long winners, and unchanged inference."""

from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from navsim.agents.drivoR_action_dit.agent import DrivoRActionDiTAgent
from navsim.agents.drivoR_action_dit.diverse_gt import DrivoRActionDiTDiverseGTAgent
from navsim.agents.drivoR_action_dit.flow import denormalize_trajectory, normalize_trajectory
from navsim.agents.drivoR_action_dit.loss import DrivoRActionDiTLoss
from navsim.agents.drivoR_action_dit.x_l1 import (
    DrivoRActionDiTXL1Agent, DrivoRActionDiTDiverseGTXL1Agent, physical_x_l1_loss,
)
from navsim.agents.drivoR_pseudo_expert.loss import multi_expert_coverage_loss
from test_action_dit_auxiliary import CONFIGS, configuration, fixture


VARIANTS = ['human_xl1_diverse8_l1', 'single_gt_long_xl1']


def config(variant):
    parent = 'human_diverse8_l1' if variant == VARIANTS[0] else 'single_gt_long'
    return OmegaConf.merge(configuration(),
        OmegaConf.load(CONFIGS/'drivoR_action_dit_dinov3_vitl_single_gt.yaml').config,
        OmegaConf.load(CONFIGS/f'drivoR_action_dit_dinov3_vitl_{parent}.yaml').config,
        OmegaConf.load(CONFIGS/f'drivoR_action_dit_dinov3_vitl_{variant}.yaml').config)


def make_agent(variant, legacy=False):
    cls = ((DrivoRActionDiTDiverseGTAgent if legacy else DrivoRActionDiTDiverseGTXL1Agent)
           if variant == VARIANTS[0] else
           (DrivoRActionDiTAgent if legacy else DrivoRActionDiTXL1Agent))
    value = cls(config(variant), lr_args={}, loss=DrivoRActionDiTLoss())
    def oracle(targets, proposals, test=True):
        assert not test and proposals.shape == (2, 64, 8, 3)
        raw = torch.full((2, 64, 7), .75)
        return raw[..., -1], raw[..., -1].amax(1), raw, None, None, None
    value.compute_score = oracle
    return value


def targets_for(fixture):
    features, original = fixture
    targets = {key:value for key,value in original.items() if not key.startswith('trajectory_multi_gt')}
    targets['trajectory_long'] = original['trajectory'] + 5
    targets['diverse_gt_like_mask'] = torch.zeros_like(original['pseudo_expert_mask'])
    targets['diverse_gt_missing'] = torch.tensor([False, True])
    targets['pseudo_expert_mask'][0, 3:] = False
    targets['pseudo_expert_mask'][1] = False
    return features, targets


def test_l1_selects_by_physical_distance_and_only_winner_receives_gradient():
    target = torch.zeros(1, 8, 3, requires_grad=True)
    poses = torch.zeros(1, 2, 8, 3)
    poses[:, 0, :, 1] = 2.0
    poses[:, 1, :, 0] = 3.0
    prediction = normalize_trajectory(poses).detach().requires_grad_()
    # Normalized L1 would prefer the 3m x error; physical L1 prefers 2m y.
    normalized_error = (prediction.detach()-normalize_trajectory(target.detach())[:,None]).abs().sum(-1).mean(-1)
    assert normalized_error.argmin(1).item() == 1
    loss = physical_x_l1_loss(prediction, target)
    torch.testing.assert_close(loss, torch.tensor(2.), atol=1e-5, rtol=1e-5)
    loss.backward()
    assert prediction.grad[:, 0].abs().sum() > 0 and not prediction.grad[:, 1].any()
    assert target.grad is None


@pytest.mark.parametrize('time', [0., .001, 50., 500., 1000.])
def test_human_and_long_l1_have_no_velocity_or_time_weight_and_choose_independent_winners(fixture, monkeypatch, time):
    features, targets = targets_for(fixture)
    value = make_agent('single_gt_long_xl1')
    model = value._drivor_model
    def forbidden(*args, **kwargs):
        raise AssertionError('Direct X L1 must not call v conversion or time weighting')
    for name in ('velocity_loss', 'x_to_v', 'training_weight'):
        monkeypatch.setattr(model.schedule, name, forbidden)
    monkeypatch.setattr(model.schedule, 'sample_timesteps', lambda b,m,device: torch.full((b,m),time))
    human = targets['trajectory'][:, None].expand(-1,64,-1,-1)
    long = targets['trajectory_long'][:, None].expand(-1,64,-1,-1)
    poses = torch.cat((human+10, long+20), dim=1).clone()
    poses[:, 7] = human[:, 7] + torch.tensor([.5, 0., 0.])
    poses[:, 64+42] = long[:, 42] + torch.tensor([1.5, 0., 0.])
    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.prediction = nn.Parameter(normalize_trajectory(poses))
            self.inputs = []
        def forward(self, noisy, times, scene, ego):
            self.inputs.append((noisy.detach(), times.detach()))
            return self.prediction
    model.action_dit = Predictor()
    output = value.forward_train(features, targets)
    assert len(model.action_dit.inputs) == 1
    noisy, times = model.action_dit.inputs[0]
    assert noisy.shape == (2,128,8,3)
    torch.testing.assert_close(times, times[:, :1].expand(-1,128))
    sigma = times[:, :64, None, None]/1000
    torch.testing.assert_close(noisy[:,:64]-noisy[:,64:],
        (1-sigma)*(normalize_trajectory(human)-normalize_trajectory(long)), atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(output['_flow_loss'], torch.tensor(2.), atol=2e-5, rtol=1e-5)
    output['_flow_loss'].backward()
    gradient = model.action_dit.prediction.grad
    active = gradient.abs().sum((-2,-1)) > 0
    assert active.sum(1).tolist() == [2,2]
    assert active[:,7].all() and active[:,64+42].all()


@pytest.mark.parametrize('variant', VARIANTS)
def test_one_forward_same_noisy_inputs_scores_and_checkpoint_as_legacy(fixture, monkeypatch, variant):
    features, targets = targets_for(fixture)
    value, legacy = make_agent(variant), make_agent(variant, legacy=True)
    legacy.load_state_dict(value.state_dict(), strict=True)
    calls, old_calls, scored = [], [], []
    value._drivor_model.action_dit.register_forward_hook(lambda module,args,out:calls.append((args,out)))
    legacy._drivor_model.action_dit.register_forward_hook(lambda module,args,out:old_calls.append((args,out)))
    value._drivor_model.scorer.register_forward_pre_hook(lambda module,args:scored.append(args[0].shape[1]))
    torch.manual_seed(33)
    new_prediction=value.forward_train(features, targets)
    torch.manual_seed(33)
    old_prediction=legacy.forward_train(features, targets)
    for key in ('proposals','pdm_score'):
        torch.testing.assert_close(new_prediction[key],old_prediction[key],atol=0,rtol=0)
    assert len(calls)==len(old_calls)==1 and scored==[64]
    torch.testing.assert_close(calls[0][0][0],old_calls[0][0][0],atol=0,rtol=0)
    expected_count=64 if variant==VARIANTS[0] else 128
    assert calls[0][1].shape==(2,expected_count,8,3)
    result=value.compute_loss(features,targets,new_prediction)
    # Closed form in physical coordinates, independent of the X L1 helper.
    physical=denormalize_trajectory(calls[0][1])
    human_error=(physical[:,:64]-targets['trajectory'][:,None]).abs().sum(-1).mean(-1)
    expected=human_error.min(1).values.mean()
    if variant==VARIANTS[1]:
        long_error=(physical[:,64:]-targets['trajectory_long'][:,None]).abs().sum(-1).mean(-1)
        expected=expected+long_error.min(1).values.mean()
    torch.testing.assert_close(new_prediction['_flow_loss'],expected)
    if variant==VARIANTS[0]:
        coverage=multi_expert_coverage_loss(physical[:,:64],targets['pseudo_experts'],targets['pseudo_expert_mask'])
        torch.testing.assert_close(result['diverse_gt_loss'],coverage)
        expected=expected+.5*coverage
        assert result['diverse_gt_count']==1.5
    torch.testing.assert_close(result['loss'],expected+result['final_score_loss'])
    result['trajectory_loss'].backward(retain_graph=True)
    model=value._drivor_model
    assert model.action_dit.head.weight.grad.abs().sum()>0
    assert model.image_backbone.neck.weight.grad.abs().sum()>0
    assert targets['pseudo_experts'].grad is None
    assert all(p.grad is None for p in model.scorer.parameters())
    model.zero_grad(set_to_none=True)
    result['final_score_loss'].backward()
    assert all(p.grad is None for p in model.action_dit.parameters())
    new_score_loss=legacy.compute_loss(features,targets,old_prediction)['final_score_loss']
    torch.testing.assert_close(result['final_score_loss'],new_score_loss,atol=0,rtol=0)

    value.eval();legacy.eval()
    value._checkpoint_path='/inference/checkpoint.ckpt'
    assert len(value.get_target_builders())==1
    with torch.no_grad():
        torch.manual_seed(73);a=value(features)
        torch.manual_seed(73);b=legacy(features)
    for key in ('proposals','trajectory','pdm_score'):
        torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
    # Training used one call; inference retains the existing Euler schedule.
    assert len(calls)==1+model.num_inference_steps
