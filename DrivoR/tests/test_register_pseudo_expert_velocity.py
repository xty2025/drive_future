"""Pseudo-expert action coverage must not invent or reassign future targets."""

import torch
import pytest

from navsim.agents.drivoR.multi_gt import multi_gt_velocity_loss
from navsim.agents.drivoR_action_dit.flow import normalize_trajectory
from navsim.agents.drivoR_action_dit_register import agent as agent_module
from test_action_dit_register import config, make_agent, oracle, sample


def expert_config(scorer=False):
    cfg = config(scorer=scorer)
    cfg.pseudo_expert = dict(enabled=True, loss_type='velocity', weight=.5,
                             path='/unused/expert/cache', top_k=8, score_thr=.8)
    return cfg


def add_experts(targets):
    targets = dict(targets)
    experts = torch.randn(1,8,8,3)*5
    experts[:,3:] = float('nan')  # Padded experts must not enter loss/matching.
    targets.update(pseudo_experts=experts,
                   pseudo_expert_mask=torch.tensor([[True]*3+[False]*5]),
                   pseudo_expert_fallback=torch.tensor([False]))
    return targets


@pytest.mark.parametrize('scorer',[False,True])
def test_same_forward_velocity_formula_weighting_and_detach(monkeypatch,sample,scorer):
    features,targets = sample
    targets = add_experts(targets)
    agent = make_agent(monkeypatch,cfg=expert_config(scorer))
    agent.compute_score = oracle
    model = agent._drivor_model
    captured = []
    model.action_dit.register_forward_hook(lambda module,args,out:captured.append((args,out)))
    prediction = agent.forward_train(features,targets)
    result = agent.compute_loss(features,targets,prediction)
    assert len(captured)==1
    inputs,(action,registers) = captured[0]
    expected,_ = multi_gt_velocity_loss(action,normalize_trajectory(targets['pseudo_experts']),
        targets['pseudo_expert_mask'],inputs[2],model.schedule,matching='nearest')
    torch.testing.assert_close(result['pseudo_expert_loss'],expected)
    torch.testing.assert_close(result['trajectory_loss'],prediction['_flow_loss']+.5*expected)
    torch.testing.assert_close(result['loss'],
        agent.loss.trajectory_weight*(prediction['_flow_loss']+.5*expected)
        +agent.loss.final_score_weight*result['final_score_loss']+prediction['_register_loss'])
    assert result['pseudo_expert_count']==3
    assert prediction['_register_matched_indices'].shape==(1,2)
    assert '_pseudo_expert_velocity_loss' not in prediction['_flow_diagnostics']
    result['pseudo_expert_loss'].backward(retain_graph=True)
    assert model.action_dit.head.weight.grad.abs().sum()>0
    assert model.image_backbone.neck.weight.grad.abs().sum()>0
    assert model.action_dit.register_head.weight.grad is None
    assert all(p.grad is None for p in model.ema_teacher.parameters())
    assert all(p.grad is None for p in model.scorer.parameters())
    model.zero_grad(set_to_none=True)
    result['final_score_loss'].backward()
    assert all(p.grad is None for p in model.action_dit.parameters())
    assert all(p.grad is None for p in model.ema_teacher.parameters())
    if scorer:
        assert model.register_score_projection[0].weight.grad.abs().sum()>0


@pytest.mark.parametrize('scorer',[False,True])
def test_experts_do_not_change_noising_or_future_gt_assignment(monkeypatch,sample,scorer):
    features,targets = sample
    targets = add_experts(targets)
    agent = make_agent(monkeypatch,cfg=expert_config(scorer))
    captured = []
    model = agent._drivor_model
    model.action_dit.register_forward_pre_hook(
        lambda module,args:captured.append(tuple(value.detach().clone() for value in args[:3])))
    torch.manual_seed(91)
    first = agent.forward_train(features,targets)
    changed = dict(targets,pseudo_experts=targets['pseudo_experts']+100)
    torch.manual_seed(91)
    second = agent.forward_train(features,changed)
    for original,repeated in zip(captured[0],captured[1]):
        torch.testing.assert_close(original,repeated,rtol=0,atol=0)
    for key in ('proposals','predicted_registers','_register_matched_indices','_flow_loss','_register_loss'):
        torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
    assert not torch.equal(first['_pseudo_expert_velocity_loss'],second['_pseudo_expert_velocity_loss'])


@pytest.mark.parametrize('scorer',[False,True])
def test_checkpoint_uses_equivalent_register_model_without_training_targets(monkeypatch,sample,scorer):
    train = make_agent(monkeypatch,cfg=expert_config(scorer))
    evaluate = make_agent(monkeypatch,scorer=scorer).eval()
    evaluate.load_state_dict(train.state_dict(),strict=True)
    def forbidden(*args,**kwargs):
        raise AssertionError('Inference must not encode GT future images')
    evaluate._drivor_model.ema_teacher.forward = forbidden
    with torch.no_grad():
        prediction = evaluate(sample[0])
    assert prediction['proposals'].shape==(1,8,8,3)
    assert prediction['predicted_registers'].shape==(1,8,8,4,16)
    assert torch.isfinite(prediction['pdm_score']).all()


def test_target_builders_load_experts_only_for_joint_training(monkeypatch):
    agent = make_agent(monkeypatch,cfg=expert_config())
    monkeypatch.setattr(agent_module,'FutureRegisterTargetBuilder',lambda *args:'future')
    monkeypatch.setattr(agent_module,'PseudoExpertTargetBuilder',lambda *args:'experts')
    assert agent.get_target_builders()[1:]==['future','experts']
    agent._checkpoint_path = '/trained/model.ckpt'
    assert len(agent.get_target_builders())==1
    agent._checkpoint_path = None
    agent._drivor_model.future_stage = 'future_only'
    assert agent.get_target_builders()[1:]==['future']
