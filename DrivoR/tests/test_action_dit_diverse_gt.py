"""Human-primary + diverse8 supervision, masks, shared forward and compatibility."""

import json

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from navsim.agents.drivoR_action_dit.agent import DrivoRActionDiTAgent
from navsim.agents.drivoR_action_dit.diverse_gt import DiverseTrajectoryTargetBuilder, DrivoRActionDiTDiverseGTAgent
from navsim.agents.drivoR_action_dit.flow import normalize_trajectory
from navsim.agents.drivoR_action_dit.loss import DrivoRActionDiTLoss
from navsim.agents.drivoR.multi_gt import multi_gt_velocity_loss
from navsim.agents.drivoR_pseudo_expert.loss import multi_expert_coverage_loss
from test_action_dit_auxiliary import CONFIGS, configuration, fixture


def diverse_config(kind='l1'):
    cfg = OmegaConf.merge(configuration(),
        OmegaConf.load(CONFIGS/'drivoR_action_dit_dinov3_vitl_single_gt.yaml').config,
        OmegaConf.load(CONFIGS/'drivoR_action_dit_dinov3_vitl_human_diverse8_l1.yaml').config)
    if kind=='vmse':
        cfg = OmegaConf.merge(cfg,OmegaConf.load(CONFIGS/'drivoR_action_dit_dinov3_vitl_human_diverse8_vmse.yaml').config)
    return cfg


def make_agent(kind):
    value = DrivoRActionDiTDiverseGTAgent(diverse_config(kind),lr_args={},loss=DrivoRActionDiTLoss())
    def oracle(targets,proposals,test=True):
        assert not test and proposals.shape==(2,64,8,3)
        raw = torch.full((2,64,7),.75)
        return raw[...,-1],raw[...,-1].amax(1),raw,None,None,None
    value.compute_score = oracle
    return value


def write_scene(root,token='scene'):
    path = root/token
    path.mkdir(parents=True,exist_ok=True)
    poses = np.arange(8*8*3,dtype=np.float64).reshape(8,8,3)/10
    poses[0]=0  # A valid stop trajectory must not be discarded as padding.
    poses[2]=np.nan
    valid = np.array([True,True,False,True,True,False,False,False])
    np.save(path/'gt_trajectories.npy',poses)
    np.save(path/'gt_valid_mask.npy',valid)
    np.save(path/'gt_like_mask.npy',np.array([False,True,False,False,False,False,False,False]))
    # Slot 3 is also padded, even though the source validity mask says True.
    (path/'trajectory.json').write_text(json.dumps(dict(target_times=(np.arange(1,9)*.5).tolist(),
        gt_padded=[False,False,True,True,False,True,True,True],sparse_coordinate_frame='current_ego_rear_axle')))
    # Scores/raw64/dense trajectories are intentionally absent: not needed.
    (path/'complete.json').write_text('{}')
    return poses


def test_selected_masks_keep_gt_like_and_stop_without_rescoring_or_human_fallback(tmp_path):
    poses = write_scene(tmp_path)
    builder = DiverseTrajectoryTargetBuilder(OmegaConf.create(dict(path=str(tmp_path),top_k=8)))
    result = builder.load('scene')
    assert result['pseudo_experts'].dtype==torch.float32
    assert result['pseudo_expert_mask'].tolist()==[True,True,False,False,True,False,False,False]
    assert result['diverse_gt_like_mask'].tolist()==[False,True,False,False,False,False,False,False]
    assert not result['pseudo_expert_fallback'] and not result['diverse_gt_missing']
    assert not result['pseudo_experts'][0].any()
    torch.testing.assert_close(result['pseudo_experts'][1],torch.tensor(poses[1],dtype=torch.float32))
    assert not result['pseudo_experts'][~result['pseudo_expert_mask']].any()


def test_incomplete_scene_has_no_auxiliary_gt_and_is_rechecked_after_completion(tmp_path):
    builder = DiverseTrajectoryTargetBuilder(OmegaConf.create(dict(path=str(tmp_path),top_k=8)))
    (tmp_path/'scene').mkdir()
    np.save(tmp_path/'scene/gt_trajectories.npy',np.full((8,8,3),np.nan))
    first = builder.load('scene')
    assert first['diverse_gt_missing'] and not first['pseudo_expert_mask'].any()
    assert not first['pseudo_expert_fallback']  # No second copy of human GT.
    write_scene(tmp_path)
    second = builder.load('scene')
    assert not second['diverse_gt_missing'] and second['pseudo_expert_mask'].sum()==3
    strict = DiverseTrajectoryTargetBuilder(OmegaConf.create(dict(path=str(tmp_path),missing_policy='error')))
    with pytest.raises(FileNotFoundError,match='not complete'):
        strict.load('unfinished')


@pytest.mark.parametrize('kind',['l1','vmse'])
def test_human_primary_plus_half_auxiliary_loss_uses_one_dit_and_scorer_forward(fixture,monkeypatch,kind):
    features,targets = fixture
    # These variants do not require the original top4 multi-GT at all.
    targets = {k:v for k,v in targets.items() if not k.startswith('trajectory_multi_gt')}
    targets['diverse_gt_like_mask'] = torch.zeros_like(targets['pseudo_expert_mask'])
    targets['diverse_gt_missing'] = torch.tensor([False,True])
    targets['pseudo_expert_mask'][0,3:]=False
    targets['pseudo_expert_mask'][1]=False
    value = make_agent(kind);model=value._drivor_model
    times = torch.tensor([[1.],[500.]]).expand(2,64)
    monkeypatch.setattr(model.schedule,'sample_timesteps',lambda *args:times)
    dit,score = [],[]
    model.action_dit.register_forward_hook(lambda module,args,out:dit.append((args,out)))
    model.scorer.register_forward_pre_hook(lambda module,args:score.append(args[0].shape[1]))
    prediction = value.forward_train(features,targets)
    result = value.compute_loss(features,targets,prediction)
    assert len(dit)==1 and score==[64]
    inputs,clean = dit[0]
    human = normalize_trajectory(targets['trajectory'])[:,None].expand_as(clean)
    expected_human = model.schedule.velocity_loss(clean,human,inputs[0],times,'best_of_k')[0]
    torch.testing.assert_close(prediction['_flow_loss'],expected_human)
    if kind=='l1':
        expected = multi_expert_coverage_loss(prediction['_pseudo_expert_proposals'],targets['pseudo_experts'],targets['pseudo_expert_mask'])
    else:
        expected,_ = multi_gt_velocity_loss(clean,normalize_trajectory(targets['pseudo_experts']),
            targets['pseudo_expert_mask'],times,model.schedule,matching='nearest')
    torch.testing.assert_close(result['diverse_gt_loss'],expected)
    torch.testing.assert_close(result['trajectory_loss'],expected_human+.5*expected)
    torch.testing.assert_close(result['loss'],expected_human+.5*expected+result['final_score_loss'])
    assert result['diverse_gt_count']==1.5 and result['diverse_gt_scene_coverage']==.5
    assert result['diverse_gt_missing_fraction']==.5
    assert 'flow_multi_gt_loss' not in result and 'flow_long_loss' not in result
    result['diverse_gt_loss'].backward(retain_graph=True)
    assert model.action_dit.head.weight.grad.abs().sum()>0
    assert model.image_backbone.neck.weight.grad.abs().sum()>0
    assert targets['pseudo_experts'].grad is None
    assert all(p.grad is None for p in model.scorer.parameters())
    model.zero_grad(set_to_none=True)
    result['final_score_loss'].backward()
    assert all(p.grad is None for p in model.action_dit.parameters())


@pytest.mark.parametrize('kind',['l1','vmse'])
def test_new_targets_do_not_change_human_noise_or_main_loss_and_empty_aux_is_zero(fixture,kind):
    features,targets=fixture
    targets=dict(targets,diverse_gt_like_mask=torch.zeros_like(targets['pseudo_expert_mask']),
                 diverse_gt_missing=torch.zeros(2,dtype=torch.bool))
    value=make_agent(kind)
    torch.manual_seed(31);first=value.forward_train(features,targets)
    changed=dict(targets,pseudo_experts=targets['pseudo_experts']+100)
    torch.manual_seed(31);second=value.forward_train(features,changed)
    for key in ('proposals','pdm_score','_flow_loss'):
        torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
    empty=dict(targets,pseudo_expert_mask=torch.zeros_like(targets['pseudo_expert_mask']))
    torch.manual_seed(31);out=value.forward_train(features,empty)
    result=value.compute_loss(features,empty,out)
    assert result['diverse_gt_loss']==0
    torch.testing.assert_close(result['loss'],result['human_trajectory_loss']+result['final_score_loss'])


@pytest.mark.parametrize('kind',['l1','vmse'])
def test_training_reader_and_plain_dinov3_checkpoint_compatibility(fixture,tmp_path,kind):
    features,_=fixture
    value=make_agent(kind)
    value._config.multi_gt.root='/not/used/old/multi_gt'
    value._config.pseudo_expert.path=str(tmp_path)
    builders=value.get_target_builders()
    assert len(builders)==2 and isinstance(builders[-1],DiverseTrajectoryTargetBuilder)
    baseline_cfg=OmegaConf.create(OmegaConf.to_container(value._config,resolve=False));del baseline_cfg['pseudo_expert']
    baseline=DrivoRActionDiTAgent(baseline_cfg,lr_args={},loss=DrivoRActionDiTLoss()).eval()
    baseline.load_state_dict(value.state_dict(),strict=True)
    value._checkpoint_path='/checkpoint.ckpt'
    value._config.pseudo_expert.path='/no/GT/at/evaluation'
    assert len(value.get_target_builders())==1
    with torch.no_grad():
        out=baseline(features)
    assert out['proposals'].shape==(2,64,8,3) and out['proposals'].dtype==torch.float32
