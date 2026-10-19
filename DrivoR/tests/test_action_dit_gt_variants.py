"""Good+poor GT concatenation and human-only Action-DiT supervision."""

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from navsim.agents.drivoR.merged_multi_gt import MergedMultiTrajectoryTargetBuilder
from navsim.agents.drivoR.multi_gt import MultiTrajectoryTargetBuilder
from navsim.agents.drivoR_action_dit.agent import DrivoRActionDiTAgent
from navsim.agents.drivoR_action_dit import model as model_module


def write_export(root, token, count, offset=0, pad_to=None):
    folder = root / token
    folder.mkdir(parents=True, exist_ok=True)
    size = count if pad_to is None else pad_to
    trajectories = np.zeros((size, 8, 3), np.float64)
    trajectories[:, :, 0] = (np.arange(size) + offset)[:, None]
    np.save(folder / 'gt_trajectories.npy', trajectories)
    np.save(folder / 'gt_valid_mask.npy', np.ones(size, bool))
    (folder / 'trajectory.json').write_text(json.dumps(dict(
        target_times=(np.arange(1, 9) * .5).tolist(),
        gt_padded=[index >= count for index in range(size)])))


def reader(root, max_gt=8):
    return MergedMultiTrajectoryTargetBuilder(OmegaConf.create(dict(
        root=str(root / 'good'), additional_roots=[str(root / 'poor')],
        max_gt=max_gt, weighting='equal', matching='nearest', missing_policy='original')))


@pytest.mark.parametrize('good_count,poor_count', [(1,0),(2,1),(3,2),(4,4)])
def test_merge_uses_actual_valid_count_and_excludes_good_padding(tmp_path, good_count, poor_count):
    write_export(tmp_path/'good', 'scene', good_count, pad_to=4)
    write_export(tmp_path/'poor', 'scene', poor_count, offset=10)
    result = reader(tmp_path).load('scene', np.full((8,3),100.))
    count = good_count + poor_count
    assert result['trajectory_multi_gt'].shape == (8,8,3)
    assert result['multi_gt_count'] == count
    assert result['multi_gt_primary_count'] == good_count
    assert result['multi_gt_additional_count'] == poor_count
    assert result['trajectory_multi_gt_valid'].tolist() == [True]*count+[False]*(8-count)
    expected = list(range(good_count)) + list(range(10,10+poor_count))
    assert result['trajectory_multi_gt'][:count,0,0].tolist() == expected
    assert not result['multi_gt_fallback']


def test_missing_poor_never_adds_another_human_fallback(tmp_path):
    write_export(tmp_path/'good','scene',2,pad_to=4)
    (tmp_path/'poor').mkdir()
    result=reader(tmp_path).load('scene',np.ones((8,3)))
    assert result['multi_gt_count'] == 2 and result['multi_gt_additional_count'] == 0


def test_primary_fallback_is_retained_once_and_poor_gt_is_still_appended(tmp_path):
    (tmp_path/'good').mkdir()
    write_export(tmp_path/'poor','scene',3,offset=10)
    result=reader(tmp_path).load('scene',np.ones((8,3)))
    assert result['multi_gt_count'] == 4 and result['multi_gt_fallback']
    assert result['multi_gt_primary_count'] == 1 and result['multi_gt_additional_count'] == 3
    torch.testing.assert_close(result['trajectory_multi_gt'][0],torch.ones(8,3))


def test_merge_reuses_mask_checks_and_refuses_silent_truncation(tmp_path):
    write_export(tmp_path/'good','scene',4)
    write_export(tmp_path/'poor','scene',4,offset=10)
    with pytest.raises(ValueError,match='merged GTs'):
        reader(tmp_path,max_gt=4).load('scene',np.zeros((8,3)))
    mask=np.array([True,False,True,False])
    np.save(tmp_path/'poor/scene/gt_valid_mask.npy',mask)
    result=reader(tmp_path).load('scene',np.zeros((8,3)))
    assert result['multi_gt_count'] == 6
    assert result['trajectory_multi_gt'][4:6,0,0].tolist() == [10.,12.]


def test_merged_cache_name_cannot_reuse_old_four_gt_targets(tmp_path):
    (tmp_path/'good').mkdir(); (tmp_path/'poor').mkdir()
    merged=reader(tmp_path)
    old=MultiTrajectoryTargetBuilder(OmegaConf.create(dict(root=str(tmp_path/'good'))))
    assert merged.get_unique_name() != old.get_unique_name()


def config(variant):
    folder=Path(__file__).resolve().parents[1]/'navsim/planning/script/config/common/agent'
    names=['drivoR','drivoR_action_dit','drivoR_action_dit_multigt','drivoR_action_dit_dinov3_vitl']
    if variant == 'single_gt_long':
        names.append('drivoR_action_dit_dinov3_vitl_single_gt')
    names.append('drivoR_action_dit_dinov3_vitl_'+variant)
    return OmegaConf.merge(*[OmegaConf.load(folder/(name+'.yaml')).config for name in names])


def test_single_gt_config_has_no_export_loader_or_long_target():
    cfg=config('single_gt')
    cfg.multi_gt.root='/does/not/exist'
    agent=DrivoRActionDiTAgent.__new__(DrivoRActionDiTAgent)
    nn.Module.__init__(agent)
    agent._config=cfg
    assert len(agent.get_target_builders()) == 1
    assert not cfg.multi_gt.enabled and cfg.long_trajectory_additional_poses == 0
    assert cfg.proposal_num == 64


def test_real_poor_export_empty_and_nonempty_rows_are_both_supported():
    root=Path(__file__).resolve().parents[2]/'multi_traj_gt_data'
    poor=root/'navtrain_poor4_multigt_dense_0p1s_f64_16gpu'
    good=root/'navtrain_multigt_top4_dense_0p1s_f64_8gpu'
    if not poor.is_dir():
        pytest.skip('Real poor GT export unavailable')
    builder=MergedMultiTrajectoryTargetBuilder(OmegaConf.create(dict(
        root=str(good),additional_roots=[str(poor)],max_gt=8,missing_policy='original')))
    for token,primary,additional in [('357a415dcb8b55a6',1,0),('572b74405e2252cb',2,2)]:
        result=builder.load(token,np.zeros((8,3)))
        assert result['multi_gt_count'] == primary+additional
        assert result['multi_gt_primary_count'] == primary
        assert result['multi_gt_additional_count'] == additional


class TinyImageEncoder(nn.Module):
    def __init__(self, options, image_size, output_dim, *args):
        super().__init__()
        self.neck=nn.Linear(3,output_dim)

    def forward(self, image):
        return self.neck(image.mean((-1,-2)))


def test_single_human_loss_matches_one_valid_multi_gt_and_runs_dit_once(monkeypatch):
    monkeypatch.setattr(model_module,'DinoV3ImageEncoder',TinyImageEncoder)
    cfg=config('single_gt')
    cfg.tf_d_model=8; cfg.tf_d_ffn=16; cfg.scorer_ref_num=1
    cfg.action_dit.update(dict(hidden_dim=8,ffn_dim=16,freq_dim=8,num_heads=2,attn_head_dim=4,
                              num_layers=1,pretrained_path='',gradient_checkpointing=False))
    model=model_module.DrivoRActionDiTModel(cfg).train()
    features=dict(image=torch.randn(2,1,3,32,32),ego_status=torch.randn(2,4,11))
    targets=dict(trajectory=torch.randn(2,8,3),trajectory_long=torch.full((2,8,3),1000.))
    calls=[]
    hook=model.action_dit.register_forward_pre_hook(lambda module,args:calls.append(tuple(args[0].shape)))
    torch.manual_seed(19)
    single=model(features,targets=targets)
    assert calls == [(2,64,8,3)]
    assert 'flow_long_loss' not in single['_flow_diagnostics']
    cfg.multi_gt.enabled=True
    targets.update(trajectory_multi_gt=targets['trajectory'][:,None],
                   trajectory_multi_gt_valid=torch.ones(2,1,dtype=torch.bool))
    torch.manual_seed(19)
    multi=model(features,targets=targets)
    hook.remove()
    torch.testing.assert_close(single['proposals'],multi['proposals'],rtol=0,atol=0)
    torch.testing.assert_close(single['_flow_loss'],multi['_flow_loss'])
    single['_flow_loss'].backward()
    assert model.action_dit.head.weight.grad.abs().sum() > 0


def test_model_gt_builder_dispatches_merged_only_for_the_new_config(tmp_path):
    write_export(tmp_path/'good','scene',4)
    write_export(tmp_path/'poor','scene',4)
    cfg=config('multigt_poor')
    cfg.multi_gt.root=str(tmp_path/'good')
    cfg.multi_gt.additional_roots=[str(tmp_path/'poor')]
    agent=DrivoRActionDiTAgent.__new__(DrivoRActionDiTAgent)
    nn.Module.__init__(agent)
    agent._config=cfg
    assert isinstance(agent.get_target_builders()[1],MergedMultiTrajectoryTargetBuilder)
    cfg.multi_gt.additional_roots=[]
    assert type(agent.get_target_builders()[1]) is MultiTrajectoryTargetBuilder


def test_human_plus_long_config_packs_two_groups_into_one_dit_forward(monkeypatch):
    monkeypatch.setattr(model_module,'DinoV3ImageEncoder',TinyImageEncoder)
    cfg=config('single_gt_long')
    assert not cfg.multi_gt.enabled and cfg.long_trajectory_additional_poses == 2
    cfg.tf_d_model=8; cfg.tf_d_ffn=16; cfg.scorer_ref_num=1
    cfg.action_dit.update(dict(hidden_dim=8,ffn_dim=16,freq_dim=8,num_heads=2,attn_head_dim=4,
                              num_layers=1,pretrained_path='',gradient_checkpointing=False))
    model=model_module.DrivoRActionDiTModel(cfg).train()
    calls=[]
    model.action_dit.register_forward_pre_hook(lambda module,args:calls.append(tuple(args[0].shape)))
    features=dict(image=torch.randn(2,1,3,32,32),ego_status=torch.randn(2,4,11))
    targets=dict(trajectory=torch.randn(2,8,3),trajectory_long=torch.randn(2,8,3))
    output=model(features,targets=targets)
    assert calls == [(2,128,8,3)]
    assert output['proposals'].shape == (2,64,8,3)
    diag=output['_flow_diagnostics']
    torch.testing.assert_close(output['_flow_loss'],diag['flow_normal_loss']+diag['flow_long_loss'])
    output['_flow_loss'].backward()
    assert model.action_dit.head.weight.grad.abs().sum() > 0
