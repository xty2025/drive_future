"""Bank supervision, scorer/generator gradient boundaries and aligned future GTs."""

import importlib.util
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from navsim.agents.drivoR import drivor_model as base_module
from navsim.agents.drivoR_world.model import (
    DrivoRWorldModel, sample_bank, bank_outcome_loss, future_cosine_loss, nearest_proposals)
from navsim.agents.drivoR_world.targets import BankTargetBuilder, FutureTargetBuilder


def config(future=False):
    cfg = dict(num_poses=8, proposal_num=64, tf_d_model=16, tf_d_ffn=32,
               num_scene_tokens=2, full_history_status=False, one_token_per_traj=True,
               ref_num=2, scorer_ref_num=2, refiner_num_heads=1, refiner_ls_values=0.,
               image_size=[32, 32], image_backbone={}, lidar_pc=[], b2d=False,
               double_score=False, agent_pred=False, area_pred=False, bev_map=False, bev_agent=False,
               noc=1., dac=1., ddc=0., ttc=5., ep=5., comfort=2.,
               bank=dict(count=16, safety_fraction=.5),
               future=dict(enabled=future, times=[.5,1,1.5,2,2.5,3,3.5,4], embedding_dim=12,
                           num_layers=2, num_heads=2, detach_context=True))
    for name in ('cam_f0','cam_l0','cam_l1','cam_l2','cam_r0','cam_r1','cam_r2','cam_b0'):
        cfg[name] = [3] if name == 'cam_f0' else []
    return OmegaConf.create(cfg)


class TinyImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_features = 16
        self.projection = nn.Linear(3, 16)

    def forward(self, images, queries):
        return queries.flatten(1, 2) + self.projection(images.mean((1, 3, 4)))[:, None]


@pytest.fixture
def models(monkeypatch):
    monkeypatch.setattr(base_module, 'ImgEncoder', TinyImageEncoder)
    torch.manual_seed(17)
    features = {'image': torch.randn(2,1,3,32,32), 'ego_status': torch.randn(2,4,11)}
    return features


def test_bank_only_preserves_original_drivor_predictions(models):
    base = base_module.DrivoRModel(config()).eval()
    world = DrivoRWorldModel(config()).eval()
    world.load_state_dict(base.state_dict(), strict=True)
    with torch.no_grad():
        a, b = base(models), world(models)
    assert set(a) == set(b)
    for key in ('proposals', 'trajectory', 'pdm_score'):
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)


def test_bank_sampling_unsafe_quota_and_no_invalid_candidates():
    trajectories = torch.randn(2,64,8,3,requires_grad=True)
    labels = torch.ones(2,64,7,requires_grad=True)
    with torch.no_grad(): labels[0,:12,0] = 0.
    valid = torch.ones(2,64,dtype=torch.bool)
    valid[1] = False
    torch.manual_seed(2)
    c, y, mask = sample_bank(trajectories, labels, valid)
    assert c.shape == (2,16,8,3) and y.shape == (2,16,7)
    assert int((y[0,:,0] == 0).sum()) >= 8
    assert len(torch.unique(c[0].flatten(1),dim=0)) == 16
    assert mask[0].all() and not mask[1].any()
    assert not c.requires_grad and not y.requires_grad


def test_bank_replacement_uses_only_valid_pool():
    c = torch.arange(64.).reshape(1,64,1,1).expand(-1,-1,8,3)
    labels = torch.ones(1,64,7)
    valid = torch.zeros(1,64,dtype=torch.bool)
    valid[0,[2,7,9]] = True
    out, _, mask = sample_bank(c, labels, valid)
    assert set(out[0,:,0,0].tolist()) == {2.,7.,9.} and mask.all()


def test_bank_formula_matches_five_bce_and_progress_l1():
    names = ['no_at_fault_collisions','drivable_area_compliance','ego_progress',
             'time_to_collision_within_bound','comfort','driving_direction_compliance']
    logits = {name: torch.tensor([[.2,-.4]],requires_grad=True) for name in names}
    y = torch.tensor([[[.5,1,.8,0,1,.5,0],[1,0,.2,1,0,1,0]]])
    expected = []
    for col in (0,1,3,4,5):
        target = y[...,col].clone()
        if col in (0,5): target[target==.5] = 0
        expected.append(F.binary_cross_entropy_with_logits(logits[names[col]],target))
    expected = torch.stack(expected).mean() + .1*(logits['ego_progress'].sigmoid()-y[...,2]).abs().mean()
    actual = bank_outcome_loss(logits,y,torch.ones(1,2,dtype=torch.bool))
    torch.testing.assert_close(actual,expected)


@pytest.mark.parametrize('future', [False,True])
def test_bank_loss_stops_generator_coordinates_but_trains_scene_and_scorer(models, future):
    model = DrivoRWorldModel(config(future)).train()
    prediction = model(models)
    targets = {'bank_candidates': torch.randn(2,64,8,3,requires_grad=True),
               'bank_subscores': torch.rand(2,64,7), 'bank_valid': torch.ones(2,64,dtype=torch.bool)}
    loss, coverage = model.bank_loss(targets,prediction)
    loss.backward()
    assert coverage == 1
    assert targets['bank_candidates'].grad is None
    assert all(p.grad is None for p in model.traj_head.parameters())
    assert model.init_feature.weight.grad is None
    assert model.hist_encoding.weight.grad.abs().sum() > 0
    assert model.image_backbone.projection.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.scorer.parameters())


def test_future_loss_detaches_direct_context_but_backprops_through_selected_trajectories(models):
    model = DrivoRWorldModel(config(True)).train()
    prediction = model(models)
    targets = {'future_gt_trajectories': torch.randn(2,4,8,3),
               'future_features': torch.randn(2,4,8,12,requires_grad=True),
               'future_valid': torch.ones(2,4,8,dtype=torch.bool)}
    targets['future_valid'][0,1:] = False
    observed = []
    hook = model.future_predictor.register_forward_pre_hook(lambda module,args:observed.append(args))
    loss = model.realized_future_loss(targets,prediction)
    hook.remove()
    indices = nearest_proposals(prediction['proposals'],targets['future_gt_trajectories'])
    selected = prediction['proposals'][torch.arange(2)[:,None],indices]
    torch.testing.assert_close(observed[0][0], selected[targets['future_valid'].any(-1)], atol=0, rtol=0)
    assert not observed[0][1].requires_grad and observed[0][1].grad_fn is None
    loss.backward()
    assert targets['future_features'].grad is None
    # This is the intentionally retained indirect path: future -> trajectory
    # generator -> current visual encoder. The predictor's direct K/V is detached.
    assert model.image_backbone.projection.weight.grad.abs().sum() > 0
    assert model.scene_embeds.grad.abs().sum() > 0
    assert model.hist_encoding.weight.grad.abs().sum() > 0
    assert model.init_feature.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.traj_head[-1].parameters())
    assert all(p.grad is None for p in model.scorer.parameters())
    assert all(p.grad is None for p in model.scorer_attention.parameters())
    assert model.pos_embed[0].weight.grad is None
    assert model.future_predictor.readout.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.future_predictor.parameters())


def test_future_branch_does_not_repeat_vision_or_trajectory_generation(models):
    model = DrivoRWorldModel(config(True)).train()
    calls = {'vision':0,'generator':0}
    def count(name):
        def hook(module,args): calls[name] += 1
        return hook
    handles = [model.image_backbone.register_forward_pre_hook(count('vision')),
               model.trajectory_decoder.register_forward_pre_hook(count('generator'))]
    prediction = model(models)
    targets = {'future_gt_trajectories':torch.randn(2,1,8,3),
               'future_features':torch.randn(2,1,8,12), 'future_valid':torch.ones(2,1,8,dtype=torch.bool)}
    model.realized_future_loss(targets,prediction)
    for handle in handles: handle.remove()
    assert calls == {'vision':1,'generator':1}


def test_multiple_gts_can_choose_the_same_proposal():
    proposals = torch.zeros(1,3,8,3,requires_grad=True)
    with torch.no_grad():
        proposals[0,1,:,0] = 10
        proposals[0,2,:,0] = 20
    gt = torch.zeros(1,2,8,3)
    gt[0,0,:,0],gt[0,1,:,0] = .1,.2
    indices = nearest_proposals(proposals,gt)
    assert indices.tolist() == [[0,0]]
    selected = proposals[torch.arange(1)[:,None],indices]
    selected.mean().backward()
    assert proposals.grad[0,0].count_nonzero() > 0
    assert proposals.grad[0,1:].count_nonzero() == 0


def test_future_loss_means_times_and_actual_gts_per_scene():
    prediction = torch.tensor([1.,0.]).repeat(2,2,8,1).requires_grad_()
    target = torch.tensor([0.,1.]).repeat(2,2,8,1)
    target[0,0] = torch.tensor([1.,0.])
    mask = torch.ones(2,2,8,dtype=torch.bool)
    mask[1,1] = False
    target[1,1] = float('nan')
    loss = future_cosine_loss(prediction,target,mask)
    torch.testing.assert_close(loss,torch.tensor(.75))
    loss.backward()
    assert prediction.grad[1,1].count_nonzero() == 0


def test_future_cache_keeps_video_slot_with_its_trajectory(tmp_path):
    token = 'abcd0123456789ab'
    folder = tmp_path/token[:2]
    folder.mkdir()
    f = np.zeros((4,8,12),np.float16)
    trajectories = np.zeros((4,8,3),np.float32)
    valid = np.zeros((4,8),bool)
    for slot in (0,2,3):
        f[slot], trajectories[slot], valid[slot] = slot+1,slot+10,True
    np.savez(folder/(token+'.npz'),features=f,trajectories=trajectories,valid=valid,times=np.arange(1,9)*.5)
    builder = FutureTargetBuilder(OmegaConf.create(dict(path=str(tmp_path),max_gt=4,
        times=[.5,1,1.5,2,2.5,3,3.5,4],embedding_dim=12,missing_policy='error')))
    result = builder.load(token)
    assert not result['future_valid'][1].any()
    assert result['future_features'][2,0,0] == 3 and result['future_gt_trajectories'][2,0,0] == 12


def test_missing_future_is_masked_without_fabricating_supervision(tmp_path):
    cfg = OmegaConf.create(dict(path=str(tmp_path), max_gt=4, times=[.5,1,1.5,2,2.5,3,3.5,4],
                                embedding_dim=12, missing_policy='ignore'))
    targets = FutureTargetBuilder(cfg).load('missing_scene')
    assert not targets['future_valid'].any()
    prediction = torch.randn(1,4,8,12,requires_grad=True)
    loss = future_cosine_loss(prediction,targets['future_features'][None],targets['future_valid'][None])
    assert loss == 0
    loss.backward()
    assert prediction.grad.count_nonzero() == 0


def test_evaluation_checkpoint_does_not_require_training_auxiliary_caches():
    from navsim.agents.drivoR_world.agent import DrivoRWorldAgent
    agent = DrivoRWorldAgent.__new__(DrivoRWorldAgent)
    nn.Module.__init__(agent)
    agent._config = config(True)
    agent._checkpoint_path = '/tmp/complete_training_checkpoint.ckpt'
    assert len(agent.get_target_builders()) == 1


def test_batched_teacher_encoding_keeps_scene_slot_and_time_alignment(tmp_path, monkeypatch):
    tools = Path(__file__).resolve().parents[1]/'tools'
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location('prepare_world_test',tools/'prepare_world4scorer_data.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokens = ['aaaa0123456789ab','bbbb0123456789ab']
    assignments = [np.asarray([(s,t) for s in (0,2) for t in range(8)]),
                   np.asarray([(0,t) for t in range(8)])]
    items = []
    for i, (token, assignment) in enumerate(zip(tokens, assignments)):
        start = 0 if i == 0 else 16
        pixels = np.arange(start,start+len(assignment),dtype=np.uint8)[:,None,None,None]*np.ones((1,2,2,3),np.uint8)
        trajectories = np.full((4 if i == 0 else 1,8,3),i+1,np.float32)
        items.append((token,trajectories,pixels,assignment))
    class Images:
        def __init__(self, records, args): pass
        def __len__(self): return len(items)
        def __getitem__(self, index): return items[index]
    class Teacher(nn.Module):
        def forward(self, x): return x[:,0,0,0,0,None,None].expand(-1,1,1024)
    from navsim.agents.drivoR_action_dit import vjepa2_1_image_encoder
    import safetensors.torch
    monkeypatch.setattr(vjepa2_1_image_encoder,'build_official_vjepa21',lambda name:Teacher())
    monkeypatch.setattr(safetensors.torch,'load_file',lambda *a,**kw:{})
    monkeypatch.setattr(module,'FutureImages',Images)
    monkeypatch.setattr(module,'scene_tokens',lambda *a:tokens)
    from argparse import Namespace
    args = Namespace(rank=0,world_size=1,mode='encode-cosmos',gt_root='',tokens_json=None,limit=None,
                     output=str(tmp_path),overwrite=False,workers=0,device='cpu',weights='',batch_size=20,size=[2,2])
    module.encode(args)
    with np.load(tmp_path/'aa'/(tokens[0]+'.npz')) as a:
        assert not a['valid'][1].any()
        np.testing.assert_allclose(a['features'][2,0,0],(8/255-.485)/.229,atol=.002)
    with np.load(tmp_path/'bb'/(tokens[1]+'.npz')) as b:
        np.testing.assert_allclose(b['features'][0,0,0],(16/255-.485)/.229,atol=.002)
        assert b['trajectories'][0,0,0] == 2


def test_frame_indices_include_future_4s_and_exclude_current(tmp_path):
    path = Path(__file__).resolve().parents[1]/'tools/extract_cosmos_future_frames.py'
    spec = importlib.util.spec_from_file_location('extract_frames_test',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    indices = module.frame_indices(np.arange(41)*.1)
    assert indices.tolist() == [5,10,15,20,25,30,35,40]
    pixels = np.zeros((41,32,32,3),np.uint8)
    pixels[...,0] = np.arange(41)[:,None,None]*5
    pixels[...,1] = 40
    video = tmp_path/'future.mp4'
    subprocess.run(['ffmpeg','-v','error','-threads','1','-f','rawvideo','-pix_fmt','rgb24',
                    '-s','32x32','-r','10','-i','pipe:0','-c:v','libx264','-crf','0',
                    '-pix_fmt','yuv444p','-threads','1',str(video)],input=pixels.tobytes(),check=True)
    decoded = module.decode_frames(video,indices,(32,32))
    np.testing.assert_allclose(decoded[...,0].mean((1,2)),indices*5,atol=2)


def test_bank_converter_skips_official_empty_scene_entries(tmp_path):
    import pickle
    path = Path(__file__).resolve().parents[1]/'tools/build_world4scorer_bank.py'
    spec = importlib.util.spec_from_file_location('world_bank_test',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scores = [{key:1. for key in module.SCORE_KEYS} for _ in range(2)]
    data = [{'token':'empty','valid':True,'trajectories_relative':[],'scores':[]},
            {'token':'valid','valid':True,'trajectories_relative':np.zeros((2,8,3)),'scores':scores}]
    source=tmp_path/'source.pkl'
    with source.open('wb') as handle: pickle.dump(data,handle)
    report=module.build(source,tmp_path/'bank')
    assert report['num_tokens']==1 and report['empty_scenes_skipped']==1
    builder=BankTargetBuilder(OmegaConf.create(dict(path=str(tmp_path/'bank'),capacity=64,missing_policy='ignore')))
    assert builder.load('valid')['bank_valid'].sum()==2
    assert not builder.load('empty')['bank_valid'].any()


def test_incomplete_bank_is_not_accepted_as_ready(tmp_path):
    np.save(tmp_path/'tokens.npy',np.asarray(['token'],dtype='S64'))
    with pytest.raises(FileNotFoundError):
        BankTargetBuilder(OmegaConf.create(dict(path=str(tmp_path),capacity=64,missing_policy='ignore')))
