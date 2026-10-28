"""Single-human supervision keeps all proposals and aligns the same winner."""

from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import torch

from navsim.agents.drivoR_action_dit.flow import normalize_trajectory
from navsim.agents.drivoR_action_dit_register import agent as agent_module
from navsim.agents.drivoR_action_dit_register import targets as target_module
from navsim.agents.drivoR_action_dit_register.losses import matched_register_loss
from test_action_dit_register import config, make_agent, oracle, sample


def single_config():
    cfg = config(scorer=True)
    cfg.multi_gt.enabled = False
    cfg.multi_gt.max_gt = 1
    cfg.multi_gt.root = '/unavailable/multi_gt'
    cfg.future_register.frames_root = '/unavailable/cosmos'
    cfg.future_register.sensor_blobs_path = '/sensors'
    return cfg


def test_single_human_best_of_candidates_reuses_teacher_and_matching(monkeypatch,sample):
    features,all_targets = sample
    targets = {k:v for k,v in all_targets.items() if k=='trajectory' or k.startswith('human_future_')}
    agent = make_agent(monkeypatch,cfg=single_config())
    agent.compute_score = oracle
    model = agent._drivor_model
    dit,teachers = [],[]
    model.action_dit.register_forward_hook(lambda module,args,out:dit.append((args,out)))
    model.ema_teacher.register_forward_hook(lambda module,args,out:teachers.append((args,out)))
    torch.manual_seed(41)
    prediction = agent.forward_train(features,targets)
    result = agent.compute_loss(features,targets,prediction)
    assert len(dit)==len(teachers)==1
    assert teachers[0][0][0].shape==(1,1,8,3,32,32)
    assert prediction['proposals'].shape==(1,8,8,3)  # Tiny model: still all 8 candidates.
    assert prediction['_register_matched_indices'].shape==(1,1)
    inputs,(action,registers) = dit[0]
    human = normalize_trajectory(targets['trajectory'])[:,None].expand_as(action)
    expected_action = model.schedule.velocity_loss(action,human,inputs[0],inputs[2],'best_of_k')[0]
    torch.testing.assert_close(prediction['_flow_loss'],expected_action)
    expected_future = matched_register_loss(prediction['predicted_registers'],teachers[0][1],
        targets['human_future_register_valid'][:,None],prediction['_register_matched_indices'],
        inputs[2],model.schedule,'velocity')
    torch.testing.assert_close(prediction['_register_loss'],expected_future)
    torch.testing.assert_close(result['loss'],expected_action+expected_future+result['final_score_loss'])
    assert result['multi_gt_count']==1 and result['future_register_gt_count']==1
    assert 'pseudo_expert_loss' not in result
    result['loss'].backward()
    assert model.action_dit.head.weight.grad.abs().sum()>0
    assert model.action_dit.register_head.weight.grad.abs().sum()>0
    assert model.image_backbone.neck.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.ema_teacher.parameters())
    # Even stale multi-GT cache entries must not change single-human supervision.
    unrelated = {k:torch.full_like(v,float('nan')) if v.is_floating_point() else v
                 for k,v in all_targets.items() if k not in targets}
    torch.manual_seed(41)
    repeated = agent.forward_train(features,{**targets,**unrelated})
    for key in ('proposals','predicted_registers','_flow_loss','_register_loss','_register_matched_indices'):
        torch.testing.assert_close(prediction[key],repeated[key],rtol=0,atol=0)


def test_single_gt_builder_uses_real_human_frames_without_external_gt_or_video_cache(monkeypatch):
    agent = make_agent(monkeypatch,cfg=single_config())
    def forbidden(*args,**kwargs):
        raise AssertionError('Single-human training must not open multi-GT / Cosmos targets')
    monkeypatch.setattr(agent_module,'FutureRegisterTargetBuilder',forbidden)
    builders = agent.get_target_builders()
    assert len(builders)==2 and isinstance(builders[1],target_module.HumanFutureRegisterTargetBuilder)
    future = OmegaConf.create(dict(image_size=[8,8],sensor_blobs_path='/sensors'))
    builder = target_module.HumanFutureRegisterTargetBuilder(future)
    paths = []
    def load(root,path):
        paths.append(path)
        return np.full((8,8,3),int(path),np.uint8)
    monkeypatch.setattr(target_module,'load_camera_image',load)
    scene = SimpleNamespace(scene_metadata=SimpleNamespace(initial_token='test',num_history_frames=4),
        frames=[SimpleNamespace(cameras=SimpleNamespace(cam_f0=SimpleNamespace(image=None,image_path=str(i))))
                for i in range(12)])
    result = builder.compute_targets(scene)
    assert paths==list(map(str,range(4,12)))
    assert set(result)=={'human_future_register_images','human_future_register_valid'}
    assert result['human_future_register_images'][:,0,0,0].tolist()==list(range(4,12))
    assert result['human_future_register_valid'].all()
    agent._checkpoint_path = '/checkpoint.ckpt'
    assert len(agent.get_target_builders())==1


def test_single_gt_and_original_register_scorer_share_checkpoint_architecture(monkeypatch,sample):
    single = make_agent(monkeypatch,cfg=single_config())
    original = make_agent(monkeypatch,scorer=True).eval()
    original.load_state_dict(single.state_dict(),strict=True)
    with torch.no_grad():
        output = original(sample[0])
    assert output['proposals'].shape==(1,8,8,3)
    assert output['predicted_registers'].shape==(1,8,8,4,16)
