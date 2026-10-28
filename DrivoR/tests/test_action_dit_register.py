"""Directed denoising, EMA teacher, paired future frames and staged contracts."""

import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import sys
import zipfile

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import pytest
import torch
from torch import nn

from navsim.agents.drivoR_action_dit.action_dit import ActionDiT
from navsim.agents.drivoR_action_dit.dinov3_image_encoder import DinoV3LoRAQV
from navsim.agents.drivoR_action_dit.flow import ContinuousFlowSchedule
from navsim.agents.drivoR_action_dit import model as base_module
from navsim.agents.drivoR_action_dit.loss import DrivoRActionDiTLoss
from navsim.agents.drivoR_action_dit_register.agent import DrivoRActionRegisterAgent
from navsim.agents.drivoR_action_dit_register.dit import ActionRegisterDiT
from navsim.agents.drivoR_action_dit_register.losses import matched_register_loss, assigned_register_loss
from navsim.agents.drivoR_action_dit_register.targets import FutureRegisterTargetBuilder
from navsim.agents.drivoR_action_dit_register.teacher import EmaDinoRegisters


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/dinov3'))
from dinov3.models.vision_transformer import DinoVisionTransformer


def native_model():
    model = DinoVisionTransformer(img_size=32,patch_size=16,embed_dim=16,depth=1,num_heads=2,
                                 ffn_ratio=2,n_storage_tokens=4,layerscale_init=1e-5,
                                 mask_k_bias=True,norm_layer='layernormbf16',pos_embed_rope_dtype='fp32')
    model.init_weights()
    model.requires_grad_(False)
    for block in model.blocks:
        block.attn.qkv = DinoV3LoRAQV(block.attn.qkv,2)
    return model.eval()


class TinyEncoder(nn.Module):
    def __init__(self, options, image_size, output_dim, *args):
        super().__init__()
        self.model = native_model()
        self.num_features = self.model.num_features
        self.neck = nn.Linear(self.num_features,output_dim)

    @property
    def native_vit(self):
        return self.model

    def forward(self, images):
        return self.neck(self.model.forward_features(images[:,0])['x_norm_patchtokens'])


def config(stage='joint', scorer=False, fusion='project'):
    folder = ROOT/'navsim/planning/script/config/common/agent'
    names = ('drivoR','drivoR_action_dit','drivoR_action_dit_multigt','drivoR_action_dit_dinov3_vitl',
             'drivoR_action_dit_dinov3_vitl_register')
    cfg = OmegaConf.merge(*[OmegaConf.load(folder/(name+'.yaml')).config for name in names])
    cfg.tf_d_model=8; cfg.tf_d_ffn=16; cfg.scorer_ref_num=1; cfg.proposal_num=8
    cfg.action_dit.update(dict(hidden_dim=8,ffn_dim=16,freq_dim=8,num_heads=2,attn_head_dim=4,
                              num_layers=2,pretrained_path='',num_inference_steps=2,gradient_checkpointing=False))
    cfg.future_register.stage=stage
    cfg.future_register.teacher_batch_size=8
    cfg.future_register.scorer.enabled=scorer
    cfg.future_register.scorer.fusion=fusion
    cfg.multi_gt.max_gt=2
    return cfg


def make_agent(monkeypatch,stage='joint',scorer=False,fusion='project',cfg=None):
    monkeypatch.setattr(base_module,'DinoV3ImageEncoder',TinyEncoder)
    cfg = config(stage,scorer,fusion) if cfg is None else cfg
    value = DrivoRActionRegisterAgent(cfg,lr_args=dict(name='AdamW',base_lr=.001,base_batch_size=1),
                                     loss=DrivoRActionDiTLoss())
    return value


@pytest.fixture
def sample():
    torch.manual_seed(17)
    features = dict(image=torch.randn(1,1,3,32,32),ego_status=torch.randn(1,4,11))
    targets = dict(trajectory=torch.randn(1,8,3),trajectory_multi_gt=torch.randn(1,2,8,3),
                   trajectory_multi_gt_valid=torch.ones(1,2,dtype=torch.bool),
                   future_register_images=torch.randint(0,256,(1,2,8,3,32,32),dtype=torch.uint8),
                   future_register_valid=torch.ones(1,2,8,dtype=torch.bool),
                   human_future_register_images=torch.randint(0,256,(1,8,3,32,32),dtype=torch.uint8),
                   human_future_register_valid=torch.ones(1,8,dtype=torch.bool))
    return features,targets


def oracle(targets,proposals,test=False):
    assert not test
    scores = torch.full((*proposals.shape[:2],7),.75,device=proposals.device)
    return scores[...,-1],scores[...,-1].amax(-1),scores,None,None,None


@pytest.mark.parametrize('checkpointing',[False,True])
def test_action_cannot_read_registers_and_registers_can_read_action(checkpointing):
    base = ActionDiT(hidden_dim=8,ffn_dim=16,freq_dim=8,num_heads=2,attn_head_dim=4,num_layers=2,
                     use_gradient_checkpointing=checkpointing)
    joint = ActionRegisterDiT(base,16).train()
    action = torch.randn(1,3,8,3,requires_grad=True)
    registers = torch.randn(1,3,32,16,requires_grad=True)
    time = torch.full((1,3),500.)
    scene,ego = torch.randn(1,7,8),torch.randn(1,8)
    original = base(action,time,scene,ego)
    a,r = joint(action,registers,time,scene,ego)
    changed_a,changed_r = joint(action,registers+10,time,scene,ego)
    torch.testing.assert_close(a,changed_a,rtol=0,atol=0)
    torch.testing.assert_close(a,original,atol=1e-6,rtol=1e-5)
    assert not torch.equal(r,changed_r)
    register_gradient, = torch.autograd.grad(a.square().sum(),registers,retain_graph=True)
    assert not register_gradient.any()
    action_gradient, = torch.autograd.grad(r.square().sum(),action)
    assert action_gradient.abs().sum()>0


def test_ema_tracks_merged_effective_qv_weights_not_the_product_of_ema_factors():
    student=native_model()
    teacher=EmaDinoRegisters(student,decay=.5,chunk_size=2)
    adapter=student.blocks[0].attn.qkv
    old=teacher.encoder.blocks[0].attn.qkv.weight.detach().clone()
    with torch.no_grad():
        adapter.q_a.weight.add_(.2); adapter.q_b.weight.fill_(.3)
        adapter.v_b.weight.fill_(.4)
    teacher.update(student)
    new=teacher.encoder.blocks[0].attn.qkv.weight
    torch.testing.assert_close(new[:16],old[:16]+.5*(adapter.q_b.weight@adapter.q_a.weight))
    torch.testing.assert_close(new[16:32],old[16:32],rtol=0,atol=0)
    torch.testing.assert_close(new[32:],old[32:]+.5*(adapter.v_b.weight@adapter.v_a.weight))
    assert teacher.updates==1 and not any(p.requires_grad for p in teacher.parameters())
    teacher.train(True)
    assert not teacher.encoder.training


def test_teacher_selects_four_native_registers_and_ignores_invalid_frames():
    student=native_model()
    teacher=EmaDinoRegisters(student,chunk_size=2)
    pixels=torch.randint(0,256,(1,2,8,3,32,32),dtype=torch.uint8)
    mask=torch.zeros(1,2,8,dtype=torch.bool); mask[0,1,3]=True
    output=teacher(pixels,mask)
    image=(pixels[0,1,3:4].float()/255.-teacher.pixel_mean)/teacher.pixel_std
    expected=teacher.encoder.forward_features(image)['x_storage_tokens']
    assert output.shape==(1,2,8,4,16) and not output.requires_grad
    torch.testing.assert_close(output[0,1,3],expected[0])
    assert not output[~mask].any()


@pytest.mark.parametrize('loss_type',['velocity','x_mse'])
def test_future_loss_uses_the_same_trajectory_winner_and_valid_time_gt_means(loss_type):
    pred=torch.zeros(1,3,8,4,2,requires_grad=True)
    target=torch.ones(1,2,8,4,2); target[:,1]*=3
    valid=torch.ones(1,2,8,dtype=torch.bool); valid[:,1,2:]=False
    target[:,1,2:]=float('nan')
    matched=torch.tensor([[1,1]])
    time=torch.full((1,3),.001)
    schedule=ContinuousFlowSchedule()
    loss=matched_register_loss(pred,target,valid,matched,time,schedule,loss_type)
    scale=schedule.training_weight(time)[0,0]/.1**2 if loss_type=='velocity' else 1
    torch.testing.assert_close(loss,torch.as_tensor(5*scale,dtype=torch.float32))
    loss.backward()
    assert not pred.grad[:,0].any() and not pred.grad[:,2].any()
    assert torch.isfinite(pred.grad).all()


def test_stage1_equal_gt_average_does_not_bias_gt_with_more_noise_samples():
    pred=torch.zeros(1,5,8,4,2)
    target=torch.ones(1,2,8,4,2); target[:,1]*=3
    assignment=torch.tensor([[0,1,0,1,0]])
    value=assigned_register_loss(pred,target,torch.ones(1,2,8,dtype=torch.bool),assignment,
                                torch.full((1,5),500.),ContinuousFlowSchedule(),'x_mse')
    assert value.item()==5.


@pytest.mark.parametrize('scorer,fusion',[(False,'flatten'),(True,'flatten'),(True,'project')])
def test_joint_forward_once_scoring_detaches_both_predictions(monkeypatch,sample,scorer,fusion):
    features,targets=sample
    agent=make_agent(monkeypatch,scorer=scorer,fusion=fusion)
    model=agent._drivor_model
    calls=[]
    model.scorer_attention.eval()  # Test the gradient path without stochastic-depth drops.
    model.action_dit.register_forward_pre_hook(lambda module,args:calls.append(1))
    agent.compute_score=oracle
    prediction=agent.forward_train(features,targets)
    result=agent.compute_loss(features,targets,prediction)
    assert calls==[1]
    assert prediction['predicted_registers'].shape==(1,8,8,4,16)
    result['final_score_loss'].backward()
    assert all(p.grad is None for p in model.action_dit.parameters())
    assert all(p.grad is None for p in model.ema_teacher.parameters())
    assert model.image_backbone.neck.weight.grad.abs().sum()>0
    assert model.scorer.pred_score['comfort'][0].weight.grad.abs().sum()>0
    if scorer and fusion=='flatten':
        assert model.pos_embed[0].in_features==24+32*16
    elif not scorer:
        assert model.pos_embed[0].in_features==24


def test_future_loss_trains_shared_dit_and_vision_but_never_scorer_or_teacher(monkeypatch,sample):
    features,targets=sample
    agent=make_agent(monkeypatch,scorer=True)
    model=agent._drivor_model
    result=agent.forward_train(features,targets)
    result['_register_loss'].backward()
    assert model.action_dit.register_head.weight.grad.abs().sum()>0
    assert model.action_dit.action_encoder.weight.grad.abs().sum()>0
    assert model.action_dit.head.weight.grad is None
    assert model.image_backbone.neck.weight.grad.abs().sum()>0
    assert model.image_backbone.model.blocks[0].attn.qkv.q_b.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.scorer.parameters())
    assert all(p.grad is None for p in model.ema_teacher.parameters())


def test_stage1_clean_gt_actions_no_action_head_no_scorer_no_oracle(monkeypatch,sample):
    features,targets=sample
    agent=make_agent(monkeypatch,stage='future_only',scorer=True)
    model=agent._drivor_model
    def forbidden(*args,**kwargs):
        raise AssertionError('Stage1 must not run scoring, oracle or action prediction head')
    model.scorer.forward=forbidden; model.action_dit.head.forward=forbidden; agent.compute_score=forbidden
    output=agent.forward_train(features,targets)
    loss=agent.compute_loss(features,targets,output)['loss']
    loss.backward()
    assert model.action_dit.register_head.weight.grad.abs().sum()>0
    assert not any(p.requires_grad for p in model.scorer.parameters())
    gt=targets['trajectory_multi_gt']
    torch.testing.assert_close(output['proposals'][:,::2],gt[:,0:1].expand(-1,4,-1,-1),atol=1e-5,rtol=1e-5)
    assert all(getattr(callback,'monitor',None)!='val/score_epoch' for callback in agent.get_training_callbacks())


def test_ema_changes_only_after_optimizer_step_and_teacher_is_excluded(monkeypatch,sample):
    features,targets=sample
    agent=make_agent(monkeypatch,stage='future_only',scorer=True)
    optimizer=agent.get_optimizers()[0]
    model=agent._drivor_model
    parameters=[p for group in optimizer.param_groups for p in group['params']]
    assert len(parameters)==len({id(p) for p in parameters})
    assert {id(p) for p in parameters}=={id(p) for p in model.parameters() if p.requires_grad}
    for _ in range(2):
        output=agent.forward_train(features,targets)
        agent.compute_loss(features,targets,output)['loss'].backward()
    assert model.ema_teacher.updates==0
    optimizer.step()
    assert model.ema_teacher.updates==1


def test_stage1_validation_skips_navsim_and_stage2_loads_model_and_ema(monkeypatch,tmp_path,sample):
    features,targets=sample
    stage1=make_agent(monkeypatch,stage='future_only',scorer=True)
    stage1._drivor_model.ema_teacher.updates.fill_(7)
    lightning=stage1.lightning_module_class(stage1)
    lightning.log=lambda *args,**kwargs:None
    stage1.compute_score=lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError('No Stage1 oracle'))
    with torch.no_grad():
        assert torch.isfinite(lightning.validation_step((features,targets),0))
    path=tmp_path/'stage1.ckpt'
    torch.save({'state_dict':{'agent.'+key:value for key,value in stage1.state_dict().items()}},path)
    cfg=config(scorer=True); cfg.future_register.init_checkpoint=str(path)
    stage2=make_agent(monkeypatch,cfg=cfg)
    for key,value in stage1.state_dict().items():
        torch.testing.assert_close(value,stage2.state_dict()[key],rtol=0,atol=0)
    assert stage2._drivor_model.ema_teacher.updates==7
    assert stage2._drivor_model.action_dit.head.weight.requires_grad
    assert stage2._drivor_model.scorer.pred_score['comfort'][0].weight.requires_grad


def test_no_future_frames_is_masked_without_breaking_stage1_graph(monkeypatch,sample):
    features,targets=sample
    targets['future_register_valid'].zero_()
    agent=make_agent(monkeypatch,stage='future_only',scorer=True)
    prediction=agent.forward_train(features,targets)
    loss=agent.compute_loss(features,targets,prediction)['loss']
    assert loss.item()==0
    loss.backward()
    assert all(p.grad is not None for p in agent._drivor_model.parameters() if p.requires_grad)


def test_joint_euler_uses_actual_sigma_without_training_floor(monkeypatch,sample):
    agent=make_agent(monkeypatch)
    model=agent._drivor_model
    clean_a=torch.randn(1,8,8,3); clean_r=torch.randn(1,8,32,16)
    calls=[]
    class Perfect(nn.Module):
        def forward(self,a,r,t,s,e):
            calls.append(t[0,0].item())
            return clean_a,clean_r
    model.action_dit=Perfect(); model.num_inference_steps=60
    a,r=model.sample_joint(torch.randn(1,4,8),torch.randn(1,8))
    torch.testing.assert_close(a,clean_a,rtol=0,atol=0)
    torch.testing.assert_close(r,clean_r,rtol=0,atol=0)
    assert len(calls)==60 and calls[-1]<100


def test_future_frame_reader_preserves_noncontiguous_gt_slots(tmp_path):
    token='testscene'
    gt_root=tmp_path/'gt'; folder=gt_root/token; folder.mkdir(parents=True)
    trajectories=np.zeros((4,8,3),np.float32)
    for i in range(4): trajectories[i]=i
    np.save(folder/'gt_trajectories.npy',trajectories)
    np.save(folder/'gt_valid_mask.npy',np.ones(4,bool))
    (folder/'trajectory.json').write_text(json.dumps(dict(gt_padded=[True,False,True,False],
        target_times=(np.arange(1,9)*.5).tolist(),gt_labels=['a','b','c','d'])))
    frames=tmp_path/'frames'/token[:2]; frames.mkdir(parents=True)
    meta=dict(size_wh=[8,8],times=(np.arange(1,9)*.5).tolist(),slots=[])
    with zipfile.ZipFile(frames/(token+'.zip'),'w') as archive:
        for slot in (3,1):
            names=[]
            for time in range(8):
                name=f'{slot}/{time}.png'; names.append(name)
                buffer=BytesIO(); Image.fromarray(np.full((8,8,3),slot*40+time,np.uint8)).save(buffer,format='PNG')
                archive.writestr(name,buffer.getvalue())
            meta['slots'].append(dict(slot=slot,label='bd'[slot//2],trajectory=trajectories[slot].tolist(),
                                      frame_indices=list(range(5,41,5)),images=names))
        archive.writestr('metadata.json',json.dumps(meta))
    reader=FutureRegisterTargetBuilder(OmegaConf.create(dict(root=str(gt_root),max_gt=4,missing_policy='original')),
        OmegaConf.create(dict(frames_root=str(tmp_path/'frames'),image_size=[8,8])))
    result=reader.load(token,np.zeros((8,3)))
    assert result['future_register_valid'].sum()==16
    assert result['future_register_images'].dtype==torch.uint8
    assert result['future_register_images'][0,0,0,0,0]==40
    assert result['future_register_images'][1,7,0,0,0]==127
    assert not result['future_register_valid'][2:].any()


def test_both_noise_anchors_use_original_human_not_the_counterfactual_gt(monkeypatch,sample):
    features,targets=sample
    agent=make_agent(monkeypatch)
    model=agent._drivor_model
    def teacher(pixels,valid):
        values=pixels.float().mean((-1,-2,-3))
        return values[...,None,None].expand(*values.shape,4,16).masked_fill(~valid[...,None,None],0.)
    model.ema_teacher.forward=teacher
    captured=[]
    hook=model.action_dit.register_forward_pre_hook(
        lambda module,args:captured.append(tuple(value.detach().clone() for value in args[:3])))
    torch.manual_seed(43)
    model.forward_train(features,targets)
    changed=dict(targets,trajectory_multi_gt=targets['trajectory_multi_gt']+100,
                 future_register_images=torch.zeros_like(targets['future_register_images']))
    torch.manual_seed(43)
    model.forward_train(features,changed)
    for first,second in zip(captured[0],captured[1]):
        torch.testing.assert_close(first,second,rtol=0,atol=0)
    changed=dict(targets,human_future_register_images=torch.full_like(targets['human_future_register_images'],255))
    torch.manual_seed(43)
    model.forward_train(features,changed)
    torch.testing.assert_close(captured[0][0],captured[2][0],rtol=0,atol=0)
    assert not torch.equal(captured[0][1],captured[2][1])
    assert not torch.equal(captured[0][1][:,0],captured[0][1][:,1])
    hook.remove()


def test_original_human_frames_are_loaded_at_plus_half_to_four_seconds(monkeypatch,tmp_path):
    from navsim.agents.drivoR_action_dit_register import targets as target_module
    gt=tmp_path/'gt'; gt.mkdir()
    frames=tmp_path/'frames'; frames.mkdir()
    reader=FutureRegisterTargetBuilder(OmegaConf.create(dict(root=str(gt),max_gt=4)),
        OmegaConf.create(dict(frames_root=str(frames),image_size=[8,8],sensor_blobs_path=str(tmp_path/'sensors'))))
    monkeypatch.setattr(reader,'load',lambda token,fallback:{})
    observed=[]
    def load_image(root,path):
        observed.append(path)
        return np.full((8,8,3),int(path),np.uint8)
    monkeypatch.setattr(target_module,'load_camera_image',load_image)
    scene=SimpleNamespace(scene_metadata=SimpleNamespace(initial_token='test',num_history_frames=4),
        frames=[SimpleNamespace(cameras=SimpleNamespace(cam_f0=SimpleNamespace(image=None,image_path=str(i)))) for i in range(14)],
        get_future_trajectory=lambda num_trajectory_frames:SimpleNamespace(poses=np.zeros((8,3))))
    result=reader.compute_targets(scene)
    assert observed==[str(i) for i in range(4,12)]
    assert result['human_future_register_images'][:,0,0,0].tolist()==list(range(4,12))
    assert result['human_future_register_valid'].all()


def test_unrequested_camera_keeps_path_metadata_without_loading_pixels(monkeypatch):
    from navsim.common import dataclasses as data
    names=('CAM_F0','CAM_L0','CAM_L1','CAM_L2','CAM_R0','CAM_R1','CAM_R2','CAM_B0')
    cameras={name:{'data_path':name+'.jpg'} for name in names}
    monkeypatch.setattr(data,'load_camera_image',lambda *args: (_ for _ in ()).throw(AssertionError('No input image requested')))
    result=data.Cameras.from_camera_dict(Path('.'),cameras,[])
    assert result.cam_f0.image is None and result.cam_f0.image_path=='CAM_F0.jpg'
