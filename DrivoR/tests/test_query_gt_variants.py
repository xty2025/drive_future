"""Fair shared initialization, GT supervision, optional future and EMA."""

from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

from navsim.agents.drivoR_action_dit.action_dit import ActionDiT
from navsim.agents.drivoR_query_register import gt_variants as module
from navsim.agents.drivoR_query_register.losses import DrivoRQueryRegisterLoss
from test_action_dit_register import TinyEncoder, sample, oracle


ROOT=Path(__file__).resolve().parents[1]
VARIANTS=('single_gt_long','single_gt_long_register','human_diverse8_l1','single_gt')


def config(variant):
    folder=ROOT/'navsim/planning/script/config/common/agent'
    names=['drivoR','drivoR_query_register_dinov3_vitl','drivoR_query_dinov3_vitl']
    if variant in ('single_gt_long_register','single_gt'):
        names.append('drivoR_query_dinov3_vitl_single_gt_long')
    names.append('drivoR_query_dinov3_vitl_'+variant)
    cfg=OmegaConf.merge(*[OmegaConf.load(folder/(name+'.yaml')).config for name in names])
    cfg.tf_d_model=8;cfg.tf_d_ffn=16;cfg.ref_num=2;cfg.scorer_ref_num=1
    cfg.refiner_num_heads=2;cfg.proposal_num=4
    cfg.query_register.gradient_checkpointing=False
    cfg.future_register.sensor_blobs_path='/unused/sensors'
    cfg.future_register.teacher_batch_size=4
    return cfg


def agent(monkeypatch,variant):
    monkeypatch.setattr(module,'DinoV3ImageEncoder',TinyEncoder)
    monkeypatch.setattr(ActionDiT,'__init__',lambda *a,**k: (_ for _ in ()).throw(AssertionError('No Action-DiT')))
    value=module.DrivoRQueryGTAgent(config(variant),
        lr_args=dict(name='AdamW',base_lr=.001,base_batch_size=1),loss=DrivoRQueryRegisterLoss())
    value.compute_score=oracle
    return value


def targets_for(sample):
    features,source=sample
    targets={key:value for key,value in source.items() if not key.startswith(('trajectory_multi_gt','future_register_'))}
    targets['trajectory_long']=targets['trajectory']+2
    targets.update(pseudo_experts=torch.randn(1,8,8,3,requires_grad=True),
        pseudo_expert_mask=torch.tensor([[True,True]+[False]*6]),
        diverse_gt_missing=torch.tensor([False]))
    return features,targets


def test_shared_parameters_have_identical_initialization_and_action_path(monkeypatch,sample):
    models=[]
    for variant in VARIANTS:
        torch.manual_seed(51)
        models.append(agent(monkeypatch,variant)._drivor_model.eval())
    baseline,with_future,diverse,human_only=models
    for key,tensor in baseline.state_dict().items():
        assert torch.equal(tensor,with_future.state_dict()[key]),key
        assert torch.equal(tensor,diverse.state_dict()[key]),key
        assert torch.equal(tensor,human_only.state_dict()[key]),key
    assert baseline.query_decoder.attention_mask.shape==(4,4)
    assert with_future.query_decoder.attention_mask.shape==(132,132)
    assert not hasattr(baseline,'future_queries') and not hasattr(diverse,'ema_teacher')
    with torch.no_grad():
        original=baseline(sample[0])
        with_future.future_queries.add_(10)
        extended=with_future(sample[0])
        extra=diverse(sample[0])
        human=human_only(sample[0])
    for key in ('proposals','pdm_score'):
        torch.testing.assert_close(original[key],extended[key],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(original[key],extra[key],atol=0,rtol=0)
        torch.testing.assert_close(original[key],human[key],atol=0,rtol=0)
    assert 'predicted_registers' not in original and 'predicted_registers' not in extra


@pytest.mark.parametrize('variant',['single_gt','single_gt_long','human_diverse8_l1'])
def test_no_register_models_construct_no_teacher_and_load_no_future_rgb(monkeypatch,sample,tmp_path,variant):
    def forbidden(*args,**kwargs):
        raise AssertionError('No future branch or target builder should be constructed')
    monkeypatch.setattr(module,'EmaDinoRegisters',forbidden)
    monkeypatch.setattr(module,'HumanFutureRegisterTargetBuilder',forbidden)
    value=agent(monkeypatch,variant)
    if variant=='human_diverse8_l1':
        value._config.pseudo_expert.path=str(tmp_path)
    assert len(value.get_target_builders())==(2 if variant=='human_diverse8_l1' else 1)
    features,targets=targets_for(sample)
    targets={key:item for key,item in targets.items() if not key.startswith('human_future_')}
    output=value.forward_train(features,targets)
    loss=value.compute_loss(features,targets,output)
    assert output['_register_loss']==0 and 'predicted_registers' not in output
    assert torch.isfinite(loss['loss'])
    prefixes=('future_queries','future_time_embedding','register_slot_embedding','register_head','ema_teacher')
    assert not any(key.startswith(prefixes) for key in value._drivor_model.state_dict())


def test_single_human_only_supervises_its_winner_without_long_gt(monkeypatch):
    model=agent(monkeypatch,'single_gt')._drivor_model
    poses=torch.zeros(1,4,8,3)
    poses[:,1]=10;poses[:,2]=20;poses[:,3]=30
    poses.requires_grad_()
    human=torch.full((1,8,3),.2)
    output=dict(proposals=poses)
    model._add_supervision(output,dict(trajectory=human))
    torch.testing.assert_close(output['_trajectory_loss'],torch.tensor(.6))
    assert output['_human_matched_indices'].tolist()==[[0]]
    assert output['_register_loss']==0
    assert '_long_matched_indices' not in output
    assert set(output['_query_diagnostics'])=={'human_trajectory_loss'}
    # A stale target cache must not silently re-enable long/auxiliary GT loss.
    extras=dict(proposals=poses)
    model._add_supervision(extras,dict(trajectory=human,
        trajectory_long=torch.full_like(human,10.3),
        pseudo_experts=torch.full((1,8,8,3),20.3),
        pseudo_expert_mask=torch.ones(1,8,dtype=torch.bool)))
    torch.testing.assert_close(extras['_trajectory_loss'],output['_trajectory_loss'])
    output['_trajectory_loss'].backward()
    assert (poses.grad.abs().sum((-2,-1))>0).tolist()==[[True,False,False,False]]


def test_long_and_human_share_proposals_but_only_human_winner_gets_future_loss(monkeypatch):
    value=agent(monkeypatch,'single_gt_long_register')
    model=value._drivor_model
    poses=torch.zeros(1,4,8,3)
    poses[:,1]=10;poses[:,2]=20;poses[:,3]=30
    poses.requires_grad_()
    registers=torch.zeros(1,4,8,4,16)
    registers[:,1]=100
    registers.requires_grad_()
    targets=dict(trajectory=torch.full((1,8,3),.2),trajectory_long=torch.full((1,8,3),10.3),
        human_future_register_images=torch.zeros(1,8,3,32,32,dtype=torch.uint8),
        human_future_register_valid=torch.tensor([[True]*6+[False]*2]))
    monkeypatch.setattr(model.ema_teacher,'forward',lambda pixels,valid:torch.ones(1,1,8,4,16))
    output=dict(proposals=poses,predicted_registers=registers)
    model._add_supervision(output,targets)
    assert output['_human_matched_indices'].tolist()==[[0]]
    assert output['_long_matched_indices'].tolist()==[[1]]
    assert output['_register_matched_indices'].tolist()==[[0]]
    torch.testing.assert_close(output['_trajectory_loss'],torch.tensor(1.5))
    torch.testing.assert_close(output['_register_loss'],torch.tensor(1.))
    output['_trajectory_loss'].backward()
    assert (poses.grad.abs().sum((-2,-1))>0).tolist()==[[True,True,False,False]]
    output['_register_loss'].backward()
    assert (registers.grad.abs().sum((-3,-2,-1))>0).tolist()==[[True,False,False,False]]
    assert not registers.grad[:,:,6:].any()


def test_diverse8_keeps_half_weight_and_valid_gt_mean_without_long_gt(monkeypatch):
    value=agent(monkeypatch,'human_diverse8_l1')
    poses=torch.zeros(1,4,8,3)
    poses[:,1]=10;poses[:,2]=20;poses[:,3]=30
    poses.requires_grad_()
    experts=torch.full((1,8,8,3),float('nan'))
    experts[:,0]=1;experts[:,1]=12
    experts.requires_grad_()
    targets=dict(trajectory=torch.full((1,8,3),.2),pseudo_experts=experts,
        pseudo_expert_mask=torch.tensor([[True,True]+[False]*6]),diverse_gt_missing=torch.tensor([False]))
    output=dict(proposals=poses)
    value._drivor_model._add_supervision(output,targets)
    # Human=0.6; two auxiliary losses=3 and 6; mean=4.5, weighted by 0.5.
    torch.testing.assert_close(output['_trajectory_loss'],torch.tensor(.6+.5*4.5))
    assert '_long_matched_indices' not in output
    assert output['_register_loss']==0
    output['_trajectory_loss'].backward()
    assert experts.grad is None
    assert not poses.grad[:,2:].any()
    targets['pseudo_expert_mask'].zero_()
    output=dict(proposals=poses)
    value._drivor_model._add_supervision(output,targets)
    torch.testing.assert_close(output['_trajectory_loss'],torch.tensor(.6))


@pytest.mark.parametrize('variant',VARIANTS)
def test_one_forward_scorer_detach_optimizer_and_inference_without_targets(monkeypatch,sample,variant):
    features,targets=targets_for(sample)
    value=agent(monkeypatch,variant).eval()
    model=value._drivor_model
    calls=[];score_calls=[]
    model.query_decoder.register_forward_hook(lambda m,a,o:calls.append(1))
    model.scorer.register_forward_pre_hook(lambda m,a:score_calls.append(a[0].shape))
    output=value.forward_train(features,targets)
    losses=value.compute_loss(features,targets,output)
    assert calls==[1] and score_calls==[torch.Size((1,4,8,3))]
    torch.testing.assert_close(losses['loss'],output['_trajectory_loss']+output['_register_loss']+losses['final_score_loss'])
    losses['final_score_loss'].backward()
    assert all(p.grad is None for p in model.query_decoder.parameters())
    assert all(p.grad is None for p in model.trajectory_head.parameters())
    if model.predict_future:
        assert all(p.grad is None for p in model.register_head.parameters())
    assert model.image_backbone.neck.weight.grad.abs().sum()>0
    model.zero_grad(set_to_none=True)
    output=value.forward_train(features,targets)
    losses=value.compute_loss(features,targets,output)
    optimizer=value.get_optimizers()[0]
    params=[p for group in optimizer.param_groups for p in group['params']]
    assert len(params)==len({id(p) for p in params})
    assert {id(p) for p in params}=={id(p) for p in model.parameters() if p.requires_grad}
    losses['loss'].backward();optimizer.step()
    if model.predict_future:
        assert model.ema_teacher.updates==1
        monkeypatch.setattr(model.ema_teacher,'forward',lambda *a:(_ for _ in ()).throw(AssertionError('Inference must not run teacher')))
    value._checkpoint_path='/checkpoint.ckpt'
    assert len(value.get_target_builders())==1
    with torch.no_grad():
        torch.manual_seed(7);first=value(features)
        torch.manual_seed(9);second=value(features)
    torch.testing.assert_close(first['proposals'],second['proposals'],atol=0,rtol=0)
    assert '_trajectory_loss' not in first


@pytest.mark.parametrize('variant',VARIANTS)
def test_full_checkpoint_load_keeps_future_mode_but_needs_no_training_gt(monkeypatch,sample,tmp_path,variant):
    value=agent(monkeypatch,variant).eval()
    checkpoint=tmp_path/'model.ckpt'
    torch.save({'state_dict':{'agent.'+key:tensor for key,tensor in value.state_dict().items()}},checkpoint)
    cfg=config(variant)
    cfg.long_trajectory_additional_poses=0  # Evaluation launcher uses this override.
    if 'pseudo_expert' in cfg:
        cfg.pseudo_expert.path='/absent/training/gt'
    restored=module.DrivoRQueryGTAgent(cfg,lr_args={},checkpoint_path=str(checkpoint),loss=DrivoRQueryRegisterLoss())
    restored.initialize();restored.eval()
    assert len(restored.get_target_builders())==1
    if restored._drivor_model.predict_future:
        monkeypatch.setattr(restored._drivor_model.ema_teacher,'forward',lambda *a:(_ for _ in ()).throw(AssertionError('No inference teacher')))
    with torch.no_grad():
        first=value(sample[0]);second=restored(sample[0])
    for key in ('proposals','trajectory','pdm_score'):
        torch.testing.assert_close(first[key],second[key],atol=0,rtol=0)
