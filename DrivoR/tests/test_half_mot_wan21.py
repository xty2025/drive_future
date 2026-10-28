"""A28 MoT/Wan2.1 variants: native loaders, directed gradients and full routing."""
import ast
import json
import logging
from pathlib import Path

from einops import rearrange
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
from safetensors.torch import save_file
import torch
from torch import nn
from torch.nn import functional as F

from navsim.agents.drivoR_action_dit.action_dit import ActionDiT
from navsim.agents.drivoR_action_dit_register.mot import ActionRegisterMoT
from test_diverse8_register import sample
from test_half_a28_ablations import (
    ROOT, BASE, tiny_agent, test_launchers_keep_half_split_output_root_and_dlc_topology as check_train,
    test_eval_launchers_restore_matching_architecture_and_force_fp32 as check_eval,
)

VARIANTS = ('a28_mot', 'a28_wan21_1p3b')


def small_mot(checkpointing=False):
    action = ActionDiT(hidden_dim=8, ffn_dim=32, freq_dim=8, num_heads=2, attn_head_dim=4,
                        num_layers=2, use_gradient_checkpointing=checkpointing)
    video = OmegaConf.create(dict(hidden_dim=16, ffn_dim=32, freq_dim=8, num_heads=2, attn_head_dim=4,
        num_layers=2, eps=1e-6, gradient_checkpointing=checkpointing, pretrained_path=None))
    return ActionRegisterMoT(action, video, register_dim=16, context_dim=8)


@pytest.mark.parametrize('variant', VARIANTS)
def test_presets_change_only_a28_generator_and_keep_half_scope(variant):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT/'navsim/planning/script/config/training')):
        overrides=['train_test_split/scene_filter=all_scenes','train_test_split_synthetic/scene_filter=all_scenes']
        cfg=compose(config_name='half_'+variant,overrides=overrides)
        base=compose(config_name='half_a28',overrides=overrides)
    a,b=cfg.agent.config,base.agent.config
    raw_a,raw_b=OmegaConf.to_container(a,resolve=False),OmegaConf.to_container(b,resolve=False)
    for key in raw_b:
        if key not in ('action_dit','future_register'):
            assert raw_a[key] == raw_b[key]
    for key,value in raw_b['future_register'].items():
        assert raw_a['future_register'][key] == value
    assert cfg.agent.scheduler_args.dataset_size==42555
    assert cfg.runtime.scheduler_dataset_size_from_data and cfg.runtime.get('training_data_scope')!='trainval'
    assert cfg.trainer.params.strategy=='deepspeed_stage_2' and cfg.trainer.params.precision=='32-true'
    assert cfg.agent.config.proposal_num==64
    if variant=='a28_mot':
        assert a.action_dit.hidden_dim==256 and a.action_dit.pretrained_path==b.action_dit.pretrained_path
        assert a.future_register.mot.video_dit.hidden_dim==3072
    else:
        assert a.action_dit.hidden_dim==1536 and a.action_dit.ffn_dim==8960 and a.action_dit.num_heads==12
        assert a.action_dit.pretrained_format=='wan21_native'


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('mode', ('training','train_eval'))
def test_new_training_routes(tmp_path,variant,mode):
    check_train(tmp_path,variant,mode)


@pytest.mark.parametrize('variant', VARIANTS)
def test_new_eval_routes(tmp_path,variant):
    check_eval(tmp_path,variant)


@pytest.mark.parametrize('checkpointing',(False,True))
def test_mot_register_reads_action_but_action_cannot_read_register(checkpointing):
    torch.manual_seed(5)
    mot=small_mot(checkpointing).train()
    action=torch.randn(2,3,8,3,requires_grad=True)
    registers=torch.randn(2,3,32,16,requires_grad=True)
    scene=torch.randn(2,7,8,requires_grad=True);ego=torch.randn(2,8)
    times=torch.full((2,3),500.)
    a,r=mot(action,registers,times,scene,ego)
    changed_a,_=mot(action,registers+5,times,scene,ego)
    torch.testing.assert_close(a,changed_a,rtol=0,atol=0)
    r_to_a,=torch.autograd.grad(a.square().sum(),registers,retain_graph=True)
    assert not r_to_a.any()
    a_to_r,scene_to_r=torch.autograd.grad(r.square().sum(),(action,scene),retain_graph=True)
    assert a_to_r.abs().sum()>0 and scene_to_r.abs().sum()>0
    video_params=list(mot.mixtures['video'].parameters())
    grads=torch.autograd.grad(a.square().sum(),video_params,allow_unused=True,retain_graph=True)
    assert all(g is None or not g.any() for g in grads)
    assert mot.attention_mask[:32].all() and not mot.attention_mask[32:,:32].any()
    assert mot.attention_mask[32:,32:].all()
    # Same candidate in a different scene/proposal must also remain independent.
    modified=action.detach().clone();modified[0,0]+=5
    _,other_r=mot(modified,registers,times,scene,ego)
    torch.testing.assert_close(r[0,1:],other_r[0,1:],rtol=0,atol=0)
    torch.testing.assert_close(r[1],other_r[1],rtol=0,atol=0)
    assert not torch.equal(r[0,0],other_r[0,0])


def test_mot_matches_actual_fastwam_per_expert_attention_and_ffn():
    """Execute the reference's real MoT/DiT methods without importing its pipeline."""
    reference=Path('/cpfs29jb/data/zhaozc/project/FastWam_re/src/fastwam/models/wan22')
    namespace=dict(torch=torch,nn=nn,F=F,rearrange=rearrange,logger=logging.getLogger('mot_reference'))
    wanted={'flash_attention','modulate','precompute_freqs_cis','rope_apply','RMSNorm',
            'SelfAttention','CrossAttention','GateModule','DiTBlock','MoT'}
    for file in ('wan_video_dit.py','mot.py'):
        nodes=[n for n in ast.parse((reference/file).read_text()).body
               if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in wanted]
        nodes.insert(0,ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0))
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(reference/file),'exec'),namespace)
    mot=small_mot().eval()
    experts={}
    for name,source in mot.mixtures.items():
        expert=nn.Module();expert.num_heads=2;expert.attn_head_dim=4;expert.use_gradient_checkpointing=False
        expert.blocks=nn.ModuleList([namespace['DiTBlock'](source.hidden_dim,4,2,32) for _ in source.blocks])
        expert.blocks.load_state_dict(source.blocks.state_dict(),strict=True)
        experts[name]=expert
    reference_mot=namespace['MoT'](experts,mot_checkpoint_mixed_attn=False).eval()
    action=torch.randn(1,3,8,3);registers=torch.randn(1,3,32,16)
    scene=torch.randn(1,7,8);ego=torch.randn(1,8);times=torch.full((1,3),500.)
    actual_a,actual_r=mot(action,registers,times,scene,ego)
    video,act=mot.mixtures['video'],mot.mixtures['action']
    pos=(mot.future_time_embedding[:,None]+mot.register_slot_embedding[None]).reshape(1,1,32,16)
    inputs={'video':(video.action_encoder(registers)+pos).flatten(0,1),'action':act.action_encoder(action).flatten(0,1)}
    mods={name:mot.modulation(expert,times,ego).flatten(0,1) for name,expert in mot.mixtures.items()}
    contexts={name:dict(context=expert.scene_projection(scene).repeat_interleave(3,dim=0)) for name,expert in mot.mixtures.items()}
    freqs=namespace['precompute_freqs_cis'](4,end=40)[:,None]
    result=reference_mot(inputs,mot.attention_mask,{'video':freqs[8:40],'action':freqs[:8]},contexts,mods)
    expected_a=act.head(result['action']).reshape_as(actual_a)
    expected_r=video.head(result['video']).reshape_as(actual_r)
    torch.testing.assert_close(actual_a,expected_a,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(actual_r,expected_r,rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize('variant',VARIANTS)
def test_new_agents_single_forward_gradient_detach_and_strict_reload(monkeypatch,sample,tmp_path,variant):
    agent=tiny_agent(monkeypatch,variant,scorer_heads=1)
    model=agent._drivor_model;features,targets=sample
    calls=[];model.action_dit.register_forward_hook(lambda *args:calls.append(1))
    pred=agent.forward_train(features,targets)
    assert calls==[1] and pred['predicted_registers'].shape==(1,4,8,4,16)
    losses=agent.compute_loss(features,targets,pred)
    losses['final_score_loss'].backward()
    assert all(p.grad is None for p in model.action_dit.parameters())
    model.zero_grad(set_to_none=True)
    losses=agent.compute_loss(features,targets,agent.forward_train(features,targets))
    losses['future_register_loss'].backward()
    if variant=='a28_mot':
        for key in ('video','action'):
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.action_dit.mixtures[key].parameters())
    assert all(p.grad is None for p in model.ema_teacher.parameters())
    configured=agent.get_optimizers();params=[p for group in configured[0].param_groups for p in group['params']]
    assert len(params)==len({id(p) for p in params})
    assert {id(p) for p in params}=={id(p) for p in model.parameters() if p.requires_grad}
    file=tmp_path/'model.ckpt';torch.save(dict(state_dict={'agent.'+k:v for k,v in agent.state_dict().items()}),file)
    restored=tiny_agent(monkeypatch,variant,str(file),scorer_heads=1);restored.initialize();restored.eval()
    monkeypatch.setattr(restored._drivor_model.ema_teacher,'forward',lambda *args:(_ for _ in ()).throw(AssertionError('No future GT at inference')))
    with torch.no_grad():
        torch.manual_seed(3);expected=agent(features)
        torch.manual_seed(3);actual=restored(features)
    torch.testing.assert_close(actual['proposals'],expected['proposals'])
    torch.testing.assert_close(actual['pdm_score'],expected['pdm_score'])


def test_single_safetensors_wan21_checkpoint_loads_native_backbone(tmp_path):
    original=ActionDiT(hidden_dim=16,ffn_dim=32,freq_dim=8,num_heads=2,attn_head_dim=4,num_layers=2)
    state={k:v for k,v in original.state_dict().items() if k.startswith(('blocks.','time_embedding.','time_projection.'))}
    state['head.head.weight']=torch.randn(64,16)
    save_file(state,tmp_path/'diffusion_pytorch_model.safetensors')
    (tmp_path/'config.json').write_text(json.dumps(dict(dim=16,ffn_dim=32,freq_dim=8,num_heads=2,num_layers=2)))
    loaded=ActionDiT(hidden_dim=16,ffn_dim=32,freq_dim=8,num_heads=2,attn_head_dim=4,num_layers=2,context_dim=8)
    report=loaded.load_wan_pretrained(tmp_path,checkpoint_format='wan21_native')
    assert report['format']=='wan21_native' and report['omitted_source_tensors']==['head.head.weight']
    for key in loaded.state_dict():
        if key in state:torch.testing.assert_close(loaded.state_dict()[key],state[key],rtol=0,atol=0)
