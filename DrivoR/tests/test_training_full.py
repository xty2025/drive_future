"""Full trainval scope, maintained runtime, scheduler length and launch routing."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from navsim.planning.script import run_training as training
from navsim.planning.script import run_training_full as full


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    'T07': ('query_human_diverse8_l1', 'drivoR_query_dinov3_vitl_human_diverse8_l1'),
    'T10': ('query_single_gt', 'drivoR_query_dinov3_vitl_single_gt'),
    'T05': ('query_single_gt_long', 'drivoR_query_dinov3_vitl_single_gt_long'),
    'A23': ('action_dit_dinov3_vitl_human_xl1_diverse8_l1', 'drivoR_action_dit_dinov3_vitl_human_xl1_diverse8_l1'),
    'A21': ('action_dit_dinov3_vitl_human_diverse8_l1', 'drivoR_action_dit_dinov3_vitl_human_diverse8_l1'),
    'A27': ('action_dit_dinov3_vitl_human_xl1_diverse8_l1_register', 'drivoR_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register'),
    'A28': ('action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_scorer', 'drivoR_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_scorer'),
}


def small_config(tmp_path):
    return OmegaConf.create(dict(train_logs=['train'], val_logs=['val'],
        output_dir=str(tmp_path), navsim_log_path='unused', sensor_blobs_path='unused',
        cache_path=None, force_cache_computation=False,
        train_test_split=dict(scene_filter=dict(log_names=['train','val','excluded'])),
        runtime={}, agent=dict(scheduler_args=dict(dataset_size=85000))))


@pytest.mark.parametrize('filtered',[True,False])
def test_trainval_union_keeps_validation_subset_and_train_only_unchanged(monkeypatch,tmp_path,filtered):
    monkeypatch.setattr(training,'instantiate',lambda spec:SimpleNamespace(log_names=None if spec.log_names is None else list(spec.log_names)))
    monkeypatch.setattr(training,'SceneLoader',lambda **kw:SimpleNamespace(logs=list(kw['scene_filter'].log_names)))
    monkeypatch.setattr(training,'Dataset',lambda **kw:kw['scene_loader'])
    agent=SimpleNamespace(get_sensor_config=lambda:None,get_feature_builders=lambda:[],get_target_builders=lambda:[])
    cfg=small_config(tmp_path)
    if not filtered:cfg.train_test_split.scene_filter.log_names=None
    ordinary,_=training.build_datasets(cfg,agent)
    assert ordinary.logs==['train']
    combined,validation=full.build_datasets(cfg,agent)
    assert combined.logs==['train','val'] and validation.logs==['val']
    assert cfg.runtime.validation_is_training_subset


def test_cache_only_config_also_gets_union_and_scoped_resume(monkeypatch,tmp_path):
    cfg=small_config(tmp_path)
    cfg.train_logs=['train','shared'];cfg.val_logs=['shared','val']
    cfg.use_cache_without_dataset=True;cfg.auto_resume=False;cfg.train_ckpt_path=None
    captured=[]
    monkeypatch.setattr(training.main,'__wrapped__',lambda effective:captured.append(effective))
    monkeypatch.setenv('RANK','0')
    full.run(cfg)
    assert len(captured)==1 and list(cfg.train_logs)==['train','shared','val']
    assert list(cfg.val_logs)==['shared','val'] and cfg.use_cache_without_dataset
    assert cfg.auto_resume is False and cfg.train_ckpt_path is None
    audit=json.loads((tmp_path/'trainval_full_scope.json').read_text())
    assert audit['combined_train_log_count']==3
    assert audit['original_train_log_count']==audit['validation_log_count']==2


def test_actual_dataset_length_updates_only_opted_in_schedule(tmp_path):
    cfg=small_config(tmp_path)
    agent=SimpleNamespace(scheduler_args=OmegaConf.create(dict(dataset_size=85000)))
    training.sync_scheduler_dataset_size(cfg,agent,range(7))
    assert agent.scheduler_args.dataset_size==85000
    cfg.runtime.scheduler_dataset_size_from_data=True
    training.sync_scheduler_dataset_size(cfg,agent,range(7))
    assert agent.scheduler_args.dataset_size==cfg.agent.scheduler_args.dataset_size==7


@pytest.mark.parametrize('rid',VARIANTS)
def test_presets_change_scope_and_schedule_without_changing_model_or_loss(rid):
    _,agent=VARIANTS[rid]
    overrides=['train_test_split/scene_filter=all_scenes','train_test_split_synthetic/scene_filter=all_scenes']
    with initialize_config_dir(version_base=None,config_dir=str(ROOT/'navsim/planning/script/config/training')):
        cfg=compose(config_name='full_'+rid.lower(),overrides=overrides)
        original=compose(config_name='default_training',overrides=[*overrides,'agent='+agent])
    assert OmegaConf.to_container(cfg.agent.config)==OmegaConf.to_container(original.agent.config)
    assert OmegaConf.to_container(cfg.agent.loss)==OmegaConf.to_container(original.agent.loss)
    assert cfg.full_training.reference_experiment==rid
    assert cfg.runtime.training_data_scope=='trainval' and cfg.runtime.scheduler_dataset_size_from_data
    assert cfg.agent.scheduler_args.dataset_size==103288 and cfg.auto_resume is False
    assert cfg.trainer.params.num_nodes==cfg.trainer.params.devices==8
    assert cfg.dataloader.params.batch_size*cfg.agent.num_gpus*cfg.trainer.params.accumulate_grad_batches==64
    if rid in ('A27', 'A28'):
        assert cfg.agent.config.future_register.get('supervision_scope', 'human_and_diverse') == 'human_and_diverse'
        assert cfg.agent.config.future_register.human_loss_type == 'l1'
        assert cfg.agent.config.future_register.scorer.enabled == (rid == 'A28')


@pytest.mark.parametrize('rid',VARIANTS)
@pytest.mark.parametrize('mode',['training','train_eval'])
def test_full_shells_select_entry_preset_and_keep_dlc_topology(tmp_path,rid,mode):
    stem,agent=VARIANTS[rid]
    fake=tmp_path/'python'
    fake.write_text('#!/usr/bin/python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n');fake.chmod(0o755)
    env={k:v for k,v in os.environ.items() if not k.startswith(('DRIVOR_','TORCHELASTIC_'))
         and k not in ('LOCAL_RANK','NODE_RANK','NNODES','NPROC_PER_NODE','SLURM_NODEID')}
    env.update(DRIVOR_PYTHON=str(fake),WORLD_SIZE='8',RANK='3',GPU_NUM='8',
               MASTER_ADDR='dlc-master-0',MASTER_PORT='23456',PAI_JOB_ID='full-test')
    # Config-only train_eval must route to full training without starting eval.
    extra=['--cfg','job'] if mode=='train_eval' else []
    result=subprocess.run(['bash',str(ROOT/f'scripts/training/run_drivor_{stem}_full_{mode}.sh'),*extra],
        env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    args=json.loads(result.stdout)
    entry=next(a for a in args if a.endswith('/run_training_full.py'))
    assert entry==str(ROOT/'navsim/planning/script/run_training_full.py')
    assert '--config-name=full_'+rid.lower() in args
    assert [a for a in args if a.startswith('agent=')][-1]=='agent='+agent
    if mode=='training':
        assert '--nnodes=8' in args and '--nproc_per_node=8' in args and '--node_rank=3' in args
    else:assert 'torch.distributed.run' not in args
    assert 'agent.num_gpus=64' in args
    assert any(a.startswith('experiment_name=training_drivoR_') and a.endswith('_trainval_full') for a in args)


def test_ordinary_launcher_still_defaults_to_train_only(tmp_path):
    fake=tmp_path/'python';fake.write_text('#!/usr/bin/python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n');fake.chmod(0o755)
    env={k:v for k,v in os.environ.items() if not k.startswith('DRIVOR_')}
    env['DRIVOR_PYTHON']=str(fake)
    result=subprocess.run(['bash',str(ROOT/'scripts/training/run_drivor_training.sh'),'--cfg','job'],
        env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    args=json.loads(result.stdout)
    assert str(ROOT/'navsim/planning/script/run_training.py') in args
    assert not any(a.startswith('--config-name=full_') for a in args)


def test_full_entry_cpu_smoke_uses_actual_size_and_fp32_monitor(monkeypatch,tmp_path):
    from pytorch_lightning.callbacks import ModelCheckpoint
    from test_epoch_pdms_checkpoints import DrivoRTinyAgent,TinyScenes
    from navsim.agents.drivoR.drivor_agent import DrivoRAgent
    agent=DrivoRTinyAgent()
    agent.scheduler_args=OmegaConf.create(dict(dataset_size=85000))
    agent.get_training_callbacks=lambda:[c for c in DrivoRAgent.get_training_callbacks(agent) if isinstance(c,ModelCheckpoint)]
    monkeypatch.setattr(training,'instantiate',lambda spec:agent)
    monkeypatch.setattr(training,'build_datasets',lambda cfg,a:(TinyScenes(),TinyScenes()))
    monkeypatch.setenv('RANK','0')
    cfg=small_config(tmp_path)
    cfg.update(dict(seed=2,train_ckpt_path=None,auto_resume=False,use_cache_without_dataset=False,validation_run=False,
        dataloader=dict(params=dict(batch_size=2,num_workers=0)),
        trainer=dict(params=dict(accelerator='cpu',devices=1,strategy='auto',precision='bf16-mixed',max_epochs=1,
            default_root_dir=str(tmp_path),logger=False,enable_model_summary=False,enable_progress_bar=False,
            num_sanity_val_steps=0,log_every_n_steps=1))))
    cfg.runtime.update(console_log_interval=0,log_local_steps=False,validation_fp32=True)
    full.run(cfg)
    audit=json.loads((tmp_path/'trainval_full_dataset.json').read_text())
    assert audit==dict(train_samples=3,validation_samples=3,validation_is_training_subset=True,scheduler_dataset_size=3)
    assert (tmp_path/'checkpoints/last.ckpt').is_file()
    assert '32-true' in (tmp_path/'val_pdms.csv').read_text()
