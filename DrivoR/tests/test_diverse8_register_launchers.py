"""Six standalone pipelines keep the same architecture for automatic FP32 test."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT=Path(__file__).resolve().parents[1]
VARIANTS=[family+'_register'+suffix for family in
    ('action_dit_human_diverse8_l1','action_dit_human_xl1_diverse8_l1','query_human_diverse8_l1')
    for suffix in ('','_scorer')]
VARIANTS += ['action_dit_human_xl1_diverse8_l1_register_human_only',
             'action_dit_human_xl1_diverse8_l1_register_scorer_human_only']


@pytest.mark.parametrize('variant',VARIANTS)
def test_train_eval_uses_correct_model_and_fp32(tmp_path,variant):
    run_pipeline(tmp_path,variant)


@pytest.mark.parametrize('rank,args,code',[(7,[],0),(0,['--cfg','job'],0),(0,[],9),
    (0,['agent.config.ref_num=6','agent.config.future_register.scorer.enabled=false'],0)])
def test_rank_config_failure_and_overrides(tmp_path,rank,args,code):
    run_pipeline(tmp_path,'query_human_diverse8_l1_register_scorer',rank,args,code)


def run_pipeline(tmp_path,variant,rank=0,args=(),code=0):
    root=tmp_path/'DrivoR';training=root/'scripts/training';evaluation=root/'scripts/evaluation'
    training.mkdir(parents=True);evaluation.mkdir(parents=True)
    for name in ['drivor_diverse8_register_common.sh','run_drivor_diverse8_register_train_eval.sh']:
        shutil.copyfile(ROOT/'scripts/training'/name,training/name)
    shutil.copyfile(ROOT/'scripts/evaluation/run_drivor_diverse8_register_evaluation.sh',evaluation/'run_drivor_diverse8_register_evaluation.sh')
    recorder=tmp_path/'record.py';recorder.write_text('import json,os,sys\nwith open(os.environ["EVENTS"],"a") as f:\n f.write(json.dumps(dict(kind=sys.argv[1],args=sys.argv[2:],experiment=os.environ["DRIVOR_EXPERIMENT"]))+"\\n")\n')
    (training/'run_drivor_diverse8_register_training.sh').write_text(
        '/usr/bin/python3 "$RECORDER" train "$@"\n'
        'p="$NAVSIM_EXP_ROOT/ke/$DRIVOR_EXPERIMENT/$DRIVOR_RUN_ID/lightning_logs/version_0/checkpoints"\n'
        'mkdir -p "$p"\ntouch "$p/last.ckpt"\nexit "$TRAIN_EXIT"\n')
    (evaluation/'run_drivor_pdm_score_evaluation.sh').write_text('test -f "$1" || exit 8\n/usr/bin/python3 "$RECORDER" eval "$@"\n')
    env={k:v for k,v in os.environ.items() if not k.startswith('DRIVOR_') and k not in ('NODE_RANK','SLURM_NODEID')}
    env.update(RANK=str(rank),WORLD_SIZE='8',DRIVOR_RUN_ID='test',NAVSIM_EXP_ROOT=str(tmp_path/'exp'),
        EVENTS=str(tmp_path/'events'),RECORDER=str(recorder),TRAIN_EXIT=str(code),DRIVOR_EVAL_PRECISION='bf16-mixed')
    result=subprocess.run(['bash',str(training/'run_drivor_diverse8_register_train_eval.sh'),variant,*args],
                          env=env,capture_output=True,text=True,timeout=10)
    assert result.returncode==code,result.stderr
    records=[json.loads(line) for line in (tmp_path/'events').read_text().splitlines()]
    evaluate=rank==0 and '--cfg' not in args and code==0
    assert [r['kind'] for r in records]==(['train','eval'] if evaluate else ['train'])
    if evaluate:
        family='action_dit' if variant.startswith('action_dit') else 'query'
        suffix=variant.removeprefix(family+'_')
        assert f'agent=drivoR_{family}_dinov3_vitl_{suffix}' in records[1]['args']
        assert records[1]['experiment']==f'eval_navtest_{family}_dinov3_vitl_{suffix}'
        assert 'trainer.params.precision=32-true' in records[1]['args']
        assert 'agent.config.future_register.init_checkpoint=null' in records[1]['args']
        for argument in args:assert argument in records[1]['args']
