"""GT controls preserve architecture through automatic FP32 navtest."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('variant',['single_gt','single_gt_long','single_gt_long_register','human_diverse8_l1'])
@pytest.mark.parametrize('rank,args,exit_code',[(0,[],0),(7,[],0),(0,['--cfg','job'],0),(0,[],7),
    (0,['agent.config.ref_num=6','agent.config.future_register.enabled=false'],0)])
def test_pipeline(tmp_path,variant,rank,args,exit_code):
    root=tmp_path/'DrivoR';training=root/'scripts/training';evaluation=root/'scripts/evaluation'
    training.mkdir(parents=True);evaluation.mkdir(parents=True)
    wrapper=f'run_drivor_query_{variant}_train_eval.sh'
    for name in (wrapper,'run_drivor_query_gt_train_eval.sh'):
        shutil.copyfile(ROOT/'scripts/training'/name,training/name)
    shutil.copyfile(ROOT/'scripts/evaluation/run_drivor_query_gt_evaluation.sh',evaluation/'run_drivor_query_gt_evaluation.sh')
    recorder=tmp_path/'record.py'
    recorder.write_text('import json,os,sys\nwith open(os.environ["EVENTS"],"a") as f:\n'
        ' f.write(json.dumps(dict(kind=sys.argv[1],args=sys.argv[2:],experiment=os.environ["DRIVOR_EXPERIMENT"]))+"\\n")\n')
    (training/'run_drivor_query_gt_training.sh').write_text(
        '/usr/bin/python3 "$RECORDER" train "$@"\n'
        'p="$NAVSIM_EXP_ROOT/ke/$DRIVOR_EXPERIMENT/$DRIVOR_RUN_ID/lightning_logs/version_0/checkpoints"\n'
        'mkdir -p "$p"\ntouch "$p/last.ckpt"\nexit "$TRAIN_EXIT"\n')
    (evaluation/'run_drivor_pdm_score_evaluation.sh').write_text(
        'test -f "$1" || exit 8\n/usr/bin/python3 "$RECORDER" eval "$@"\n')
    env={k:v for k,v in os.environ.items() if not k.startswith('DRIVOR_') and k not in ('NODE_RANK','SLURM_NODEID')}
    env.update(RANK=str(rank),WORLD_SIZE='8',DRIVOR_RUN_ID='controls_test',NAVSIM_EXP_ROOT=str(tmp_path/'exp'),
        EVENTS=str(tmp_path/'events'),RECORDER=str(recorder),TRAIN_EXIT=str(exit_code),DRIVOR_EVAL_PRECISION='bf16-mixed')
    result=subprocess.run(['bash',str(training/wrapper),*args],env=env,capture_output=True,text=True,timeout=10)
    assert result.returncode==exit_code,result.stderr
    records=[json.loads(line) for line in (tmp_path/'events').read_text().splitlines()]
    evaluate=rank==0 and '--cfg' not in args and exit_code==0
    assert [row['kind'] for row in records]==(['train','eval'] if evaluate else ['train'])
    assert records[0]['args'][0]==variant
    assert records[0]['experiment']==f'training_drivoR_query_dinov3_vitl_navtrain_{variant}'
    if evaluate:
        assert records[1]['experiment']==f'eval_navtest_query_dinov3_vitl_{variant}'
        assert f'agent=drivoR_query_dinov3_vitl_{variant}' in records[1]['args']
        assert 'trainer.params.precision=32-true' in records[1]['args']
        assert 'agent.config.long_trajectory_additional_poses=0' in records[1]['args']
        for argument in args:
            assert argument in records[1]['args']
