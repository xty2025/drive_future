"""New opt-in X L1 entries retain head-node-only FP32 evaluation."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('variant,parent', [('human_xl1_diverse8_l1','human_diverse8_l1'),
                                            ('single_gt_long_xl1','single_gt_long')])
@pytest.mark.parametrize('rank,args,train_exit', [(0,[],0),(7,[],0),(0,['--cfg','job'],0),(0,[],7)])
def test_pipeline(tmp_path,variant,parent,rank,args,train_exit):
    root=tmp_path/'DrivoR';training=root/'scripts/training';evaluation=root/'scripts/evaluation'
    training.mkdir(parents=True);evaluation.mkdir(parents=True)
    wrapper=f'run_drivor_action_dit_dinov3_vitl_{variant}_train_eval.sh'
    for name in (wrapper,'run_drivor_action_dit_dinov3_vitl_aux_train_eval.sh'):
        shutil.copyfile(ROOT/'scripts/training'/name,training/name)
    recorder=tmp_path/'record.py'
    recorder.write_text('import json,os,sys\nwith open(os.environ["EVENTS"],"a") as f:\n'
        ' f.write(json.dumps(dict(kind=sys.argv[1],args=sys.argv[2:],experiment=os.environ["DRIVOR_EXPERIMENT"]))+"\\n")\n')
    (training/f'run_drivor_action_dit_dinov3_vitl_{parent}_training.sh').write_text(
        '/usr/bin/python3 "$RECORDER" train "$@"\n'
        'p="$NAVSIM_EXP_ROOT/ke/$DRIVOR_EXPERIMENT/$DRIVOR_RUN_ID/lightning_logs/version_0/checkpoints"\n'
        'mkdir -p "$p"\ntouch "$p/last.ckpt"\nexit "$TRAIN_EXIT"\n')
    (evaluation/'run_drivor_pdm_score_evaluation.sh').write_text(
        'test -f "$1" || exit 8\n/usr/bin/python3 "$RECORDER" eval "$@"\n')
    env={k:v for k,v in os.environ.items() if not k.startswith('DRIVOR_') and k not in ('NODE_RANK','SLURM_NODEID')}
    env.update(RANK=str(rank),WORLD_SIZE='8',DRIVOR_RUN_ID='x_l1_test',NAVSIM_EXP_ROOT=str(tmp_path/'exp'),
               EVENTS=str(tmp_path/'events'),RECORDER=str(recorder),TRAIN_EXIT=str(train_exit),DRIVOR_EVAL_PRECISION='bf16-mixed')
    result=subprocess.run(['bash',str(training/wrapper),*args],env=env,capture_output=True,text=True,timeout=10)
    assert result.returncode==train_exit,result.stderr
    records=[json.loads(line) for line in (tmp_path/'events').read_text().splitlines()]
    evaluate=rank==0 and not args and train_exit==0
    assert [r['kind'] for r in records]==(['train','eval'] if evaluate else ['train'])
    assert records[0]['experiment']==f'training_drivoR_action_dit_dinov3_vitl_navtrain_{variant}'
    assert f'agent=drivoR_action_dit_dinov3_vitl_{variant}' in records[0]['args']
    if evaluate:
        assert records[1]['experiment']==f'eval_navtest_action_dit_dinov3_vitl_{variant}'
        assert 'trainer.params.precision=32-true' in records[1]['args']
        assert 'agent=drivoR_action_dit_dinov3_vitl' in records[1]['args']
        assert 'agent.config.multi_gt.enabled=false' in records[1]['args']
