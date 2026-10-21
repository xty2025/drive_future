"""The two human+diverse8 variants train independently and evaluate in FP32."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize('variant',['human_diverse8_l1','human_diverse8_vmse'])
@pytest.mark.parametrize('rank,args',[(0,[]),(7,[]),(0,['--cfg','job'])])
def test_pipeline_uses_new_experiment_and_head_only_fp32_eval(tmp_path,variant,rank,args):
    root=tmp_path/'DrivoR';training=root/'scripts/training';evaluation=root/'scripts/evaluation'
    training.mkdir(parents=True);evaluation.mkdir(parents=True)
    source=Path(__file__).resolve().parents[1]/'scripts/training/run_drivor_action_dit_dinov3_vitl_aux_train_eval.sh'
    script=training/source.name;shutil.copyfile(source,script)
    (training/f'run_drivor_action_dit_dinov3_vitl_{variant}_training.sh').write_text(
        'echo train >> "$EVENTS"\n'
        'path="$NAVSIM_EXP_ROOT/ke/$DRIVOR_EXPERIMENT/$DRIVOR_RUN_ID/lightning_logs/version_0/checkpoints"\n'
        'mkdir -p "$path"\ntouch "$path/last.ckpt"\n')
    (evaluation/'run_drivor_pdm_score_evaluation.sh').write_text(
        'echo eval >> "$EVENTS"\n'
        'test -f "$1"\n'
        '[[ "$1" == *navtrain_' + variant + '/test_run/* ]]\n'
        '[[ " $* " == *" trainer.params.precision=32-true "* ]]\n'
        '[[ " $* " == *" agent.config.multi_gt.enabled=false "* ]]\n')
    env=os.environ.copy()
    for key in ('NODE_RANK','DRIVOR_TRAIN_EXPERIMENT','DRIVOR_EVAL_PRECISION'):
        env.pop(key,None)
    env.update(RANK=str(rank),WORLD_SIZE='8',DRIVOR_RUN_ID='test_run',EVENTS=str(tmp_path/'events'),
               NAVSIM_EXP_ROOT=str(tmp_path/'exp'))
    result=subprocess.run(['bash',str(script),variant,*args],env=env,capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert (tmp_path/'events').read_text().splitlines()==(['train','eval'] if rank==0 and not args else ['train'])
