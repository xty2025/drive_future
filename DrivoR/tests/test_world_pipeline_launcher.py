"""Verify preparation/training/evaluation ordering without GPUs or DLC processes."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize('variant,rank,arguments,expected', [
    ('vjepa2_world',0,[],['prepare:real','train','eval:vjepa2_world']),
    ('vjepa2_world_multigt',0,[],['prepare:cosmos','train','eval:vjepa2_world_multigt']),
    ('vjepa2_world_multigt',1,[],['prepare:cosmos','train']),
    ('vjepa2_1_vitl',0,[],['train','eval:vjepa2_1_vitl']),
    ('vjepa2_world',0,['--cfg','job'],['train']),
])
def test_pipeline_order_and_head_only_evaluation(tmp_path,variant,rank,arguments,expected):
    training=tmp_path/'DrivoR/scripts/training'
    evaluation=tmp_path/'DrivoR/scripts/evaluation'
    training.mkdir(parents=True)
    evaluation.mkdir(parents=True)
    source=Path(__file__).resolve().parents[1]/'scripts/training/run_drivor_world_train_eval.sh'
    script=training/source.name
    shutil.copyfile(source,script)
    (training/'run_drivor_world_data_preparation.sh').write_text('echo "prepare:$1" >> "$EVENTS"\n')
    (training/f'run_drivor_{variant}_training.sh').write_text(
        'echo train >> "$EVENTS"\n'
        'ckpt="$NAVSIM_EXP_ROOT/ke/$DRIVOR_EXPERIMENT/$DRIVOR_RUN_ID/lightning_logs/version_5/checkpoints"\n'
        'mkdir -p "$ckpt"\n'
        'touch "$ckpt/last.ckpt"\n')
    (evaluation/'run_drivor_world_evaluation.sh').write_text(
        'echo "eval:$1" >> "$EVENTS"\n'
        'test -f "$2"\n'
        '[[ "$2" == */version_5/checkpoints/last.ckpt ]]\n')
    env=os.environ.copy()
    for key in ('DRIVOR_TRAIN_EXPERIMENT','DRIVOR_PREPARE_FUTURE','DRIVOR_FUTURE_FEATURES','NODE_RANK'):
        env.pop(key,None)
    env.update(EVENTS=str(tmp_path/'events'),RANK=str(rank),WORLD_SIZE='8',DRIVOR_RUN_ID='test_run',
               NAVSIM_EXP_ROOT=str(tmp_path/'exp'),DRIVOR_WORLD_DATA_ROOT=str(tmp_path/'data'))
    result=subprocess.run(['bash',str(script),variant,*arguments],env=env,text=True,capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert (tmp_path/'events').read_text().splitlines()==expected
