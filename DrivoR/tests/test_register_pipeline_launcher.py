"""Joint register variants must evaluate the matching architecture in FP32."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('variant',[
    'register','register_scorer','register_scorer_single_gt','pseudo_expert_vmse_register','pseudo_expert_vmse_register_scorer'])
@pytest.mark.parametrize('rank,arguments',[(0,[]),(7,[]),(0,['--cfg','job'])])
def test_train_then_head_only_fp32_eval(tmp_path,variant,rank,arguments):
    training = tmp_path/'DrivoR/scripts/training'
    evaluation = tmp_path/'DrivoR/scripts/evaluation'
    training.mkdir(parents=True); evaluation.mkdir(parents=True)
    script = training/'run_drivor_action_dit_dinov3_vitl_register_train_eval.sh'
    shutil.copyfile(ROOT/'scripts/training'/script.name,script)
    shutil.copyfile(ROOT/'scripts/evaluation/run_drivor_action_dit_dinov3_vitl_register_evaluation.sh',
                    evaluation/'run_drivor_action_dit_dinov3_vitl_register_evaluation.sh')
    (training/f'run_drivor_action_dit_dinov3_vitl_{variant}_training.sh').write_text(
        'echo train >> "$EVENTS"\n'
        'ckpt="$NAVSIM_EXP_ROOT/ke/$DRIVOR_EXPERIMENT/$DRIVOR_RUN_ID/lightning_logs/version_5/checkpoints"\n'
        'mkdir -p "$ckpt"\n'
        'touch "$ckpt/last.ckpt"\n')
    capture = tmp_path/'capture.py'
    capture.write_text('import json, os, sys\nfrom pathlib import Path\n'
        'Path(os.environ["EVAL_ARGS"]).write_text(json.dumps(sys.argv[1:]))\n')
    (evaluation/'run_drivor_pdm_score_evaluation.sh').write_text(
        'echo eval >> "$EVENTS"\n'
        'test -f "$1"\n'
        f'"{sys.executable}" "{capture}" "$@"\n')
    env = os.environ.copy()
    for key in ('NODE_RANK','DRIVOR_TRAIN_EXPERIMENT','DRIVOR_EVAL_PRECISION'):
        env.pop(key,None)
    env.update(RANK=str(rank),WORLD_SIZE='8',GPU_NUM='8',DRIVOR_RUN_ID='test_run',
        NAVSIM_EXP_ROOT=str(tmp_path/'exp'),EVENTS=str(tmp_path/'events'),EVAL_ARGS=str(tmp_path/'args.json'))
    result = subprocess.run(['bash',str(script),variant,*arguments],env=env,capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    should_evaluate = rank==0 and not arguments
    assert (tmp_path/'events').read_text().splitlines()==(['train','eval'] if should_evaluate else ['train'])
    if should_evaluate:
        args = json.loads((tmp_path/'args.json').read_text())
        suffix = 'multigt_'+variant if variant.startswith('pseudo_') else variant
        assert f'navtrain_{suffix}/test_run/' in args[0]
        model = 'register_scorer' if 'scorer' in variant else 'register'
        assert 'agent=drivoR_action_dit_dinov3_vitl_'+model in args
        assert 'trainer.params.precision=32-true' in args
        assert 'agent.config.multi_gt.enabled=false' in args
