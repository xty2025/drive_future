"""All four query-register launchers preserve model options in FP32 eval."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('variant', ['single_gt', 'multigt', 'scorer_single_gt', 'scorer_multigt'])
@pytest.mark.parametrize('rank,args,exit_code', [(0, [], 0), (7, [], 0),
    (0, ['--cfg', 'job'], 0), (0, [], 7),
    (0, ['agent.config.ref_num=6', 'agent.config.future_register.scorer.enabled=true'], 0)])
def test_pipeline(tmp_path, variant, rank, args, exit_code):
    root = tmp_path / 'DrivoR'
    training, evaluation = root / 'scripts/training', root / 'scripts/evaluation'
    training.mkdir(parents=True)
    evaluation.mkdir(parents=True)
    wrapper = f'run_drivor_query_register_{variant}_train_eval.sh'
    for name in (wrapper, 'run_drivor_query_register_train_eval.sh'):
        shutil.copyfile(ROOT / 'scripts/training' / name, training / name)
    shutil.copyfile(ROOT / 'scripts/evaluation/run_drivor_query_register_evaluation.sh',
                    evaluation / 'run_drivor_query_register_evaluation.sh')
    recorder = tmp_path / 'record.py'
    recorder.write_text(
        'import json,os,sys\n'
        'with open(os.environ["EVENTS"],"a") as f:\n'
        ' f.write(json.dumps(dict(kind=sys.argv[1],args=sys.argv[2:],'
        'experiment=os.environ["DRIVOR_EXPERIMENT"]))+"\\n")\n')
    (training / 'run_drivor_query_register_training.sh').write_text(
        '/usr/bin/python3 "$RECORDER" train "$@"\n'
        'p="$NAVSIM_EXP_ROOT/ke/$DRIVOR_EXPERIMENT/$DRIVOR_RUN_ID/lightning_logs/version_0/checkpoints"\n'
        'mkdir -p "$p"\ntouch "$p/last.ckpt"\nexit "$TRAIN_EXIT"\n')
    (evaluation / 'run_drivor_pdm_score_evaluation.sh').write_text(
        'test -f "$1" || exit 8\n/usr/bin/python3 "$RECORDER" eval "$@"\n')
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('DRIVOR_') and k not in ('NODE_RANK', 'SLURM_NODEID')}
    env.update(RANK=str(rank), WORLD_SIZE='8', DRIVOR_RUN_ID='query_test',
               NAVSIM_EXP_ROOT=str(tmp_path / 'exp'), EVENTS=str(tmp_path / 'events'),
               RECORDER=str(recorder), TRAIN_EXIT=str(exit_code), DRIVOR_EVAL_PRECISION='bf16-mixed')
    result = subprocess.run(['bash', str(training / wrapper), *args], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == exit_code, result.stderr
    records = [json.loads(line) for line in (tmp_path / 'events').read_text().splitlines()]
    evaluate = rank == 0 and '--cfg' not in args and exit_code == 0
    assert [r['kind'] for r in records] == (['train', 'eval'] if evaluate else ['train'])
    assert records[0]['args'][0] == variant
    assert records[0]['experiment'] == f'training_drivoR_query_register_dinov3_vitl_navtrain_{variant}'
    if evaluate:
        assert records[1]['experiment'] == f'eval_navtest_query_register_dinov3_vitl_{variant}'
        assert 'trainer.params.precision=32-true' in records[1]['args']
        assert 'agent.config.multi_gt.enabled=false' in records[1]['args']
        selected_agent = [arg for arg in records[1]['args'] if arg.startswith('agent=')][-1]
        assert selected_agent == 'agent=drivoR_query_register_dinov3_vitl_' + variant
        for argument in args:
            assert argument in records[1]['args']
