"""Human/long supervision survives adding ordinary-group CLOVER coverage."""

import json
import os
from pathlib import Path
import shutil
import subprocess

from omegaconf import OmegaConf
import pytest
import torch

from navsim.agents.drivoR_action_dit import model as model_module
from navsim.agents.drivoR_action_dit.agent import DrivoRActionDiTAgent
from navsim.agents.drivoR_action_dit.auxiliary import DrivoRActionDiTAuxiliaryAgent
from navsim.agents.drivoR_action_dit.loss import DrivoRActionDiTLoss
from test_action_dit_gt_variants import TinyImageEncoder, config


ROOT = Path(__file__).resolve().parents[1]
NAME = 'drivoR_action_dit_dinov3_vitl_single_gt_long_pseudo_expert'
TRAIN_EXP = 'training_drivoR_action_dit_dinov3_vitl_navtrain_single_gt_long_pseudo_expert'
EVAL_EXP = 'eval_navtest_action_dit_dinov3_vitl_single_gt_long_pseudo_expert'


def test_coverage_preserves_human_long_and_scorer_and_only_supervises_normal_group(monkeypatch):
    monkeypatch.setattr(model_module, 'DinoV3ImageEncoder', TinyImageEncoder)
    base_config = config('single_gt_long')
    base_config.tf_d_model = 8
    base_config.tf_d_ffn = 16
    base_config.scorer_ref_num = 1
    base_config.action_dit.update(dict(hidden_dim=8, ffn_dim=16, freq_dim=8,
                                      num_heads=2, attn_head_dim=4, num_layers=1,
                                      pretrained_path='', gradient_checkpointing=False))
    extra_config = OmegaConf.merge(base_config, OmegaConf.load(
        ROOT / 'navsim/planning/script/config/common/agent' / (NAME + '.yaml')).config)
    base = DrivoRActionDiTAgent(base_config, lr_args={}, loss=DrivoRActionDiTLoss())
    extra = DrivoRActionDiTAuxiliaryAgent(extra_config, lr_args={}, loss=DrivoRActionDiTLoss())
    extra.load_state_dict(base.state_dict(), strict=True)
    oracle_calls = []

    def oracle(targets, proposals, test=True):
        assert proposals.shape == (2, 64, 8, 3) and not test
        oracle_calls.append(proposals.shape)
        raw = torch.full((2, 64, 7), .75)
        return raw[..., -1], raw[..., -1].amax(1), raw, None, None, None

    base.compute_score = extra.compute_score = oracle
    features = dict(image=torch.randn(2, 1, 3, 32, 32), ego_status=torch.randn(2, 4, 11))
    targets = dict(trajectory=torch.randn(2, 8, 3), trajectory_long=torch.randn(2, 8, 3),
                   pseudo_experts=torch.randn(2, 8, 8, 3, requires_grad=True),
                   pseudo_expert_mask=torch.tensor([[True, True] + [False] * 6,
                                                     [True] + [False] * 7]),
                   pseudo_expert_fallback=torch.zeros(2, dtype=torch.bool))
    with torch.no_grad():
        targets['pseudo_experts'][~targets['pseudo_expert_mask']] = float('nan')
    calls = dict(image=0, dit=[], scorer=[])
    captured = []

    def image_call(module, args):
        calls['image'] += 1

    def dit_call(module, args, output):
        calls['dit'].append(tuple(args[0].shape))
        output.retain_grad()
        captured.append(output)

    model = extra._drivor_model
    model.image_backbone.register_forward_pre_hook(image_call)
    model.action_dit.register_forward_hook(dit_call)
    model.scorer.register_forward_pre_hook(lambda module, args: calls['scorer'].append(args[0].shape[1]))
    torch.manual_seed(37)
    original = base.forward_train(features, targets)
    torch.manual_seed(37)
    augmented = extra.forward_train(features, targets)
    for key in ('proposals', 'pdm_score', '_flow_loss'):
        torch.testing.assert_close(original[key], augmented[key], rtol=0, atol=0)
    old_loss = base.compute_loss(features, targets, original)
    new_loss = extra.compute_loss(features, targets, augmented)
    assert calls == dict(image=1, dit=[(2, 128, 8, 3)], scorer=[64])
    assert len(oracle_calls) == 2
    diag = augmented['_flow_diagnostics']
    torch.testing.assert_close(augmented['_flow_loss'], diag['flow_normal_loss'] + diag['flow_long_loss'])
    torch.testing.assert_close(old_loss['final_score_loss'], new_loss['final_score_loss'], rtol=0, atol=0)
    torch.testing.assert_close(new_loss['loss'], old_loss['loss'] + .5 * new_loss['pseudo_expert_loss'])

    expected_scenes = []
    for b in range(2):
        experts = targets['pseudo_experts'][b, targets['pseudo_expert_mask'][b]].detach()
        distances = (augmented['proposals'][b, :, None] - experts[None]).abs().sum(-1).mean(-1)
        expected_scenes.append(distances.amin(0).mean())
    torch.testing.assert_close(new_loss['pseudo_expert_loss'], torch.stack(expected_scenes).mean())
    new_loss['pseudo_expert_loss'].backward()
    assert captured[0].grad[:, :64].abs().sum() > 0
    assert not captured[0].grad[:, 64:].any()
    assert model.action_dit.head.weight.grad.abs().sum() > 0
    assert model.image_backbone.neck.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.scorer.parameters())
    assert all(p.grad is None for p in model.scorer_attention.parameters())
    assert targets['pseudo_experts'].grad is None


@pytest.mark.parametrize('rank,args,train_exit', [(0, [], 0), (7, [], 0),
                                                (0, ['--cfg', 'job'], 0), (0, [], 7)])
def test_new_pipeline_keeps_agent_and_experiment_and_runs_only_head_fp32_eval(tmp_path, rank, args, train_exit):
    root = tmp_path / 'DrivoR'
    training, evaluation = root / 'scripts/training', root / 'scripts/evaluation'
    training.mkdir(parents=True)
    evaluation.mkdir(parents=True)
    name = 'run_drivor_action_dit_dinov3_vitl_single_gt_long_pseudo_expert_train_eval.sh'
    for script_name in (name, 'run_drivor_action_dit_dinov3_vitl_aux_train_eval.sh'):
        shutil.copyfile(ROOT / 'scripts/training' / script_name, training / script_name)
    recorder = tmp_path / 'record.py'
    recorder.write_text(
        'import json,os,sys\n'
        'with open(os.environ["EVENTS"],"a") as f:\n'
        ' f.write(json.dumps(dict(kind=sys.argv[1],args=sys.argv[2:],'
        'experiment=os.environ["DRIVOR_EXPERIMENT"],run_id=os.environ["DRIVOR_RUN_ID"]))+"\\n")\n')
    (training / 'run_drivor_action_dit_dinov3_vitl_single_gt_long_training.sh').write_text(
        '/usr/bin/python3 "$RECORDER" train "$@"\n'
        'path="$NAVSIM_EXP_ROOT/ke/$DRIVOR_EXPERIMENT/$DRIVOR_RUN_ID/lightning_logs/version_0/checkpoints"\n'
        'mkdir -p "$path"\ntouch "$path/last.ckpt"\nexit "$TRAIN_EXIT"\n')
    (evaluation / 'run_drivor_pdm_score_evaluation.sh').write_text(
        'test -f "$1" || exit 8\n/usr/bin/python3 "$RECORDER" eval "$@"\n')
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('DRIVOR_') and k not in ('NODE_RANK', 'SLURM_NODEID')}
    env.update(RANK=str(rank), WORLD_SIZE='8', DRIVOR_RUN_ID='test_long_pe',
               NAVSIM_EXP_ROOT=str(tmp_path / 'exp'), EVENTS=str(tmp_path / 'events'),
               RECORDER=str(recorder), TRAIN_EXIT=str(train_exit), DRIVOR_EVAL_PRECISION='bf16-mixed')
    result = subprocess.run(['bash', str(training / name), *args], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == train_exit, result.stderr
    events = [json.loads(line) for line in (tmp_path / 'events').read_text().splitlines()]
    evaluate = rank == 0 and not args and train_exit == 0
    assert [e['kind'] for e in events] == (['train', 'eval'] if evaluate else ['train'])
    assert events[0]['experiment'] == TRAIN_EXP
    assert 'agent=' + NAME in events[0]['args']
    if evaluate:
        assert events[1]['experiment'] == EVAL_EXP
        assert events[1]['run_id'] == 'test_long_pe'
        assert 'trainer.params.precision=32-true' in events[1]['args']
        assert 'agent=drivoR_action_dit_dinov3_vitl' in events[1]['args']
        assert 'agent.config.multi_gt.enabled=false' in events[1]['args']
