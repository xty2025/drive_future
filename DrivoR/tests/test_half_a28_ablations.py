"""Independent half-data variants: routing, targets, attention and gradients."""
import json
import os
from pathlib import Path
import subprocess

from hydra import compose, initialize_config_dir
from hydra._internal.utils import get_args_parser
from hydra.utils import get_class
from omegaconf import OmegaConf
import pytest
from safetensors.torch import save_file
import torch
from torch import nn
from torch.nn import functional as F

from navsim.agents.drivoR_action_dit import model as base_module
from navsim.agents.drivoR_action_dit.action_dit import ActionDiT
from navsim.agents.drivoR_action_dit.flow import normalize_trajectory
from navsim.agents.drivoR_action_dit.loss import DrivoRActionDiTLoss
from navsim.agents.drivoR_action_dit_register.dit import ActionRegisterDiT
from navsim.agents.drivoR_action_dit_register.diverse_losses import grouped_anchor_indices
from navsim.agents.drivoR_action_dit_register.teacher import EmaDinoRegisters, SharedDinoTargets
from navsim.agents.drivoR_action_dit_register import vae_teacher
from test_action_dit_register import TinyEncoder, native_model, oracle
from test_diverse8_register import compose as compose_agent, sample

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('a23', 'a27', 'a28', 'a28_random_dit', 'a28_patch16', 'a28_register_first',
            'a28_bidirectional', 'a28_shared_vit', 'a28_frozen_vit', 'a28_dit1024',
            'a28_wan5b', 'a28_vae16', 'a28_grouped_noise')
BASE = 'drivoR_action_dit_dinov3_vitl_human_xl1_diverse8_l1'


def agent_name(variant):
    return {'a23': BASE, 'a27': BASE + '_register', 'a28': BASE + '_register_scorer'}.get(
        variant, 'drivoR_half_' + variant)


class TinyVae(nn.Module):
    z_dim = 48

    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 48, 1).requires_grad_(False)
        self.seen_shapes = []

    def single_encode(self, image, device):
        self.seen_shapes.append(tuple(image.shape))
        assert image.shape[2] == 1
        return self.projection(image[:, :, 0]).unsqueeze(2)


def tiny_agent(monkeypatch, variant, checkpoint='', scorer_heads=2):
    raw = compose_agent(agent_name(variant))
    cfg = raw.config
    cfg.tf_d_model = 8; cfg.tf_d_ffn = 16; cfg.proposal_num = 4
    cfg.ref_num = 2; cfg.scorer_ref_num = 1; cfg.refiner_num_heads = scorer_heads
    cfg.pseudo_expert.path = '/absent/gt'
    cfg.action_dit.update(dict(hidden_dim=16 if variant in ('a28_dit1024', 'a28_wan5b', 'a28_wan21_1p3b') else 8,
        ffn_dim=32, freq_dim=8, num_heads=2, attn_head_dim=4, num_layers=2,
        num_inference_steps=2, gradient_checkpointing=False))
    if variant != 'a28_random_dit':
        cfg.action_dit.pretrained_path = None
    if 'future_register' in cfg:
        cfg.future_register.frames_root = '/absent/frames'
        cfg.future_register.rollouts_root = '/absent/videos'
        cfg.future_register.sensor_blobs_path = '/absent/sensors'
        cfg.future_register.teacher_batch_size = 8
        if cfg.future_register.get('generator_kind') == 'mot':
            cfg.future_register.mot.video_dit.update(dict(hidden_dim=16, ffn_dim=32, freq_dim=8,
                num_heads=2, attn_head_dim=4, num_layers=2, pretrained_path=None, gradient_checkpointing=False))
    monkeypatch.setattr(base_module, 'DinoV3ImageEncoder', TinyEncoder)
    monkeypatch.setattr(vae_teacher, 'load_wan_vae', lambda options: TinyVae())
    result = get_class(raw._target_)(cfg, lr_args=dict(name='AdamW', base_lr=.001, base_batch_size=1),
        checkpoint_path=checkpoint, loss=DrivoRActionDiTLoss())
    result.compute_score = oracle
    return result.eval()


@pytest.mark.parametrize('variant', VARIANTS)
def test_primary_config_uses_same_half_scope_and_independent_a28_parent(variant):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'navsim/planning/script/config/training')):
        cfg = compose(config_name='half_' + variant, overrides=[
            'train_test_split/scene_filter=all_scenes', 'train_test_split_synthetic/scene_filter=all_scenes'])
    assert cfg.agent.scheduler_args.dataset_size == 42555
    assert cfg.runtime.scheduler_dataset_size_from_data and cfg.runtime.get('training_data_scope') != 'trainval'
    assert cfg.cache_path is None and not cfg.use_cache_without_dataset
    assert cfg.agent.config.tf_d_model == 256 and cfg.agent.config.tf_d_ffn == 1024
    assert cfg.agent.config.proposal_num == 64 and cfg.agent.config.pseudo_expert.weight == .5
    assert cfg.agent.config.pseudo_expert.source == 'diverse8' and not cfg.agent.config.multi_gt.enabled
    assert cfg.agent.config.long_trajectory_additional_poses == 0
    if variant.startswith('a28_'):
        raw = OmegaConf.load(ROOT / 'navsim/planning/script/config/common/agent' / (agent_name(variant) + '.yaml'))
        assert list(raw.defaults) == [BASE + '_register_scorer', '_self_']
        future = cfg.agent.config.future_register
        assert future.scorer.enabled and future.get('supervision_scope', 'human_and_diverse') == 'human_and_diverse'
        assert future.human_loss_type == 'l1' and future.loss_type == 'velocity'
    if variant == 'a28_wan5b':
        assert cfg.trainer.params.strategy == 'deepspeed_stage_2' and cfg.trainer.params.precision == '32-true'
        assert cfg.agent.config.action_dit.hidden_dim == 3072
    if variant == 'a28_random_dit':
        assert cfg.agent.config.action_dit.pretrained_path is None


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('mode', ('training', 'train_eval'))
def test_launchers_keep_half_split_output_root_and_dlc_topology(tmp_path, variant, mode):
    fake = tmp_path / 'python'
    fake.write_text('#!/usr/bin/python3\nimport os,json,sys\nprint(json.dumps(dict(argv=sys.argv[1:],root=os.environ["NAVSIM_EXP_ROOT"],cache=os.environ["DRIVOR_DIVERSE8_FRAMES_ROOT"])))\n')
    fake.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('DRIVOR_', 'TORCHELASTIC_'))
           and k not in ('LOCAL_RANK', 'NODE_RANK', 'NNODES', 'NPROC_PER_NODE', 'SLURM_NODEID')}
    env.update(DRIVOR_PYTHON=str(fake), WORLD_SIZE='8', RANK='3', GPU_NUM='8', MASTER_ADDR='dlc-master-0',
               MASTER_PORT='23456', PAI_JOB_ID='half-ablation-test', NAVSIM_EXP_ROOT='/wrong-old-output')
    result = subprocess.run(['bash', str(ROOT / f'scripts/training/run_drivor_half_{variant}_{mode}.sh'),
        'train_test_split=navtrain', 'use_cache_without_dataset=true', '--cfg', 'job'],
        capture_output=True, text=True, env=env, timeout=20)
    assert result.returncode == 0, result.stderr
    recorded = json.loads(result.stdout)
    args = recorded['argv']
    assert recorded['root'] == str(ROOT / 'exp/navtrain_half_ablations')
    assert recorded['cache'] == str(ROOT / 'exp/world4scorer_data/frames_diverse8')
    entry = str(ROOT / 'navsim/planning/script/run_training.py')
    parsed = get_args_parser().parse_args(args[args.index(entry) + 1:])
    assert parsed.config_name == 'half_' + variant and parsed.cfg == 'job'
    assert 'train_test_split=navtrain_half' in parsed.overrides
    assert [v for v in parsed.overrides if v.startswith('train_test_split=')][-1] == 'train_test_split=navtrain_half'
    assert [v for v in parsed.overrides if v.startswith('agent=')][-1] == 'agent=' + agent_name(variant)
    assert 'agent.num_gpus=64' in parsed.overrides and 'agent.scheduler_args.dataset_size=42555' in parsed.overrides
    assert 'trainer.params.num_nodes=8' in parsed.overrides and '++trainer.params.devices=8' in parsed.overrides
    assert 'experiment_name=training_drivoR_half_' + variant in parsed.overrides


@pytest.mark.parametrize('variant', VARIANTS)
def test_eval_launchers_restore_matching_architecture_and_force_fp32(tmp_path, variant):
    fake = tmp_path / 'python'
    fake.write_text('#!/usr/bin/python3\nimport os,json,sys\nprint(json.dumps(dict(argv=sys.argv[1:],root=os.environ["NAVSIM_EXP_ROOT"])))\n')
    fake.chmod(0o755)
    ckpt = tmp_path / 'last.ckpt'; ckpt.touch()
    env = {k: v for k, v in os.environ.items() if not k.startswith('DRIVOR_')}
    env.update(DRIVOR_PYTHON=str(fake), DRIVOR_RUN_ID='half-eval-smoke')
    result = subprocess.run(['bash', str(ROOT / 'scripts/evaluation/run_drivor_half_ablation_evaluation.sh'),
        variant, str(ckpt), 'trainer.params.precision=bf16-mixed', '--cfg', 'job'],
        env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    recorded = json.loads(result.stdout)
    args = recorded['argv']
    entry = str(ROOT / 'navsim/planning/script/run_pdm_score_multi_gpu.py')
    parsed = get_args_parser().parse_args(args[args.index(entry) + 1:])
    assert parsed.cfg == 'job'
    assert [x for x in parsed.overrides if x.startswith('agent=')][-1] == 'agent=' + agent_name(variant)
    assert [x for x in parsed.overrides if x.startswith('trainer.params.precision=')] == ['trainer.params.precision=32-true']
    assert 'agent.config.action_dit.pretrained_path=null' in parsed.overrides
    assert recorded['root'] == str(ROOT / 'exp/navtrain_half_ablations')


@pytest.mark.parametrize('variant', VARIANTS)
def test_all_models_one_forward_shapes_detached_score_and_checkpoint_roundtrip(monkeypatch, sample, tmp_path, variant):
    agent = tiny_agent(monkeypatch, variant)
    features, targets = sample
    model = agent._drivor_model
    calls = []
    model.action_dit.register_forward_hook(lambda *args: calls.append(1))
    output = agent.forward_train(features, targets)
    losses = agent.compute_loss(features, targets, output)
    assert calls == [1] and output['proposals'].shape == (1, 4, 8, 3)
    assert torch.isfinite(losses['loss'])
    if variant != 'a23':
        per_frame = 16 if variant in ('a28_patch16', 'a28_vae16') else 4
        dim = 48 if variant == 'a28_vae16' else 16
        assert output['predicted_registers'].shape == (1, 4, 8, per_frame, dim)
        assert output['_register_matched_indices'].shape == (1, 9)
    losses['final_score_loss'].backward()
    assert all(p.grad is None for p in model.action_dit.parameters())
    model.zero_grad(set_to_none=True)
    losses = agent.compute_loss(features, targets, agent.forward_train(features, targets))
    losses['loss'].backward()
    assert model.action_dit.head.weight.grad.abs().sum() > 0
    if variant == 'a28_frozen_vit':
        assert all(not p.requires_grad and p.grad is None for p in model.image_backbone.native_vit.parameters())
        assert model.image_backbone.neck.weight.grad.abs().sum() > 0
    if variant == 'a28_shared_vit':
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.image_backbone.native_vit.parameters())
        assert model.ema_teacher.encoder is model.image_backbone.native_vit
        assert list(model.ema_teacher.parameters()) == []
    if variant in ('a28_dit1024', 'a28_wan5b'):
        assert model.action_dit.scene_projection.weight.grad.abs().sum() > 0
        assert model.pos_embed[-1].out_features == 8  # Scorer stayed narrow.
    checkpoint = tmp_path / 'model.ckpt'
    torch.save({'state_dict': {'agent.' + k: v for k, v in agent.state_dict().items()}}, checkpoint)
    restored = tiny_agent(monkeypatch, variant, str(checkpoint))
    restored.initialize()
    if variant != 'a23':
        monkeypatch.setattr(restored._drivor_model.ema_teacher, 'forward',
                            lambda *args: (_ for _ in ()).throw(AssertionError('No teacher during inference')))
    with torch.no_grad():
        torch.manual_seed(15); expected = agent(features)
        torch.manual_seed(15); actual = restored(features)
    torch.testing.assert_close(expected['proposals'], actual['proposals'])
    torch.testing.assert_close(expected['pdm_score'], actual['pdm_score'])


@pytest.mark.parametrize('order,mode', [('register_first', 'register_reads_action'), ('action_first', 'bidirectional')])
def test_attention_layout_has_requested_dependency_gradients(order, mode):
    dit = ActionDiT(hidden_dim=8, ffn_dim=16, freq_dim=8, num_heads=2, attn_head_dim=4,
                    num_layers=2, use_gradient_checkpointing=True)
    joint = ActionRegisterDiT(dit, 16, token_order=order, attention_mode=mode).train()
    action = torch.randn(1, 3, 8, 3, requires_grad=True)
    registers = torch.randn(1, 3, 32, 16, requires_grad=True)
    args = (torch.full((1, 3), 500.), torch.randn(1, 7, 8), torch.randn(1, 8))
    a, r = joint(action, registers, *args)
    r_to_a, = torch.autograd.grad(a.square().sum(), registers, retain_graph=True)
    a_to_r, = torch.autograd.grad(r.square().sum(), action)
    assert a_to_r.abs().sum() > 0
    if mode == 'bidirectional':
        assert r_to_a.abs().sum() > 0 and joint.attention_mask.all()
    else:
        assert not r_to_a.any() and not joint.attention_mask[32:, :32].any()
        assert joint.attention_mask[:32, 32:].all() and joint.action_slice == slice(32, 40)


def test_patch_targets_are_spatial_means_and_shared_teacher_has_no_target_gradient():
    student = native_model()
    pixels = torch.rand(1, 1, 8, 3, 96, 96, requires_grad=True) * 255
    valid = torch.ones(1, 1, 8, dtype=torch.bool)
    teacher = EmaDinoRegisters(student, target_kind='patches', pooling_grid=(4, 4))
    output = teacher(pixels, valid)
    image = (pixels[0, 0, :1].detach() / 255 - teacher.pixel_mean) / teacher.pixel_std
    with torch.no_grad():
        patches = teacher.encoder.forward_features(image)['x_norm_patchtokens'].transpose(1, 2).reshape(1, 16, 6, 6)
        expected = F.adaptive_avg_pool2d(patches, (4, 4)).flatten(2).transpose(1, 2)
    torch.testing.assert_close(output[0, 0, :1], expected)
    shared = SharedDinoTargets(student)
    assert not shared(pixels, valid).requires_grad and not output.requires_grad
    assert shared.encoder is student and set(shared.state_dict()) == {'updates'}
    shared.update(student)
    assert shared.updates == 0


def test_vae_targets_preserve_all_eight_times_use_16_spatial_bins_and_detach(monkeypatch):
    encoder = TinyVae()
    monkeypatch.setattr(vae_teacher, 'load_wan_vae', lambda cfg: encoder)
    teacher = vae_teacher.WanVaeTargets({}, pooling_grid=(4, 4), chunk_size=3)
    pixels = torch.randint(0, 256, (1, 2, 8, 3, 32, 32), dtype=torch.uint8)
    valid = torch.ones(1, 2, 8, dtype=torch.bool); valid[:, 1, 3] = False
    output = teacher(pixels, valid)
    assert output.shape == (1, 2, 8, 16, 48) and not output.requires_grad
    assert sum(shape[0] for shape in encoder.seen_shapes) == 15
    assert all(shape[2] == 1 for shape in encoder.seen_shapes)
    assert not output[:, 1, 3].any()
    expected = F.adaptive_avg_pool2d(encoder.projection(pixels[0, 0, 7:8].float() / 127.5 - 1), (4, 4))
    torch.testing.assert_close(output[0, 0, 7], expected.flatten(2).transpose(1, 2)[0])


@pytest.mark.parametrize('valid,counts', [([True] * 8, [8] + [7] * 8),
    ([True, False, True] + [False] * 5, [22, 21, 0, 21, 0, 0, 0, 0, 0]),
    ([False] * 8, [64] + [0] * 8)])
def test_group_counts_follow_valid_gt_ids_and_remainder_goes_to_human(valid, counts):
    assignment = grouped_anchor_indices(torch.tensor([valid]), 64)
    assert assignment.shape == (1, 64)
    assert torch.bincount(assignment[0], minlength=9).tolist() == counts


def test_grouped_noises_use_same_gt_for_action_and_future_without_changing_matching(monkeypatch, sample):
    agent = tiny_agent(monkeypatch, 'a28_grouped_noise')
    model = agent._drivor_model
    model.proposal_num = 64
    features, targets = sample
    targets['pseudo_expert_mask'].fill_(True)
    targets['future_register_valid'].fill_(True)
    targets['pseudo_experts'] = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1).expand(1, 8, 8, 3)
    teacher = torch.arange(9, dtype=torch.float32).reshape(1, 9, 1, 1, 1).expand(1, 9, 8, 4, 16)
    monkeypatch.setattr(model.ema_teacher, 'forward', lambda *args: teacher)
    monkeypatch.setattr(torch, 'randn_like', torch.zeros_like)
    monkeypatch.setattr(model.schedule, 'sample_timesteps', lambda b, m, device: torch.full((b, m), 500., device=device))
    captured = []
    action_prediction = torch.randn(1, 64, 8, 3)
    future_prediction = torch.randn(1, 64, 32, 16)
    def predict(actions, future, *args):
        captured.append((actions, future))
        return action_prediction, future_prediction
    monkeypatch.setattr(model.action_dit, 'forward', predict)
    grouped = model.forward_train(features, targets)
    ids = grouped_anchor_indices(targets['pseudo_expert_mask'], 64)
    clean = normalize_trajectory(torch.cat((targets['trajectory'][:, None], targets['pseudo_experts']), dim=1))
    sigma = model.schedule.sigma(torch.tensor(500.))
    torch.testing.assert_close(captured[0][0], (1 - sigma) * clean[torch.zeros_like(ids), ids])
    torch.testing.assert_close(captured[0][1], (1 - sigma) * teacher[torch.zeros_like(ids), ids].flatten(2, 3))
    model.input_gt_assignment = 'original'
    original = model.forward_train(features, targets)
    for key in ('_flow_loss', '_register_loss', '_register_matched_indices'):
        torch.testing.assert_close(grouped[key], original[key])


def test_wider_fastwam_loader_keeps_new_context_adapters_and_loads_all_backbone(tmp_path):
    source = ActionDiT(hidden_dim=16, ffn_dim=32, freq_dim=8, num_heads=2, attn_head_dim=4, num_layers=2)
    state = {key: value for key, value in source.state_dict().items() if not key.startswith(('head.', 'action_encoder.'))}
    path = tmp_path / 'pretrained.pt'
    torch.save(dict(meta=source.checkpoint_meta, backbone_state_dict=state), path)
    target = ActionDiT(hidden_dim=16, ffn_dim=32, freq_dim=8, num_heads=2, attn_head_dim=4, num_layers=2, context_dim=8)
    report = target.load_pretrained(path)
    assert 'scene_projection.weight' in report['random_io_tensors']
    for key, tensor in state.items():
        torch.testing.assert_close(target.state_dict()[key], tensor, rtol=0, atol=0)


def test_native_wan_loader_copies_sharded_blocks_and_time_weights_exactly(tmp_path):
    source = ActionDiT(hidden_dim=16, ffn_dim=32, freq_dim=8, num_heads=2, attn_head_dim=4, num_layers=2)
    state = {k: v for k, v in source.state_dict().items() if k.startswith(('blocks.', 'time_embedding.', 'time_projection.'))}
    state['patch_embedding.weight'] = torch.randn(16, 48, 1, 2, 2)
    keys = list(state)
    groups = (keys[::2], keys[1::2])
    weight_map = {}
    for i, group in enumerate(groups):
        name = f'model-{i}.safetensors'
        save_file({key: state[key] for key in group}, tmp_path / name)
        weight_map.update({key: name for key in group})
    (tmp_path / 'config.json').write_text(json.dumps(dict(dim=16, ffn_dim=32, freq_dim=8, num_heads=2, num_layers=2)))
    (tmp_path / 'diffusion_pytorch_model.safetensors.index.json').write_text(json.dumps(dict(weight_map=weight_map)))
    target = ActionDiT(hidden_dim=16, ffn_dim=32, freq_dim=8, num_heads=2, attn_head_dim=4, num_layers=2, context_dim=8)
    report = target.load_wan_pretrained(tmp_path)
    assert report['interpolation'] is False and report['omitted_source_tensors'] == ['patch_embedding.weight']
    for key, tensor in state.items():
        if key in target.state_dict():
            torch.testing.assert_close(target.state_dict()[key], tensor, rtol=0, atol=0)
