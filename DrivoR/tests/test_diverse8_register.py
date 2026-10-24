"""Human/D8 pairing, unchanged trajectory objectives, detached scoring and EMA."""

from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from navsim.agents.drivoR_action_dit import model as base_module
from navsim.agents.drivoR_action_dit.flow import normalize_trajectory
from navsim.agents.drivoR_action_dit.loss import DrivoRActionDiTLoss
from navsim.agents.drivoR_action_dit_register.diverse_gt import DrivoRActionDiverseRegisterAgent
from navsim.agents.drivoR_query_register import gt_variants
from navsim.agents.drivoR_query_register.diverse_gt import DrivoRQueryDiverseRegisterAgent
from navsim.agents.drivoR_query_register.losses import DrivoRQueryRegisterLoss
from test_action_dit_register import TinyEncoder, oracle


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ('action_dit_human_diverse8_l1', 'action_dit_human_xl1_diverse8_l1', 'query_human_diverse8_l1')


def agent_name(variant):
    prefix = 'action_dit' if variant.startswith('action_dit') else 'query'
    return f'drivoR_{prefix}_dinov3_vitl_' + variant.removeprefix(prefix + '_')


def compose(name):
    raw = OmegaConf.load(ROOT / 'navsim/planning/script/config/common/agent' / (name + '.yaml'))
    parents = [compose(value) for value in raw.get('defaults', []) if isinstance(value, str) and value != '_self_']
    return OmegaConf.merge(*parents, raw) if parents else raw


def make_agent(monkeypatch, variant, checkpoint=''):
    cfg = compose(agent_name(variant)).config
    cfg.tf_d_model = 8; cfg.tf_d_ffn = 16; cfg.proposal_num = 4
    cfg.ref_num = 2; cfg.scorer_ref_num = 1; cfg.refiner_num_heads = 2
    cfg.pseudo_expert.path = '/absent/GT'
    cfg.future_register.frames_root = '/absent/frames'
    cfg.future_register.rollouts_root = '/absent/videos'
    cfg.future_register.sensor_blobs_path = '/absent/sensors'
    cfg.future_register.teacher_batch_size = 4
    monkeypatch.setattr(base_module, 'DinoV3ImageEncoder', TinyEncoder)
    monkeypatch.setattr(gt_variants, 'DinoV3ImageEncoder', TinyEncoder)
    if variant.startswith('action_dit'):
        cfg.action_dit.update(dict(hidden_dim=8, ffn_dim=16, freq_dim=8, num_heads=2, attn_head_dim=4,
            num_layers=2, pretrained_path='', num_inference_steps=2, gradient_checkpointing=False))
        cls, loss = DrivoRActionDiverseRegisterAgent, DrivoRActionDiTLoss()
    else:
        cfg.query_register.gradient_checkpointing = False
        cls, loss = DrivoRQueryDiverseRegisterAgent, DrivoRQueryRegisterLoss()
    result = cls(cfg, lr_args=dict(name='AdamW', base_lr=.001, base_batch_size=1),
                 checkpoint_path=checkpoint, loss=loss)
    result.compute_score = oracle
    return result


@pytest.fixture
def sample():
    torch.manual_seed(11)
    features = dict(image=torch.randn(1, 1, 3, 32, 32), ego_status=torch.randn(1, 4, 11))
    valid = torch.tensor([[True, True] + [False] * 6])
    gt = torch.randn(1, 8, 8, 3).masked_fill(~valid[..., None, None], float('nan'))
    targets = dict(trajectory=torch.randn(1, 8, 3), pseudo_experts=gt,
        pseudo_expert_mask=valid, diverse_gt_missing=torch.tensor([False]),
        future_register_images=torch.randint(0, 256, (1, 8, 8, 3, 32, 32), dtype=torch.uint8),
        future_register_valid=valid[..., None].expand(-1, -1, 8).clone(),
        human_future_register_images=torch.randint(0, 256, (1, 8, 3, 32, 32), dtype=torch.uint8),
        human_future_register_valid=torch.ones(1, 8, dtype=torch.bool))
    return features, targets


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('scorer', [False, True])
def test_one_forward_original_score_gradients_ema_and_no_gt_inference(monkeypatch, sample, family, scorer, tmp_path):
    variant = family + '_register' + ('_scorer' if scorer else '')
    value = make_agent(monkeypatch, variant).eval()
    model = value._drivor_model
    features, targets = sample
    calls = []
    trunk = model.action_dit if family.startswith('action_dit') else model.query_decoder
    trunk.register_forward_hook(lambda *args: calls.append(1))
    output = value.forward_train(features, targets)
    losses = value.compute_loss(features, targets, output)
    assert calls == [1]
    assert output['proposals'].shape == (1, 4, 8, 3)
    assert output['predicted_registers'].shape == (1, 4, 8, 4, 16)
    assert output['_register_matched_indices'].shape == (1, 9)
    assert (output['_register_matched_indices'][:, 3:] == -1).all()
    losses['final_score_loss'].backward()
    assert all(p.grad is None for p in trunk.parameters())
    head = model.action_dit.register_head if family.startswith('action_dit') else model.register_head
    assert all(p.grad is None for p in head.parameters())
    assert model.image_backbone.neck.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    output = value.forward_train(features, targets)
    losses = value.compute_loss(features, targets, output)
    torch.testing.assert_close(losses['loss'], losses['trajectory_loss'] + losses['future_register_loss'] + losses['final_score_loss'])
    assert losses['future_register_gt_count'] == 3
    assert torch.isfinite(losses['loss'])
    optimizer = value.get_optimizers()[0]
    params = [p for group in optimizer.param_groups for p in group['params']]
    assert len(params) == len({id(p) for p in params})
    assert {id(p) for p in params} == {id(p) for p in model.parameters() if p.requires_grad}
    losses['loss'].backward(); optimizer.step()
    assert model.ema_teacher.updates == 1
    assert all(p.grad is None for p in model.ema_teacher.parameters())
    checkpoint = tmp_path / 'full.ckpt'
    torch.save({'state_dict': {'agent.' + k: v for k, v in value.state_dict().items()}}, checkpoint)
    restored = make_agent(monkeypatch, variant, str(checkpoint))
    restored.initialize(); restored.eval()
    assert len(restored.get_target_builders()) == 1
    monkeypatch.setattr(restored._drivor_model.ema_teacher, 'forward',
        lambda *args: (_ for _ in ()).throw(AssertionError('Inference must not read future teacher/GT')))
    with torch.no_grad():
        torch.manual_seed(13); expected = value(features)
        torch.manual_seed(13); actual = restored(features)
    torch.testing.assert_close(expected['proposals'], actual['proposals'])
    torch.testing.assert_close(expected['pdm_score'], actual['pdm_score'])


@pytest.mark.parametrize('family', FAMILIES)
def test_each_gt_uses_its_own_trajectory_cost_and_same_register_index(monkeypatch, sample, family):
    value = make_agent(monkeypatch, family + '_register').eval()
    model = value._drivor_model
    features, targets = sample
    physical = torch.zeros(1, 4, 8, 3)
    physical[:, 0, :, 0] = 1.
    physical[:, 1, :, 2] = .1
    physical[:, 2, :, 0] = 20.
    physical[:, 3, :, 0] = 50.
    physical.requires_grad_()
    targets['trajectory'].zero_()
    targets['pseudo_experts'][:, :2].zero_()
    targets['pseudo_experts'][:, 0, :, 0] = 20.2
    targets['pseudo_experts'][:, 1, :, 0] = 20.4
    registers = torch.zeros(1, 4, 8, 4, 16, requires_grad=True)
    teacher = torch.zeros(1, 9, 8, 4, 16)
    teacher[:, 0] = 1; teacher[:, 1] = 2; teacher[:, 2] = 4
    monkeypatch.setattr(model.ema_teacher, 'forward', lambda *args: teacher)
    if family.startswith('action_dit'):
        monkeypatch.setattr(model.schedule, 'sample_timesteps', lambda b, m, device: torch.full((b, m), 500., device=device))
        seen = []
        def predict(action, future, times, scene, ego):
            seen.append((action.detach(), future.detach()))
            return normalize_trajectory(physical), registers.flatten(2, 3)
        monkeypatch.setattr(model.action_dit, 'forward', predict)
        output = value.forward_train(features, targets)
        assert len(seen) == 1
        # Independent noises for the four candidates; a single forward for all GTs.
        assert not torch.equal(seen[0][0][:, 0], seen[0][0][:, 1])
        human_winner = 0 if 'human_xl1' not in family else 1
    else:
        output = dict(proposals=physical, predicted_registers=registers)
        model._add_supervision(output, targets)
        human_winner = 1
    assert output['_register_matched_indices'].tolist() == [[human_winner, 2, 2] + [-1] * 6]
    if family == 'query_human_diverse8_l1':
        torch.testing.assert_close(output['_trajectory_loss'], torch.tensor(.25), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(output['_register_loss'], torch.tensor(7 / 3))
    elif 'human_xl1' in family:
        torch.testing.assert_close(output['_flow_loss'], torch.tensor(.25), atol=1e-5, rtol=1e-5)
    output['_register_loss'].backward()
    active = (registers.grad.abs().sum((-3, -2, -1)) > 0)[0].tolist()
    assert active == [i in (human_winner, 2) for i in range(4)]


def test_query_future_addition_preserves_shared_baseline_initialization(monkeypatch):
    from test_query_gt_variants import agent as baseline_agent
    torch.manual_seed(27)
    baseline = baseline_agent(monkeypatch, 'human_diverse8_l1')._drivor_model
    torch.manual_seed(27)
    extended = make_agent(monkeypatch, 'query_human_diverse8_l1_register')._drivor_model
    for name, tensor in baseline.state_dict().items():
        torch.testing.assert_close(tensor, extended.state_dict()[name], atol=0, rtol=0)
