"""Learned-query planning, directed attention, matching and EMA contracts."""

from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from navsim.agents.drivoR.transformer_decoder import TransformerDecoder
from navsim.agents.drivoR_action_dit.action_dit import ActionDiT
from navsim.agents.drivoR_action_dit_register.targets import HumanFutureRegisterTargetBuilder
from navsim.agents.drivoR_query_register import model as model_module
from navsim.agents.drivoR_query_register import targets as target_module
from navsim.agents.drivoR_query_register.agent import DrivoRQueryRegisterAgent
from navsim.agents.drivoR_query_register.decoder import QueryRegisterDecoder, query_attention_mask
from navsim.agents.drivoR_query_register.losses import (
    DrivoRQueryRegisterLoss, match_trajectories, matched_future_loss,
)
from test_action_dit_register import TinyEncoder, oracle, sample


ROOT = Path(__file__).resolve().parents[1]


def config(multigt=True, scorer=False):
    folder = ROOT / 'navsim/planning/script/config/common/agent'
    names = ['drivoR', 'drivoR_query_register_dinov3_vitl',
             'drivoR_query_register_dinov3_vitl_' + ('multigt' if multigt else 'single_gt')]
    cfg = OmegaConf.merge(*[OmegaConf.load(folder / (name + '.yaml')).config for name in names])
    cfg.tf_d_model, cfg.tf_d_ffn = 8, 16
    cfg.ref_num, cfg.scorer_ref_num, cfg.refiner_num_heads = 2, 1, 2
    cfg.proposal_num = 4
    cfg.query_register.gradient_checkpointing = False
    cfg.future_register.teacher_batch_size = 4
    cfg.future_register.sensor_blobs_path = '/sensors'
    cfg.future_register.scorer.enabled = scorer
    cfg.multi_gt.max_gt = 2 if multigt else 1
    return cfg


def agent(monkeypatch, multigt=True, scorer=False):
    monkeypatch.setattr(model_module, 'DinoV3ImageEncoder', TinyEncoder)
    def forbidden(*args, **kwargs):
        raise AssertionError('The ordinary Transformer must never construct an Action-DiT')
    monkeypatch.setattr(ActionDiT, '__init__', forbidden)
    value = DrivoRQueryRegisterAgent(config(multigt, scorer),
        lr_args=dict(name='AdamW', base_lr=.001, base_batch_size=1), loss=DrivoRQueryRegisterLoss())
    value.compute_score = oracle
    return value


@pytest.mark.parametrize('checkpointing', [False, True])
def test_directed_decoder_keeps_drivor_action_result_and_isolates_future_groups(checkpointing):
    cfg = config()
    cfg.query_register.gradient_checkpointing = checkpointing
    decoder = QueryRegisterDecoder(cfg, 32).eval()
    original = TransformerDecoder(proj_drop=.1, drop_path=.2, config=cfg).eval()
    original.load_state_dict(decoder.state_dict(), strict=True)
    action = torch.randn(2, 4, 8, requires_grad=True)
    future = torch.randn(2, 4, 32, 8, requires_grad=True)
    scene = torch.randn(2, 7, 8)
    a, r = decoder(action, future, scene)
    torch.testing.assert_close(a, original(action, scene)[-1], rtol=1e-5, atol=1e-6)
    changed = future.detach().clone()
    changed[:, 1, :, 0] += 7
    a2, r2 = decoder(action, changed, scene)
    torch.testing.assert_close(a, a2, rtol=0, atol=0)
    torch.testing.assert_close(r[:, 0], r2[:, 0], rtol=0, atol=0)
    assert not torch.equal(r[:, 1], r2[:, 1])
    future_gradient, = torch.autograd.grad(a.square().sum(), future, retain_graph=True)
    assert not future_gradient.any()
    action_gradient, = torch.autograd.grad(r.square().sum(), action)
    assert action_gradient.abs().sum() > 0
    if checkpointing:
        decoder.train()
        a, r = decoder(action, future, scene)
        (a.square().mean() + r.square().mean()).backward()
        assert decoder.layers[0].self_attn.attn.in_proj_weight.grad.abs().sum() > 0


def test_mask_allows_optional_bidirectional_reads_only_within_the_candidate():
    mask = query_attention_mask(3, 32, 'bidirectional')
    assert not mask[1, 3 + 32:3 + 64].any()
    assert mask[1, 3:3 + 32].all()
    assert mask[3:3 + 32, 1:3].all()
    assert not mask[:3, :3].any()


@pytest.mark.parametrize('multigt', [False, True])
@pytest.mark.parametrize('scorer', [False, True])
def test_one_pass_predictions_do_not_depend_on_gt_and_teacher_is_inactive_at_inference(monkeypatch, sample, multigt, scorer):
    features, targets = sample
    value = agent(monkeypatch, multigt, scorer)
    model = value._drivor_model
    assert not hasattr(model, 'action_dit') and not hasattr(model, 'schedule')
    calls, teachers = [], []
    model.query_decoder.register_forward_hook(lambda module, args, out: calls.append(1))
    model.ema_teacher.register_forward_hook(lambda module, args, out: teachers.append(args[0].shape))
    torch.manual_seed(43)
    first = value.forward_train(features, targets)
    assert len(calls) == len(teachers) == 1
    assert teachers[0][1] == (2 if multigt else 1)
    assert first['proposals'].shape == (1, 4, 8, 3)
    assert first['predicted_registers'].shape == (1, 4, 8, 4, 16)
    changed = dict(targets, trajectory=targets['trajectory'] + 100,
                   trajectory_multi_gt=targets['trajectory_multi_gt'] + 100,
                   future_register_images=255 - targets['future_register_images'],
                   human_future_register_images=255 - targets['human_future_register_images'])
    torch.manual_seed(43)
    second = value.forward_train(features, changed)
    for key in ('proposals', 'predicted_registers', 'pdm_score'):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    losses = value.compute_loss(features, targets, first)
    torch.testing.assert_close(losses['loss'], first['_trajectory_loss'] + first['_register_loss'] + losses['final_score_loss'])
    assert torch.isfinite(losses['loss'])
    value.eval()
    model.ema_teacher.forward = lambda *a, **k: (_ for _ in ()).throw(AssertionError('No inference teacher'))
    with torch.no_grad():
        torch.manual_seed(1)
        a = value(features)
        torch.manual_seed(999)
        b = value(features)
    for key in ('proposals', 'predicted_registers', 'pdm_score'):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert '_trajectory_loss' not in a and len(calls) == 4


@pytest.mark.parametrize('future_loss,expected', [('l1', 2.), ('mse', 5.)])
def test_shared_winner_only_receives_losses_and_valid_frames_and_gts_are_averaged(future_loss, expected):
    proposals = torch.zeros(1, 3, 8, 3)
    proposals[:, 0] = 10
    proposals[:, 2] = -10
    proposals.requires_grad_()
    gt = torch.ones(1, 3, 8, 3)
    gt[:, 1] = 2
    gt[:, 2] = float('nan')
    gt.requires_grad_()
    valid = torch.tensor([[True, True, False]])
    trajectory, indices = match_trajectories(proposals, gt, valid)
    assert indices.tolist() == [[1, 1, -1]]
    torch.testing.assert_close(trajectory, torch.tensor(4.5))
    trajectory.backward()
    assert proposals.grad[:, 1].abs().sum() > 0
    assert not proposals.grad[:, [0, 2]].any() and gt.grad is None
    predicted = torch.zeros(1, 3, 8, 4, 2, requires_grad=True)
    target = torch.ones(1, 3, 8, 4, 2)
    target[:, 1] = 3
    frames = valid[:, :, None].expand(-1, -1, 8).clone()
    frames[:, 1, 2:] = False
    target[~frames] = float('nan')
    target.requires_grad_()
    future = matched_future_loss(predicted, target, frames, indices, future_loss)
    torch.testing.assert_close(future, torch.tensor(expected))
    future.backward()
    assert predicted.grad[:, 1].abs().sum() > 0
    assert not predicted.grad[:, [0, 2]].any() and target.grad is None


@pytest.mark.parametrize('scorer', [False, True])
def test_score_loss_detaches_both_predictions_but_trains_vision_ego_and_scorer(monkeypatch, sample, scorer):
    features, targets = sample
    value = agent(monkeypatch, scorer=scorer)
    # In a one-scene test DropPath may legitimately drop the entire visual
    # cross-attention residual. Disable it to test the structural gradient path.
    value._drivor_model.scorer_attention.eval()
    prediction = value.forward_train(features, targets)
    value.compute_loss(features, targets, prediction)['final_score_loss'].backward()
    model = value._drivor_model
    for module in (model.action_queries, model.query_decoder, model.trajectory_head, model.register_head, model.ema_teacher):
        assert all(p.grad is None for p in module.parameters())
    assert model.future_queries.grad is None
    assert model.image_backbone.neck.weight.grad.abs().sum() > 0
    assert model.hist_encoding.weight.grad.abs().sum() > 0
    assert model.scorer.pred_score['comfort'][0].weight.grad.abs().sum() > 0


def test_future_loss_reaches_query_representation_and_lora_but_not_action_head_or_teacher(monkeypatch, sample):
    value = agent(monkeypatch)
    features, targets = sample
    output = value.forward_train(features, targets)
    output['_register_loss'].backward()
    model = value._drivor_model
    assert model.action_queries.weight.grad.abs().sum() > 0
    assert model.future_queries.grad.abs().sum() > 0
    assert model.query_decoder.layers[0].self_attn.attn.in_proj_weight.grad.abs().sum() > 0
    assert model.image_backbone.model.blocks[0].attn.qkv.q_b.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.trajectory_head.parameters())
    assert all(p.grad is None for p in model.scorer.parameters())
    assert all(p.grad is None for p in model.ema_teacher.parameters())


def test_ema_updates_after_optimizer_and_missing_future_keeps_a_zero_loss_graph(monkeypatch, sample):
    value = agent(monkeypatch)
    features, targets = sample
    targets['future_register_valid'].zero_()
    optimizer = value.get_optimizers()[0]
    model = value._drivor_model
    params = [p for group in optimizer.param_groups for p in group['params']]
    assert len(params) == len({id(p) for p in params})
    assert {id(p) for p in params} == {id(p) for p in model.parameters() if p.requires_grad}
    prediction = value.forward_train(features, targets)
    assert prediction['_register_loss'] == 0
    value.compute_loss(features, targets, prediction)['loss'].backward()
    assert model.future_queries.grad is not None and not model.future_queries.grad.any()
    assert model.ema_teacher.updates == 0
    optimizer.step()
    assert model.ema_teacher.updates == 1


@pytest.mark.parametrize('scorer', [False, True])
def test_single_and_multi_checkpoints_share_architecture_and_single_builder_needs_no_cosmos(monkeypatch, sample, scorer):
    single = agent(monkeypatch, multigt=False, scorer=scorer)
    multi = agent(monkeypatch, multigt=True, scorer=scorer)
    multi.load_state_dict(single.state_dict(), strict=True)
    single._config.multi_gt.root = '/absent/gt'
    single._config.future_register.frames_root = '/absent/videos'
    builders = single.get_target_builders()
    assert len(builders) == 2 and isinstance(builders[1], HumanFutureRegisterTargetBuilder)
    single._checkpoint_path = '/checkpoint.ckpt'
    assert len(single.get_target_builders()) == 1
    with torch.no_grad():
        a = single.eval()(sample[0])
        b = multi.eval()(sample[0])
    torch.testing.assert_close(a['proposals'], b['proposals'], rtol=0, atol=0)


def test_multigt_targets_skip_unused_human_anchor_and_pair_human_fallback(monkeypatch):
    builder = target_module.QueryFutureRegisterTargetBuilder.__new__(target_module.QueryFutureRegisterTargetBuilder)
    builder.image_size, builder.sensor_blobs_path = (8, 8), Path('/sensors')
    def result(fallback):
        return dict(multi_gt_fallback=torch.tensor(fallback),
                    future_register_images=torch.zeros(2, 8, 3, 8, 8, dtype=torch.uint8),
                    future_register_valid=torch.zeros(2, 8, dtype=torch.bool))
    builder.load = lambda *args: result(False)
    scene = SimpleNamespace(scene_metadata=SimpleNamespace(initial_token='scene'),
                            get_future_trajectory=lambda **kwargs: SimpleNamespace(poses=None))
    def forbidden(*args):
        raise AssertionError('Counterfactual targets do not need an extra human noise anchor')
    monkeypatch.setattr(target_module, '_human_future_targets', forbidden)
    assert not builder.compute_targets(scene)['future_register_valid'].any()
    builder.load = lambda *args: result(True)
    monkeypatch.setattr(target_module, '_human_future_targets', lambda *args: {
        'human_future_register_images': torch.ones(8, 3, 8, 8, dtype=torch.uint8),
        'human_future_register_valid': torch.ones(8, dtype=torch.bool)})
    fallback = builder.compute_targets(scene)
    assert fallback['future_register_valid'][0].all()
    assert not fallback['future_register_valid'][1].any()
