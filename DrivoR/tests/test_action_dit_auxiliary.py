"""Action-DiT Bank/coverage integration, gradient boundaries and baseline parity."""

from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from navsim.agents.drivoR_action_dit import model as model_module
from navsim.agents.drivoR_action_dit.agent import DrivoRActionDiTAgent
from navsim.agents.drivoR_action_dit.auxiliary import DrivoRActionDiTAuxiliaryAgent
from navsim.agents.drivoR_action_dit.loss import DrivoRActionDiTLoss
from navsim.agents.drivoR_pseudo_expert.loss import multi_expert_coverage_loss


CONFIGS = Path(__file__).resolve().parents[1] / 'navsim/planning/script/config/common/agent'


def configuration(variant=None):
    names = ['drivoR', 'drivoR_action_dit', 'drivoR_action_dit_multigt', 'drivoR_action_dit_dinov3_vitl']
    if variant == 'pseudo_expert_vmse':
        names.append('drivoR_action_dit_dinov3_vitl_pseudo_expert')
    if variant:
        names.append('drivoR_action_dit_dinov3_vitl_' + variant)
    cfg = OmegaConf.merge(*[OmegaConf.load(CONFIGS / (name + '.yaml')).config for name in names])
    cfg.tf_d_model, cfg.tf_d_ffn, cfg.scorer_ref_num = 8, 16, 2
    cfg.action_dit.update(dict(hidden_dim=8, ffn_dim=16, freq_dim=8, num_heads=2,
                               attn_head_dim=4, num_layers=2, pretrained_path='',
                               num_inference_steps=2, gradient_checkpointing=False))
    return cfg


class TinyImageEncoder(nn.Module):
    def __init__(self, config, image_size, output_dim, *args):
        super().__init__()
        self.neck = nn.Linear(3, output_dim)

    def forward(self, images):
        return self.neck(images.mean((-1, -2)))


@pytest.fixture
def fixture(monkeypatch):
    monkeypatch.setattr(model_module, 'DinoV3ImageEncoder', TinyImageEncoder)
    torch.manual_seed(17)
    features = dict(image=torch.randn(2, 1, 3, 32, 32), ego_status=torch.randn(2, 4, 11))
    targets = dict(trajectory=torch.randn(2, 8, 3), trajectory_multi_gt=torch.randn(2, 4, 8, 3),
                   trajectory_multi_gt_valid=torch.tensor([[True,True,True,True],[True,False,False,False]]),
                   bank_candidates=torch.randn(2, 64, 8, 3, requires_grad=True),
                   bank_subscores=torch.rand(2, 64, 7), bank_valid=torch.ones(2, 64, dtype=torch.bool),
                   pseudo_experts=torch.randn(2, 8, 8, 3, requires_grad=True),
                   pseudo_expert_mask=torch.ones(2, 8, dtype=torch.bool),
                   pseudo_expert_fallback=torch.zeros(2, dtype=torch.bool))
    return features, targets


def agent(variant=None):
    cls = DrivoRActionDiTAuxiliaryAgent if variant else DrivoRActionDiTAgent
    value = cls(configuration(variant), lr_args={}, loss=DrivoRActionDiTLoss())
    calls = []

    def oracle(targets, proposals, test=True):
        assert proposals.shape == (2, 64, 8, 3) and not test
        calls.append(tuple(proposals.shape))
        raw = torch.full((2, 64, 7), .75)
        return raw[..., -1], raw[..., -1].amax(1), raw, None, None, None

    value.compute_score = oracle
    return value, calls


@pytest.mark.parametrize('variant', ['bank', 'pseudo_expert', 'pseudo_expert_vmse'])
def test_auxiliary_keeps_same_model_predictions_and_multi_gt_loss(fixture, variant):
    features, targets = fixture
    original, _ = agent()
    extra, oracle_calls = agent(variant)
    extra.load_state_dict(original.state_dict(), strict=True)
    torch.manual_seed(23)
    a = original.forward_train(features, targets)
    torch.manual_seed(23)
    b = extra.forward_train(features, targets)
    for key in ('proposals', 'pdm_score', '_flow_loss'):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    old = original.compute_loss(features, targets, a)
    new = extra.compute_loss(features, targets, b)
    loss_name = 'bank_loss' if variant == 'bank' else 'pseudo_expert_loss'
    torch.testing.assert_close(new['loss'], old['loss'] + .5 * new[loss_name])
    for key in ('flow_multi_gt_loss', 'final_score_loss'):
        torch.testing.assert_close(new[key], old[key], rtol=0, atol=0)
    assert len(oracle_calls) == 1
    assert ('pseudo_expert_loss' in new) == variant.startswith('pseudo_expert')
    assert ('bank_loss' in new) == (variant == 'bank')


@pytest.mark.parametrize('variant,scorer_sizes', [('bank',[64,16]), ('pseudo_expert',[64]), ('pseudo_expert_vmse',[64])])
def test_one_visual_and_dit_forward_with_shared_scorer(fixture, variant, scorer_sizes):
    features, targets = fixture
    value, _ = agent(variant)
    model = value._drivor_model
    calls = dict(image=0, dit=0, scorer=[])

    def count(name):
        def hook(module, args):
            calls[name] += 1
        return hook

    hooks = [model.image_backbone.register_forward_pre_hook(count('image')),
             model.action_dit.register_forward_pre_hook(count('dit')),
             model.scorer.register_forward_pre_hook(lambda module,args:calls['scorer'].append(args[0].shape[1]))]
    prediction = value.forward_train(features, targets)
    result = value.compute_loss(features, targets, prediction)
    result['loss'].backward()
    for hook in hooks:
        hook.remove()
    assert calls == dict(image=1, dit=1, scorer=scorer_sizes)
    assert torch.isfinite(result['loss'])


def test_bank_gradients_train_same_scorer_scene_and_ego_but_not_dit(fixture):
    features, targets = fixture
    value, _ = agent('bank')
    model = value._drivor_model
    prediction = value.forward_train(features, targets)
    value.compute_loss(features, targets, prediction)['bank_loss'].backward()
    assert targets['bank_candidates'].grad is None
    assert all(parameter.grad is None for parameter in model.action_dit.parameters())
    assert model.image_backbone.neck.weight.grad.abs().sum() > 0
    assert model.hist_encoding.weight.grad.abs().sum() > 0
    assert model.pos_embed[0].weight.grad.abs().sum() > 0
    assert model.scorer.pred_score['comfort'][0].weight.grad.abs().sum() > 0


def test_coverage_retains_x_prediction_gradient_and_uses_physical_l1(fixture):
    features, targets = fixture
    value, _ = agent('pseudo_expert')
    model = value._drivor_model
    prediction = value.forward_train(features, targets)
    assert prediction['_pseudo_expert_proposals'].requires_grad
    assert not prediction['proposals'].requires_grad
    torch.testing.assert_close(prediction['_pseudo_expert_proposals'], prediction['proposals'], rtol=0, atol=0)
    result = value.compute_loss(features, targets, prediction)
    expected = multi_expert_coverage_loss(prediction['proposals'], targets['pseudo_experts'], targets['pseudo_expert_mask'])
    torch.testing.assert_close(result['pseudo_expert_loss'], expected)
    result['pseudo_expert_loss'].backward()
    assert targets['pseudo_experts'].grad is None
    assert model.action_dit.head.weight.grad.abs().sum() > 0
    assert model.image_backbone.neck.weight.grad.abs().sum() > 0
    assert model.hist_encoding.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.scorer.parameters())
    assert all(parameter.grad is None for parameter in model.scorer_attention.parameters())


@pytest.mark.parametrize('variant', ['pseudo_expert', 'pseudo_expert_vmse'])
def test_experts_do_not_change_noisy_input_or_original_multi_gt_supervision(fixture, variant):
    features, targets = fixture
    value, _ = agent(variant)
    torch.manual_seed(29)
    a = value.forward_train(features, targets)
    changed = dict(targets, pseudo_experts=targets['pseudo_experts'].detach() + 100)
    torch.manual_seed(29)
    b = value.forward_train(features, changed)
    torch.testing.assert_close(a['proposals'], b['proposals'], rtol=0, atol=0)
    torch.testing.assert_close(a['_flow_loss'], b['_flow_loss'], rtol=0, atol=0)


@pytest.mark.parametrize('variant', ['bank', 'pseudo_expert', 'pseudo_expert_vmse'])
def test_fp32_inference_is_checkpoint_compatible_and_needs_no_auxiliary_cache(fixture, variant):
    features, _ = fixture
    value, _ = agent(variant)
    baseline, _ = agent()
    baseline.load_state_dict(value.state_dict(), strict=True)
    value.eval(); baseline.eval()
    value._checkpoint_path = '/inference/checkpoint.ckpt'
    value._config.multi_gt.enabled = False
    key = 'pseudo_expert' if variant.startswith('pseudo_expert') else variant
    value._config[key].path = '/absent/training/cache'
    assert len(value.get_target_builders()) == 1
    with torch.no_grad():
        torch.manual_seed(31)
        a = value(features)
        torch.manual_seed(31)
        b = baseline(features)
    assert '_pseudo_expert_proposals' not in a
    assert '_pseudo_expert_velocity_loss' not in a
    assert a['proposals'].dtype == torch.float32
    torch.testing.assert_close(a['proposals'], b['proposals'], rtol=0, atol=0)
    torch.testing.assert_close(a['pdm_score'], b['pdm_score'], rtol=0, atol=0)


@pytest.mark.parametrize('variant', ['bank', 'pseudo_expert', 'pseudo_expert_vmse'])
def test_configs_only_add_one_auxiliary_objective(variant):
    cfg = configuration(variant)
    base = configuration()
    key = 'pseudo_expert' if variant.startswith('pseudo_expert') else variant
    options = cfg.pop(key)
    assert options.enabled
    if variant == 'pseudo_expert_vmse':
        assert options.loss_type == 'velocity'
    assert OmegaConf.to_container(cfg, resolve=False) == OmegaConf.to_container(base, resolve=False)
    assert cfg.multi_gt.enabled and cfg.multi_gt.weighting == 'equal'
    assert cfg.multi_gt.matching == 'nearest' and cfg.multi_gt.input_gt_assignment == 'original'
