"""DrivoR's four-camera query features, shared by Action-DiT and its scorer."""

from copy import deepcopy
from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch
from torch import nn
from timm.models.vision_transformer import VisionTransformer

from navsim.agents.drivoR.drivor_model import DrivoRModel
from navsim.agents.drivoR_action_dit.model import DrivoRActionDiTModel
from navsim.agents.drivoR_action_dit import image_encoder
from navsim.common.dataclasses import SensorConfig


def configs():
    folder = Path(__file__).resolve().parents[1] / 'navsim/planning/script/config/common/agent'
    base = OmegaConf.load(folder / 'drivoR.yaml')
    action = OmegaConf.load(folder / 'drivoR_action_dit.yaml')
    multiview = OmegaConf.load(folder / 'drivoR_action_dit_multiview.yaml')
    original = OmegaConf.merge(base, action)
    changed = OmegaConf.merge(original, multiview)
    return original, changed


def tiny_model(monkeypatch):
    _, config = configs()
    config = config.config
    config.image_size = [28, 28]
    config.tf_d_model, config.tf_d_ffn = 8, 16
    config.image_backbone.use_grid_mask = False
    config.action_dit.update(dict(hidden_dim=8, ffn_dim=16, freq_dim=8, num_heads=2,
                                  attn_head_dim=4, num_layers=2, pretrained_path=''))

    def make_vit(*args, **kwargs):
        return VisionTransformer(img_size=kwargs['img_size'], patch_size=14,
            embed_dim=8, depth=1, num_heads=2, mlp_ratio=2, num_classes=0,
            reg_tokens=4, no_embed_class=True)

    monkeypatch.setattr(image_encoder.timm, 'create_model', make_vit)
    return DrivoRActionDiTModel(config)


def batch(batch_size=1):
    features = {'image': torch.randn(batch_size, 4, 3, 28, 28),
                'ego_status': torch.randn(batch_size, 4, 11)}
    features['ego_status'][:, -1, :3] = 0
    targets = {'trajectory': torch.randn(batch_size, 8, 3),
               'trajectory_long': torch.randn(batch_size, 8, 3)}
    return features, targets


def test_configuration_only_changes_visual_input():
    native, multi = configs()
    assert native.config.visual_feature_mode == 'native_patches'
    assert native.config.cam_l0 == native.config.cam_r0 == native.config.cam_b0 == []
    assert multi.config.visual_feature_mode == 'drivor_queries'
    assert multi.config.num_scene_tokens == 16
    camera_names = ('cam_f0', 'cam_l0', 'cam_l1', 'cam_l2', 'cam_r0', 'cam_r1', 'cam_r2', 'cam_b0')
    sensors = SensorConfig(**{name: list(multi.config[name]) for name in (*camera_names, 'lidar_pc')})
    assert set(sensors.get_sensors_at_iteration(3)) == {'cam_f0', 'cam_l0', 'cam_r0', 'cam_b0'}
    assert all(sensors.get_sensors_at_iteration(index) == [] for index in (0, 1, 2, 4))
    comparable = deepcopy(multi.config)
    for key in ('visual_feature_mode', 'num_scene_tokens', *camera_names):
        comparable[key] = native.config[key]
    assert OmegaConf.to_container(comparable) == OmegaConf.to_container(native.config)
    assert OmegaConf.to_container(multi.loss) == OmegaConf.to_container(native.loss)
    assert multi._target_ == native._target_


def test_query_features_match_the_original_drivor_model(monkeypatch):
    model = tiny_model(monkeypatch).eval()
    reference = DrivoRModel(deepcopy(model._config)).eval()
    reference.image_backbone.load_state_dict(model.image_backbone.encoder.state_dict())
    with torch.no_grad():
        reference.scene_embeds.copy_(model.image_backbone.scene_embeds)
    features, _ = batch()
    original_context = []
    handle = reference.trajectory_decoder.register_forward_pre_hook(
        lambda module, args: original_context.append(args[1].detach().clone()))
    try:
        with torch.no_grad():
            reference(features)
            actual, _ = model.encode_scene(features)
    finally:
        handle.remove()
    assert actual.shape == (1, 64, 8)
    torch.testing.assert_close(actual, original_context[0], rtol=0, atol=0)


def test_queries_enter_dino_and_each_camera_keeps_its_own_16_outputs(monkeypatch):
    model = tiny_model(monkeypatch).eval()
    assert model.image_backbone.scene_embeds.shape == (1, 4, 16, 8)
    inputs = []
    handle = model.image_backbone.native_vit.blocks[0].register_forward_pre_hook(
        lambda module, args: inputs.append(tuple(args[0].shape)))
    features, _ = batch()
    try:
        with torch.no_grad():
            original, _ = model.encode_scene(features)
            changed_images = features['image'].clone()
            changed_images[:, 3] = torch.randn_like(changed_images[:, 3])
            changed, _ = model.encode_scene(dict(features, image=changed_images))
    finally:
        handle.remove()
    # Tiny image: 16 queries + CLS + 4 registers + 4 patches, independently per camera.
    assert inputs == [(4, 25, 8), (4, 25, 8)]
    torch.testing.assert_close(original[:, :48], changed[:, :48], rtol=0, atol=0)
    assert not torch.allclose(original[:, 48:], changed[:, 48:])


def test_dit_and_scorer_share_64_query_features_and_train_in_one_pass(monkeypatch):
    model = tiny_model(monkeypatch).train()
    features, targets = batch()
    dit_inputs, action_context, scorer_context = [], [], []
    handles = [
        model.action_dit.register_forward_pre_hook(
            lambda module, args: dit_inputs.append(tuple(args[0].shape))),
        model.action_dit.blocks[0].cross_attn.k.register_forward_pre_hook(
            lambda module, args: action_context.append(args[0])),
        model.scorer_attention.register_forward_pre_hook(
            lambda module, args: scorer_context.append(args[1])),
    ]
    monkeypatch.setattr(model, 'sample_normalized', lambda *a, **k: pytest.fail('Training ran Euler'))
    try:
        with torch.autocast('cpu', dtype=torch.bfloat16):
            result = model(features, targets=targets)
            assert result['_scene_context'].shape == (1, 64, 8)
            assert action_context[0] is scorer_context[0] is result['_scene_context']
            loss = result['_flow_loss'] + sum(x.float().square().mean() for x in result['pred_logit'].values())
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    assert dit_inputs == [(1, 128, 8, 3)]
    assert result['proposals'].shape == (1, 64, 8, 3)
    assert not result['proposals'].requires_grad
    assert torch.isfinite(loss)
    assert model.image_backbone.scene_embeds.grad.abs().sum() > 0
    assert model.image_backbone.encoder.neck.weight.grad.abs().sum() > 0
    assert model.action_dit.head.weight.grad.abs().sum() > 0
    assert model.scorer.pred_score['comfort'][0].weight.grad.abs().sum() > 0


def test_score_gradient_trains_queries_but_not_action_dit(monkeypatch):
    model = tiny_model(monkeypatch).train()
    features, targets = batch()
    output = model(features, targets=targets)
    sum(value.square().mean() for value in output['pred_logit'].values()).backward()
    assert all(parameter.grad is None for parameter in model.action_dit.parameters())
    assert model.image_backbone.scene_embeds.grad.abs().sum() > 0
    assert model.hist_encoding.weight.grad.abs().sum() > 0
