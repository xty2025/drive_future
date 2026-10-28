"""V-JEPA 2 image adaptation, native RoPE parity and frozen-base LoRA contracts."""

from functools import partial
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file
from torch import nn

from navsim.agents.drivoR_action_dit import vjepa2_image_encoder as encoder_module


ROOT = Path(__file__).resolve().parents[1]


def tiny_model():
    native, _ = encoder_module.official_vjepa2_modules()
    return native.VisionTransformer(img_size=(32, 32), patch_size=16, num_frames=64,
                                    tubelet_size=2, embed_dim=48, depth=2, num_heads=4,
                                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                    use_rope=True, use_sdpa=True, handle_nonsquare_inputs=True)


@pytest.fixture
def encoder_factory(tmp_path, monkeypatch):
    reference = tiny_model().eval()
    path = tmp_path / 'target_encoder.safetensors'
    save_file(reference.state_dict(), str(path))
    monkeypatch.setattr(encoder_module, 'build_official_vjepa2', lambda name: tiny_model())
    def make(*, cameras=1, queries=0, checkpointing=True):
        cfg = OmegaConf.create(dict(model_name='vjepa2_vit_large', model_weights=str(path),
                                    use_lora=True, lora_rank=4, use_grid_mask=False,
                                    gradient_checkpointing=checkpointing))
        return encoder_module.VJepa2ImageEncoder(
            cfg, [48, 32], 8, cameras, queries, 'drivor_queries' if queries else 'native_patches')
    return make, reference


def test_image_is_repeated_into_one_tubelet_and_matches_official_features(encoder_factory):
    make, reference = encoder_factory
    encoder = make().eval()
    images = torch.randn(2, 1, 3, 32, 48)
    observed = []
    hook = encoder.native_vit.patch_embed.register_forward_pre_hook(lambda module, args: observed.append(args[0]))
    try:
        with torch.no_grad():
            actual = encoder(images)
            video = images[:, 0].unsqueeze(2).repeat(1, 1, 2, 1, 1)
            expected = encoder.neck(reference(video))
    finally:
        hook.remove()
    torch.testing.assert_close(observed[0], video, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert actual.shape == (2, 6, 8)
    assert encoder.native_vit.pos_embed is None and encoder.num_prefix_tokens == 0
    assert encoder.scene_embeds is None


def test_query_attention_zero_prefix_matches_native_rope(encoder_factory):
    make, _ = encoder_factory
    attention = make().native_vit.blocks[0].attn
    x = torch.randn(2, 6, 48)
    expected = attention(x, T=1, H_patches=2, W_patches=3)
    actual = encoder_module.query_attention(attention, x, 0, 1, 2, 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('cameras', [1, 4])
def test_queries_are_unrotated_and_cameras_are_independent_videos(encoder_factory, monkeypatch, cameras):
    make, _ = encoder_factory
    encoder = make(cameras=cameras, queries=16, checkpointing=False).train()
    _, native = encoder_module.official_vjepa2_modules()
    rotate = native.rotate_queries_or_keys
    rotated_lengths, positions = [], []
    def capture(x, pos):
        rotated_lengths.append(x.shape[-2]); positions.append(pos.detach().clone())
        return rotate(x, pos)
    monkeypatch.setattr(native, 'rotate_queries_or_keys', capture)
    images = torch.randn(1, cameras, 3, 32, 48)
    videos, lengths = [], []
    stem_hook = encoder.native_vit.patch_embed.register_forward_pre_hook(lambda module, args: videos.append(args[0]))
    qkv = encoder.native_vit.blocks[0].attn.qkv
    q_hook = qkv.register_forward_pre_hook(lambda module, args: lengths.append(args[0].shape[1]))
    try:
        output = encoder(images)
        (output - torch.randn_like(output)).square().mean().backward()
    finally:
        stem_hook.remove(); q_hook.remove()
    assert videos[0].shape == (cameras, 3, 2, 32, 48)
    torch.testing.assert_close(videos[0][:, :, 0], images[0], rtol=0, atol=0)
    torch.testing.assert_close(videos[0][:, :, 1], images[0], rtol=0, atol=0)
    assert lengths[0] == 22 and set(rotated_lengths) == {6}  # 16 queries + 6 patches.
    assert positions[0].tolist() == [0] * 6
    assert positions[2].tolist() == [0, 0, 0, 1, 1, 1]
    assert positions[4].tolist() == [0, 1, 2, 0, 1, 2]
    assert output.shape == (1, cameras * 16, 8)
    assert encoder.scene_embeds.grad.abs().sum() > 0
    assert qkv.q_b.weight.grad.abs().sum() > 0 and qkv.v_b.weight.grad.abs().sum() > 0
    assert encoder.neck.weight.grad.abs().sum() > 0
    assert not encoder.native_vit.training and encoder.training
    for name, parameter in encoder.native_vit.named_parameters():
        if not any(f'.{part}.' in name for part in ('q_a', 'q_b', 'v_a', 'v_b')):
            assert not parameter.requires_grad and parameter.grad is None, name


@pytest.mark.parametrize('queries', [0, 16])
def test_checkpointing_preserves_outputs_and_adaptation_gradients(encoder_factory, queries):
    make, _ = encoder_factory
    plain = make(queries=queries, checkpointing=False).train()
    checked = make(queries=queries, checkpointing=True).train()
    checked.load_state_dict(plain.state_dict())
    images = torch.randn(1, 1, 3, 32, 48)
    a, b = plain(images), checked(images)
    torch.testing.assert_close(a, b)
    target = torch.randn_like(a)
    (a - target).square().mean().backward(); (b - target).square().mean().backward()
    for (_, p), (_, q) in zip(plain.named_parameters(), checked.named_parameters()):
        if p.requires_grad:
            assert p.grad is not None and q.grad is not None
            torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-5)


def test_vjepa_config_preserves_dinov3_training_recipe_except_backbone():
    folder = ROOT / 'navsim/planning/script/config/common/agent'
    base = OmegaConf.merge(*[OmegaConf.load(folder / (name + '.yaml')).config for name in (
        'drivoR', 'drivoR_action_dit', 'drivoR_action_dit_multigt')])
    dino = OmegaConf.merge(base, OmegaConf.load(folder / 'drivoR_action_dit_dinov3_vitl.yaml').config)
    vjepa = OmegaConf.merge(base, OmegaConf.load(folder / 'drivoR_action_dit_vjepa2_vitl.yaml').config)
    for key in dino:
        if key != 'image_backbone':
            assert dino[key] == vjepa[key], key
    assert vjepa.image_backbone.family == 'vjepa2'
    assert vjepa.image_backbone.lora_rank == 32 and not vjepa.image_backbone.finetune
    assert vjepa.image_size == [1152, 672]
    assert vjepa.multi_gt.weighting == 'equal' and vjepa.multi_gt.matching == 'nearest'
