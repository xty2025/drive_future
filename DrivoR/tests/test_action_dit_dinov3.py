"""Native DINOv3 loading, prefix/RoPE layout and frozen-base LoRA contracts."""

from pathlib import Path
import sys

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file
from torch.nn import functional as F

from navsim.agents.drivoR_action_dit import dinov3_image_encoder as encoder_module


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party/dinov3'))
from dinov3.models.vision_transformer import DinoVisionTransformer


def tiny_model():
    model = DinoVisionTransformer(img_size=32, patch_size=16, embed_dim=32, depth=2,
                                  num_heads=4, ffn_ratio=4, n_storage_tokens=4,
                                  layerscale_init=1e-5, mask_k_bias=True,
                                  norm_layer='layernormbf16', pos_embed_rope_dtype='fp32',
                                  pos_embed_rope_rescale_coords=2)
    model.init_weights()
    return model


def export_hf(model):
    native = model.state_dict()
    result = {'embeddings.cls_token': native['cls_token'],
              'embeddings.register_tokens': native['storage_tokens'],
              'embeddings.mask_token': native['mask_token'].unsqueeze(1),
              'embeddings.patch_embeddings.weight': native['patch_embed.proj.weight'],
              'embeddings.patch_embeddings.bias': native['patch_embed.proj.bias'],
              'norm.weight': native['norm.weight'], 'norm.bias': native['norm.bias']}
    for index in range(len(model.blocks)):
        src, dst = f'blocks.{index}.', f'layer.{index}.'
        q, k, v = native[src + 'attn.qkv.weight'].chunk(3)
        for part, value in zip(('q', 'k', 'v'), (q, k, v)):
            result[dst + f'attention.{part}_proj.weight'] = value.contiguous()
        q_bias, _, v_bias = native[src + 'attn.qkv.bias'].chunk(3)
        result[dst + 'attention.q_proj.bias'] = q_bias.contiguous()
        result[dst + 'attention.v_proj.bias'] = v_bias.contiguous()
        for norm in ('norm1', 'norm2'):
            for kind in ('weight', 'bias'):
                result[dst + norm + '.' + kind] = native[src + norm + '.' + kind]
        for origin, target in (('attn.proj', 'attention.o_proj'), ('mlp.fc1', 'mlp.up_proj'),
                               ('mlp.fc2', 'mlp.down_proj')):
            for kind in ('weight', 'bias'):
                result[dst + target + '.' + kind] = native[src + origin + '.' + kind]
        result[dst + 'layer_scale1.lambda1'] = native[src + 'ls1.gamma']
        result[dst + 'layer_scale2.lambda1'] = native[src + 'ls2.gamma']
    return {key: value.clone() for key, value in result.items()}


@pytest.fixture
def encoder_factory(tmp_path, monkeypatch):
    reference = tiny_model().eval()
    path = tmp_path / 'model.safetensors'
    save_file(export_hf(reference), str(path))
    monkeypatch.setattr(encoder_module, 'build_official_dinov3', lambda name: tiny_model())
    def make(*, width=48, height=32, cameras=1, queries=0, checkpointing=True):
        cfg = OmegaConf.create(dict(model_name='dinov3_vits16', model_weights=str(path),
                                    use_lora=True, lora_rank=4, use_grid_mask=False,
                                    gradient_checkpointing=checkpointing))
        mode = 'drivor_queries' if queries else 'native_patches'
        return encoder_module.DinoV3ImageEncoder(cfg, [width, height], 8, cameras, queries, mode)
    return make, reference


def test_modelscope_conversion_and_native_feature_forward_match_official(encoder_factory):
    make, reference = encoder_factory
    encoder = make().eval()
    images = torch.randn(2, 1, 3, 32, 48)
    with torch.no_grad():
        expected = encoder.neck(reference.forward_features(images[:, 0])['x_norm_patchtokens'])
        actual = encoder(images)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert actual.shape == (2, 6, 8)
    assert encoder.scene_embeds is None
    assert encoder.load_report['consumed_tensors'] == encoder.load_report['source_tensors']
    # Compare Q/K/V projections against independently split HF-format linears.
    source = export_hf(reference)
    tokens = torch.randn(2, 11, 32)
    expected_qkv = torch.cat((F.linear(tokens, source['layer.0.attention.q_proj.weight'],
                                      source['layer.0.attention.q_proj.bias']),
                              F.linear(tokens, source['layer.0.attention.k_proj.weight']),
                              F.linear(tokens, source['layer.0.attention.v_proj.weight'],
                                       source['layer.0.attention.v_proj.bias'])), dim=-1)
    torch.testing.assert_close(encoder.native_vit.blocks[0].attn.qkv(tokens), expected_qkv)


def test_input_goes_directly_to_native_patch_embed_without_padding(encoder_factory):
    make, reference = encoder_factory
    encoder = make(width=47, height=31).eval()
    images = torch.randn(1, 1, 3, 31, 47)
    captured = []
    hook = encoder.native_vit.patch_embed.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    try:
        with torch.no_grad():
            actual = encoder(images)
            expected = encoder.neck(reference.forward_features(images[:, 0])['x_norm_patchtokens'])
    finally:
        hook.remove()
    torch.testing.assert_close(captured[0], images[:, 0])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert actual.shape == (1, 2, 8)  # Native patch embedding uses floor(H/16)*floor(W/16).


@pytest.mark.parametrize('cameras', [1, 4])
def test_scene_queries_keep_native_prefixes_and_patch_rope(encoder_factory, cameras):
    make, _ = encoder_factory
    encoder = make(cameras=cameras, queries=16).train()
    tokens_seen, rope_sizes = [], []
    qkv = encoder.native_vit.blocks[0].attn.qkv
    q_hook = qkv.register_forward_pre_hook(lambda module, args: tokens_seen.append(tuple(args[0].shape)))
    r_hook = encoder.native_vit.rope_embed.register_forward_hook(lambda module, args, output: rope_sizes.append(output[0].shape[0]))
    try:
        output = encoder(torch.randn(1, cameras, 3, 32, 48))
        (output - torch.randn_like(output)).square().mean().backward()
    finally:
        q_hook.remove(); r_hook.remove()
    # 16 queries + CLS + 4 storage + 6 patches; RoPE covers only the six patches.
    assert tokens_seen[0] == (cameras, 27, 32)
    assert rope_sizes == [6]
    assert output.shape == (1, cameras * 16, 8)
    assert encoder.scene_embeds.grad is not None and encoder.scene_embeds.grad.abs().sum() > 0
    assert not encoder.native_vit.training and encoder.training
    assert qkv.base.weight.grad is None and not qkv.base.weight.requires_grad
    assert qkv.q_b.weight.grad.abs().sum() > 0 and qkv.v_b.weight.grad.abs().sum() > 0
    assert encoder.neck.weight.grad.abs().sum() > 0
    for name, parameter in encoder.native_vit.named_parameters():
        if not any(f'.{part}.' in name for part in ('q_a', 'q_b', 'v_a', 'v_b')):
            assert not parameter.requires_grad and parameter.grad is None, name


@pytest.mark.parametrize('queries', [0, 16])
def test_gradient_checkpointing_preserves_values_and_lora_gradients(encoder_factory, queries):
    make, _ = encoder_factory
    plain = make(queries=queries, checkpointing=False).train()
    checked = make(queries=queries, checkpointing=True).train()
    checked.load_state_dict(plain.state_dict())
    images = torch.randn(1, 1, 3, 32, 48)
    out_plain, out_checked = plain(images), checked(images)
    torch.testing.assert_close(out_plain, out_checked)
    target = torch.randn_like(out_plain)
    (out_plain - target).square().mean().backward()
    (out_checked - target).square().mean().backward()
    for (_, a), (_, b) in zip(plain.named_parameters(), checked.named_parameters()):
        if a.requires_grad:
            assert a.grad is not None and b.grad is not None
            torch.testing.assert_close(a.grad, b.grad, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize('shape', [(5, 32), (2, 5, 32)])
def test_qv_lora_does_not_modify_key_projection(encoder_factory, shape):
    make, _ = encoder_factory
    adapter = make().native_vit.blocks[0].attn.qkv
    with torch.no_grad():
        adapter.q_b.weight.normal_(); adapter.v_b.weight.normal_()
    x = torch.randn(*shape)
    q, k, v = adapter(x).chunk(3, -1)
    base_q, base_k, base_v = adapter.base(x).chunk(3, -1)
    torch.testing.assert_close(k, base_k, rtol=0, atol=0)
    assert not torch.equal(q, base_q) and not torch.equal(v, base_v)


@pytest.mark.parametrize('size', ['vits', 'vitl'])
def test_dinov3_config_replaces_backbone_and_sets_aligned_resize(size):
    folder = ROOT / 'navsim/planning/script/config/common/agent'
    parent = OmegaConf.merge(*[OmegaConf.load(folder / (name + '.yaml')).config for name in (
        'drivoR', 'drivoR_action_dit', 'drivoR_action_dit_multigt')])
    variant = OmegaConf.load(folder / f'drivoR_action_dit_dinov3_{size}.yaml')
    assert variant.defaults[0] == 'drivoR_action_dit_multigt'
    override = variant.config
    combined = OmegaConf.merge(parent, override)
    assert set(override.keys()) == {'image_backbone', 'image_size', 'multi_gt'}
    for key in parent:
        if key not in ('image_backbone', 'image_size', 'multi_gt'):
            assert combined[key] == parent[key], key
    assert combined.image_backbone.family == 'dinov3'
    assert combined.image_backbone.use_lora and not combined.image_backbone.finetune
    assert combined.image_backbone.lora_rank == 32
    assert combined.image_size == [1152, 672]
    for key in parent.multi_gt:
        assert combined.multi_gt[key] == parent.multi_gt[key]
    assert combined.multi_gt.matching == 'nearest' and combined.multi_gt.weighting == 'equal'
    assert 'human_weight' not in combined.multi_gt and 'other_weight' not in combined.multi_gt
