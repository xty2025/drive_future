"""Build and load the Wan2.2 -> NVS Action DiT backbone payload at 256 dims.

This reproduces SimWAM's ``scripts/preprocess_action_dit_backbone.py`` exactly --
sequential per-dimension 1D linear interpolation with ``align_corners=True`` and
``alpha = sqrt(src_last_dim / target_last_dim)`` scaling -- but reads the Wan
safetensors shards directly instead of instantiating the 5B video DiT, and maps
Wan's 30 blocks onto however many NVS Action DiT layers are configured.

``action_encoder.``, ``head.`` and the NVS-only condition projections are left
randomly initialized, matching SimWAM's ``ACTION_BACKBONE_SKIP_PREFIXES``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .wan_dit import NvsActionDiT

WAN_INDEX_FILENAME = "diffusion_pytorch_model.safetensors.index.json"


def interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def resize_tensor_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == tuple(target_shape):
        return src
    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(f"cannot reduce rank for resize: src={tuple(src.shape)} target={target_shape}")
        out = out.squeeze(0)
    for dim, new_size in enumerate(target_shape):
        if out.shape[dim] == new_size:
            continue
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv_perm = [0] * out.ndim
        for position, source_axis in enumerate(perm):
            inv_perm[source_axis] = position
        permuted = out.permute(*perm).contiguous()
        prefix_shape = permuted.shape[:-1]
        permuted = interpolate_last_dim(permuted, new_size).reshape(*prefix_shape, new_size)
        out = permuted.permute(*inv_perm).contiguous()
    if tuple(out.shape) != tuple(target_shape):
        raise ValueError(f"resize produced {tuple(out.shape)} for target {target_shape}")
    return out.to(dtype=src.dtype)


def select_source_layers(source_layers: int, target_layers: int) -> list[int]:
    """Uniform stride over Wan's depth; identity when the layer counts match."""

    if target_layers < 1:
        raise ValueError("target layer count must be positive")
    if target_layers > source_layers:
        raise ValueError(f"cannot expand {source_layers} Wan blocks into {target_layers} layers")
    return [index * source_layers // target_layers for index in range(target_layers)]


class _WanShardReader:
    def __init__(self, wan_root: Path) -> None:
        index_path = wan_root / WAN_INDEX_FILENAME
        if not index_path.is_file():
            raise FileNotFoundError(f"missing Wan index file: {index_path}")
        weight_map = json.loads(index_path.read_text())["weight_map"]
        self._root = wan_root
        self._weight_map = weight_map
        self._handles: dict[str, Any] = {}

    def keys(self) -> set[str]:
        return set(self._weight_map)

    def get(self, key: str) -> torch.Tensor:
        shard = self._weight_map.get(key)
        if shard is None:
            raise KeyError(f"Wan checkpoint has no tensor `{key}`")
        handle = self._handles.get(shard)
        if handle is None:
            from safetensors import safe_open

            handle = safe_open(str(self._root / shard), framework="pt", device="cpu")
            self._handles[shard] = handle
        return handle.get_tensor(key)

    def close(self) -> None:
        self._handles.clear()


def _source_block_count(reader: _WanShardReader) -> int:
    indices = set()
    for key in reader.keys():
        if key.startswith("blocks."):
            indices.add(int(key.split(".", 2)[1]))
    if not indices:
        raise ValueError("Wan checkpoint exposes no `blocks.*` tensors")
    if indices != set(range(max(indices) + 1)):
        raise ValueError("Wan block indices are not contiguous from zero")
    return max(indices) + 1


def build_action_dit_backbone(
    wan_root: str | Path,
    target: NvsActionDiT,
    apply_alpha_scaling: bool = True,
) -> dict[str, Any]:
    """Interpolate Wan's video-DiT weights into ``target``'s backbone shapes."""

    wan_root = Path(wan_root)
    reader = _WanShardReader(wan_root)
    try:
        source_layers = _source_block_count(reader)
        layer_map = select_source_layers(source_layers, target.num_layers)
        state = target.state_dict()
        backbone_keys = NvsActionDiT.backbone_key_set(state.keys())
        payload: dict[str, torch.Tensor] = {}
        copied = interpolated = 0
        for key in sorted(backbone_keys):
            if key.startswith("blocks."):
                _, index, remainder = key.split(".", 2)
                source_key = f"blocks.{layer_map[int(index)]}.{remainder}"
            else:
                source_key = key
            src = reader.get(source_key).to(torch.float32)
            want = state[key]
            if tuple(src.shape) == tuple(want.shape):
                value = src
                copied += 1
            else:
                value = resize_tensor_to_shape(src, tuple(want.shape))
                if apply_alpha_scaling and src.ndim >= 2 and src.shape[-1] != want.shape[-1]:
                    value = value * (float(src.shape[-1]) / float(want.shape[-1])) ** 0.5
                interpolated += 1
            payload[key] = value.detach().to(dtype=want.dtype).contiguous()
    finally:
        reader.close()
    return {
        "policy": {
            "skip_prefixes": list(NvsActionDiT.BACKBONE_SKIP_PREFIXES),
            "alpha_scaling": bool(apply_alpha_scaling),
            "interpolation": "sequential_1d_linear_align_corners_true",
            "source": str(wan_root),
            "source_layers": source_layers,
            "layer_map": layer_map,
        },
        "backbone_state_dict": payload,
        "meta": target.meta(),
        "stats": {"copied": copied, "interpolated": interpolated},
    }


def load_action_dit_backbone(target: NvsActionDiT, payload_path: str | Path) -> dict[str, Any]:
    """Validate a payload against ``target`` and load it with SimWAM's strictness."""

    payload_path = Path(payload_path)
    if not payload_path.is_file():
        raise FileNotFoundError(f"Action DiT backbone payload missing: {payload_path}")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid payload type from {payload_path}: {type(payload)}")
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise ValueError(f"`meta` missing in {payload_path}")
    expected = target.meta()
    for key in NvsActionDiT.META_KEYS:
        if key not in meta:
            raise ValueError(f"`meta.{key}` missing in {payload_path}")
        if key == "eps":
            if abs(float(meta[key]) - float(expected[key])) > 1e-12:
                raise ValueError(f"`meta.{key}` mismatch: expected {expected[key]}, got {meta[key]}")
        elif int(meta[key]) != int(expected[key]):
            raise ValueError(f"`meta.{key}` mismatch: expected {expected[key]}, got {meta[key]}")
    backbone = payload.get("backbone_state_dict")
    if not isinstance(backbone, dict):
        raise ValueError(f"`backbone_state_dict` must be a dict in {payload_path}")
    state = target.state_dict()
    expected_keys = NvsActionDiT.backbone_key_set(state.keys())
    provided = set(backbone.keys())
    missing = sorted(expected_keys - provided)
    unexpected = sorted(provided - expected_keys)
    if missing or unexpected:
        raise ValueError(f"backbone key mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
    merged = dict(state)
    for key in expected_keys:
        value = backbone[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"`backbone_state_dict[{key}]` must be a Tensor")
        if tuple(value.shape) != tuple(merged[key].shape):
            raise ValueError(f"shape mismatch for `{key}`: {tuple(value.shape)} vs {tuple(merged[key].shape)}")
        merged[key] = value.to(device=merged[key].device, dtype=merged[key].dtype)
    target.load_state_dict(merged, strict=True)
    return {"policy": payload.get("policy", {}), "loaded_keys": len(expected_keys)}
