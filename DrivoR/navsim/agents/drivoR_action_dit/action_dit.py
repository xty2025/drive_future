"""FastWAM/Wan Action-DiT with visual cross attention and shared candidate K/V.

Parameter names/shapes of time_embedding, time_projection and blocks match
FastWam_re/src/fastwam/models/wan22/action_dit.py. No external FastWAM imports,
text encoder, text embeddings, VAE, or video model are instantiated.
"""

import logging
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

log = logging.getLogger(__name__)


def sinusoidal_embedding_1d(dim, position):
    frequencies = 10000 ** (-torch.arange(dim // 2, device=position.device,
                                           dtype=torch.float64) / (dim // 2))
    angles = torch.outer(position.double(), frequencies)
    return torch.cat((angles.cos(), angles.sin()), dim=-1).to(position.dtype)


def apply_rope(x, num_heads):
    """Reference 1D RoPE; token positions restart for each candidate trajectory."""
    batch, length, width = x.shape
    head_dim = width // num_heads
    x_complex = torch.view_as_complex(x.double().reshape(batch, length, num_heads,
                                                         head_dim // 2, 2))
    freq = 1. / (10000 ** (torch.arange(0, head_dim, 2, device=x.device,
                                       dtype=torch.float64) / head_dim))
    angles = torch.outer(torch.arange(length, device=x.device), freq)
    rotations = torch.polar(torch.ones_like(angles), angles)[None, :, None]
    return torch.view_as_real(x_complex * rotations).reshape_as(x).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, attn_head_dim, eps):
        super().__init__()
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        width = num_heads * attn_head_dim
        self.q, self.k, self.v = (nn.Linear(hidden_dim, width) for _ in range(3))
        self.o = nn.Linear(width, hidden_dim)
        # Reference RMSNorm is over all attention channels, before head splitting.
        self.norm_q, self.norm_k = RMSNorm(width, eps), RMSNorm(width, eps)

    def forward(self, x):
        b, m, t, d = x.shape
        flat = x.reshape(b * m, t, d)
        q = apply_rope(self.norm_q(self.q(flat)), self.num_heads)
        k = apply_rope(self.norm_k(self.k(flat)), self.num_heads)
        v = self.v(flat)
        def split(value):
            return value.reshape(b * m, t, self.num_heads, self.attn_head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(split(q), split(k), split(v))
        out = out.transpose(1, 2).reshape(b, m, t, -1)
        return self.o(out)


class SharedSceneCrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, attn_head_dim, eps):
        super().__init__()
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        width = num_heads * attn_head_dim
        self.q, self.k, self.v = (nn.Linear(hidden_dim, width) for _ in range(3))
        self.o = nn.Linear(width, hidden_dim)
        self.norm_q, self.norm_k = RMSNorm(width, eps), RMSNorm(width, eps)

    def forward(self, x, scene):
        b, m, t, _ = x.shape
        q = self.norm_q(self.q(x)).reshape(b, m * t, self.num_heads,
                                          self.attn_head_dim).transpose(1, 2)
        k = self.norm_k(self.k(scene)).reshape(b, scene.shape[1], self.num_heads,
                                              self.attn_head_dim).transpose(1, 2)
        v = self.v(scene).reshape(b, scene.shape[1], self.num_heads,
                                 self.attn_head_dim).transpose(1, 2)
        # Flatten queries, not the batch: K/V are projected once per scene and
        # never repeat/expand into B*M copies of the visual patch context.
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, m, t, -1)
        return self.o(out)


class ActionBlock(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, num_heads, attn_head_dim, eps):
        super().__init__()
        self.self_attn = SelfAttention(hidden_dim, num_heads, attn_head_dim, eps)
        self.cross_attn = SharedSceneCrossAttention(hidden_dim, num_heads, attn_head_dim, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, ffn_dim),
                                 nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, hidden_dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim ** 0.5)

    def forward(self, x, scene, t_mod):
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = (
            t_mod + self.modulation.to(t_mod)).chunk(6, dim=2)
        x = x + gate_sa * self.self_attn(self.norm1(x) * (1 + scale_sa) + shift_sa)
        x = x + self.cross_attn(self.norm3(x), scene)
        return x + gate_ff * self.ffn(self.norm2(x) * (1 + scale_ff) + shift_ff)


class ActionDiT(nn.Module):
    def __init__(self, hidden_dim=256, action_dim=3, ffn_dim=1024, freq_dim=256,
                 num_heads=24, attn_head_dim=128, num_layers=30, eps=1e-6,
                 use_gradient_checkpointing=True, context_dim=None):
        super().__init__()
        if attn_head_dim % 2 or freq_dim % 2:
            raise ValueError("RoPE head width and sinusoidal width must be even")
        self.hidden_dim = hidden_dim
        self.freq_dim = freq_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.checkpoint_meta = dict(hidden_dim=hidden_dim, ffn_dim=ffn_dim,
                                    freq_dim=freq_dim, num_heads=num_heads,
                                    attn_head_dim=attn_head_dim, num_layers=num_layers, eps=eps)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, hidden_dim), nn.SiLU(),
                                            nn.Linear(hidden_dim, hidden_dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))
        self.blocks = nn.ModuleList([ActionBlock(hidden_dim, ffn_dim, num_heads,
                                                 attn_head_dim, eps) for _ in range(num_layers)])
        # The reference ActionDiT actually uses this plain Linear head; its
        # separate ActionHead class is unused by the requested configuration.
        self.head = nn.Linear(hidden_dim, action_dim)
        # Keep DrivoR perception/Scorer at 256 even for a wider trajectory DiT.
        # Identities add no state-dict keys to the existing 256-wide models.
        context_dim = hidden_dim if context_dim is None else int(context_dim)
        self.scene_projection = (nn.Identity() if context_dim == hidden_dim
                                 else nn.Linear(context_dim, hidden_dim))
        self.ego_projection = (nn.Identity() if context_dim == hidden_dim
                               else nn.Linear(context_dim, hidden_dim))
        self.load_report = None

    def load_pretrained(self, path):
        """Load every compatible non-text tensor; fail on missing backbone keys.

        Only the text_embedding branch is deliberately removed. Prepared FastWAM
        payloads can omit action_encoder/head under their published policy; full
        payloads containing compatible I/O layers load them as well.
        """
        payload = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
        meta = payload.get("meta", {})
        for key, value in self.checkpoint_meta.items():
            if key not in meta or meta[key] != value:
                raise ValueError(f"ActionDiT checkpoint meta.{key}: {meta.get(key)!r} != {value!r}")
        state = payload.get("backbone_state_dict", payload.get("state_dict"))
        if not isinstance(state, dict):
            raise ValueError("ActionDiT checkpoint requires backbone_state_dict or state_dict")
        own = self.state_dict()
        skipped_text = [name for name in state if name.startswith("text_embedding.")]
        loaded = {name: value for name, value in state.items() if name not in skipped_text}
        unexpected = sorted(set(loaded) - set(own))
        missing = sorted(set(own) - set(loaded))
        missing_backbone = [name for name in missing if not name.startswith(
            ("action_encoder.", "head.", "scene_projection.", "ego_projection."))]
        bad_shapes = [name for name in loaded if name in own and loaded[name].shape != own[name].shape]
        if unexpected or missing_backbone or bad_shapes:
            raise ValueError(f"ActionDiT checkpoint mismatch: unexpected={unexpected}, "
                             f"missing_backbone={missing_backbone}, wrong_shape={bad_shapes}")
        self.load_state_dict(loaded, strict=not missing)
        loaded_numel = sum(value.numel() for value in loaded.values())
        total_numel = sum(value.numel() for value in own.values())
        self.load_report = dict(path=str(Path(path).resolve()), loaded_tensors=len(loaded),
                                model_tensors=len(own), loaded_numel=loaded_numel,
                                model_numel=total_numel, fraction=loaded_numel / total_numel,
                                omitted_text_tensors=skipped_text, random_io_tensors=missing,
                                metadata=meta)
        log.info("ActionDiT checkpoint: %d/%d tensors; %.6f%% parameters; random I/O=%s; removed text=%s",
                 len(loaded), len(own), 100 * self.load_report["fraction"], missing, skipped_text)
        return self.load_report

    def load_wan_pretrained(self, path):
        """Copy the native Wan 5B blocks/time weights, without interpolation.

        Video patch I/O and text embeddings do not represent trajectories or
        DINO features. Those adapters are new; all 30 full-width blocks load.
        Read one tensor at a time, without materializing another 5B state dict.
        """
        import json
        from safetensors import safe_open

        root = Path(path)
        metadata = json.loads((root / "config.json").read_text())
        for name, key in (("hidden_dim", "dim"), ("ffn_dim", "ffn_dim"),
                          ("freq_dim", "freq_dim"), ("num_heads", "num_heads"),
                          ("num_layers", "num_layers")):
            if self.checkpoint_meta[name] != metadata[key]:
                raise ValueError(f"Wan {key}={metadata[key]} differs from configured {name}")
        index = json.loads((root / "diffusion_pytorch_model.safetensors.index.json").read_text())
        weight_map = index["weight_map"]
        own = self.state_dict()
        wanted = {key for key in own if key.startswith(("blocks.", "time_embedding.", "time_projection."))}
        missing = sorted(wanted - weight_map.keys())
        if missing:
            raise ValueError(f"Missing Wan backbone tensors: {missing}")
        loaded_numel = 0
        with torch.no_grad():
            for filename in sorted({weight_map[key] for key in wanted}):
                with safe_open(str(root / filename), framework="pt", device="cpu") as source:
                    for key in sorted(k for k in wanted if weight_map[k] == filename):
                        tensor = source.get_tensor(key)
                        if tensor.shape != own[key].shape:
                            raise ValueError(f"Wan tensor {key}: {tuple(tensor.shape)} != {tuple(own[key].shape)}")
                        own[key].copy_(tensor)
                        loaded_numel += tensor.numel()
        self.load_report = dict(path=str(root.resolve()), format="wan22_native", metadata=metadata,
            loaded_tensors=len(wanted), loaded_numel=loaded_numel,
            random_io_tensors=sorted(set(own) - wanted),
            omitted_source_tensors=sorted(set(weight_map) - wanted), interpolation=False)
        log.info("Native Wan: loaded %d tensors / %d parameters; new trajectory/DINO adapters",
                 len(wanted), loaded_numel)
        return self.load_report

    def forward(self, noisy_action, timesteps, scene_context, ego_embedding):
        scene_context = self.scene_projection(scene_context)
        ego_embedding = self.ego_projection(ego_embedding)
        b, m, _, _ = noisy_action.shape
        time = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim,
                                                           timesteps.float().reshape(-1)))
        condition = time.reshape(b, m, self.hidden_dim) + ego_embedding[:, None]
        t_mod = self.time_projection(condition).reshape(b, m, 6, self.hidden_dim)
        x = self.action_encoder(noisy_action)
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, scene_context, t_mod, use_reentrant=False)
            else:
                x = block(x, scene_context, t_mod)
        return self.head(x)
