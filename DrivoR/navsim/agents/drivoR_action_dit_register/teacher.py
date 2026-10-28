"""EMA of the effective DINOv3 backbone, including its learned Q/V LoRA updates."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from navsim.agents.drivoR_action_dit.dinov3_image_encoder import DinoV3LoRAQV


class EmaDinoRegisters(nn.Module):
    """Frozen native encoder; returns the four final normalized storage tokens.

    Teacher Q/V matrices track EMA(W_base + B @ A), rather than independently
    averaging A and B. Other native weights remain the copied frozen weights.
    The student's downstream 1024->256 projection is not part of this target.
    """

    def __init__(self, student, decay=0.999, chunk_size=4, target_kind='registers', pooling_grid=(4, 4)):
        super().__init__()
        self.decay = float(decay)
        self.chunk_size = int(chunk_size)
        if not 0 <= self.decay < 1 or self.chunk_size < 1:
            raise ValueError("EMA decay must be in [0,1), chunk_size must be positive")
        self.encoder = deepcopy(student)
        self.dim = int(student.num_features)
        self.target_kind = target_kind
        self.pooling_grid = tuple(int(value) for value in pooling_grid)
        if target_kind not in ('registers', 'patches'):
            raise ValueError('DINO future targets must be registers or patches')
        self.register_count = (int(student.n_storage_tokens) if target_kind == 'registers'
                               else self.pooling_grid[0] * self.pooling_grid[1])
        if target_kind == 'registers' and self.register_count != 4:
            raise ValueError("This variant requires DINOv3's four native register tokens")
        self._adapter_names = []
        # Replace copied LoRA wrappers with native, merged QKV linears. The
        # teacher therefore represents an EMA of actual forward-pass weights.
        for name, module in student.named_modules():
            if isinstance(module, DinoV3LoRAQV):
                parent_name, child_name = name.rsplit('.', 1)
                merged = deepcopy(module.base)
                with torch.no_grad():
                    width = module.in_features
                    merged.weight[:width].add_(module.q_b.weight @ module.q_a.weight)
                    merged.weight[2 * width:].add_(module.v_b.weight @ module.v_a.weight)
                setattr(self.encoder.get_submodule(parent_name), child_name, merged)
                self._adapter_names.append(name)
        self.encoder.requires_grad_(False)
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("pixel_mean", torch.tensor([.485,.456,.406]).view(1,3,1,1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor([.229,.224,.225]).view(1,3,1,1), persistent=False)
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def update(self, student):
        for name in self._adapter_names:
            module = student.get_submodule(name)
            target = self.encoder.get_submodule(name)
            width = module.in_features
            # Compute the EMA in FP32 even when optimizer hooks are called
            # inside a mixed-precision training context.
            with torch.autocast(device_type=target.weight.device.type, enabled=False):
                query = module.base.weight[:width].float() + module.q_b.weight.float() @ module.q_a.weight.float()
                value = module.base.weight[2 * width:].float() + module.v_b.weight.float() @ module.v_a.weight.float()
                target.weight[:width].lerp_(query.to(target.weight.dtype), 1 - self.decay)
                target.weight[2 * width:].lerp_(value.to(target.weight.dtype), 1 - self.decay)
        self.updates.add_(1)

    @torch.no_grad()
    def forward(self, pixels, valid):
        """pixels [B,K,8,3,H,W] uint8 -> targets [B,K,8,4,1024]."""
        valid = valid.bool()
        output = torch.zeros(*valid.shape, self.register_count, self.dim,
                             device=pixels.device, dtype=torch.float32)
        flat = pixels[valid]
        if not len(flat):
            return output
        chunks = []
        device_type = pixels.device.type
        for start in range(0, len(flat), self.chunk_size):
            image = flat[start:start + self.chunk_size].float() / 255.
            image = (image - self.pixel_mean) / self.pixel_std
            with torch.autocast(device_type=device_type,
                                enabled=torch.is_autocast_enabled(device_type),
                                dtype=torch.get_autocast_dtype(device_type), cache_enabled=False):
                features = self.encoder.forward_features(image)
                if self.target_kind == 'registers':
                    values = features["x_storage_tokens"]
                else:
                    patch_size = self.encoder.patch_embed.patch_size
                    if isinstance(patch_size, int):
                        patch_size = (patch_size, patch_size)
                    grid_h, grid_w = image.shape[-2] // patch_size[0], image.shape[-1] // patch_size[1]
                    patches = features['x_norm_patchtokens'].transpose(1, 2).reshape(len(image), self.dim, grid_h, grid_w)
                    values = F.adaptive_avg_pool2d(patches, self.pooling_grid).flatten(2).transpose(1, 2)
            chunks.append(values.float())
        output[valid] = torch.cat(chunks)
        return output


class SharedDinoTargets(nn.Module):
    """The live student ViT is read under no-grad; no EMA or second ViT copy.

    Keep its reference outside _modules so parameters/state_dict/optimizer do
    not register the same encoder twice. The owning image_backbone moves it.
    """

    def __init__(self, student, chunk_size=4):
        super().__init__()
        object.__setattr__(self, '_shared_encoder', student)
        self.chunk_size = int(chunk_size)
        self.dim = int(student.num_features)
        self.register_count = int(student.n_storage_tokens)
        self.target_kind = 'registers'
        self.pooling_grid = (4, 4)
        self.register_buffer('updates', torch.zeros((), dtype=torch.long))
        self.register_buffer('pixel_mean', torch.tensor([.485,.456,.406]).view(1,3,1,1), persistent=False)
        self.register_buffer('pixel_std', torch.tensor([.229,.224,.225]).view(1,3,1,1), persistent=False)
        self.train(False)

    @property
    def encoder(self):
        return self._shared_encoder

    def train(self, mode=True):
        super().train(False)
        self.encoder.eval()
        return self

    def update(self, student):
        pass  # Targets already use the current weights, including current LoRA.

    forward = EmaDinoRegisters.forward


def build_future_teacher(student, options):
    target_kind = str(options.get('target_kind', 'dinov3_registers'))
    if target_kind == 'wan_vae':
        from .vae_teacher import WanVaeTargets
        return WanVaeTargets(options.vae, pooling_grid=options.get('pooling_grid', [4, 4]),
                             chunk_size=options.teacher_batch_size)
    mode = str(options.get('teacher_mode', 'ema'))
    if mode == 'shared':
        if target_kind != 'dinov3_registers':
            raise ValueError('Shared-ViT ablations keep the native register targets')
        return SharedDinoTargets(student, chunk_size=options.teacher_batch_size)
    if mode != 'ema' or target_kind not in ('dinov3_registers', 'dinov3_patches'):
        raise ValueError(f'Unknown future teacher: {mode}/{target_kind}')
    return EmaDinoRegisters(student, decay=options.ema_decay, chunk_size=options.teacher_batch_size,
        target_kind='patches' if target_kind == 'dinov3_patches' else 'registers',
        pooling_grid=options.get('pooling_grid', [4, 4]))
