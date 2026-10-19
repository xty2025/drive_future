"""EMA of the effective DINOv3 backbone, including its learned Q/V LoRA updates."""

from copy import deepcopy

import torch
from torch import nn

from navsim.agents.drivoR_action_dit.dinov3_image_encoder import DinoV3LoRAQV


class EmaDinoRegisters(nn.Module):
    """Frozen native encoder; returns the four final normalized storage tokens.

    Teacher Q/V matrices track EMA(W_base + B @ A), rather than independently
    averaging A and B. Other native weights remain the copied frozen weights.
    The student's downstream 1024->256 projection is not part of this target.
    """

    def __init__(self, student, decay=0.999, chunk_size=4):
        super().__init__()
        self.decay = float(decay)
        self.chunk_size = int(chunk_size)
        if not 0 <= self.decay < 1 or self.chunk_size < 1:
            raise ValueError("EMA decay must be in [0,1), chunk_size must be positive")
        self.encoder = deepcopy(student)
        self.dim = int(student.num_features)
        self.register_count = int(student.n_storage_tokens)
        if self.register_count != 4:
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
                values = self.encoder.forward_features(image)["x_storage_tokens"]
            chunks.append(values.float())
        output[valid] = torch.cat(chunks)
        return output
