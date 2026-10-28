"""Frozen Wan2.2 VAE targets: independent future frames, spatial pooling."""

import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def load_wan_vae(options):
    # Reuse the supplied implementation directly, without its pipeline loader,
    # video DiT, text encoder, model registry, network access or file hashing.
    source = Path(options.source_file)
    spec = importlib.util.spec_from_file_location('_drivor_wan22_vae', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.WanVideoVAE38()
    path = Path(options.checkpoint_path)
    if path.is_dir():
        path = path / 'Wan2.2_VAE.pth'
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    state = state.get('model_state', state)
    if all(key.startswith('model.') for key in state):
        model.load_state_dict(state, strict=True)
    else:
        model.model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


class WanVaeTargets(nn.Module):
    def __init__(self, options, pooling_grid=(4, 4), chunk_size=4):
        super().__init__()
        self.encoder = load_wan_vae(options)
        self.dim = int(self.encoder.z_dim)
        self.pooling_grid = tuple(int(value) for value in pooling_grid)
        self.register_count = self.pooling_grid[0] * self.pooling_grid[1]
        self.chunk_size = int(chunk_size)
        self.register_buffer('updates', torch.zeros((), dtype=torch.long))
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        self.encoder.eval()
        return self

    def update(self, student):
        pass  # The VAE is permanently frozen; DINO is not its EMA source.

    @torch.no_grad()
    def forward(self, pixels, valid):
        """[B,K,8,3,H,W] -> [B,K,8,16,48], in native normalized latent units.

        Future frames become independent length-one videos in the batch. The
        VAE therefore never compresses the eight supervised time positions.
        encode returns the scaled posterior mean, without posterior sampling.
        """
        valid = valid.bool()
        output = torch.zeros(*valid.shape, self.register_count, self.dim,
                             device=pixels.device, dtype=torch.float32)
        frames = pixels[valid]
        values = []
        for start in range(0, len(frames), self.chunk_size):
            image = frames[start:start + self.chunk_size].float() / 127.5 - 1.
            with torch.autocast(device_type=pixels.device.type,
                               enabled=torch.is_autocast_enabled(pixels.device.type),
                               dtype=torch.get_autocast_dtype(pixels.device.type), cache_enabled=False):
                latent = self.encoder.single_encode(image.unsqueeze(2), pixels.device)
                if latent.shape[2] != 1:
                    raise ValueError('Independent frame encoding must retain exactly one latent time position')
                pooled = F.adaptive_avg_pool2d(latent[:, :, 0], self.pooling_grid)
            values.append(pooled.flatten(2).transpose(1, 2).float())
        if values:
            output[valid] = torch.cat(values)
        return output
