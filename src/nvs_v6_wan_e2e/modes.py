"""Immutable external mode codes and structured action source anchors."""

from __future__ import annotations

import hashlib
import json

import torch


class ImmutableModeBuffer:
    def __init__(self, count: int = 64, grid: tuple[int, int] = (8, 8), device: torch.device | None = None) -> None:
        if count != grid[0] * grid[1]:
            raise ValueError("v6 mode count must match the configured grid")
        velocity = torch.tensor([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], device=device)
        yaw_rate = torch.tensor([-0.30, -0.18, -0.08, 0.0, 0.04, 0.08, 0.18, 0.30], device=device)
        code = torch.cartesian_prod(velocity, yaw_rate).reshape(count, 2)
        self._code = code
        self._hash = hashlib.sha256(code.detach().cpu().numpy().tobytes()).hexdigest()

    @property
    def count(self) -> int:
        return self._code.shape[0]

    @property
    def codebook_hash(self) -> str:
        return self._hash

    def to(self, device: torch.device) -> "ImmutableModeBuffer":
        result = object.__new__(ImmutableModeBuffer)
        result._code = self._code.to(device)
        result._hash = self._hash
        return result

    def expand_for_batch(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        code = self._code if device is None else self._code.to(device)
        return code.unsqueeze(0).expand(batch_size, -1, -1)

    def subset(self, count: int) -> "ImmutableModeBuffer":
        if count < 1 or count > self.count:
            raise ValueError("mode subset must be within the immutable codebook")
        result = object.__new__(ImmutableModeBuffer)
        result._code = self._code[:count].clone()
        result._hash = hashlib.sha256(result._code.detach().cpu().numpy().tobytes()).hexdigest()
        return result

    def state(self) -> dict[str, object]:
        return {"count": self.count, "hash": self.codebook_hash, "source": "v6_fallback_velocity_yaw_grid"}


class StructuredSourceInitializer:
    """Small-angle unicycle rollout of each mode code, in DrivoR's metric units.

    The mode code is ``(delta_speed [m/s], yaw_rate [rad/s])`` applied to the
    current observable speed ``ego_history[:, -1, 3]`` (DrivoR's ``vx``).  The
    output is ``[B,M,8,3]`` with ``x,y`` in metres and ``heading`` in radians --
    no rescaling, exactly like DrivoR's ``state_size=3`` trajectory space.
    """

    def __init__(self, jitter_std: float = 0.4, heading_jitter_std: float = 0.05, seed: int = 17) -> None:
        self.jitter_std = jitter_std
        self.heading_jitter_std = heading_jitter_std
        self.seed = seed

    def clean(self, ego_history: torch.Tensor, modes: torch.Tensor, steps: int = 8) -> torch.Tensor:
        if modes.ndim != 3 or modes.shape[-1] != 2:
            raise ValueError("modes must be [B,M,2]")
        if ego_history.shape[-1] < 4:
            raise ValueError("ego_history must carry DrivoR's pose3 + velocity2 + acceleration2")
        speed = ego_history[:, -1, 3].clamp_min(0.0).unsqueeze(1)
        speed = (speed + modes[..., 0]).clamp_min(0.0)
        yaw_rate = modes[..., 1]
        times = torch.arange(1, steps + 1, device=modes.device, dtype=modes.dtype) * 0.5
        x = speed.unsqueeze(-1) * times
        y = 0.5 * (speed * yaw_rate).unsqueeze(-1) * times.square()
        heading = yaw_rate.unsqueeze(-1) * times
        return torch.stack((x, y, heading), dim=-1)

    def noisy(self, ego_history: torch.Tensor, modes: torch.Tensor, steps: int = 8, generator: torch.Generator | None = None) -> torch.Tensor:
        clean = self.clean(ego_history, modes, steps)
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        noise[..., 2] = noise[..., 2] * (self.heading_jitter_std / max(self.jitter_std, 1e-8))
        return clean + noise * self.jitter_std
