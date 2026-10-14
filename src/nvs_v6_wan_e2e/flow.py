"""Rectified-flow schedule for the ActionDiT, ported from SimWAM.

:class:`WanContinuousFlowMatchScheduler`, :func:`norm_odo` and :func:`denorm_odo`
are verbatim copies of the SimWAM implementations (``vendor/SimWAM/src/simwam/
models/wan22/schedulers/scheduler_continuous.py`` and ``vendor/SimWAM/src/simwam/
datasets/navsim/navsim_dataset.py``), so the noise schedule, the ``shift``-warped
timestep sampling, the bell-shaped loss weight and the ``[-1, 1]`` action
normalization all behave exactly as they do there.

One deliberate NVS substitution
------------------------------
SimWAM's flow runs between the GT action at ``sigma=0`` and *Gaussian noise* at
``sigma=1``.  NVS instead uses the **structured 64-anchor grid** as the
``sigma=1`` endpoint, so ``noise`` in the ported formulas is the anchor prior::

    x_sigma  = (1 - sigma) * gt   + sigma * anchor      # scheduler.add_noise
    v_target =              anchor - gt                 # scheduler.training_target
    gt_hat   = x_sigma - sigma * v_hat                  # exact inverse at any sigma

``x_sigma`` is written in normalized units above; :class:`ActionFlow` hands the
model metric trajectories and keeps ``v`` normalized, applying the constant
``range / 2`` Jacobian of :func:`denorm_odo` when it inverts.

Everything else is unchanged, which is what makes the two ends of the v6
contract fall out for free:

* ``sigma = 0`` gives ``x_sigma == gt``: Stage 1's noise-free diverse action.
  ``training_weight`` is also 0 there (the bell vanishes at both ends), so the
  velocity term is automatically off and Stage 1 trains only the representation.
* ``sigma = 1`` gives ``x_sigma == anchor``: one forward, no rollout, and with
  the near-zero ``Head`` initialization ``v_hat ~= 0`` so ``gt_hat == anchor``.
  The anchor grid's oracle-best PDMS is therefore the *starting* point of
  training rather than something it has to rediscover.

Anchors are metric ego-local trajectories, so the interpolation is done on the
``[-1, 1]`` normalized copy and mapped back with :func:`denorm_odo`.  Because
``norm_odo`` is affine and ``sigma + (1 - sigma) == 1``, interpolating before or
after normalization is identical; :class:`ActionFlow` keeps the model boundary
metric and the flow arithmetic normalized.
"""

from __future__ import annotations

import torch

#: SimWAM ``navsim_dataset.norm_odo`` constants: ``2 * (v + bias) / range - 1``.
#: Fitted on navtrain's absolute 8x0.5 s ego-local trajectories.
ODO_NORM_BIAS = (1.57, 19.68, 1.67)
ODO_NORM_RANGE = (66.74, 42.00, 3.53)
#: Largest entry of ``d(denorm_odo)/d(norm) == range / 2``.  A normalized-space
#: velocity of this magnitude is one metre, which is what sets the near-zero
#: ``Head`` init in :mod:`nvs_v6_wan_e2e.wan_dit`.
ODO_METRIC_JACOBIAN_MAX = max(ODO_NORM_RANGE) / 2.0


def norm_odo(trajectory: torch.Tensor) -> torch.Tensor:
    """SimWAM's ``norm_odo``: metric ``[..., 3]`` -> ``[-1, 1]`` (affine, no clipping)."""
    x = 2 * (trajectory[..., 0:1] + 1.57) / 66.74 - 1
    y = 2 * (trajectory[..., 1:2] + 19.68) / 42 - 1
    heading = 2 * (trajectory[..., 2:3] + 1.67) / 3.53 - 1
    return torch.cat([x, y, heading], dim=-1)


def denorm_odo(normalized: torch.Tensor) -> torch.Tensor:
    """SimWAM's ``denorm_odo``: exact inverse of :func:`norm_odo`."""
    x = (normalized[..., 0:1] + 1) * 66.74 / 2 - 1.57
    y = (normalized[..., 1:2] + 1) * 42 / 2 - 19.68
    heading = (normalized[..., 2:3] + 1) * 3.53 / 2 - 1.67
    return torch.cat([x, y, heading], dim=-1)


def velocity_to_metric_scale(normalized_velocity: torch.Tensor) -> torch.Tensor:
    """Convert a normalized-space flow velocity into metres/radians.

    ``d(denorm_odo)/d(norm)`` is the constant ``range / 2`` per dimension, so
    ``denorm_odo(x - sigma * v) == denorm_odo(x) - sigma * (range / 2) * v``.
    Returning the scaled velocity keeps the caller in metric units end to end.
    """
    scale = torch.tensor(
        [value / 2.0 for value in ODO_NORM_RANGE],
        device=normalized_velocity.device,
        dtype=normalized_velocity.dtype,
    )
    return normalized_velocity * scale


class WanContinuousFlowMatchScheduler:
    """Continuous-time Flow-Matching scheduler with shift-based sampling.

    Verbatim from SimWAM's ``scheduler_continuous.py``; ``add_noise`` and
    ``training_target`` are reused with the anchor prior in place of Gaussian
    noise, which needs no change to this class.
    """

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, eps: float = 1e-10):
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}")
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.eps = float(eps)
        self._y_min, self._weight_norm_const = self._precompute_training_weight_stats()

    @staticmethod
    def _phi(u: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * u / (1.0 + (shift - 1.0) * u)

    def _precompute_training_weight_stats(self) -> tuple[float, float]:
        steps = self.num_train_timesteps
        u_grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        t_grid = self._phi(u_grid, self.shift) * float(steps)
        y_grid = torch.exp(-2.0 * ((t_grid - (steps / 2.0)) / steps) ** 2)
        y_min = float(y_grid.min().item())
        y_shifted_grid = y_grid - y_min
        norm_const = float(y_shifted_grid.mean().item())
        return y_min, norm_const

    def sample_training_t(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}")
        u = torch.rand((batch_size,), device=device, dtype=torch.float32)
        sigma = self._phi(u, self.shift)
        timestep = sigma * float(self.num_train_timesteps)
        return timestep.to(dtype=dtype)

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        t = timestep.to(dtype=torch.float32)
        steps = float(self.num_train_timesteps)
        y = torch.exp(-2.0 * ((t - (steps / 2.0)) / steps) ** 2)
        y_shifted = y - self._y_min
        weight = y_shifted / (self._weight_norm_const + self.eps)
        if weight.numel() == 1:
            return weight.reshape(())
        return weight

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigma = (timestep / float(self.num_train_timesteps)).to(
            original_samples.device, dtype=original_samples.dtype
        )
        if sigma.ndim == 0:
            return (1 - sigma) * original_samples + sigma * noise
        sigma = sigma.view(-1, *([1] * (original_samples.ndim - 1)))
        return (1 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return noise - sample

    def build_inference_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
        shift = self.shift if shift_override is None else float(shift_override)
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")

        u_steps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)
        sigma_steps = self._phi(u_steps, shift)
        timesteps = sigma_steps[:-1] * float(self.num_train_timesteps)
        deltas = sigma_steps[1:] - sigma_steps[:-1]
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    @staticmethod
    def step(model_output: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        delta = delta.to(sample.device, dtype=sample.dtype)
        if delta.ndim == 0:
            return sample + model_output * delta
        delta = delta.view(-1, *([1] * (sample.ndim - 1)))
        return sample + model_output * delta


#: ``sigma`` of the deployed single forward: the anchor prior itself.
INFERENCE_SIGMA = 1.0


class ActionFlow:
    """Builds one noisy training batch and inverts one forward, in metric units.

    The scheduler is stateless with respect to the model, so a single instance is
    shared by both stages; only ``sigma`` differs (Stage 1 pins it to 0, Stage 2
    samples it per scene per forward).
    """

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0) -> None:
        self.scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps, shift)

    @property
    def num_train_timesteps(self) -> int:
        return self.scheduler.num_train_timesteps

    def sample_timestep(
        self, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Sample SimWAM's shift-warped continuous timestep, one per scene.

        ``shape`` is ``(B,)``: every candidate of a scene shares its ``sigma``.
        Per-candidate sigmas would make the Hungarian/dense comparison unfair --
        a candidate at ``sigma ~= 0`` is handed its GT and cannot lose, so it
        would win the matching against candidates that had to denoise.
        """
        count = 1
        for extent in shape:
            count *= int(extent)
        return self.scheduler.sample_training_t(count, device, dtype).view(shape)

    def noisy_action(
        self, anchor: torch.Tensor, data: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpolate anchor(prior) -> data(GT) and return ``(x_sigma, v_target)``.

        ``x_sigma`` is metric ``[B, M, 8, 3]`` because that is the unit the rest of
        the v6 pipeline speaks.  ``v_target`` stays in *normalized* space, which is
        what the ``Head`` emits: the per-dimension ``range / 2`` factors differ by
        19x (33.37 m vs 1.765 rad), so an MSE in metric space would once again let
        ``x`` swamp ``heading``.
        """
        if anchor.shape != data.shape:
            raise ValueError(f"anchor {tuple(anchor.shape)} and data {tuple(data.shape)} must match")
        normalized = self.scheduler.add_noise(norm_odo(data), norm_odo(anchor), timestep)
        velocity = self.scheduler.training_target(norm_odo(data), norm_odo(anchor), timestep)
        return denorm_odo(normalized), velocity

    def recover(self, noisy_action: torch.Tensor, velocity: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Single-forward inverse ``x0 = x_sigma - sigma * v`` -- no rollout.

        ``velocity`` is the ``Head``'s normalized-space prediction; it is scaled to
        metric here so the caller can stay in metres.
        """
        sigma = self._sigma_for(timestep, noisy_action)
        return noisy_action - sigma * velocity_to_metric_scale(velocity)

    def weight(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.scheduler.training_weight(timestep)

    def _sigma_for(self, timestep: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        sigma = (timestep.to(device=reference.device, dtype=reference.dtype) / float(self.num_train_timesteps))
        return sigma.view(*sigma.shape, *([1] * (reference.ndim - sigma.ndim)))


__all__ = [
    "INFERENCE_SIGMA",
    "ODO_METRIC_JACOBIAN_MAX",
    "ODO_NORM_BIAS",
    "ODO_NORM_RANGE",
    "ActionFlow",
    "WanContinuousFlowMatchScheduler",
    "denorm_odo",
    "norm_odo",
    "velocity_to_metric_scale",
]
