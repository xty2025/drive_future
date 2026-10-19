"""The absolute NAVSIM normalization and continuous schedule used by FastWAM.

Reference: FastWam_re/src/fastwam/datasets/navsim/navsim_dataset.py and
models/wan22/schedulers/scheduler_continuous.py. Time is the *noise* fraction.
"""

import torch


def normalize_trajectory(trajectory: torch.Tensor) -> torch.Tensor:
    bias = trajectory.new_tensor((1.57, 19.68, 1.67))
    span = trajectory.new_tensor((66.74, 42.0, 3.53))
    return 2 * (trajectory + bias) / span - 1


def denormalize_trajectory(trajectory: torch.Tensor) -> torch.Tensor:
    bias = trajectory.new_tensor((1.57, 19.68, 1.67))
    span = trajectory.new_tensor((66.74, 42.0, 3.53))
    return (trajectory + 1) * span / 2 - bias


class ContinuousFlowSchedule:
    def __init__(self, num_train_timesteps=1000, train_shift=5.0, infer_shift=5.0,
                 sigma_min=0.1):
        if num_train_timesteps <= 0 or train_shift <= 0 or infer_shift <= 0:
            raise ValueError("Flow timestep count and shifts must be positive")
        if not 0 < sigma_min <= 1:
            raise ValueError("sigma_min must be in (0, 1]")
        self.num_train_timesteps = int(num_train_timesteps)
        self.train_shift = float(train_shift)
        self.infer_shift = float(infer_shift)
        self.sigma_min = float(sigma_min)
        grid = torch.linspace(1., 0., self.num_train_timesteps + 1,
                              dtype=torch.float64)[:-1]
        sigma_grid = self.shift(grid, self.train_shift)
        y = torch.exp(-2 * (sigma_grid - 0.5).square())
        self.y_min = float(y.min())
        self.weight_norm = float((y - self.y_min).mean())

    @staticmethod
    def shift(u, amount):
        return amount * u / (1 + (amount - 1) * u)

    def sample_timesteps(self, batch_size, candidates, device):
        # FastWAM samples one time per scene. All candidates of that scene use
        # the same noise level so best-of-K compares like-for-like velocities.
        u = torch.rand(batch_size, 1, device=device, dtype=torch.float32)
        return (self.shift(u, self.train_shift) * self.num_train_timesteps).expand(-1, candidates)

    def sigma(self, timesteps):
        return timesteps.float() / self.num_train_timesteps

    def training_weight(self, timesteps):
        y = torch.exp(-2 * (self.sigma(timesteps) - 0.5).square())
        return (y - self.y_min) / (self.weight_norm + 1e-10)

    def add_noise(self, clean, noise, timesteps):
        sigma = self.sigma(timesteps)[..., None, None]
        return (1 - sigma) * clean.float() + sigma * noise.float()

    def x_to_v(self, pred_x, noisy, timesteps):
        """Bounded conversion in FP32: v_hat = (x_sigma - x_hat) / sigma_safe."""
        sigma_safe = self.sigma(timesteps).clamp(self.sigma_min, 1.)[..., None, None]
        return (noisy.float() - pred_x.float()) / sigma_safe

    def velocity_loss(self, pred_x, clean, noisy, timesteps, reduction="mean"):
        """Match reference sample-mode loss while retaining a zero at x_hat=x_GT.

        Both v predictions and targets use the *same* bounded denominator:
          v_hat=(x_sigma-x_hat)/sigma_safe
          v_target=(x_sigma-x_GT)/sigma_safe
        Hence their error is (x_GT-x_hat)/sigma_safe, including sigma < sigma_min.
        Clamping only v_hat but retaining noise-x_GT as the target would be biased.
        """
        v_pred = self.x_to_v(pred_x, noisy, timesteps)
        v_target = self.x_to_v(clean, noisy, timesteps)
        # Compute the algebraically equal residual directly to avoid cancellation
        # between two potentially large velocities under mixed precision.
        sigma_safe = self.sigma(timesteps).clamp(self.sigma_min, 1.)[..., None, None]
        error = (clean.float() - pred_x.float()) / sigma_safe
        per_candidate = error.square().mean(dim=(-2, -1))
        per_candidate = per_candidate * self.training_weight(timesteps)
        if reduction == "mean":
            loss = per_candidate.mean()
        elif reduction == "best_of_k":
            loss = per_candidate.min(dim=1).values.mean()
        else:
            raise ValueError(f"Unknown flow reduction: {reduction}")
        return loss, v_pred, v_target

    def inference_schedule(self, num_steps, device):
        if num_steps < 1:
            raise ValueError("num_inference_steps must be at least 1")
        u = torch.linspace(1., 0., num_steps + 1, device=device, dtype=torch.float32)
        sigma = self.shift(u, self.infer_shift)
        return sigma[:-1] * self.num_train_timesteps, sigma[1:] - sigma[:-1]
