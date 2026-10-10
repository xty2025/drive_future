"""Single-assignment v6 losses and gradient/freeze checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from .data import PairTargets


@dataclass
class Assignment:
    gt_indices: torch.Tensor
    candidate_indices: torch.Tensor

    def as_pairs(self) -> list[tuple[int, int]]:
        return list(zip(self.gt_indices.tolist(), self.candidate_indices.tolist()))


def hungarian_xy(predicted: torch.Tensor, ground_truth: torch.Tensor, valid: torch.Tensor) -> list[Assignment]:
    if predicted.ndim != 4 or ground_truth.ndim != 4:
        raise ValueError("predicted and ground_truth must be [B,M,8,3] and [B,K,8,3]")
    assignments: list[Assignment] = []
    for batch in range(predicted.shape[0]):
        valid_indices = torch.nonzero(valid[batch], as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=predicted.device)
            assignments.append(Assignment(empty, empty))
            continue
        cost = torch.cdist(
            ground_truth[batch, valid_indices, :, :2].reshape(valid_indices.numel(), -1).float(),
            predicted[batch, :, :, :2].reshape(predicted.shape[1], -1).float(),
            p=1,
        ) / ground_truth.shape[-2]
        gt_cpu, candidate_cpu = linear_sum_assignment(cost.detach().cpu().numpy())
        assignments.append(Assignment(valid_indices[torch.as_tensor(gt_cpu, device=valid_indices.device)], torch.as_tensor(candidate_cpu, device=predicted.device)))
    return assignments


def paired_residual_velocity_mse(endpoint: torch.Tensor, source: torch.Tensor, gt_encoded: torch.Tensor, assignment: list[Assignment]) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for batch, match in enumerate(assignment):
        if match.gt_indices.numel():
            predicted_delta = endpoint[batch, match.candidate_indices] - source[batch, match.candidate_indices]
            target_delta = gt_encoded[batch, match.gt_indices] - source[batch, match.candidate_indices]
            losses.append((predicted_delta - target_delta).square().mean())
    if not losses:
        return endpoint.sum() * 0.0
    return torch.stack(losses).mean()


def paired_future_l1(predicted_register: torch.Tensor, targets: PairTargets, assignment: list[Assignment]) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for batch, match in enumerate(assignment):
        for gt_index, candidate_index in match.as_pairs():
            valid = targets.future_chunk_valid[batch, gt_index]
            weights = targets.future_quality[batch, gt_index].to(predicted_register.device) * valid.to(predicted_register.device)
            if weights.sum() <= 0:
                continue
            error = (predicted_register[batch, candidate_index] - targets.future_register[batch, gt_index].detach()).abs().mean(dim=-1)
            losses.append((error * weights).sum() / weights.sum())
    if not losses:
        return predicted_register.sum() * 0.0
    return torch.stack(losses).mean()


def masked_soft_bce(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """DrivoR's six-head score loss, with evaluator-failure masks.

    The head order is ``NC,DAC,DDC,TTC,EP,C``.  This follows
    ``DrivoRLoss.score_loss``: NC/DDC map the 0.5 class to 0, TTC ignores its
    sentinel, every head is reduced independently, and the six losses are
    summed.  ``valid`` is the adapter extension needed for failed online oracle
    rows; it does not change any valid DrivoR label.
    """
    if logits.shape != targets.shape or valid.shape != logits.shape:
        raise ValueError("score logits, targets, and head mask must share [B,M,6]")
    if logits.shape[-1] != 6:
        raise ValueError("DrivoR score loss requires six heads")
    transformed = targets.to(logits.dtype).clone()
    transformed[..., 0] = torch.where(transformed[..., 0] == 0.5, 0.0, transformed[..., 0])
    transformed[..., 2] = torch.where(transformed[..., 2] == 0.5, 0.0, transformed[..., 2])
    mask = valid.to(logits.dtype)
    # The official training target uses 2.0 as TTC invalid, independent of the
    # online adapter's validity flag.
    mask[..., 3] = mask[..., 3] * targets[..., 3].ne(2.0).to(logits.dtype)
    if weights is not None:
        mask = mask * weights.to(mask.dtype)
    transformed = torch.where(mask.bool(), transformed, torch.zeros_like(transformed))
    element = nn.functional.binary_cross_entropy_with_logits(logits, transformed, reduction="none")
    per_head_numerator = (element * mask).sum(dim=(0, 1))
    per_head_denominator = mask.sum(dim=(0, 1)).clamp_min(1.0)
    return (per_head_numerator / per_head_denominator).sum()


def assert_disjoint_parameter_groups(*groups: Iterable[nn.Parameter]) -> None:
    seen: set[int] = set()
    for group in groups:
        current = {id(parameter) for parameter in group}
        if seen & current:
            raise AssertionError("optimizer parameter groups overlap")
        seen.update(current)


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)
