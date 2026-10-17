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


def paired_residual_velocity_mse(endpoint: torch.Tensor, source: torch.Tensor, gt_state: torch.Tensor, assignment: list[Assignment]) -> torch.Tensor:
    """DrivoR-native metric residual loss: ``(endpoint - source)`` vs ``(gt - source)``."""

    losses: list[torch.Tensor] = []
    for batch, match in enumerate(assignment):
        if match.gt_indices.numel():
            predicted_delta = endpoint[batch, match.candidate_indices] - source[batch, match.candidate_indices]
            target_delta = gt_state[batch, match.gt_indices] - source[batch, match.candidate_indices]
            losses.append((predicted_delta - target_delta).square().mean())
    if not losses:
        return endpoint.sum() * 0.0
    return torch.stack(losses).mean()


def candidate_gt_l1_ade(proposals: torch.Tensor, ground_truth: torch.Tensor, gt_valid: torch.Tensor) -> torch.Tensor:
    """``[B,M,K]`` L1 ADE for every (candidate, GT) pair, DrivoR's exact metric.

    ``torch.linalg.norm(..., dim=-1, ord=1)`` sums ``|dx| + |dy| + |dheading|``,
    then ``.mean(-1)`` averages over the eight 0.5 s poses.  Because L1's gradient
    is ``sign(error)``, every state dimension contributes the same gradient
    magnitude regardless of its unit -- metres and radians need no reweighting,
    which is exactly why DrivoR uses L1 rather than the L2 residual NVS had.

    Invalid GT columns are set to ``+inf`` so ``amin`` skips them; callers must
    check ``isfinite`` before dividing.
    """
    if proposals.ndim != 4 or ground_truth.ndim != 4:
        raise ValueError("proposals and ground_truth must be [B,M,8,3] and [B,K,8,3]")
    if proposals.shape[-2:] != ground_truth.shape[-2:]:
        raise ValueError(f"trajectory contract mismatch: {tuple(proposals.shape[-2:])} vs {tuple(ground_truth.shape[-2:])}")
    if gt_valid.shape != ground_truth.shape[:2]:
        raise ValueError("gt_valid must be [B,K] aligned with ground_truth")
    distance = torch.linalg.norm(proposals[:, :, None] - ground_truth[:, None, :], dim=-1, ord=1).mean(-1)
    return distance.masked_fill(~gt_valid[:, None, :].bool(), float("inf"))


def hungarian_matched_l1(proposals: torch.Tensor, ground_truth: torch.Tensor, assignment: list[Assignment]) -> torch.Tensor:
    """DrivoR's exact L1 ADE, restricted to the Hungarian-matched pairs.

    With one valid GT per scene this is bit-identical to DrivoR's
    ``norm(p - gt[:, None], dim=-1, ord=1).mean(-1).amin(1).mean()``, because the
    unique match is also the unique winner.  With K GTs the Hungarian cost matrix
    assigns each GT a *distinct* candidate, so K different candidates receive the
    full-weight gradient instead of all K collapsing onto the same winner.
    """
    losses: list[torch.Tensor] = []
    for batch, match in enumerate(assignment):
        if match.gt_indices.numel() == 0:
            continue
        predicted = proposals[batch, match.candidate_indices]
        target = ground_truth[batch, match.gt_indices].detach()
        losses.append(torch.linalg.norm(predicted - target, dim=-1, ord=1).mean())
    if not losses:
        return proposals.sum() * 0.0
    return torch.stack(losses).mean()


def _masked_mean(values: torch.Tensor, weights: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    """Weighted mean over the finite entries; graph-preserving zero if there are none."""
    mask = (torch.isfinite(values) & weights.bool()).to(values.dtype)
    denominator = (weights * mask).sum()
    if float(denominator) <= 0.0:
        return fallback.sum() * 0.0
    return (torch.where(mask.bool(), values, torch.zeros_like(values)) * weights * mask).sum() / denominator


def drivor_dense_loss(proposals: torch.Tensor, ground_truth: torch.Tensor, gt_valid: torch.Tensor, gate_m: float = 0.0) -> torch.Tensor:
    """Small-weight term pulling every candidate toward its nearest valid GT.

    Hungarian matching gives K of the 64 candidates gradient and leaves the rest
    completely free, which is exactly the reported failure: "only the few
    GT-supervised candidates are good, overall quality is poor".  DrivoR's own
    remedy for unsupervised proposals is a weak shaping term plus
    :func:`drivor_diversity_loss`; this is that weak term.  Callers are expected to
    weight it well below :func:`hungarian_matched_l1` so it regularizes the set
    without overriding the matched supervision.

    ``gate_m > 0`` reweights each candidate by ``exp(-d / gate_m)``, keeping the
    pull local so a far-away anchor is not dragged across the whole scene.
    ``gate_m <= 0`` disables the gate and yields the plain mean over all
    candidates -- which also makes this value a direct read-out of overall set
    quality in metres, so it doubles as the "all-64 ADE" training metric.
    """
    per_candidate = candidate_gt_l1_ade(proposals, ground_truth, gt_valid).amin(dim=2)
    if gate_m > 0:
        weights = torch.exp(-torch.nan_to_num(per_candidate, posinf=0.0).clamp_min(0.0) / gate_m)
    else:
        weights = torch.ones_like(per_candidate)
    return _masked_mean(per_candidate, weights, proposals)


def drivor_diversity_loss(proposals: torch.Tensor) -> torch.Tensor:
    """DrivoR's ``diversity_loss`` verbatim: maximize the smallest pairwise gap."""
    distance = torch.linalg.norm(proposals[:, :, None] - proposals[:, None], dim=-1, ord=1).mean(-1)
    distance = distance + (distance == 0)
    return -distance.amin(1).amin(1).mean()


def multigt_trajectory_loss(
    proposals: torch.Tensor,
    ground_truth: torch.Tensor,
    gt_valid: torch.Tensor,
    *,
    dense_weight: float = 0.1,
    diversity_weight: float = 1.0,
    dense_gate_m: float = 8.0,
    assignment: list[Assignment] | None = None,
) -> dict[str, torch.Tensor]:
    """DrivoR's trajectory objective for the 64-candidate Multi-GT head.

    ``matched`` is DrivoR's own L1 minADE at full weight, applied to the Hungarian
    pairs.  ``dense`` and ``diversity`` are the two terms that keep the remaining
    candidates from being unsupervised: a weak nearest-GT pull and DrivoR's
    anti-collapse spread penalty.  This replaces the previous hard-Hungarian L2,
    which supervised one candidate per scene and was equal-weight over metres and
    radians, so ``x`` (<=76 m) swamped ``heading`` (~1.2 rad).  The matching cost
    in :func:`hungarian_xy` is still position-only; heading enters the loss.
    """
    if assignment is None:
        assignment = hungarian_xy(proposals.detach(), ground_truth, gt_valid)
    matched = hungarian_matched_l1(proposals, ground_truth, assignment)
    dense = drivor_dense_loss(proposals, ground_truth, gt_valid, gate_m=dense_gate_m)
    diversity = drivor_diversity_loss(proposals)
    total = matched + dense_weight * dense + diversity_weight * diversity
    return {"loss_trajectory": total, "matched_ade": matched, "dense_ade": dense, "diversity": diversity}


def flow_velocity_loss(velocity: torch.Tensor, velocity_target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """SimWAM's ``v_loss``: MSE on the rectified-flow velocity, bell-weighted by ``t``.

    ``velocity``/``velocity_target`` are ``[B,M,8,3]`` in the normalized
    ``[-1,1]`` action space, so the three dimensions are comparable and the MSE
    cannot be dominated by metres again.  ``weight`` is
    :meth:`ActionFlow.weight` at the sampled timestep, ``[B]``: SimWAM's
    shift-normalized Gaussian bell, which is exactly 0 at ``t=0`` and
    ``t=num_train_timesteps``.  That is what lets Stage 1 pin ``sigma = 0`` and
    get an automatically-zero velocity term without a separate code path.
    """
    if velocity.shape != velocity_target.shape:
        raise ValueError(f"velocity {tuple(velocity.shape)} and target {tuple(velocity_target.shape)} must match")
    if weight.shape != (velocity.shape[0],):
        raise ValueError(f"weight must be [B] aligned with velocity, got {tuple(weight.shape)}")
    per_sample = torch.nn.functional.mse_loss(
        velocity.float(), velocity_target.float(), reduction="none"
    ).mean(dim=(1, 2, 3))
    return (per_sample * weight.to(per_sample.device, per_sample.dtype)).mean()


def nearest_gt_index(proposals: torch.Tensor, ground_truth: torch.Tensor, gt_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[B,M]`` nearest-valid-GT index per candidate, plus a ``[B,M]`` has-GT flag.

    The index is a hard, detached choice: it selects *which* GT supervises a
    candidate and must never carry gradient itself.
    """
    distance = candidate_gt_l1_ade(proposals.detach(), ground_truth, gt_valid)
    return distance.argmin(dim=2), torch.isfinite(distance).any(dim=2)


def gather_flow_data(anchor: torch.Tensor, ground_truth: torch.Tensor, gt_valid: torch.Tensor) -> torch.Tensor:
    """Per-candidate ``sigma = 0`` endpoint of the rectified flow: ``[B,M,8,3]``.

    Each candidate flows toward its nearest valid GT, measured on the *anchor* so
    the target is fixed for a given candidate index and does not chase the model's
    own output.  A scene with no valid GT falls back to ``data == anchor``, whose
    velocity target is exactly zero: that candidate is asked to stay put rather
    than to invent a trajectory.
    """
    index, supervised = nearest_gt_index(anchor, ground_truth, gt_valid)
    batch, candidates = index.shape
    gathered = torch.gather(
        ground_truth, 1, index.view(batch, candidates, 1, 1).expand(batch, candidates, *anchor.shape[2:])
    )
    return torch.where(supervised.view(batch, candidates, 1, 1), gathered, anchor)


def dense_future_l1(predicted_register: torch.Tensor, targets: PairTargets, proposals: torch.Tensor) -> torch.Tensor:
    """Stage-2 future loss: every candidate is supervised by its nearest GT's future.

    A Hungarian-only pairing could reach just the K candidates the match happened to
    pick, so 60+ of the 64 register rows would get zero future gradient even though
    Stage 1 had trained them -- and the future queries share the DiT trunk with the
    action tokens, so an untrained row is an untrained trajectory too.  Pairing each
    candidate with the GT it is closest to makes the future branch dense and
    consistent with :func:`drivor_dense_loss`.
    """
    index, supervised = nearest_gt_index(proposals, targets.gt_trajectory, targets.gt_valid)
    b, m = index.shape
    tokens, width = targets.future_register.shape[2], targets.future_register.shape[3]
    future = torch.gather(targets.future_register, 1, index.view(b, m, 1, 1).expand(b, m, tokens, width))
    per_token = index.view(b, m, 1).expand(b, m, tokens)
    # The validity mask and quality live on the GT axis, so they must be gathered
    # with the same index -- otherwise a candidate inherits another GT's validity.
    weight = torch.gather(targets.future_quality, 1, per_token) * torch.gather(targets.future_chunk_valid, 1, per_token)
    weight = weight * supervised.view(b, m, 1).to(weight.dtype)
    denominator = weight.sum()
    if float(denominator) <= 0:
        return predicted_register.sum() * 0.0
    # ``mean(dim=-1)`` reduces the 1024 feature width to one number per register token
    # and the weighted mean then reduces the token axis, so widening the target from one
    # token per frame to four leaves the loss's scale untouched -- it stays an
    # element-mean over the target, not a sum over it.
    error = (predicted_register - future.detach()).abs().mean(dim=-1)
    return (error * weight).sum() / denominator


def _quality_weighted_chunk_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    chunk_valid: torch.Tensor,
    quality: torch.Tensor,
) -> torch.Tensor | None:
    """Quality- and validity-weighted L1 over the register targets of one row.

    Returns ``None`` when the row carries no usable token, so callers can skip it
    instead of dividing by a zero weight sum.  The reduction is an element-mean: the
    feature width first, then a weighted mean over the token axis, so the value does not
    scale with how many tokens a frame contributes.
    """

    weights = quality.to(prediction.device) * chunk_valid.to(prediction.device)
    if weights.sum() <= 0:
        return None
    error = (prediction - target.detach()).abs().mean(dim=-1)
    return (error * weights).sum() / weights.sum()


def identity_future_l1(predicted_register: torch.Tensor, targets: PairTargets) -> torch.Tensor:
    """Stage-1 future loss over the pairing the dataset already provides.

    Stage 1 feeds ``targets.gt_trajectory`` in as the clean action, so register row
    ``k`` is the DiT's read-out for GT ``k`` and is supervised by GT ``k``'s
    per-frame future: no Hungarian assignment, and every valid GT row receives
    gradient rather than only the few candidates an anchor codebook happened to
    match.

    All four axes are checked, not just ``(B, K)``.  A row-count mismatch is the
    one that breaks the pairing, but a width mismatch -- ``future_head`` built at
    ``d_model`` while the cache's standardizer says the Teacher's own feature width
    -- is the likelier one, and unchecked it surfaces as a broadcast ``RuntimeError``
    from inside ``_quality_weighted_chunk_l1`` that names neither tensor nor axis.
    """

    if predicted_register.shape != targets.future_register.shape:
        raise ValueError(
            "identity pairing needs one register row per GT row, shaped exactly like the "
            "cached target, got "
            f"{tuple(predicted_register.shape)} against {tuple(targets.future_register.shape)} "
            "as [B, K, register_tokens, register_target_dim]"
        )
    losses: list[torch.Tensor] = []
    for batch in range(predicted_register.shape[0]):
        for gt_index in torch.nonzero(targets.gt_valid[batch], as_tuple=False).flatten().tolist():
            loss = _quality_weighted_chunk_l1(
                predicted_register[batch, gt_index],
                targets.future_register[batch, gt_index],
                targets.future_chunk_valid[batch, gt_index],
                targets.future_quality[batch, gt_index],
            )
            if loss is not None:
                losses.append(loss)
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
