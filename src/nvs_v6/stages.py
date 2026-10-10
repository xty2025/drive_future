"""Stage 1/2 training steps following the v6 gradient and matching contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .data import Observation, PairTargets
from .losses import (
    assert_disjoint_parameter_groups,
    hungarian_xy,
    paired_future_l1,
    paired_residual_velocity_mse,
)
from .model import DirectedPairDiT, parameter_groups
from .lora import freeze_base_keep_lora
from .modes import ImmutableModeBuffer, StructuredSourceInitializer
from .oracle import OnlineOracle
from .scorer import ScratchDrivoRScorer


@dataclass
class Stage1Result:
    loss_future: torch.Tensor
    assignment: list[Any]


@dataclass
class Stage2Result:
    loss_action: torch.Tensor
    loss_future: torch.Tensor
    loss_score: torch.Tensor
    assignment: list[Any]
    oracle: Any


def _set_generator_trainability(model: DirectedPairDiT, stage: int) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for name, parameter in model.named_parameters():
        if (
            ".base." in name
            or name.endswith("student.patch.weight")
            or name.endswith("student.patch.bias")
            or name.startswith("eta.student.norm.")
            or name.startswith("_unused")
        ):
            parameter.requires_grad_(False)
    # Re-applying stage trainability must never unfreeze the DINOv3 base.
    # Only LoRA adapters (plus the NVS scene-query/projection modules outside
    # the backbone) remain trainable.
    student = model.eta.student
    if hasattr(student, "backbone"):
        freeze_base_keep_lora(student.backbone)
    if stage == 1:
        for parameter in model.theta.parameters():
            parameter.requires_grad_(False)
        model.theta_time_embedding.requires_grad_(False)


def _set_scorer_trainability(scorer: ScratchDrivoRScorer, enabled: bool) -> None:
    for parameter in scorer.parameters():
        parameter.requires_grad_(enabled)


def build_optimizers(model: DirectedPairDiT, scorer: ScratchDrivoRScorer, stage: int, lr: float = 2e-4, scorer_lr: float = 5e-4) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    _set_generator_trainability(model, stage)
    _set_scorer_trainability(scorer, stage == 2)
    groups = parameter_groups(model)
    generator_groups = [groups["eta"], groups["phi"]] if stage == 1 else [groups["eta"], groups["theta"], groups["phi"]]
    generator_parameters = [parameter for group in generator_groups for parameter in group]
    # Keep a complete scorer optimizer in both stages.  Stage 1 parameters are
    # frozen and therefore have no gradients; Stage 2 enables this exact set.
    scorer_parameters = list(scorer.parameters())
    generator_ids = [id(parameter) for parameter in generator_parameters]
    scorer_ids = [id(parameter) for parameter in scorer_parameters]
    if len(generator_ids) != len(set(generator_ids)) or len(scorer_ids) != len(set(scorer_ids)):
        raise AssertionError("an optimizer parameter set contains duplicate Parameters")
    expected_generator_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if set(generator_ids) != expected_generator_ids:
        raise AssertionError("trainable generator Parameters are missing from optimizer groups")
    if set(scorer_ids) != {id(parameter) for parameter in scorer.parameters()}:
        raise AssertionError("scorer optimizer does not contain every scorer Parameter exactly once")
    assert_disjoint_parameter_groups(generator_parameters, scorer_parameters)
    return torch.optim.AdamW(generator_parameters, lr=lr), torch.optim.AdamW(scorer_parameters, lr=scorer_lr)


def stage1_step(
    observation: Observation,
    targets: PairTargets,
    model: DirectedPairDiT,
    optimizer: torch.optim.Optimizer,
    modes: ImmutableModeBuffer,
    source_initializer: StructuredSourceInitializer,
) -> Stage1Result:
    model.train()
    _set_generator_trainability(model, stage=1)
    optimizer.zero_grad(set_to_none=True)
    mode_codes = modes.expand_for_batch(observation.rgb_history.shape[0], observation.rgb_history.device)
    predicted_register = model.forward_register_only(observation, mode_codes)
    with torch.no_grad():
        clean_source = source_initializer.clean(observation.ego_history, mode_codes)
        clean_trajectory = _decode_clean(clean_source)
        assignment = hungarian_xy(clean_trajectory, targets.gt_trajectory, targets.gt_valid)
    loss_future = paired_future_l1(predicted_register, targets, assignment)
    loss_future.backward()
    optimizer.step()
    return Stage1Result(loss_future.detach(), assignment)


def stage2_step(
    observation: Observation,
    targets: PairTargets,
    scene_ids: list[str],
    model: DirectedPairDiT,
    scorer: ScratchDrivoRScorer,
    oracle: OnlineOracle,
    opt_generator: torch.optim.Optimizer,
    opt_scorer: torch.optim.Optimizer,
    modes: ImmutableModeBuffer,
    source_initializer: StructuredSourceInitializer,
    lambda_future: float = 0.1,
    lambda_action: float = 1.0,
    lambda_score: float = 1.0,
    batch_id: str = "",
) -> Stage2Result:
    model.train()
    scorer.train()
    _set_generator_trainability(model, stage=2)
    _set_scorer_trainability(scorer, True)
    opt_generator.zero_grad(set_to_none=True)
    opt_scorer.zero_grad(set_to_none=True)
    mode_codes = modes.expand_for_batch(observation.rgb_history.shape[0], observation.rgb_history.device)
    # DrivoR proposals are deterministic for a fixed scene/model.  Keep the
    # ActionDiT source deterministic as well so score supervision and inference
    # refer to the same candidate identities.
    source = source_initializer.clean(observation.ego_history, mode_codes)
    outputs = model.forward_pairs(observation, source, mode_codes)
    # Score supervision is isolated to psi: all generator-side inputs are
    # detached before the scorer.  The compatibility scorer may ignore the
    # register argument, but passing it detached keeps the pipeline boundary
    # explicit without creating a Score -> eta/theta/phi gradient path.
    score_logits = scorer(
        outputs.scene_context.detach(),
        outputs.trajectory.detach(),
        outputs.predicted_register.detach(),
        detach_inputs=True,
    )
    with torch.no_grad():
        oracle_labels = oracle.evaluate(scene_ids, outputs.trajectory.detach(), batch_id=batch_id)
        assignment = hungarian_xy(outputs.trajectory.detach(), targets.gt_trajectory, targets.gt_valid)
    loss_action = paired_residual_velocity_mse(outputs.endpoint, source, targets.gt_encoded, assignment)
    loss_future = paired_future_l1(outputs.predicted_register, targets, assignment)
    if not hasattr(scorer, "source_loss"):
        raise TypeError("production Stage 2 requires official DrivoRLoss.score_loss")
    loss_score = scorer.source_loss(score_logits, oracle_labels)
    (
        lambda_action * loss_action
        + lambda_future * loss_future
        + lambda_score * loss_score
    ).backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    # DrivoR's training config sets gradient_clip_val=0.0 for the scorer path.
    opt_generator.step()
    opt_scorer.step()
    if hasattr(scorer, "record_comparison"):
        scorer.record_comparison(outputs.trajectory.detach(), oracle_labels)
    return Stage2Result(loss_action.detach(), loss_future.detach(), loss_score.detach(), assignment, oracle_labels)


def _decode_clean(encoded: torch.Tensor) -> torch.Tensor:
    x = encoded[..., 0] * 20.0
    y = encoded[..., 1] * 20.0
    yaw = torch.atan2(encoded[..., 2], encoded[..., 3])
    return torch.stack((x, y, yaw), dim=-1)
