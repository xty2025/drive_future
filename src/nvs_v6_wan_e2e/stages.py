"""Stage 1/2 training steps following the v6 gradient and matching contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .data import Observation, PairTargets
from .losses import (
    assert_disjoint_parameter_groups,
    dense_future_l1,
    flow_velocity_loss,
    gather_flow_data,
    hungarian_xy,
    identity_future_l1,
    multigt_trajectory_loss,
)
from .model import DirectedPairDiT, parameter_groups
from .vision import freeze_backbone_keep_lora
from .modes import ImmutableModeBuffer, StructuredSourceInitializer
from .oracle import OnlineOracle
from .scorer import ScratchDrivoRScorer


@dataclass
class Stage1Result:
    loss_future: torch.Tensor
    supervised_rows: int


@dataclass
class Stage2Result:
    loss_action: torch.Tensor
    loss_future: torch.Tensor
    loss_score: torch.Tensor
    assignment: list[Any]
    oracle: Any
    trajectory_terms: dict[str, torch.Tensor] = field(default_factory=dict)
    loss_velocity: torch.Tensor = field(default_factory=lambda: torch.zeros(()))
    timestep: torch.Tensor | None = None


def _set_generator_trainability(model: DirectedPairDiT, stage: int) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for name, parameter in model.named_parameters():
        if (
            ".qkv.qkv." in name
            or name.endswith("student.patch.weight")
            or name.endswith("student.patch.bias")
            or name.startswith("_unused")
        ):
            parameter.requires_grad_(False)
    # Re-applying stage trainability must never unfreeze the DINOv3 base.  Only
    # DrivoR's q/v LoRA adapters plus the NVS modules outside the backbone
    # (scene_embeds, neck, hist_encoding, scene_blocks) stay trainable.
    student = model.eta.student
    if hasattr(student, "backbone"):
        freeze_backbone_keep_lora(student.backbone)
    if stage == 1:
        for parameter in model.theta.parameters():
            parameter.requires_grad_(False)


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
) -> Stage1Result:
    """Trajectory-conditioned future representation training.

    The register is conditioned on each scene's own GT trajectories, so the row
    pairing is the identity and no anchor codebook or Hungarian assignment takes
    part: row ``k`` must predict the compressed future of GT ``k``.  ``modes`` is
    deliberately not passed, because a mode code constant per row index would let
    the network identify the GT from its row alone and skip the trajectory.
    """

    model.train()
    _set_generator_trainability(model, stage=1)
    optimizer.zero_grad(set_to_none=True)
    source = targets.gt_trajectory
    predicted_register = model.forward_register_only(observation, source)
    loss_future = identity_future_l1(predicted_register, targets)
    loss_future.backward()
    optimizer.step()
    supervised_rows = int(targets.gt_valid.sum())
    return Stage1Result(loss_future.detach(), supervised_rows)


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
    lambda_velocity: float = 1.0,
    dense_weight: float = 0.1,
    diversity_weight: float = 1.0,
    dense_gate_m: float = 8.0,
    batch_id: str = "",
) -> Stage2Result:
    """Joint generator+scorer step with DrivoR's gradient path and dense multi-GT targets.

    The ActionDiT is run as a rectified flow (see :mod:`nvs_v6_wan_e2e.flow`): one
    timestep is sampled per scene per forward, the anchor grid is interpolated
    toward each candidate's nearest GT, and the same forward both predicts the
    velocity and inverts it back to a trajectory -- no rollout.  Two terms
    supervise that trajectory:

    * ``loss_velocity`` -- SimWAM's ``v_loss``, MSE against ``anchor - GT`` in the
      normalized action space, bell-weighted by ``t``.
    * ``loss_action`` -- DrivoR's own objective: L1 on the Hungarian-matched
      (GT, candidate) pairs at full weight, plus a weak dense nearest-GT pull and
      ``diversity_loss`` so the unmatched candidates are shaped rather than left
      free.

    The score gradient path is DrivoR's verbatim: proposals enter the scorer
    detached while the Scene context stays attached, so ``loss_score`` shapes eta
    but never theta/phi.
    """
    model.train()
    scorer.train()
    _set_generator_trainability(model, stage=2)
    _set_scorer_trainability(scorer, True)
    opt_generator.zero_grad(set_to_none=True)
    opt_scorer.zero_grad(set_to_none=True)
    batch = observation.rgb_history.shape[0]
    device = observation.rgb_history.device
    mode_codes = modes.expand_for_batch(batch, device)
    # DrivoR proposals are deterministic for a fixed scene/model.  Keep the
    # ActionDiT source deterministic as well so score supervision and inference
    # refer to the same candidate identities; the flow's randomness lives in the
    # sampled timestep, not in the anchor grid.
    source = source_initializer.clean(observation.ego_history, mode_codes)
    timestep = model.flow.sample_timestep((batch,), device, source.dtype)
    flow_data = gather_flow_data(source, targets.gt_trajectory, targets.gt_valid)
    flow = model.forward_flow(observation, source, mode_codes, flow_data, timestep)
    outputs = flow.pairs
    # DrivoR-exact end-to-end: the Scene context is passed attached so the score
    # loss shapes eta (SceneEncoder + DINOv3 LoRA + scene_blocks).  Proposals and
    # the (ignored) register stay detached, so there is still no Score ->
    # theta/phi gradient path.
    score_logits = scorer(
        outputs.scene_context,
        outputs.trajectory.detach(),
        outputs.predicted_register.detach(),
        detach_inputs=True,
    )
    with torch.no_grad():
        oracle_labels = oracle.evaluate(scene_ids, outputs.trajectory.detach(), batch_id=batch_id)
        assignment = hungarian_xy(outputs.trajectory.detach(), targets.gt_trajectory, targets.gt_valid)
    trajectory_terms = multigt_trajectory_loss(
        outputs.trajectory,
        targets.gt_trajectory,
        targets.gt_valid,
        dense_weight=dense_weight,
        diversity_weight=diversity_weight,
        dense_gate_m=dense_gate_m,
        assignment=assignment,
    )
    loss_action = trajectory_terms["loss_trajectory"]
    loss_velocity = flow_velocity_loss(flow.velocity, flow.velocity_target, model.flow.weight(timestep))
    loss_future = dense_future_l1(outputs.predicted_register, targets, outputs.trajectory)
    if not hasattr(scorer, "source_loss"):
        raise TypeError("production Stage 2 requires official DrivoRLoss.score_loss")
    loss_score = scorer.source_loss(score_logits, oracle_labels)
    (
        lambda_action * loss_action
        + lambda_velocity * loss_velocity
        + lambda_future * loss_future
        + lambda_score * loss_score
    ).backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    # DrivoR's training config sets gradient_clip_val=0.0 for the scorer path.
    opt_generator.step()
    opt_scorer.step()
    if hasattr(scorer, "record_comparison"):
        scorer.record_comparison(outputs.trajectory.detach(), oracle_labels)
    return Stage2Result(
        loss_action.detach(),
        loss_future.detach(),
        loss_score.detach(),
        assignment,
        oracle_labels,
        {key: value.detach() for key, value in trajectory_terms.items()},
        loss_velocity.detach(),
        timestep.detach(),
    )
