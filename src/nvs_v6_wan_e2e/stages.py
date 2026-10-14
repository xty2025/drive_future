"""Stage 1 training step and the shared optimizer/trainability plumbing.

Stage 2 does not live here.  Its production path is ``run_navtrain_wan_e2e_ddp.py``,
which needs grad-accumulation, an LR schedule and the intermediate flow tensors
(``velocity``, ``predicted_register``, ``logits``) for logging -- none of which fit
a self-contained ``zero_grad/backward/step`` helper.  Keeping a second Stage 2 here
would only drift from it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import Observation, PairTargets
from .losses import assert_disjoint_parameter_groups, identity_future_l1
from .model import DirectedPairDiT, parameter_groups
from .vision import freeze_backbone_keep_lora
from .scorer import ScratchDrivoRScorer


@dataclass
class Stage1Result:
    loss_future: torch.Tensor
    supervised_rows: int


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

    The input is noise-free by construction: ``forward_register_only`` runs the
    clean trajectory, which is the ``sigma = 0`` end of the same flow Stage 2
    noises.  Only ``phi`` learns here -- ``theta`` is frozen by
    ``_set_generator_trainability(model, 1)``.
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
