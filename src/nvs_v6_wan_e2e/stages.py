"""Stage 1 training step and the shared optimizer/trainability plumbing.

Stage 2 does not live here.  Its production path is ``run_navtrain_wan_e2e_ddp.py``,
which needs grad-accumulation, an LR schedule and the intermediate flow tensors
(``velocity``, ``predicted_register``, ``logits``) for logging -- none of which fit
a self-contained ``zero_grad/backward/step`` helper.  Keeping a second Stage 2 here
would only drift from it.
"""

from __future__ import annotations

import math
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
        # The future queries live inside theta now, so Stage 1 trains the DiT
        # trunk rather than freezing it.  The velocity head is the one thing Stage 1
        # freezes: nothing in the Stage 1 loss reads it, and leaving it trainable
        # would put its near-zero anchor-preserving initialization at the mercy of
        # whatever optimizer replaces AdamW (which skips grad-less parameters).
        for name, parameter in model.theta.named_parameters():
            if name.startswith("head."):
                parameter.requires_grad_(False)


def _set_scorer_trainability(scorer: ScratchDrivoRScorer, enabled: bool) -> None:
    for parameter in scorer.parameters():
        parameter.requires_grad_(enabled)


def build_optimizers(model: DirectedPairDiT, scorer: ScratchDrivoRScorer, stage: int, lr: float = 2e-4, scorer_lr: float = 5e-4) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    _set_generator_trainability(model, stage)
    _set_scorer_trainability(scorer, stage == 2)
    groups = parameter_groups(model)
    generator_parameters = [parameter for group in groups.values() for parameter in group]
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


def schedule_factor(step: int, total: int, warmup: int) -> float:
    """Linear warmup then cosine decay to 5% of base LR.

    Shared by both DDP trainers so a ``--lr`` on a command line means the same curve
    in Stage 1 and Stage 2.  ``step`` is 1-based: the first optimizer step runs at
    ``1 / warmup`` of the base rate rather than at zero, which is what keeps a warm
    start from taking a full-size AdamW update on an unestimated second moment.
    """

    if warmup > 0 and step <= warmup:
        return max(step, 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def stage1_step(
    observation: Observation,
    targets: PairTargets,
    model: DirectedPairDiT,
    optimizer: torch.optim.Optimizer,
) -> Stage1Result:
    """Trajectory-conditioned future representation training.

    The future queries are conditioned on each scene's own GT trajectories, so the
    row pairing is the identity and no anchor codebook or Hungarian assignment
    takes part: row ``k`` must predict the compressed future of GT ``k``.  The
    mode code is *zeroed* rather than per-row, because a code constant per row
    index would let the network identify the GT from its row alone and skip the
    trajectory.

    The input is noise-free by construction: ``forward_register_only`` runs the
    clean trajectory, which is the ``sigma = 0`` end of the same flow Stage 2
    noises.  Everything except the velocity head learns here -- which is exactly
    "train the future representation": no action-generation objective is optimized
    in this stage.  The single segment mask applies, so the register reads the
    clean action while the action cannot read the register back; that is the same
    information flow Stage 2 and the deployed path run, which is what makes
    ``loss_future`` comparable across stages.
    """

    model.train()
    _set_generator_trainability(model, stage=1)
    optimizer.zero_grad(set_to_none=True)
    source = targets.gt_trajectory
    modes = torch.zeros((*source.shape[:2], 2), device=source.device, dtype=source.dtype)
    predicted_register = model.forward_register_only(observation, source, modes)
    loss_future = identity_future_l1(predicted_register, targets)
    loss_future.backward()
    optimizer.step()
    supervised_rows = int(targets.gt_valid.sum())
    return Stage1Result(loss_future.detach(), supervised_rows)
