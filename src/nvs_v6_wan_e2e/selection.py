"""DrivoR source-compatible score aggregation and proposal selection."""

from __future__ import annotations

import torch


HEAD_NAMES = ("NC", "DAC", "DDC", "TTC", "EP", "C")

# navsim/planning/script/config/common/agent/drivoR.yaml
DRIVOR_WEIGHTS = {
    "NC": 1.0,
    "DAC": 1.0,
    "DDC": 0.0,
    "TTC": 5.0,
    "EP": 5.0,
    "C": 2.0,
}


def aggregate_score_logits(logits: torch.Tensor) -> torch.Tensor:
    """Match ``DrivoRModel.forward`` exactly (up to an argmax-neutral scale)."""
    if logits.shape[-1] != 6:
        raise ValueError("DrivoR scorer must emit six named heads")
    probabilities = torch.sigmoid(logits)
    nc, dac, ddc, ttc, ep, comfort = probabilities.unbind(dim=-1)
    return (
        DRIVOR_WEIGHTS["NC"] * nc.log()
        + DRIVOR_WEIGHTS["DAC"] * dac.log()
        + DRIVOR_WEIGHTS["DDC"] * ddc.log()
        + (
            DRIVOR_WEIGHTS["TTC"] * ttc
            + DRIVOR_WEIGHTS["EP"] * ep
            + DRIVOR_WEIGHTS["C"] * comfort
        ).log()
    )


def select_best_candidate(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    utility = aggregate_score_logits(logits)
    index = utility.argmax(dim=1)
    return index, utility
