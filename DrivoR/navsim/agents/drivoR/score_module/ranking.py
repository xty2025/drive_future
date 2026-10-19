"""Numerically stable form of DrivoR's proposal ranking equation."""

import math

import torch
from torch.nn import functional as F


def aggregate_score_logits(logits, config):
    log_probabilities = {
        name: F.logsigmoid(value.float()) for name, value in logits.items()
    }
    # Evaluate the weighted probability sum in log-space to retain ranking even
    # when every probability would underflow in fp16/bf16/float32.
    weighted_terms = [
        log_probabilities[name] + math.log(weight)
        for name, weight in (
            ("time_to_collision_within_bound", config.ttc),
            ("ego_progress", config.ep),
            ("comfort", config.comfort),
        )
        if weight != 0
    ]
    score = torch.logsumexp(torch.stack(weighted_terms), dim=0)
    for name, weight in (
        ("no_at_fault_collisions", config.noc),
        ("drivable_area_compliance", config.dac),
        ("driving_direction_compliance", config.ddc),
    ):
        # A disabled DDC head must not produce 0 * -inf = NaN.
        if weight != 0:
            score = score + weight * log_probabilities[name]
    return score
