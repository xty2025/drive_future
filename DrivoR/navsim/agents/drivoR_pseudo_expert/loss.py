"""Add CLOVER's final-proposal coverage term to the existing DrivoR objective."""

from navsim.agents.drivoR.layers.losses.drivor_loss import DrivoRLoss


def multi_expert_coverage_loss(proposals, pseudo_experts, mask):
    """Mean over scenes of mean valid-expert nearest-proposal L1(x,y,heading).

    Coordinates are physical ego-relative poses. Sum the three state dimensions,
    average the eight times, then independently minimize over proposals per GT.
    As in CLOVER, amin shares gradients if several proposals tie exactly.
    """
    mask = mask.bool()
    experts = pseudo_experts.detach().float().masked_fill(~mask[..., None, None], 0.)
    distance = (proposals.float()[:, :, None] - experts[:, None]).abs().sum(-1).mean(-1)
    per_expert = distance.amin(dim=1)
    return ((per_expert * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()


class DrivoRPseudoExpertLoss(DrivoRLoss):
    def __init__(self, pseudo_expert_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.pseudo_expert_weight = float(pseudo_expert_weight)

    def forward(self, targets, pred, config, scoring_function=None):
        result = super().forward(targets, pred, config, scoring_function)
        if self.pseudo_expert_weight == 0:
            return result
        coverage = multi_expert_coverage_loss(
            pred["proposals"], targets["pseudo_experts"], targets["pseudo_expert_mask"])
        result["pseudo_expert_loss"] = coverage
        result["pseudo_expert_count"] = targets["pseudo_expert_mask"].sum(-1).float().mean()
        result["pseudo_expert_fallback_fraction"] = targets["pseudo_expert_fallback"].float().mean()
        added = self.pseudo_expert_weight * coverage
        result["trajectory_loss"] = result["trajectory_loss"] + added
        result["loss"] = result["loss"] + self.trajectory_weight * added
        return result
