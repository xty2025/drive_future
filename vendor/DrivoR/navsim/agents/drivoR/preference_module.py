"""Preference objectives for PASP-DrivoR.

The policy distribution is always over the model's current dynamic proposal
slots.  External BeyondDrive trajectories therefore never enter a DPO ratio;
they are used only by the scorer ranking and repulsive-distance objectives.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def unified_target(
    proposals: torch.Tensor,
    target_trajectory: torch.Tensor,
    pdm_scores: torch.Tensor,
    human_weight: float = 1.0,
    pdm_weight: float = 1.0,
    human_temperature: float = 1.0,
    target_temperature: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """DriveDPO-style dense target distribution over current proposals."""
    distance = torch.linalg.norm(proposals - target_trajectory[:, None], dim=-1).mean(-1)
    human_log_prob = F.log_softmax(-distance / max(human_temperature, eps), dim=1)
    pdm_log_prob = F.log_softmax(pdm_scores.clamp_min(0.0) / max(target_temperature, eps), dim=1)
    return F.softmax(human_weight * human_log_prob + pdm_weight * pdm_log_prob, dim=1)


def mine_pair(
    proposals: torch.Tensor,
    target_trajectory: torch.Tensor,
    pdm_scores: torch.Tensor,
    unsafe_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mine DriveDPO chosen/rejected slot indices and a valid-pair mask."""
    target = unified_target(proposals, target_trajectory, pdm_scores)
    chosen = target.argmax(dim=1)
    distance = torch.linalg.norm(proposals - target_trajectory[:, None], dim=-1).mean(-1)
    unsafe = pdm_scores < unsafe_threshold
    unsafe_distance = distance.masked_fill(~unsafe, float("inf"))
    rejected = unsafe_distance.argmin(dim=1)
    valid = torch.isfinite(unsafe_distance.gather(1, rejected[:, None]).squeeze(1))
    valid &= rejected != chosen
    return chosen, rejected, valid


def preference_losses(
    proposals: torch.Tensor,
    policy_logits: torch.Tensor,
    reference_policy_logits: Optional[torch.Tensor],
    target_trajectory: torch.Tensor,
    pdm_scores: torch.Tensor,
    scorer_scores: Optional[torch.Tensor] = None,
    negative_trajectory: Optional[torch.Tensor] = None,
    unsafe_threshold: float = 0.2,
    dpo_beta: float = 0.1,
    unified_weight: float = 1.0,
    dpo_weight: float = 1.0,
    rank_weight: float = 0.1,
    rd_weight: float = 0.1,
    rd_margin: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Compute dense policy, slot-DPO, scorer ranking, and external RD losses."""
    with torch.no_grad():
        pdm = pdm_scores.detach()
        target = unified_target(proposals.detach(), target_trajectory.detach(), pdm)
        chosen, rejected, valid = mine_pair(
            proposals.detach(), target_trajectory.detach(), pdm, unsafe_threshold
        )

    log_probs = F.log_softmax(policy_logits, dim=1)
    unified = F.kl_div(log_probs, target, reduction="batchmean")

    # DPO requires a reference independent of the current policy update.  A
    # fixed uniform slot prior is the from-scratch reference; callers may pass
    # logits from a separately frozen reference policy when one is available.
    if reference_policy_logits is None:
        ref_log_probs = torch.full_like(log_probs, -torch.log(
            log_probs.new_tensor(float(log_probs.shape[1]))
        ))
    else:
        if reference_policy_logits.shape != policy_logits.shape:
            raise ValueError(
                "reference policy shape mismatch: "
                f"current={tuple(policy_logits.shape)} reference={tuple(reference_policy_logits.shape)}"
            )
        ref_log_probs = F.log_softmax(reference_policy_logits.detach(), dim=1)
    chosen_logp = log_probs.gather(1, chosen[:, None]).squeeze(1)
    rejected_logp = log_probs.gather(1, rejected[:, None]).squeeze(1)
    ref_chosen = ref_log_probs.gather(1, chosen[:, None]).squeeze(1)
    ref_rejected = ref_log_probs.gather(1, rejected[:, None]).squeeze(1)
    dpo_per_sample = -F.logsigmoid(dpo_beta * ((chosen_logp - rejected_logp) - (ref_chosen - ref_rejected)))
    dpo = _masked_mean(dpo_per_sample, valid)

    zero = policy_logits.sum() * 0.0
    rank = zero
    if scorer_scores is not None:
        scorer_chosen = scorer_scores.gather(1, chosen[:, None]).squeeze(1)
        scorer_rejected = scorer_scores.gather(1, rejected[:, None]).squeeze(1)
        rank = _masked_mean(F.softplus(-(scorer_chosen - scorer_rejected)), valid)

    rd = zero
    if negative_trajectory is not None:
        chosen_traj = proposals[torch.arange(proposals.shape[0], device=proposals.device), chosen]
        distance = torch.linalg.norm(chosen_traj - negative_trajectory, dim=-1).mean(-1)
        rd = -distance.clamp_max(rd_margin).mean()

    total = unified_weight * unified + dpo_weight * dpo + rank_weight * rank + rd_weight * rd
    return {
        "preference_loss": total,
        "unified_loss": unified,
        "dpo_loss": dpo,
        "rank_loss": rank,
        "rd_loss": rd,
        "preference_pair_rate": valid.float().mean(),
    }
