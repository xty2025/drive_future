"""Flow trajectory supervision plus the unmodified official DrivoR score loss."""

import torch

from navsim.agents.drivoR.layers.losses.drivor_loss import DrivoRLoss


class DrivoRActionDiTLoss(DrivoRLoss):
    def forward(self, targets, pred, config, scoring_function, flow_loss):
        proposals = pred["proposals"]
        final_scores, best_scores, raw7, gt_states, gt_valid, gt_areas = scoring_function(
            targets, proposals, test=False)
        sub_losses, score_loss, _, _, _ = self.score_loss(
            pred["pred_logit"], pred["pred_logit2"], pred["pred_agents_states"],
            pred["pred_area_logit"], raw7, gt_states, gt_valid, gt_areas, None)
        selected = pred["pdm_score"].detach().argmax(dim=1)
        rows = torch.arange(len(selected), device=selected.device)
        result = {
            "loss": self.trajectory_weight * flow_loss + self.final_score_weight * score_loss,
            "trajectory_loss": flow_loss,
            "final_score_loss": score_loss,
            "score": final_scores[rows, selected].mean(),
            "best_score": best_scores.mean(),
        }
        result.update(zip(("da_loss", "ttc_loss", "noc_loss", "progress_loss", "ddc_loss", "comfort_loss"),
                          sub_losses))
        return result
