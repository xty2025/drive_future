"""Direct trajectory L1, matched future regression, and official scorer loss."""

import torch

from navsim.agents.drivoR.layers.losses.drivor_loss import DrivoRLoss


def match_trajectories(proposals, target, valid):
    """Independent nearest proposal per GT; average valid GTs then scenes."""
    valid = valid.bool()
    target = target.detach().float().masked_fill(~valid[..., None, None], 0.)
    cost = (proposals.float()[:, None] - target[:, :, None]).abs().sum(-1).mean(-1)
    values, indices = cost.min(-1)
    loss = (values.masked_fill(~valid, 0.).sum(-1) / valid.sum(-1).clamp_min(1)).mean()
    return loss, indices.masked_fill(~valid, -1)


def matched_future_loss(predicted, target, valid, matched, loss_type='l1'):
    """Use exactly the Action winner for each GT's [8,4,D] future targets."""
    rows = torch.arange(len(predicted), device=predicted.device)[:, None]
    chosen = predicted[rows, matched.detach().clamp_min(0)]
    valid = valid.bool() & (matched >= 0)[..., None]
    target = target.detach().float().masked_fill(~valid[..., None, None], 0.)
    difference = chosen.float() - target
    if loss_type == 'l1':
        error = difference.abs()
    elif loss_type == 'mse':
        error = difference.square()
    else:
        raise ValueError('Direct future-register loss must be l1 or mse')
    per_frame = error.mean((-1, -2))
    per_gt = per_frame.masked_fill(~valid, 0.).sum(-1) / valid.sum(-1).clamp_min(1)
    gt_valid = valid.any(-1)
    return (per_gt.masked_fill(~gt_valid, 0.).sum(-1) / gt_valid.sum(-1).clamp_min(1)).mean()


class DrivoRQueryRegisterLoss(DrivoRLoss):
    def forward(self, targets, pred, config, scoring_function):
        scores, best_scores, raw7, gt_states, gt_valid, gt_areas = scoring_function(
            targets, pred['proposals'], test=False)
        parts, score_loss, _, _, _ = self.score_loss(
            pred['pred_logit'], pred['pred_logit2'], pred['pred_agents_states'],
            pred['pred_area_logit'], raw7, gt_states, gt_valid, gt_areas, None)
        trajectory = pred['_trajectory_loss']
        future = pred['_register_loss']
        chosen = pred['pdm_score'].detach().argmax(1)
        rows = torch.arange(len(chosen), device=chosen.device)
        result = {
            'loss': self.trajectory_weight * trajectory
                    + float(config.future_register.loss_weight) * future
                    + self.final_score_weight * score_loss,
            'trajectory_loss': trajectory, 'future_register_loss': future,
            'final_score_loss': score_loss,
            'score': scores[rows, chosen].mean(), 'best_score': best_scores.mean(),
        }
        result.update(zip(('da_loss', 'ttc_loss', 'noc_loss', 'progress_loss', 'ddc_loss', 'comfort_loss'), parts))
        result.update(pred['_query_diagnostics'])
        return result
