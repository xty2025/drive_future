"""Register denoising losses, paired with GT identities or trajectory matching."""

import torch


def register_error(predicted, target, valid, timesteps, schedule, loss_type):
    """Inputs [B,N,F,R,D]; return masked mean over valid future times [B,N]."""
    valid = valid.bool()
    target = target.detach().float().masked_fill(~valid[...,None,None],0.)
    difference = predicted.float()-target
    if loss_type == 'velocity':
        sigma = schedule.sigma(timesteps).clamp(schedule.sigma_min,1.)
        difference = difference / sigma[...,None,None,None]
    elif loss_type != 'x_mse':
        raise ValueError(f'Unknown future-register loss: {loss_type}')
    per_time = difference.square().mean((-1,-2))
    per_candidate = per_time.masked_fill(~valid,0.).sum(-1)/valid.sum(-1).clamp_min(1)
    if loss_type == 'velocity':
        per_candidate = per_candidate*schedule.training_weight(timesteps)
    return per_candidate


def matched_register_loss(predicted, target, valid, matched_indices, timesteps, schedule, loss_type):
    """Each GT supervises its trajectory-matched candidate; duplicate winners are allowed."""
    rows = torch.arange(len(predicted),device=predicted.device)[:,None]
    indices = matched_indices.detach().clamp_min(0)
    selected = predicted[rows,indices]
    times = timesteps.gather(1,indices)
    valid = valid.bool() & (matched_indices>=0)[...,None]
    error = register_error(selected,target,valid,times,schedule,loss_type)
    gt_valid = valid.any(-1)
    return (error.masked_fill(~gt_valid,0.).sum(-1)/gt_valid.sum(-1).clamp_min(1)).mean()


def assigned_register_loss(predicted, target, valid, assignment, timesteps, schedule, loss_type):
    """Stage1: every register-noise sample uses its paired clean GT action.

    Average assigned noise samples inside each GT, then GTs inside the scene;
    22/21/21 assignments for K=3 therefore still give equal GT weights.
    """
    rows = torch.arange(len(predicted),device=predicted.device)[:,None]
    paired_target = target[rows,assignment]
    paired_valid = valid[rows,assignment]
    error = register_error(predicted,paired_target,paired_valid,timesteps,schedule,loss_type)
    present = paired_valid.any(-1)
    sums = error.new_zeros(target.shape[:2]).scatter_add(1,assignment,error*present)
    counts = error.new_zeros(target.shape[:2]).scatter_add(1,assignment,present.to(error.dtype))
    per_gt = sums/counts.clamp_min(1)
    return (per_gt.sum(-1)/(counts>0).sum(-1).clamp_min(1)).mean()
