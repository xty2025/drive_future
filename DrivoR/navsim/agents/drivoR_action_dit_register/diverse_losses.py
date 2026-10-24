"""Shared pairing helpers: keep the human and diverse8 trajectory winners."""

import torch


def future_supervision_scope(options):
    scope = str(options.get('supervision_scope', 'human_and_diverse'))
    if scope not in ('human_and_diverse', 'human_only'):
        raise ValueError('future_register.supervision_scope must be human_and_diverse or human_only')
    return scope


@torch.no_grad()
def physical_gt_matches(proposals, targets, valid):
    gt = targets.detach().float().masked_fill(~valid.bool()[..., None, None], 0.)
    cost = (proposals.detach().float()[:, None] - gt[:, :, None]).abs().sum(-1).mean(-1)
    return cost.argmin(-1).masked_fill(~valid.bool(), -1)


def paired_future_targets(teacher, targets, supervision_scope='human_and_diverse'):
    """Encode supervised futures once per GT, never once per proposal."""
    human_valid = targets['human_future_register_valid'].bool()[:, None]
    human_pixels = targets['human_future_register_images'][:, None]
    if supervision_scope == 'human_only':
        return teacher(human_pixels, human_valid), human_valid
    valid = torch.cat((human_valid,
        targets['future_register_valid'].bool() & targets['pseudo_expert_mask'].bool()[..., None]), dim=1)
    pixels = torch.cat((human_pixels, targets['future_register_images']), dim=1)
    return teacher(pixels, valid), valid


def grouped_anchor_indices(diverse_valid, candidates):
    """Split candidates over human + valid D8; every remainder goes to human.

    Slot 0 is human, slots 1..8 keep the original D8 identities. For 9 GTs and
    64 candidates this is 8 human + 7 per D8. Padding GTs receive no candidates.
    This only selects noisy inputs; output matching remains independent.
    """
    assignments = []
    for valid in diverse_valid.bool():
        ids = torch.cat((valid.new_zeros(1, dtype=torch.long), valid.nonzero().flatten() + 1))
        counts = torch.full_like(ids, candidates // len(ids))
        counts[0] += candidates % len(ids)
        assignments.append(torch.repeat_interleave(ids, counts))
    return torch.stack(assignments)
