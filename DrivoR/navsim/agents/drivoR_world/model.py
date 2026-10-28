"""World4Scorer bank supervision and an independent matched-proposal future branch.

Reference: World4Scorer (Apache-2.0), shared_future_predictor.py and clover_aux_loss.
The future branch uses eight trajectory tokens with self/cross attention. It
supervises GT-nearest proposals against frozen V-JEPA 2.1 features. Only its
direct current-scene K/V path is detached; selected trajectories backpropagate
normally through the generator, including its upstream visual features.
"""

import torch
from torch import nn
from torch.nn import functional as F

from navsim.agents.drivoR.drivor_model import DrivoRModel
from navsim.agents.drivoR.transformer_decoder import Block


class IndependentFuturePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.tf_d_model
        self.trajectory_encoder = nn.Sequential(nn.Linear(3, config.tf_d_ffn), nn.GELU(),
                                                nn.Linear(config.tf_d_ffn, dim))
        self.time_embedding = nn.Parameter(torch.randn(1, len(config.future.times), dim) * 1e-6)
        self.blocks = nn.ModuleList([
            Block(dim=dim, num_heads=int(config.future.get('num_heads', 8)),
                  mlp_ratio=config.tf_d_ffn / dim, proj_drop=0.1, drop_path=0.2)
            for _ in range(int(config.future.get('num_layers', config.scorer_ref_num)))])
        self.out_norm = nn.LayerNorm(config.tf_d_model)
        self.readout = nn.Linear(dim, int(config.future.embedding_dim))

    def forward(self, trajectories, scene):
        queries = self.trajectory_encoder(trajectories) + self.time_embedding
        for block in self.blocks:
            queries = block(queries, scene)
        return self.readout(self.out_norm(queries))


@torch.no_grad()
def nearest_proposals(proposals, ground_truth):
    """Each GT independently selects its nearest physical L1 proposal; repeats allowed."""
    cost = (proposals.detach().float()[:, None] - ground_truth.detach().float()[:, :, None]).abs().sum(-1).mean(-1)
    return cost.argmin(-1)


def sample_bank(candidates, labels, valid, count=16, safety_fraction=0.5):
    """Reference sampling: unsafe quota, other valid entries, then replacement.

    Torch's per-rank model RNG controls sampling, independently of DataLoader
    worker RNG. Unavailable scenes retain a zero mask and never create labels.
    """
    picked, masks = [], []
    for row in range(len(candidates)):
        pool = torch.where(valid[row])[0]
        if not len(pool):
            picked.append(torch.zeros(count, dtype=torch.long, device=valid.device))
            masks.append(torch.zeros(count, dtype=torch.bool, device=valid.device))
            continue
        unsafe = pool[(labels[row, pool, 0] < 0.5) | (labels[row, pool, 1] < 0.5)]
        n_unsafe = min(int(count * safety_fraction), len(unsafe))
        selected = unsafe[torch.randperm(len(unsafe), device=valid.device)[:n_unsafe]]
        rest = pool[~torch.isin(pool, selected)]
        selected = torch.cat((selected, rest[torch.randperm(len(rest), device=valid.device)[:count - n_unsafe]]))
        if len(selected) < count:
            selected = torch.cat((selected, pool[torch.randint(len(pool), (count - len(selected),), device=valid.device)]))
        picked.append(selected)
        masks.append(torch.ones(count, dtype=torch.bool, device=valid.device))
    indices = torch.stack(picked)
    rows = torch.arange(len(candidates), device=indices.device)[:, None]
    return candidates[rows, indices].detach(), labels[rows, indices].detach(), torch.stack(masks)


def bank_outcome_loss(logits, labels, valid):
    """L_bank = mean(five BCE heads) + 0.1 * progress L1, as in the reference."""
    labels = labels.float()
    pairs = [("no_at_fault_collisions", 0), ("drivable_area_compliance", 1),
             ("time_to_collision_within_bound", 3), ("comfort", 4),
             ("driving_direction_compliance", 5)]
    losses = []
    for name, column in pairs:
        target = labels[..., column]
        mask = valid & torch.isfinite(target) & (target >= 0) & (target <= 1)
        target = torch.where(mask, target, 0.)
        if column in (0, 5):
            target = torch.where(target == 0.5, 0., target)
        per = F.binary_cross_entropy_with_logits(logits[name].float(), target, reduction="none")
        losses.append((per * mask).sum() / mask.sum().clamp_min(1))
    target = labels[..., 2]
    mask = valid & torch.isfinite(target) & (target >= 0) & (target <= 1)
    error = (logits["ego_progress"].float().sigmoid() - torch.where(mask, target, 0.)).abs()
    return torch.stack(losses).mean() + 0.1 * (error * mask).sum() / mask.sum().clamp_min(1)


def future_cosine_loss(predicted, target, valid):
    """Average valid times inside GT, then valid GTs inside scene, then scenes."""
    target = target.detach().float()
    target = target.masked_fill(~valid[..., None], 0.)
    distance = 1.0 - F.cosine_similarity(predicted.float(), target, dim=-1)
    per_gt = (distance * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    gt_valid = valid.any(-1)
    per_scene = (per_gt * gt_valid).sum(-1) / gt_valid.sum(-1).clamp_min(1)
    return per_scene.mean()


class DrivoRWorldModel(DrivoRModel):
    def __init__(self, config):
        super().__init__(config)
        self.future_enabled = bool(config.future.enabled)
        if self.future_enabled:
            self.future_predictor = IndependentFuturePredictor(config)
            self.future_times = len(config.future.times)
            self.future_dim = int(config.future.embedding_dim)

    def _add_training_context(self, output, scene_features, ego_token):
        if self.training:
            output["_scene_context"] = scene_features
            output["_ego_token"] = ego_token

    def bank_loss(self, targets, prediction):
        options = self._config.bank
        scene, ego = prediction["_scene_context"], prediction["_ego_token"]
        candidates, labels, valid = sample_bank(
            targets["bank_candidates"], targets["bank_subscores"], targets["bank_valid"],
            count=int(options.count), safety_fraction=float(options.safety_fraction))
        state = self._score_features(candidates, scene, ego)
        logits = self.scorer(candidates, state)[0]
        return bank_outcome_loss(logits, labels, valid), valid.any(-1).float().mean()

    def realized_future_loss(self, targets, prediction):
        scene = prediction['_scene_context']
        if self._config.future.get('detach_context', True):
            scene = scene.detach()
        gt = targets['future_gt_trajectories']
        valid = targets['future_valid'].bool()
        gt_valid = valid.any(-1)
        indices = nearest_proposals(prediction['proposals'], gt)
        candidates = prediction['proposals']
        rows = torch.arange(len(candidates), device=candidates.device)[:, None]
        selected = candidates[rows, indices]
        prediction['_future_matched_indices'] = indices
        if not gt_valid.any():
            # Keep the auxiliary parameters in the DDP graph for a rank whose
            # batch has no available future labels, without fabricating targets.
            return self.future_predictor(selected[:1, 0], scene[:1]).sum() * 0.
        b, k = gt.shape[:2]
        context = scene[:, None].expand(-1, k, -1, -1)[gt_valid]
        values = self.future_predictor(selected[gt_valid], context)
        predicted = values.new_zeros(b, k, self.future_times, self.future_dim)
        predicted[gt_valid] = values
        return future_cosine_loss(predicted, targets['future_features'], valid)
