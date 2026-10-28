"""Variable-GT targets and independent GT-to-proposal matching contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from navsim.agents.drivoR.multi_gt import (
    MultiTrajectoryTargetBuilder, multi_gt_l1_loss, multi_gt_velocity_loss,
    balanced_gt_inputs, multi_gt_enabled, drive_jepa_multi_gt_l1_loss, mean_valid_gt_winners,
    drive_jepa_multi_gt_velocity_loss,
)
from navsim.agents.drivoR.layers.losses.drivor_loss import DrivoRLoss
from navsim.agents.drivoR_action_dit import model as model_module
from navsim.agents.drivoR_action_dit.flow import ContinuousFlowSchedule, normalize_trajectory
from navsim.agents.drivoR_action_dit.loss import DrivoRActionDiTLoss


def target(gt, valid):
    return {'trajectory_multi_gt': gt, 'trajectory_multi_gt_valid': valid}


def write_gt(root, token, trajectories, mask=None, padded=None, labels=None):
    folder = root / token
    folder.mkdir(parents=True)
    n = len(trajectories)
    np.save(folder / 'gt_trajectories.npy', trajectories)
    np.save(folder / 'gt_valid_mask.npy', np.ones(n, bool) if mask is None else mask)
    metadata = {
        'target_times': (np.arange(1, 9) * 0.5).tolist(),
        'gt_padded': [False] * n if padded is None else padded,
    }
    if labels is not None:
        metadata['gt_labels'] = labels
    (folder / 'trajectory.json').write_text(json.dumps(metadata))


@pytest.mark.parametrize('count', [1, 2, 3, 4])
def test_reader_uses_actual_nonpadded_gt_count_and_keeps_stationary_gt(tmp_path, count):
    gt = np.zeros((4, 8, 3))
    for index in range(count):
        gt[index, :, 0] = index * 2
    gt[count:] = gt[0]
    write_gt(tmp_path, 'scene', gt, padded=[False] * count + [True] * (4 - count))
    builder = MultiTrajectoryTargetBuilder(OmegaConf.create(dict(root=str(tmp_path), max_gt=4)))
    result = builder.load('scene', np.zeros((8, 3)))
    assert result['multi_gt_count'].item() == count
    assert result['trajectory_multi_gt_valid'].tolist() == [True] * count + [False] * (4 - count)
    assert result['trajectory_multi_gt'].shape == (4, 8, 3)
    assert result['trajectory_multi_gt'].dtype == torch.float32
    assert not result['multi_gt_fallback']
    assert builder.get_unique_name() != 'drivor_target'


def test_reader_combines_validity_and_padding_masks(tmp_path):
    gt = np.zeros((4, 8, 3))
    gt[0] = float('nan')
    gt[1, :, 0], gt[2, :, 0], gt[3, :, 0] = 1, 2, 3
    write_gt(tmp_path, 'scene', gt, np.array([False, True, True, True]), [False, False, True, False])
    result = MultiTrajectoryTargetBuilder(OmegaConf.create(dict(root=str(tmp_path)))).load('scene', np.zeros((8, 3)))
    assert result['multi_gt_count'] == 2
    torch.testing.assert_close(result['trajectory_multi_gt'][:2, 0, 0], torch.tensor([1., 3.]))
    assert torch.isfinite(result['trajectory_multi_gt']).all()


def test_reader_missing_policy_is_explicit_and_fallback_is_reported(tmp_path):
    strict = MultiTrajectoryTargetBuilder(OmegaConf.create(dict(root=str(tmp_path), missing_policy='error')))
    with pytest.raises(FileNotFoundError):
        strict.load('missing', np.ones((8, 3)))
    fallback = MultiTrajectoryTargetBuilder(OmegaConf.create(dict(root=str(tmp_path), missing_policy='original')))
    result = fallback.load('missing', np.ones((8, 3)))
    assert result['multi_gt_count'] == 1 and result['multi_gt_fallback']
    torch.testing.assert_close(result['trajectory_multi_gt'][0], torch.ones(8, 3))


def test_real_export_padding_is_not_counted_as_four_gts():
    root = Path(__file__).resolve().parents[2] / 'multi_traj_gt_data/navtrain_multigt_top4_dense_0p1s_f64_8gpu'
    if not root.is_dir():
        pytest.skip('Real multi-GT export unavailable')
    builder = MultiTrajectoryTargetBuilder(OmegaConf.create(dict(root=str(root))))
    for token, count in [('1aa44d46e4ab5bc7', 1), ('654ad3d9062c5e32', 4)]:
        result = builder.load(token, np.zeros((8, 3)))
        assert result['multi_gt_count'] == count
        assert result['trajectory_multi_gt_valid'].sum() == count


def test_weighted_reader_tracks_human_identity_after_filtering_and_uses_separate_cache(tmp_path):
    gt = np.zeros((4, 8, 3))
    write_gt(tmp_path, 'scene', gt, mask=np.array([False, True, True, True]),
             padded=[False, False, False, True],
             labels=['stop', 'human_log', 'progress_070', 'human_log'])
    builder = MultiTrajectoryTargetBuilder(OmegaConf.create(dict(
        root=str(tmp_path), weighting='drive_jepa', missing_policy='original')))
    result = builder.load('scene', np.zeros((8, 3)))
    assert result['multi_gt_count'] == 2
    assert result['trajectory_multi_gt_is_human'].tolist() == [True, False, False, False]
    assert builder.get_unique_name() == 'drivor_multi_gt_human_v1'
    fallback = builder.load('missing', np.zeros((8, 3)))
    assert fallback['trajectory_multi_gt_is_human'].tolist() == [True, False, False, False]
    equal = MultiTrajectoryTargetBuilder(OmegaConf.create(dict(root=str(tmp_path))))
    assert equal.get_unique_name() == 'drivor_multi_gt_v1'
    assert 'trajectory_multi_gt_is_human' not in equal.load('scene', np.zeros((8, 3)))


@pytest.mark.parametrize('other_count', [0, 1, 2, 3])
def test_jepa_weighting_preserves_human_one_and_shares_auxiliary_half(other_count):
    proposals = torch.zeros(1, 5, 8, 3)
    proposals[0, :, :, 0] = torch.tensor([1., 12., 23., 34., 1000.])[:, None]
    proposals.requires_grad_()
    gt = torch.zeros(1, 4, 8, 3)
    valid = torch.zeros(1, 4, dtype=torch.bool)
    human = torch.zeros_like(valid)
    for i in range(other_count):
        gt[0, i, :, 0] = (i + 1) * 10
    # The human can appear last, and the primary target must come from NAVSIM.
    gt[0, other_count, :, 0] = 900
    valid[0, :other_count + 1] = True
    human[0, other_count] = True
    targets = dict(trajectory=torch.zeros(1, 8, 3), **target(gt, valid),
                   trajectory_multi_gt_is_human=human)
    loss, indices, diagnostics = drive_jepa_multi_gt_l1_loss(proposals, targets)
    expected_other = torch.arange(2., other_count + 2).mean() if other_count else torch.tensor(0.)
    torch.testing.assert_close(loss, 1 + .5 * expected_other)
    torch.testing.assert_close(diagnostics['multi_gt_human_loss'], torch.tensor(1.))
    torch.testing.assert_close(diagnostics['multi_gt_other_loss'], expected_other)
    assert diagnostics['multi_gt_other_count'] == other_count
    assert indices[0, 0] == 0 and indices[0, other_count + 1] == -1
    loss.backward()
    totals = proposals.grad[0, :, :, 0].sum(-1)
    torch.testing.assert_close(totals[0], torch.tensor(1.))
    if other_count:
        torch.testing.assert_close(totals[1:other_count + 1], torch.full((other_count,), .5 / other_count))
    assert torch.count_nonzero(totals[other_count + 1:]) == 0


def test_jepa_auxiliary_loss_averages_all_scenes_including_human_only_scenes():
    proposals = torch.zeros(2, 3, 8, 3)
    proposals[0, :, :, 0] = torch.tensor([1., 12., 21.])[:, None]
    proposals[1, :, :, 0] = torch.tensor([2., 18., 70.])[:, None]
    gt = torch.zeros(2, 4, 8, 3)
    gt[0, :, :, 0] = torch.tensor([10., 0., 20., 0.])[:, None]
    valid = torch.tensor([[True, True, True, False], [True, False, False, False]])
    human = torch.tensor([[False, True, False, False], [True, False, False, False]])
    targets = dict(trajectory=torch.zeros(2, 8, 3), **target(gt, valid),
                   trajectory_multi_gt_is_human=human)
    loss, _, diagnostics = drive_jepa_multi_gt_l1_loss(proposals, targets)
    torch.testing.assert_close(loss, torch.tensor(1.875))  # mean(1 + .5*mean(2,1), 2)
    torch.testing.assert_close(diagnostics['multi_gt_other_loss'], torch.tensor(.75))


def test_weighted_human_and_other_gt_can_still_select_the_same_proposal():
    proposals = torch.ones(1, 2, 8, 3)
    proposals[:, 1] = 20
    proposals.requires_grad_()
    gt = torch.zeros(1, 2, 8, 3)
    gt[:, 1] = .1
    targets = dict(trajectory=gt[:, 0], **target(gt, torch.ones(1, 2, dtype=torch.bool)),
                   trajectory_multi_gt_is_human=torch.tensor([[True, False]]))
    loss, indices, _ = drive_jepa_multi_gt_l1_loss(proposals, targets)
    assert indices.tolist() == [[0, -1, 0]]
    loss.backward()
    assert torch.count_nonzero(proposals.grad[:, 1]) == 0


@pytest.mark.parametrize('matching', ['nearest', 'hungarian'])
def test_each_gt_chooses_a_proposal_and_unmatched_outputs_have_zero_regression_gradient(matching):
    values = torch.tensor([0.2, 9.7, 30., -20.])
    proposals = torch.zeros(1, 4, 8, 3)
    proposals[0, :, :, 0] = values[:, None]
    proposals.requires_grad_()
    gt = torch.zeros(1, 4, 8, 3)
    gt[:, 1, :, 0] = 10
    gt[:, 2:] = float('nan')
    valid = torch.tensor([[True, True, False, False]])
    loss, indices = multi_gt_l1_loss(proposals, target(gt, valid), matching=matching)
    torch.testing.assert_close(loss, torch.tensor(0.25), atol=1e-6, rtol=0)
    assert indices.tolist() == [[0, 1, -1, -1]]
    loss.backward()
    assert proposals.grad[:, :2].abs().sum() > 0
    assert torch.count_nonzero(proposals.grad[:, 2:]) == 0


def test_matching_is_independent_not_hungarian_and_ties_select_one():
    proposals = torch.zeros(1, 3, 8, 3)
    proposals[:, :2, :, 0] = 1
    proposals[:, 2, :, 0] = 20
    proposals.requires_grad_()
    gt = torch.zeros(1, 2, 8, 3)
    gt[:, 1, :, 0] = 0.1
    loss, indices = multi_gt_l1_loss(proposals, target(gt, torch.ones(1, 2, dtype=torch.bool)))
    assert indices.tolist() == [[0, 0]]
    loss.backward()
    assert proposals.grad[:, 0].abs().sum() > 0
    assert torch.count_nonzero(proposals.grad[:, 1:]) == 0


def test_hungarian_minimizes_joint_cost_instead_of_greedy_or_shared_winners():
    cost = torch.tensor([[[1., 2., 100.], [1.1, 50., 100.]]], requires_grad=True)
    valid = torch.ones(1, 2, dtype=torch.bool)
    loss, indices = mean_valid_gt_winners(cost, valid, matching='hungarian')
    assert indices.tolist() == [[1, 0]]  # GT0 gives up its own nearest proposal.
    torch.testing.assert_close(loss, torch.tensor(1.55))
    loss.backward()
    expected = torch.tensor([[[0., .5, 0.], [.5, 0., 0.]]])
    torch.testing.assert_close(cost.grad, expected)
    nearest, _ = mean_valid_gt_winners(cost.detach(), valid)
    torch.testing.assert_close(nearest, torch.tensor(1.05))


@pytest.mark.parametrize('count', [1, 2, 3, 4])
def test_hungarian_assigns_exactly_one_distinct_proposal_per_valid_gt(count):
    cost = torch.arange(64, dtype=torch.float32)[None, None].expand(1, 4, -1).clone().requires_grad_()
    valid = torch.arange(4)[None] < count
    loss, indices = mean_valid_gt_winners(cost, valid, matching='hungarian')
    selected = indices[valid]
    assert selected.unique().numel() == count
    assert (indices[~valid] == -1).all()
    torch.testing.assert_close(loss, torch.arange(count, dtype=torch.float32).mean())
    loss.backward()
    assert torch.count_nonzero(cost.grad) == count
    torch.testing.assert_close(cost.grad[cost.grad != 0], torch.full((count,), 1 / count))


def test_hungarian_excludes_padding_and_averages_each_scene_by_its_actual_gt_count():
    cost = torch.full((2, 4, 3), float('nan'))
    cost[0, 1] = torch.tensor([10., 20., 30.])
    cost[1, 0] = torch.tensor([1., 9., 30.])
    cost[1, 3] = torch.tensor([9., 1., 30.])
    cost.requires_grad_()
    valid = torch.tensor([[False, True, False, False], [True, False, False, True]])
    loss, indices = mean_valid_gt_winners(cost, valid, matching='hungarian')
    assert indices.tolist() == [[-1, 0, -1, -1], [0, -1, -1, 1]]
    torch.testing.assert_close(loss, torch.tensor(5.5))
    loss.backward()
    assert torch.isfinite(cost.grad).all()
    assert torch.count_nonzero(cost.grad[~valid]) == 0


def test_hungarian_rejects_too_few_proposals_and_handles_empty_mask():
    with pytest.raises(ValueError, match='at least as many proposals'):
        mean_valid_gt_winners(torch.zeros(1, 3, 2), torch.ones(1, 3, dtype=torch.bool), matching='hungarian')
    cost = torch.randn(1, 4, 64, requires_grad=True)
    loss, indices = mean_valid_gt_winners(cost, torch.zeros(1, 4, dtype=torch.bool), matching='hungarian')
    assert loss.item() == 0 and (indices == -1).all()
    loss.backward()
    assert torch.count_nonzero(cost.grad) == 0


def test_hungarian_config_is_equal_weight_and_uses_existing_equal_target_cache(tmp_path):
    builder = MultiTrajectoryTargetBuilder(OmegaConf.create(dict(
        root=str(tmp_path), matching='hungarian', weighting='equal')))
    assert builder.get_unique_name() == 'drivor_multi_gt_v1'
    with pytest.raises(ValueError, match='weighting=equal'):
        MultiTrajectoryTargetBuilder(OmegaConf.create(dict(
            root=str(tmp_path), matching='hungarian', weighting='drive_jepa')))
    folder = Path(__file__).resolve().parents[1] / 'navsim/planning/script/config/common/agent'
    for name in ('drivoR_multigt_hungarian', 'drivoR_action_dit_multigt_hungarian',
                 'drivoR_action_dit_multiview_multigt_hungarian'):
        options = OmegaConf.load(folder / (name + '.yaml')).config.multi_gt
        assert options.matching == 'hungarian' and options.weighting == 'equal'
        assert 'human_weight' not in options and 'other_weight' not in options


def test_reduction_averages_gts_inside_each_scene():
    proposals = torch.zeros(2, 2, 8, 3)
    proposals[0, :, :, 0] = torch.tensor([10., 30.])[:, None]
    proposals[1, :, :, 0] = torch.tensor([1., 9.])[:, None]
    gt = torch.zeros(2, 2, 8, 3)
    gt[1, 1, :, 0] = 10
    loss, _ = multi_gt_l1_loss(proposals, target(gt, torch.tensor([[True, False], [True, True]])))
    torch.testing.assert_close(loss, torch.tensor(5.5))  # mean(scene0=10, scene1=mean(1,1))


@pytest.mark.parametrize('time', [0.0, 0.001, 50.0, 500.0, 1000.0])
@pytest.mark.parametrize('matching', ['nearest', 'hungarian'])
def test_multi_gt_v_loss_matches_both_x_to_v_conversions(time, matching):
    schedule = ContinuousFlowSchedule(sigma_min=0.1)
    gt = torch.zeros(1, 2, 8, 3)
    gt[:, 1] = 2
    pred = torch.stack((gt[:, 1] + .1, gt[:, 0] + .2, gt[:, 1] + 10), dim=1).requires_grad_()
    times = torch.full((1, 3), time)
    valid = torch.ones(1, 2, dtype=torch.bool)
    loss, indices = multi_gt_velocity_loss(pred, gt, valid, times, schedule, matching=matching)
    assert indices.tolist() == [[1, 0]]
    noisy = torch.randn_like(pred)
    sigma = schedule.sigma(times).clamp_min(.1)[..., None, None]
    v_pred = (noisy - pred) / sigma
    v_gt = (noisy[:, None] - gt[:, :, None]) / sigma[:, None]
    cost = (v_pred[:, None] - v_gt).square().mean((-2, -1))
    expected = (cost.min(-1).values * schedule.training_weight(times)[:, :1]).mean()
    torch.testing.assert_close(loss, expected, atol=2e-4, rtol=2e-5)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert torch.count_nonzero(pred.grad[:, 2]) == 0


def test_balanced_inputs_use_all_valid_gts_without_fixing_loss_matching():
    gt = torch.randn(2, 4, 8, 3)
    valid = torch.tensor([[True, False, True, True], [False, True, False, False]])
    anchors, indices = balanced_gt_inputs(gt, valid, 64, torch.zeros(2, 8, 3))
    assert anchors.shape == (2, 64, 8, 3)
    assert torch.bincount(indices[0], minlength=4).tolist() == [22, 0, 21, 21]
    assert torch.bincount(indices[1], minlength=4).tolist() == [0, 64, 0, 0]


def score_inputs(proposals):
    b, m = proposals.shape[:2]
    heads = ('no_at_fault_collisions','drivable_area_compliance','ego_progress',
             'time_to_collision_within_bound','comfort','driving_direction_compliance')
    return dict(proposals=proposals, proposal_list=[proposals], pdm_score=torch.zeros(b, m),
                pred_logit={name: torch.zeros(b, m) for name in heads}, pred_logit2=None,
                pred_agents_states=None, pred_area_logit=None, agent_states=None, bev_semantic_map=None)


def oracle(targets, proposals, test):
    assert not test
    raw = torch.full((*proposals.shape[:2], 7), .75, device=proposals.device)
    return raw[..., -1], raw[..., -1].amax(1), raw, None, None, None


def test_drivor_legacy_loss_stays_identical_and_score_loss_does_not_change():
    proposals = torch.randn(2, 64, 8, 3, requires_grad=True)
    normal, long = torch.randn(2, 8, 3), torch.randn(2, 8, 3)
    targets = dict(trajectory=normal, trajectory_long=long)
    criterion = DrivoRLoss(inter_weight=0, prev_weight=0)
    legacy = criterion(targets, score_inputs(proposals), OmegaConf.create({}), oracle)
    expected = sum((proposals-g[:,None]).abs().sum(-1).mean(-1).amin(1).mean() for g in (normal, long))
    torch.testing.assert_close(legacy['trajectory_loss'], expected)
    targets.update(target(normal[:,None], torch.ones(2, 1, dtype=torch.bool)))
    changed = criterion(targets, score_inputs(proposals), OmegaConf.create({'multi_gt':{'enabled':True}}), oracle)
    torch.testing.assert_close(changed['trajectory_loss'], (proposals-normal[:,None]).abs().sum(-1).mean(-1).amin(1).mean())
    torch.testing.assert_close(changed['final_score_loss'], legacy['final_score_loss'], rtol=0, atol=0)
    assert not multi_gt_enabled(SimpleNamespace(long_trajectory_additional_poses=2))


def test_weighted_drivor_criterion_changes_only_trajectory_term_and_equal_mode_remains_available():
    proposals = torch.randn(2, 64, 8, 3, requires_grad=True)
    normal = torch.randn(2, 8, 3)
    gt = torch.stack((normal, normal + 2), dim=1)
    targets = dict(trajectory=normal, **target(gt, torch.ones(2, 2, dtype=torch.bool)),
                   trajectory_multi_gt_is_human=torch.tensor([[True, False], [True, False]]))
    criterion = DrivoRLoss(inter_weight=0, prev_weight=0)
    cfg = OmegaConf.create({'multi_gt': {'enabled': True, 'weighting': 'drive_jepa',
                                       'human_weight': 1., 'other_weight': .5}})
    weighted = criterion(targets, score_inputs(proposals), cfg, oracle)
    human_loss = (proposals - normal[:, None]).abs().sum(-1).mean(-1).min(1).values.mean()
    other_loss = (proposals - gt[:, 1, None]).abs().sum(-1).mean(-1).min(1).values.mean()
    torch.testing.assert_close(weighted['trajectory_loss'], human_loss + .5 * other_loss)
    cfg.multi_gt.weighting = 'equal'
    equal = criterion(targets, score_inputs(proposals), cfg, oracle)
    torch.testing.assert_close(equal['trajectory_loss'], (human_loss + other_loss) / 2)
    torch.testing.assert_close(weighted['final_score_loss'], equal['final_score_loss'], rtol=0, atol=0)
    assert 'multi_gt_human_loss' not in equal


class TinyImageEncoder(nn.Module):
    def __init__(self, config, image_size, output_dim):
        super().__init__()
        self.neck = nn.Linear(3, output_dim)
    def forward(self, images):
        return self.neck(images.mean((-2, -1)))


@pytest.mark.parametrize('other_count', [0, 1, 2, 3])
@pytest.mark.parametrize('other_weight', [.5, .25])
def test_weighted_velocity_loss_shares_auxiliary_weight_and_keeps_sigma_floor(other_count, other_weight):
    schedule = ContinuousFlowSchedule(sigma_min=.1)
    pred = torch.zeros(1, 5, 8, 3)
    pred[0, :, :, 0] = torch.tensor([1., 12., 23., 34., 1000.])[:, None]
    pred.requires_grad_()
    gt = torch.zeros(1, 4, 8, 3)
    valid = torch.zeros(1, 4, dtype=torch.bool)
    human = torch.zeros_like(valid)
    for i in range(other_count):
        gt[0, i, :, 0] = (i + 1) * 10
    # Ignore exported human coordinates: primary supervision uses the original GT.
    gt[0, other_count, :, 0] = 900
    valid[0, :other_count + 1] = True
    human[0, other_count] = True
    times = torch.full((1, 5), 50.)  # sigma=.05 must use the .1 training floor.
    loss, indices, diagnostics = drive_jepa_multi_gt_velocity_loss(
        pred, torch.zeros(1, 8, 3), gt, valid, human, times, schedule, other_weight=other_weight)
    scale = schedule.training_weight(times)[0, 0] / (3 * .1 ** 2)
    errors = torch.arange(2., other_count + 2)
    expected_other = errors.square().mean() if other_count else torch.tensor(0.)
    torch.testing.assert_close(loss, (1 + other_weight * expected_other) * scale)
    torch.testing.assert_close(diagnostics['multi_gt_human_loss'], scale)
    torch.testing.assert_close(diagnostics['multi_gt_other_loss'], expected_other * scale)
    assert diagnostics['multi_gt_other_count'] == other_count
    assert indices[0, 0] == 0 and indices[0, other_count + 1] == -1
    loss.backward()
    totals = pred.grad[0, :, :, 0].sum(-1)
    torch.testing.assert_close(totals[0], 2 * scale)
    if other_count:
        torch.testing.assert_close(totals[1:other_count + 1],
                                   2 * errors * scale * other_weight / other_count)
    assert torch.count_nonzero(totals[other_count + 1:]) == 0


def test_weighted_velocity_loss_averages_empty_auxiliary_scenes_as_zero():
    schedule = ContinuousFlowSchedule()
    pred = torch.zeros(2, 3, 8, 3)
    pred[0, :, :, 0] = torch.tensor([1., 12., 21.])[:, None]
    pred[1, :, :, 0] = torch.tensor([2., 18., 70.])[:, None]
    gt = torch.zeros(2, 4, 8, 3)
    gt[0, :, :, 0] = torch.tensor([10., 0., 20., 0.])[:, None]
    valid = torch.tensor([[True, True, True, False], [True, False, False, False]])
    human = torch.tensor([[False, True, False, False], [True, False, False, False]])
    times = torch.full((2, 3), 500.)
    loss, _, diagnostics = drive_jepa_multi_gt_velocity_loss(
        pred, torch.zeros(2, 8, 3), gt, valid, human, times, schedule)
    scale = schedule.training_weight(times)[0, 0] / (3 * .5 ** 2)
    torch.testing.assert_close(diagnostics['multi_gt_human_loss'], 2.5 * scale)
    torch.testing.assert_close(diagnostics['multi_gt_other_loss'], 1.25 * scale)
    torch.testing.assert_close(loss, 3.125 * scale)


def test_weighted_action_dit_reuses_one_forward_and_preserves_original_gt_noising(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / 'navsim/planning/script/config/common/agent'
    cfg = OmegaConf.merge(*[OmegaConf.load(folder / (name + '.yaml')).config for name in (
        'drivoR', 'drivoR_action_dit', 'drivoR_action_dit_multigt', 'drivoR_action_dit_multigt_weighted')])
    assert cfg.multi_gt.matching == 'nearest' and cfg.multi_gt.input_gt_assignment == 'original'
    assert cfg.multi_gt.human_weight == 1 and cfg.multi_gt.other_weight == .5
    cfg.multi_gt.human_weight, cfg.multi_gt.other_weight = 1.25, .2
    cfg.tf_d_model, cfg.tf_d_ffn, cfg.scorer_ref_num = 8, 16, 2
    cfg.action_dit.update(dict(hidden_dim=8, ffn_dim=16, freq_dim=8, num_heads=2,
                               attn_head_dim=4, num_layers=2, pretrained_path=''))
    monkeypatch.setattr(model_module, 'NativePatchImageEncoder', TinyImageEncoder)
    model = model_module.DrivoRActionDiTModel(cfg).train()
    features = {'image': torch.randn(2, 1, 3, 28, 28), 'ego_status': torch.randn(2, 4, 11)}
    gt = torch.randn(2, 4, 8, 3)
    valid = torch.tensor([[True, True, True, True], [True, False, False, False]])
    human = torch.tensor([[False, False, True, False], [True, False, False, False]])
    original = torch.stack((gt[0, 2], gt[1, 0]))
    targets = dict(trajectory=original, **target(gt, valid), trajectory_multi_gt_is_human=human)
    calls, clean_inputs = [], []
    hook = model.action_dit.register_forward_pre_hook(lambda module, args: calls.append(tuple(args[0].shape)))
    add_noise = model.schedule.add_noise
    def capture(clean, noise, times):
        clean_inputs.append(clean.detach().clone())
        return add_noise(clean, noise, times)
    monkeypatch.setattr(model.schedule, 'add_noise', capture)
    monkeypatch.setattr(model, 'sample_normalized', lambda *a, **k: pytest.fail('Weighted training ran Euler'))
    try:
        with torch.autocast('cpu', dtype=torch.bfloat16):
            pred = model(features, targets=targets)
            result = DrivoRActionDiTLoss()(targets, pred, cfg, oracle, pred['_flow_loss'])
        diagnostics = pred['_flow_diagnostics']
        expected = 1.25 * diagnostics['multi_gt_human_loss'] + .2 * diagnostics['multi_gt_other_loss']
        torch.testing.assert_close(pred['_flow_loss'].detach(), expected)
        result['loss'].backward()
    finally:
        hook.remove()
    assert calls == [(2, 64, 8, 3)] and len(clean_inputs) == 1
    torch.testing.assert_close(clean_inputs[0], normalize_trajectory(original)[:, None].expand(-1, 64, -1, -1))
    assert diagnostics['multi_gt_other_count'] == 1.5
    assert pred['proposals'].shape == (2, 64, 8, 3) and not pred['proposals'].requires_grad
    assert torch.isfinite(result['loss']) and model.action_dit.head.weight.grad.abs().sum() > 0
    assert model.scorer.pred_score['comfort'][0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('assignment', ['balanced', 'original', None])
@pytest.mark.parametrize('matching', ['nearest', 'hungarian'])
def test_action_dit_multi_gt_uses_one_shared_64_proposal_set(monkeypatch, assignment, matching):
    folder=Path(__file__).resolve().parents[1]/'navsim/planning/script/config/common/agent'
    cfg=OmegaConf.merge(OmegaConf.load(folder/'drivoR.yaml').config,
                        OmegaConf.load(folder/'drivoR_action_dit.yaml').config,
                        OmegaConf.load(folder/'drivoR_action_dit_multigt.yaml').config)
    cfg.tf_d_model, cfg.tf_d_ffn, cfg.scorer_ref_num = 8, 16, 2
    cfg.multi_gt.matching = matching
    assert cfg.multi_gt.input_gt_assignment == 'original'
    if assignment is None:
        del cfg.multi_gt.input_gt_assignment
    else:
        cfg.multi_gt.input_gt_assignment = assignment
    cfg.action_dit.update(dict(hidden_dim=8, ffn_dim=16, freq_dim=8, num_heads=2,
                               attn_head_dim=4, num_layers=2, pretrained_path=''))
    monkeypatch.setattr(model_module, 'NativePatchImageEncoder', TinyImageEncoder)
    model=model_module.DrivoRActionDiTModel(cfg).train()
    features={'image':torch.randn(2,1,3,28,28),'ego_status':torch.randn(2,4,11)}
    gt=torch.randn(2,4,8,3)
    valid=torch.tensor([[True,True,True,True],[True,False,False,False]])
    # Human GT need not be the first row in the exported multi-GT set.
    original=torch.stack((gt[0,2],gt[1,0]))
    targets=dict(trajectory=original, **target(gt,valid))
    calls=[]; clean_inputs=[]
    handle=model.action_dit.register_forward_pre_hook(lambda module,args: calls.append(tuple(args[0].shape)))
    add_noise=model.schedule.add_noise
    def capture(clean,noise,times):
        clean_inputs.append(clean.detach().clone())
        assert times.shape==(2,64)
        torch.testing.assert_close(times,times[:,:1].expand(-1,64))
        return add_noise(clean,noise,times)
    monkeypatch.setattr(model.schedule,'add_noise',capture)
    monkeypatch.setattr(model,'sample_normalized',lambda *a,**k: pytest.fail('Multi-GT training ran Euler'))
    try:
        with torch.autocast('cpu',dtype=torch.bfloat16):
            pred=model(features,targets=targets)
            result=DrivoRActionDiTLoss()(targets,pred,cfg,oracle,pred['_flow_loss'])
        result['loss'].backward()
    finally:
        handle.remove()
    assert calls==[(2,64,8,3)]
    assert pred['proposals'].shape==(2,64,8,3) and not pred['proposals'].requires_grad
    assert 'flow_multi_gt_loss' in pred['_flow_diagnostics']
    assert 'flow_long_loss' not in pred['_flow_diagnostics']
    if matching == 'hungarian':
        torch.testing.assert_close(pred['_flow_diagnostics']['multi_gt_matched_proposals'],
                                   pred['_flow_diagnostics']['multi_gt_count'])
    assert len(clean_inputs)==1
    normalized=normalize_trajectory(gt)
    for index in range(64):
        expected=normalized[0,index%4] if assignment=='balanced' else normalize_trajectory(original)[0]
        torch.testing.assert_close(clean_inputs[0][0,index],expected)
    assert torch.isfinite(result['loss']) and model.action_dit.head.weight.grad.abs().sum()>0
    assert model.scorer.pred_score['comfort'][0].weight.grad.abs().sum()>0
