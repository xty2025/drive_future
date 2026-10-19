"""CLOVER selection parity, coverage gradients and preservation of DrivoR losses."""

import importlib.util
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from navsim.agents.drivoR.layers.losses.drivor_loss import DrivoRLoss
from navsim.agents.drivoR_pseudo_expert.agent import DrivoRPseudoExpertAgent
from navsim.agents.drivoR_pseudo_expert.loss import DrivoRPseudoExpertLoss, multi_expert_coverage_loss
from navsim.agents.drivoR_pseudo_expert.selection import select_candidates, include_human_expert
from navsim.agents.drivoR_pseudo_expert.targets import PseudoExpertTargetBuilder


@pytest.mark.parametrize("kind", ["diverse", "score_ties", "duplicates", "low_scores", "empty", "spare_slot"])
def test_selection_matches_clover_reference(kind):
    reference = Path(__file__).resolve().parents[2] / "CLOVER/navsim/agents/clover/pseudo_expert_utils.py"
    if not reference.is_file():
        pytest.skip("CLOVER reference checkout unavailable")
    spec = importlib.util.spec_from_file_location("clover_pseudo_expert_reference", reference)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(17)
    trajectories = rng.standard_normal((30, 8, 3)).astype(np.float32)
    scores = rng.uniform(.75, 1., len(trajectories))
    if kind == "score_ties":
        scores[:] = .9
    if kind in ("duplicates", "spare_slot"):
        trajectories[:] = trajectories[0]
    if kind == "low_scores":
        scores[:] = .79
    if kind == "empty":
        trajectories, scores = trajectories[:0], scores[:0]
    human = rng.standard_normal((8, 3)).astype(np.float32) + 20
    selected, mask = select_candidates(trajectories, scores)
    actual = include_human_expert(selected, mask, human)
    expected = module.select_pseudo_experts(
        {"trajectories": trajectories, "scores": [{"pdm_score": value} for value in scores]}, human)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_coverage_means_over_valid_experts_then_scenes_and_allows_shared_winner():
    proposals = torch.zeros(2, 3, 8, 3)
    proposals[:, :, :, 0] = torch.tensor([0., 10., 100.])[None, :, None]
    proposals.requires_grad_()
    gt = torch.zeros(2, 3, 8, 3)
    gt[0, :, :, 0] = torch.tensor([1., 2., 9.])[:, None]
    gt[1, 0, :, 0] = 4
    gt[1, 1:] = float("nan")
    gt.requires_grad_()
    mask = torch.tensor([[True, True, True], [True, False, False]])
    loss = multi_expert_coverage_loss(proposals, gt, mask)
    torch.testing.assert_close(loss, torch.tensor(((1 + 2 + 1) / 3 + 4) / 2))
    loss.backward()
    assert gt.grad is None
    assert not proposals.grad[:, 2].any()
    assert not proposals.grad[1, 1].any()
    # First two experts independently select the same first proposal.
    torch.testing.assert_close(proposals.grad[0, 0, :, 0], torch.full((8,), -2 / (3 * 2 * 8)))


def test_coverage_sums_xyz_in_physical_coordinates_and_keeps_bf16_gradients():
    proposals = torch.zeros(1, 1, 8, 3, dtype=torch.bfloat16, requires_grad=True)
    gt = torch.tensor([1., 2., 3.]).expand(1, 1, 8, 3)
    value = multi_expert_coverage_loss(proposals, gt, torch.ones(1, 1, dtype=torch.bool))
    assert value.dtype == torch.float32 and value.item() == 6.
    value.backward()
    assert torch.isfinite(proposals.grad).all()


def test_coverage_exact_ties_follow_clover_amin_gradient_sharing():
    proposals = torch.zeros(1, 2, 8, 3, requires_grad=True)
    loss = multi_expert_coverage_loss(proposals, torch.ones(1, 1, 8, 3), torch.ones(1, 1, dtype=torch.bool))
    loss.backward()
    torch.testing.assert_close(proposals.grad, torch.full_like(proposals, -1 / 16))


def make_targets_and_prediction():
    torch.manual_seed(2)
    proposals = torch.randn(2, 64, 8, 3, requires_grad=True)
    previous = torch.randn_like(proposals, requires_grad=True)
    names = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
             "time_to_collision_within_bound", "comfort", "driving_direction_compliance")
    prediction = dict(proposals=proposals, proposal_list=[previous, proposals], pdm_score=torch.zeros(2, 64),
                      pred_logit={key: torch.randn(2, 64, requires_grad=True) for key in names},
                      pred_logit2=None, pred_agents_states=None, pred_area_logit=None,
                      agent_states=None, bev_semantic_map=None)
    targets = dict(trajectory=torch.randn(2, 8, 3), trajectory_long=torch.randn(2, 8, 3),
                   pseudo_experts=torch.randn(2, 8, 8, 3), pseudo_expert_mask=torch.ones(2, 8, dtype=torch.bool),
                   pseudo_expert_fallback=torch.tensor([False, True]))
    return targets, prediction


def oracle(targets, proposals, test):
    assert proposals.shape[1] == 64 and not test
    labels = torch.full((*proposals.shape[:2], 7), .75)
    return labels[..., -1], labels[..., -1].amax(1), labels, None, None, None


def test_only_final_coverage_is_added_and_original_gt_long_and_scorer_are_preserved():
    targets, prediction = make_targets_and_prediction()
    cfg = OmegaConf.create({})
    base = DrivoRLoss(inter_weight=0, prev_weight=.3, trajectory_weight=2)
    criterion = DrivoRPseudoExpertLoss(inter_weight=0, prev_weight=.3, trajectory_weight=2)
    calls = []

    def counted_oracle(*args, **kwargs):
        calls.append(1)
        return oracle(*args, **kwargs)

    old = base(targets, prediction, cfg, oracle)
    new = criterion(targets, prediction, cfg, counted_oracle)
    coverage = multi_expert_coverage_loss(prediction["proposals"], targets["pseudo_experts"], targets["pseudo_expert_mask"])
    assert len(calls) == 1
    torch.testing.assert_close(new["loss"], old["loss"] + coverage)
    torch.testing.assert_close(new["trajectory_loss"], old["trajectory_loss"] + .5 * coverage)
    for name in ("final_score_loss", "min_loss", "min_loss0", "da_loss", "progress_loss"):
        torch.testing.assert_close(new[name], old[name], rtol=0, atol=0)
    new["pseudo_expert_loss"].backward()
    assert prediction["proposals"].grad is not None
    assert prediction["proposal_list"][0].grad is None
    assert all(logit.grad is None for logit in prediction["pred_logit"].values())


def test_zero_coverage_weight_is_exact_base_objective():
    targets, prediction = make_targets_and_prediction()
    cfg = OmegaConf.create({})
    a = DrivoRLoss(inter_weight=0)(targets, prediction, cfg, oracle)
    b = DrivoRPseudoExpertLoss(pseudo_expert_weight=0, inter_weight=0)(targets, prediction, cfg, oracle)
    assert a.keys() == b.keys()
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)


def write_pack(root):
    np.save(root / "tokens.npy", np.array([b"one", b"none"], dtype="S64"))
    experts = np.zeros((2, 8, 8, 3), np.float32)
    valid = np.zeros((2, 8), bool)
    valid[0, 0] = True
    np.save(root / "pseudo_experts.npy", experts)
    np.save(root / "pseudo_expert_mask.npy", valid)
    (root / "manifest.json").write_text(json.dumps(dict(top_k=8, score_thr=.8)))


def test_target_reader_adds_human_only_when_needed_and_never_modifies_mmap(tmp_path):
    write_pack(tmp_path)
    reader = PseudoExpertTargetBuilder(OmegaConf.create(dict(path=str(tmp_path), top_k=8, score_thr=.8)))
    for token in ("one", "none", "missing"):
        result = reader.load(token, np.ones((8, 3), np.float32) * 10)
        assert result["pseudo_expert_mask"].sum() == (2 if token == "one" else 1)
        assert result["pseudo_expert_fallback"] == (token != "one")
    same = reader.load("one", np.zeros((8, 3), np.float32))
    assert same["pseudo_expert_mask"].sum() == 1
    assert np.load(tmp_path / "pseudo_expert_mask.npy").sum() == 1


def test_config_mismatch_or_missing_pack_fails_without_silent_loss_disable(tmp_path):
    cfg = OmegaConf.create(dict(path=str(tmp_path), top_k=8, score_thr=.8))
    with pytest.raises(FileNotFoundError):
        PseudoExpertTargetBuilder(cfg)
    write_pack(tmp_path)
    cfg.score_thr = .9
    with pytest.raises(ValueError, match="rebuild"):
        PseudoExpertTargetBuilder(cfg)


def test_evaluation_checkpoint_needs_no_pseudo_expert_data():
    agent = DrivoRPseudoExpertAgent.__new__(DrivoRPseudoExpertAgent)
    torch.nn.Module.__init__(agent)
    agent._checkpoint_path = "/some/inference.ckpt"
    agent._config = OmegaConf.create({})
    assert len(agent.get_target_builders()) == 1
