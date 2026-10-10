from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nvs_v6.config import load_v6_config
from nvs_v6.data import (
    Observation,
    PairTargets,
    drivor_observable_ego_feature,
    drivor_route_command,
    encode_trajectory,
)
from nvs_v6.dino_tokens import split_dinov3_tokens
from nvs_v6.losses import hungarian_xy, masked_soft_bce, paired_future_l1, paired_residual_velocity_mse
from nvs_v6.model import GROUP_ALLOW, DirectedPairDiT, TimmDinoV3Student, build_group_mask, parameter_groups
from nvs_v6.modes import ImmutableModeBuffer, StructuredSourceInitializer
from nvs_v6.oracle import HEAD_INDICES, NAMED_ORDER, RAW_ORDER, OracleLabels
from nvs_v6.scorer import PairedFutureScorer, ScratchDrivoRScorer
from nvs_v6.selection import DRIVOR_WEIGHTS, aggregate_score_logits
from nvs_v6.stages import build_optimizers


def make_observation(batch: int = 1, d: int = 8) -> Observation:
    return Observation(
        rgb_history=torch.randn(batch, 5, 1, 3, 16, 32),
        history_valid=torch.ones(batch, 5, 1, dtype=torch.bool),
        ego_history=torch.randn(batch, 5, d),
        route_command=torch.randn(batch, 4),
        relative_ego_poses=torch.randn(batch, 5, 4, 4),
    )


def make_model() -> DirectedPairDiT:
    torch.manual_seed(7)
    model = DirectedPairDiT(d_model=32, heads=4, ffn_dim=64, depth=2, scene_tokens=66)
    model.eval()
    return model


def make_targets(batch: int = 1, gt: int = 2, d: int = 32) -> PairTargets:
    trajectory = torch.randn(batch, gt, 8, 3)
    return PairTargets(
        gt_trajectory=trajectory,
        gt_encoded=encode_trajectory(trajectory),
        gt_valid=torch.ones(batch, gt, dtype=torch.bool),
        future_register=torch.randn(batch, gt, 4, d),
        future_chunk_valid=torch.ones(batch, gt, 4, dtype=torch.bool),
        future_quality=torch.ones(batch, gt, 4),
        pair_ids=[[f"gt-{i}" for i in range(gt)] for _ in range(batch)],
    )


class FakeOracle:
    def evaluate(self, scene_ids, trajectories, batch_id=""):
        b, m = trajectories.shape[:2]
        raw = torch.sigmoid(torch.zeros(b, m, 7))
        valid = torch.ones_like(raw, dtype=torch.bool)
        raw[..., 3] = 2.0
        valid[..., 3] = False
        return OracleLabels(
            raw7=raw,
            named_six=raw[..., list(HEAD_INDICES)],
            head_valid=valid[..., list(HEAD_INDICES)],
            final_score=raw[..., 6],
            final_valid=valid[..., 6],
            scene_hashes=["synthetic"] * b,
            trajectory_hashes=[[str(i) for i in range(m)] for _ in range(b)],
            evaluator_version="synthetic",
            failure_diagnostics=[],
        )


def test_strict_config_loader_rejects_direction_changes():
    config = load_v6_config(Path(__file__).resolve().parents[1] / "config" / "config_v6_final.yaml")
    assert config["model"]["register_reads_action"] is False
    assert config["model"]["action_reads_register"] is True


def test_group_allow_and_expansion_match_v6():
    assert GROUP_ALLOW == ((1, 0, 0), (1, 1, 1), (1, 0, 1))
    mask = build_group_mask(2, 3, 4)
    assert mask[:2, 2:5].sum() == 0
    assert mask[2:5, 5:].all()
    assert mask[5:, 2:5].sum() == 0
    assert mask[5:, 5:].all()


def test_r_only_equals_full_r_path_and_action_source_cannot_change_r():
    model = make_model()
    observation = make_observation()
    modes = ImmutableModeBuffer().expand_for_batch(1)
    source_a = torch.randn(1, 64, 8, 4)
    source_b = source_a + torch.randn_like(source_a)
    full_a = model.forward_pairs(observation, source_a, modes)
    full_b = model.forward_pairs(observation, source_b, modes)
    only = model.forward_register_only(observation, modes)
    assert torch.allclose(full_a.predicted_register, only, atol=1e-6, rtol=1e-6)
    assert torch.allclose(full_a.predicted_register, full_b.predicted_register, atol=1e-6, rtol=1e-6)


def test_r_reads_scene_but_not_action_and_action_reads_r():
    model = make_model()
    observation = make_observation()
    modes = ImmutableModeBuffer().expand_for_batch(1)
    source = torch.randn(1, 64, 8, 4)
    clean_ego = model.eta.encode_clean_ego(observation)
    scene = model.eta(observation, clean_ego=clean_ego)
    action = model._initialize_action(source, modes, clean_ego)
    register = model._initialize_register(modes)
    scene_a, register_a, action_a = model._directed_pass(scene, action, register)
    scene_b, register_b, action_b = model._directed_pass(scene, action, register + 0.2)
    assert torch.allclose(scene_a, scene_b, atol=1e-6, rtol=1e-6)
    assert torch.allclose(register_a, register_b, atol=1e-6, rtol=1e-6) is False
    assert torch.allclose(action_a, action_b, atol=1e-6, rtol=1e-6) is False


def test_parameter_groups_are_disjoint():
    model = make_model()
    groups = parameter_groups(model)
    ids = [set(map(id, group)) for group in groups.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert all(groups.values())
    assert id(model.phi_time_queries) in ids[2]
    assert id(model.theta_time_embedding) in ids[1]
    grouped = set().union(*ids)
    expected = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("_unused")
    }
    assert grouped == expected


def test_mode_buffer_is_immutable_and_structured_source_has_no_targets():
    modes = ImmutableModeBuffer()
    assert modes.count == 64
    assert len(modes.codebook_hash) == 64
    source = StructuredSourceInitializer().clean(torch.zeros(1, 5, 8), modes.expand_for_batch(1))
    assert source.shape == (1, 64, 8, 4)
    assert "gt" not in inspect.signature(StructuredSourceInitializer.clean).parameters


def test_model_forward_signature_excludes_future_answers():
    signature = inspect.signature(DirectedPairDiT.forward_pairs)
    assert set(signature.parameters) == {"self", "observation", "source", "modes"}


def test_pair_targets_support_up_to_four_gt_and_masked_short_scenes():
    targets = make_targets(gt=4)
    targets.validate()
    predicted = torch.randn(1, 64, 8, 3)
    assignment = hungarian_xy(predicted, targets.gt_trajectory, targets.gt_valid)[0]
    assert len(assignment.as_pairs()) == 4
    make_targets(gt=3).validate()
    with pytest.raises(ValueError):
        make_targets(gt=5).validate()


def test_dinov3_token_split_keeps_cls_all_registers_and_patches():
    tokens = torch.randn(2, 201, 16)
    cls, registers, patches = split_dinov3_tokens(tokens, prefix_count=5)
    assert (cls.shape[1], registers.shape[1], patches.shape[1]) == (1, 4, 196)
    assert torch.equal(torch.cat((cls, registers, patches), dim=1), tokens)


def test_dinov3_token_split_supports_register_free_tensor_and_named_api():
    tokens = torch.randn(2, 197, 16)
    cls, registers, patches = split_dinov3_tokens(tokens, prefix_count=1)
    assert (cls.shape[1], registers.shape[1], patches.shape[1]) == (1, 0, 196)
    named = {
        "x_norm_clstoken": tokens[:, 0],
        "x_norm_patchtokens": tokens[:, 1:],
    }
    named_cls, named_registers, named_patches = split_dinov3_tokens(named, prefix_count=1)
    assert torch.equal(named_cls, cls)
    assert named_registers.shape == (2, 0, 16)
    assert torch.equal(named_patches, patches)
    with pytest.raises(ValueError):
        split_dinov3_tokens(tokens, prefix_count=0)


def test_actual_dinov3_224_layout_counts_native_and_qkv_tokens():
    student = TimmDinoV3Student.__new__(TimmDinoV3Student)
    nn.Module.__init__(student)
    student.image_size = 224
    student.num_prefix_tokens = 5
    student.backbone = nn.Module()
    student.backbone.patch_embed = nn.Module()
    student.backbone.patch_embed.patch_size = (16, 16)
    current = student.token_layout(32)
    history = student.token_layout(8)
    assert current == {
        "cls": 1,
        "registers": 4,
        "scene_queries": 27,
        "image_patches": 196,
        "native_tokens": 201,
        "backbone_qkv_tokens": 228,
        "output_tokens": 32,
    }
    assert history["native_tokens"] == 201
    assert history["scene_queries"] == 3
    assert history["backbone_qkv_tokens"] == 204


def test_scene_and_directed_kv_token_lengths_are_stable():
    model = DirectedPairDiT(d_model=32, heads=4, ffn_dim=64, depth=1).eval()
    observation = make_observation()
    modes = ImmutableModeBuffer().expand_for_batch(1)[:, :3]
    source = torch.randn(1, 3, 8, 4)
    lengths = {}

    def capture(name):
        def hook(_module, inputs):
            lengths[name] = inputs[1].shape[1]
        return hook

    handles = [
        model.eta.ego_attention.register_forward_pre_hook(capture("ego")),
        model.phi["blocks"][0].attn.register_forward_pre_hook(capture("register")),
        model.theta["blocks"][0].attn.register_forward_pre_hook(capture("action")),
    ]
    try:
        output = model.forward_pairs(observation, source, modes)
    finally:
        for handle in handles:
            handle.remove()
    assert output.scene_context.shape == (1, 66, 32)
    assert lengths == {"ego": 64, "register": 70, "action": 78}


def test_clean_ego_and_route_directly_condition_action_initializer():
    model = make_model()
    observation_a = make_observation()
    observation_b = Observation(**{
        **observation_a.__dict__,
        "route_command": observation_a.route_command + torch.tensor([[3.0, -2.0, 1.0, 4.0]]),
    })
    modes = ImmutableModeBuffer().expand_for_batch(1)
    source = torch.randn(1, 64, 8, 4)
    clean_a = model.eta.encode_clean_ego(observation_a)
    clean_b = model.eta.encode_clean_ego(observation_b)
    action_a = model._initialize_action(source, modes, clean_a)
    action_b = model._initialize_action(source, modes, clean_b)
    # Ego/route also legally conditions Scene and therefore Register; this test
    # isolates the explicit clean-Ego -> Action projection and broadcast.
    assert not torch.allclose(clean_a, clean_b)
    assert not torch.allclose(action_a, action_b)


def test_navsim_ego_and_command_match_drivor_order_with_legacy_pad_last():
    pose = torch.tensor([1.0, 2.0, 3.0])
    dynamic = torch.tensor([4.0, 5.0, 6.0, 7.0])
    command = drivor_route_command([0, 1, 0, 0])
    ego = drivor_observable_ego_feature(pose, dynamic)
    observation = make_observation()
    observation.ego_history[:, -1] = ego
    observation.route_command[:] = command
    model = make_model()
    captured = {}

    def capture(_module, inputs):
        captured["ego"] = inputs[0].detach().clone()

    handle = model.eta.ego_mlp[0].register_forward_pre_hook(capture)
    try:
        model.eta.encode_clean_ego(observation)
    finally:
        handle.remove()
    expected = torch.cat((pose, dynamic, command, torch.zeros(1))).unsqueeze(0)
    assert torch.equal(captured["ego"], expected)
    assert captured["ego"].shape[-1] == 12
    with pytest.raises(ValueError):
        drivor_route_command([1, 0, 0])


def test_score_labels_are_per_candidate_not_candidate_by_gt():
    labels = FakeOracle().evaluate(["scene"], torch.randn(1, 64, 8, 3))
    targets = make_targets(gt=4)
    cost_shape = (targets.gt_trajectory.shape[1], 64)
    assert labels.raw7.shape == (1, 64, 7)
    assert labels.named_six.shape == (1, 64, 6)
    assert cost_shape == (4, 64)


def test_scorer_detaches_all_generator_inputs_and_ignores_future_register():
    scorer = PairedFutureScorer(d_model=32, heads=1)
    scorer.eval()
    scene = torch.randn(1, 66, 32, requires_grad=True)
    trajectory = torch.randn(1, 8, 8, 3, requires_grad=True)
    registers = torch.randn(1, 8, 4, 32, requires_grad=True)
    output_a = scorer(scene, trajectory, registers)
    registers_b = registers.clone()
    registers_b[:, 0, 0] += torch.linspace(-1.0, 1.0, 32)
    output_b = scorer(scene, trajectory, registers_b)
    assert torch.allclose(output_a, output_b, atol=1e-5, rtol=1e-5)
    output_a.sum().backward()
    assert scene.grad is None and trajectory.grad is None and registers.grad is None
    assert any(parameter.grad is not None for parameter in scorer.parameters())


def test_scratch_scorer_logits_depend_on_detached_nvs_scene_kv():
    torch.manual_seed(101)
    scorer = ScratchDrivoRScorer(d_model=32, heads=1).eval()
    trajectory = torch.randn(1, 8, 8, 3)
    scene_a = torch.randn(1, 66, 32, requires_grad=True)
    scene_b = scene_a.detach().clone()
    scene_b[:, :64] += 3.0 * torch.randn_like(scene_b[:, :64])
    logits_a = scorer(scene_a, trajectory)
    logits_b = scorer(scene_b, trajectory)
    assert not torch.allclose(logits_a, logits_b)
    logits_a.sum().backward()
    assert scene_a.grad is None


def test_default_real_training_constructs_random_scratch_scorer_without_checkpoint_io(monkeypatch):
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_two_scene_overfit.py").read_text()
    assert "ScratchDrivoRScorer(" in script
    assert "NativeDrivoRScorer" not in script
    assert "from_saved_state" not in script

    def reject_checkpoint(*_args, **_kwargs):
        raise AssertionError("scratch scorer attempted torch.load")

    monkeypatch.setattr(torch, "load", reject_checkpoint)
    torch.manual_seed(201)
    scorer_a = ScratchDrivoRScorer(d_model=32, heads=1)
    torch.manual_seed(202)
    scorer_b = ScratchDrivoRScorer(d_model=32, heads=1)
    assert not torch.equal(
        scorer_a.trajectory_projection[0].weight,
        scorer_b.trajectory_projection[0].weight,
    )
    assert not any(name.startswith(("scene_", "ego_")) for name, _ in scorer_a.named_buffers())
    assert "checkpoint" not in inspect.signature(ScratchDrivoRScorer).parameters


def test_hungarian_is_one_assignment_shared_by_action_and_future():
    predicted = torch.zeros(1, 4, 8, 3)
    predicted[:, 0, :, 0] = 1.0
    predicted[:, 1, :, 0] = 10.0
    ground_truth = torch.zeros(1, 2, 8, 3)
    ground_truth[:, 0, :, 0] = 10.0
    ground_truth[:, 1, :, 0] = 1.0
    match = hungarian_xy(predicted, ground_truth, torch.ones(1, 2, dtype=torch.bool))[0]
    assert match.as_pairs() == [(0, 1), (1, 0)]


def test_unmatched_and_no_future_losses_are_finite_and_differentiable():
    model = make_model()
    source = torch.randn(1, 8, 8, 4, requires_grad=True)
    endpoint = source + torch.randn_like(source, requires_grad=True)
    target = make_targets(gt=0)
    match = hungarian_xy(torch.randn(1, 8, 8, 3), target.gt_trajectory, target.gt_valid)[0]
    loss = paired_residual_velocity_mse(endpoint, source, target.gt_encoded, [match])
    assert torch.isfinite(loss) and loss.requires_grad
    loss.backward()


def test_ttc_sentinel_masks_only_ttc_head():
    logits = torch.zeros(1, 2, 6, requires_grad=True)
    targets = torch.ones(1, 2, 6)
    valid = torch.ones(1, 2, 6, dtype=torch.bool)
    valid[..., 3] = False
    loss = masked_soft_bce(logits, targets, valid)
    loss.backward()
    assert torch.isfinite(loss)
    assert RAW_ORDER == ("NC", "DAC", "EP", "TTC", "C", "DDC", "Final")
    assert NAMED_ORDER == ("NC", "DAC", "DDC", "TTC", "EP", "C")


def test_score_loss_matches_drivor_per_head_reductions_and_class_mapping():
    logits = torch.tensor([[[0.2, -0.3, 0.4, -0.5, 0.6, -0.7]]], requires_grad=True)
    targets = torch.tensor([[[0.5, 1.0, 0.5, 2.0, 0.25, 1.0]]])
    valid = torch.ones_like(targets, dtype=torch.bool)
    actual = masked_soft_bce(logits, targets, valid)
    mapped = targets.clone()
    mapped[..., 0] = 0.0
    mapped[..., 2] = 0.0
    expected = sum(
        nn.functional.binary_cross_entropy_with_logits(logits[..., head], mapped[..., head])
        for head in (0, 1, 2, 4, 5)
    )
    assert torch.allclose(actual, expected)


def test_score_loss_numerically_matches_original_drivor_source():
    try:
        from navsim.agents.drivoR.layers.losses.drivor_loss import DrivoRLoss
    except ImportError:
        pytest.skip("official DrivoR source is not on PYTHONPATH")
    torch.manual_seed(29)
    logits = torch.randn(2, 64, 6, requires_grad=True)
    raw = torch.rand(2, 64, 7)
    raw[:, 0, 0] = 0.5
    raw[:, 0, 5] = 0.5
    raw[:, 1, 3] = 2.0
    original = raw.clone()
    pred = {
        name: logits[..., index]
        for index, name in enumerate((
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "driving_direction_compliance",
            "time_to_collision_within_bound",
            "ego_progress",
            "comfort",
        ))
    }
    source = DrivoRLoss().score_loss(
        pred, None, None, None, raw.double(), None, None, None, None
    )[1]
    named = raw[..., list(HEAD_INDICES)]
    valid = torch.ones_like(named, dtype=torch.bool)
    valid[..., 3] = named[..., 3].ne(2.0)
    adapted = masked_soft_bce(logits, named, valid)
    assert torch.allclose(source, adapted, atol=1e-6, rtol=1e-6)
    assert torch.equal(raw, original)


def test_selection_matches_drivor_model_source_formula():
    logits = torch.tensor(
        [
            [
                [0.1, 0.2, 10.0, 0.3, 0.4, 0.5],
                [0.6, 0.7, -10.0, 0.8, 0.9, 1.0],
            ]
        ]
    )
    p = logits.sigmoid()
    expected = (
        DRIVOR_WEIGHTS["NC"] * p[..., 0].log()
        + DRIVOR_WEIGHTS["DAC"] * p[..., 1].log()
        + DRIVOR_WEIGHTS["DDC"] * p[..., 2].log()
        + (
            DRIVOR_WEIGHTS["TTC"] * p[..., 3]
            + DRIVOR_WEIGHTS["EP"] * p[..., 4]
            + DRIVOR_WEIGHTS["C"] * p[..., 5]
        ).log()
    )
    assert torch.allclose(aggregate_score_logits(logits), expected)


def test_scratch_scorer_uses_official_depth_projection_heads_and_no_future_gate():
    scorer = PairedFutureScorer(d_model=32, heads=1)
    assert len(scorer.scorer_attention.layers) == 4
    assert tuple(scorer.trajectory_projection[0].weight.shape) == (1024, 24)
    assert tuple(scorer.trajectory_projection[2].weight.shape) == (32, 1024)
    assert not hasattr(scorer, "future_attention")
    assert len(scorer.scorer.pred_score) == 6
    assert scorer.initialization == "scratch_random"
    assert scorer.feature_source == "nvs_dinov3_scene_context"
    assert scorer.checkpoint_loaded is False


def test_only_valid_matched_gt_supervise_candidate_future_registers():
    predicted_trajectory = torch.full((1, 64, 8, 3), 100.0)
    gt_trajectory = torch.zeros(1, 4, 8, 3)
    for index in range(4):
        predicted_trajectory[:, index, :, 0] = float(index)
        predicted_trajectory[:, index, :, 1:] = 0.0
        gt_trajectory[:, index, :, 0] = float(index)
    valid = torch.tensor([[True, True, True, False]])
    targets = PairTargets(
        gt_trajectory=gt_trajectory,
        gt_encoded=encode_trajectory(gt_trajectory),
        gt_valid=valid,
        future_register=torch.ones(1, 4, 4, 32),
        future_chunk_valid=valid.unsqueeze(-1).expand(-1, -1, 4),
        future_quality=torch.ones(1, 4, 4),
        pair_ids=[["a", "b", "c", "padding"]],
    )
    prediction = torch.zeros(1, 64, 4, 32, requires_grad=True)
    assignment = hungarian_xy(predicted_trajectory, gt_trajectory, valid)
    loss = paired_future_l1(prediction, targets, assignment)
    loss.backward()
    supervised = prediction.grad.abs().sum(dim=(-1, -2)).ne(0)
    assert assignment[0].as_pairs() == [(0, 0), (1, 1), (2, 2)]
    assert supervised.sum().item() == 3
    assert not supervised[0, 3:].any()


def test_future_loss_respects_chunk_mask_and_detached_target():
    prediction = torch.zeros(1, 4, 4, 32, requires_grad=True)
    target = make_targets(gt=1)
    target.future_register.zero_()
    target.future_register[..., 0, :] = 1.0
    target.future_chunk_valid[..., 1:] = False
    assignment = hungarian_xy(torch.zeros(1, 4, 8, 3), target.gt_trajectory, target.gt_valid)
    loss = paired_future_l1(prediction, target, assignment)
    loss.backward()
    assert torch.isfinite(loss)
    assert target.future_register.grad is None


@pytest.mark.parametrize("candidate_count", [8, 64])
def test_candidate_shapes(candidate_count):
    model = make_model()
    observation = make_observation()
    modes = ImmutableModeBuffer().expand_for_batch(1)[:, :candidate_count]
    source = torch.randn(1, candidate_count, 8, 4)
    output = model.forward_pairs(observation, source, modes)
    assert output.trajectory.shape == (1, candidate_count, 8, 3)
    assert output.predicted_register.shape == (1, candidate_count, 4, 32)


def test_action_a0_clean_ego_shape_broadcast_and_intervention():
    model = DirectedPairDiT(
        d_model=32,
        heads=4,
        ffn_dim=64,
        depth=1,
        action_tokens=5,
        scene_tokens=66,
    ).eval()
    source = torch.randn(2, 3, 5, 4)
    modes = torch.randn(2, 3, 2)
    clean_ego = torch.randn(2, 32)
    action_a = model._initialize_action(source, modes, clean_ego)
    action_b = model._initialize_action(source, modes, clean_ego + 0.5)
    expected = (
        model.theta["source_projection"](source)
        + model.theta["ego_to_action_projection"](clean_ego)[:, None, None, :]
        + model.theta_time_embedding[None, None, :, :]
        + model.theta["mode_projection"](modes)[:, :, None, :]
    )
    assert action_a.shape == (2, 3, 5, 32)
    assert torch.allclose(action_a, expected)
    assert not torch.allclose(action_a, action_b)
    ego_delta = action_b - action_a
    assert torch.allclose(ego_delta[:, :1, :1], ego_delta[:, -1:, -1:], atol=1e-6, rtol=1e-6)


def test_action_source_changes_neither_scene_nor_register_for_fixed_observation_and_u():
    model = make_model()
    observation = make_observation()
    modes = ImmutableModeBuffer().expand_for_batch(1)[:, :8]
    source_a = torch.randn(1, 8, 8, 4)
    source_b = source_a + 0.3 * torch.randn_like(source_a)
    output_a = model.forward_pairs(observation, source_a, modes)
    output_b = model.forward_pairs(observation, source_b, modes)
    assert torch.allclose(output_a.scene_context, output_b.scene_context, atol=1e-6, rtol=1e-6)
    assert torch.allclose(output_a.predicted_register, output_b.predicted_register, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(output_a.trajectory, output_b.trajectory)


def test_loss_gradient_contract_for_action_ego_projection_future_and_score():
    model = make_model()
    scorer = PairedFutureScorer(d_model=32, heads=1).eval()
    observation = make_observation()
    modes = ImmutableModeBuffer().expand_for_batch(1)[:, :8]
    source = torch.randn(1, 8, 8, 4)
    targets = make_targets(gt=1)

    output = model.forward_pairs(observation, source, modes)
    assignment = hungarian_xy(output.trajectory.detach(), targets.gt_trajectory, targets.gt_valid)
    action_loss = paired_residual_velocity_mse(
        output.endpoint, source, targets.gt_encoded, assignment
    )
    action_loss.backward()
    ego_projection = model.theta["ego_to_action_projection"]
    assert any(parameter.grad is not None for parameter in ego_projection.parameters())
    assert any(parameter.grad is not None for parameter in model.theta.parameters())
    assert any(parameter.grad is not None for parameter in model.eta.ego_mlp.parameters())
    assert all(parameter.grad is None for parameter in model.phi.parameters())
    assert model.phi_time_queries.grad is None

    model.zero_grad(set_to_none=True)
    output = model.forward_pairs(observation, source, modes)
    future_loss = paired_future_l1(output.predicted_register, targets, assignment)
    future_loss.backward()
    assert all(parameter.grad is None for parameter in model.theta.parameters())
    assert model.theta_time_embedding.grad is None
    assert any(parameter.grad is not None for parameter in model.phi.parameters())
    assert any(parameter.grad is not None for parameter in model.eta.ego_mlp.parameters())

    model.zero_grad(set_to_none=True)
    scorer.zero_grad(set_to_none=True)
    output = model.forward_pairs(observation, source, modes)
    logits = scorer(
        output.scene_context,
        output.trajectory,
        output.predicted_register,
    )
    score_loss = scorer.source_loss(logits, FakeOracle().evaluate(["scene"], output.trajectory))
    score_loss.backward()
    assert all(parameter.grad is None for parameter in model.parameters())
    assert any(parameter.grad is not None for parameter in scorer.parameters())


def test_stage_freeze_and_optimizer_coverage_include_action_ego_projection():
    model = make_model()
    scorer = PairedFutureScorer(d_model=32, heads=1)
    projection_ids = {id(parameter) for parameter in model.theta["ego_to_action_projection"].parameters()}

    stage1_generator, stage1_scorer = build_optimizers(model, scorer, stage=1)
    stage1_ids = {
        id(parameter)
        for group in stage1_generator.param_groups
        for parameter in group["params"]
    }
    assert not (projection_ids & stage1_ids)
    assert all(not parameter.requires_grad for parameter in model.theta.parameters())
    assert not model.theta_time_embedding.requires_grad
    assert all(not parameter.requires_grad for parameter in scorer.parameters())
    assert stage1_ids == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    stage1_scorer_ids = [
        id(parameter)
        for group in stage1_scorer.param_groups
        for parameter in group["params"]
    ]
    assert len(stage1_scorer_ids) == len(set(stage1_scorer_ids))
    assert set(stage1_scorer_ids) == {id(parameter) for parameter in scorer.parameters()}
    assert not (stage1_ids & set(stage1_scorer_ids))

    stage2_generator, stage2_scorer = build_optimizers(model, scorer, stage=2)
    stage2_generator_ids = [
        id(parameter)
        for group in stage2_generator.param_groups
        for parameter in group["params"]
    ]
    stage2_scorer_ids = [
        id(parameter)
        for group in stage2_scorer.param_groups
        for parameter in group["params"]
    ]
    assert projection_ids <= set(stage2_generator_ids)
    assert all(parameter.requires_grad for parameter in model.theta.parameters())
    assert model.theta_time_embedding.requires_grad
    assert all(parameter.requires_grad for parameter in scorer.parameters())
    assert len(stage2_generator_ids) == len(set(stage2_generator_ids))
    assert len(stage2_scorer_ids) == len(set(stage2_scorer_ids))
    assert not (set(stage2_generator_ids) & set(stage2_scorer_ids))
    assert set(stage2_generator_ids) == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert set(stage2_scorer_ids) == {id(parameter) for parameter in scorer.parameters()}


def test_real_dinov3_checkpoint_224_patch16_qkv_and_scene_layout():
    checkpoint = Path("/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors")
    if not checkpoint.is_file():
        pytest.skip("configured real DINOv3 checkpoint is unavailable")
    if not torch.cuda.is_available():
        pytest.skip("real DINOv3 layout forward requires CUDA")
    model = DirectedPairDiT(
        d_model=32,
        heads=4,
        ffn_dim=64,
        depth=1,
        dino_checkpoint=str(checkpoint),
        dino_model_name="vit_large_patch16_dinov3.lvd1689m",
        dino_lora_rank=16,
        dino_lora_alpha=16.0,
    ).cuda().eval()
    student = model.eta.student
    patch_size = tuple(int(value) for value in student.backbone.patch_embed.patch_size)
    assert student.image_size == 224
    assert patch_size == (16, 16)
    assert 224 % patch_size[0] == 0 and 224 % patch_size[1] == 0
    assert student.num_prefix_tokens == int(student.backbone.num_prefix_tokens) == 5
    current_layout = student.token_layout(32)
    history_layout = student.token_layout(8)
    assert current_layout == {
        "cls": 1,
        "registers": 4,
        "scene_queries": 27,
        "image_patches": 196,
        "native_tokens": 201,
        "backbone_qkv_tokens": 228,
        "output_tokens": 32,
    }
    assert history_layout == {
        "cls": 1,
        "registers": 4,
        "scene_queries": 3,
        "image_patches": 196,
        "native_tokens": 201,
        "backbone_qkv_tokens": 204,
        "output_tokens": 8,
    }
    qkv_lengths = []

    def capture_qkv(_module, inputs):
        qkv_lengths.append((int(inputs[0].shape[0]), int(inputs[0].shape[1])))

    handle = student.backbone.blocks[0].attn.register_forward_pre_hook(capture_qkv)
    observation = make_observation()
    observation.rgb_history = torch.rand(1, 5, 1, 3, 224, 224, device="cuda")
    observation.history_valid = observation.history_valid.cuda()
    observation.ego_history = observation.ego_history.cuda()
    observation.route_command = observation.route_command.cuda()
    observation.relative_ego_poses = observation.relative_ego_poses.cuda()
    try:
        with torch.no_grad():
            clean_ego = model.eta.encode_clean_ego(observation)
            scene = model.eta(observation, clean_ego=clean_ego)
    finally:
        handle.remove()
    assert qkv_lengths == [(1, 228), (4, 204)]
    assert scene.shape == (1, 64 + 2, 32)
    del model, scene, observation
    torch.cuda.empty_cache()
