"""Contract tests for the DrivoR-scene + Wan2.2-ActionDiT v6 generator.

Every assertion here is a contract the training scripts rely on: the directed
``GROUP_ALLOW`` information flow, DrivoR's 11-wide single ego token, metric
3-wide trajectories with no rescaling, the Wan block geometry, and the official
DrivoR scorer/loss/selection formulas.
"""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
# vendor/SimWAM/src is on the path so the flow tests can diff NVS's port against
# the upstream scheduler source rather than against hard-coded numbers.
for path in (REPO_ROOT / "src", REPO_ROOT / "vendor" / "DrivoR", REPO_ROOT / "vendor" / "DrivoR" / "nuplan-devkit",
             REPO_ROOT / "vendor" / "SimWAM" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from nvs_v6_wan_e2e.config import load_v6_config
from nvs_v6_wan_e2e.data import (
    Observation,
    PairTargets,
    drivor_observable_ego_feature,
    drivor_route_command,
)
from nvs_v6_wan_e2e.losses import (
    hungarian_xy,
    identity_future_l1,
    masked_soft_bce,
    paired_future_l1,
    paired_residual_velocity_mse,
)
from nvs_v6_wan_e2e.model import (
    EGO_OBSERVABLE_DIM,
    EGO_STATUS_DIM,
    GROUP_ALLOW,
    TRAJECTORY_STATE_DIM,
    DirectedPairDiT,
    FlowOutputs,
    build_group_mask,
    parameter_groups,
)
from nvs_v6_wan_e2e.modes import ImmutableModeBuffer, StructuredSourceInitializer
from nvs_v6_wan_e2e.oracle import HEAD_INDICES, NAMED_ORDER, RAW_ORDER, OracleLabels
from nvs_v6_wan_e2e.selection import DRIVOR_WEIGHTS, aggregate_score_logits
from nvs_v6_wan_e2e.stages import build_optimizers, stage1_step
from nvs_v6_wan_e2e.vision import (
    DRIVOR_IMAGE_HW,
    DRIVOR_LORA_RANK,
    DRIVOR_NUM_SCENE_TOKENS,
    DRIVOR_OFFICIAL_IMAGE_HW,
    DRIVOR_RIGHT_ZERO_PAD_PX,
    LoRAQKVtimm,
    pad_to_patch_multiple,
)
from nvs_v6_wan_e2e.wan_dit import (
    MODE_SCALE,
    RMSNorm,
    WAN_CONDITION_RMS,
    DiTBlock,
    Head,
    NvsActionDiT,
    modulate,
    rms_unit,
    sinusoidal_embedding_1d,
)
from nvs_v6_wan_e2e.flow import (
    INFERENCE_SIGMA,
    ODO_METRIC_JACOBIAN_MAX,
    ODO_NORM_RANGE,
    ActionFlow,
    WanContinuousFlowMatchScheduler,
    denorm_odo,
    norm_odo,
    velocity_to_metric_scale,
)
from nvs_v6_wan_e2e.wan_init import load_action_dit_backbone, select_source_layers

DINOV3_CHECKPOINT = Path("/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors")
WAN_PAYLOAD = REPO_ROOT / "checkpoints" / "ActionDiT_Wan22_256hdim_3outdim.pt"

D_MODEL = 32
NUM_SCENE_TOKENS = 4
FRAMES = 5
SCENE_TOKENS = FRAMES * NUM_SCENE_TOKENS + 1
NAMED_ORDER_TO_HEAD = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "comfort",
)


def _drivor_importable() -> bool:
    try:
        from navsim.agents.drivoR.score_module.scorer import Scorer  # noqa: F401
    except ImportError:
        return False
    return True


requires_drivor = pytest.mark.skipif(not _drivor_importable(), reason="official DrivoR source is not on PYTHONPATH")


def make_observation(batch: int = 1) -> Observation:
    return Observation(
        rgb_history=torch.rand(batch, FRAMES, 1, 3, 32, 48),
        history_valid=torch.ones(batch, FRAMES, 1, dtype=torch.bool),
        ego_history=torch.randn(batch, FRAMES, EGO_STATUS_DIM),
        route_command=torch.randn(batch, 4),
        relative_ego_poses=torch.randn(batch, FRAMES, 4, 4),
    )


def make_model(depth: int = 2, **overrides) -> DirectedPairDiT:
    torch.manual_seed(7)
    kwargs = dict(
        d_model=D_MODEL,
        heads=4,
        ffn_dim=64,
        depth=depth,
        num_scene_tokens=NUM_SCENE_TOKENS,
        frames=FRAMES,
        attn_head_dim=8,
        freq_dim=D_MODEL,
    )
    kwargs.update(overrides)
    return DirectedPairDiT(**kwargs).eval()


def make_targets(batch: int = 1, gt: int = 2, d: int = D_MODEL) -> PairTargets:
    return PairTargets(
        gt_trajectory=torch.randn(batch, gt, 8, TRAJECTORY_STATE_DIM),
        gt_valid=torch.ones(batch, gt, dtype=torch.bool),
        future_register=torch.randn(batch, gt, 4, d),
        future_chunk_valid=torch.ones(batch, gt, 4, dtype=torch.bool),
        future_quality=torch.ones(batch, gt, 4),
        pair_ids=[[{"gt_id": str(index)} for index in range(gt)] for _ in range(batch)],
    )


def _open_trajectory_path(model: DirectedPairDiT) -> DirectedPairDiT:
    """Give ``phi.trajectory_projection`` a non-zero weight so it can be observed.

    The projection is zero-initialised on purpose: the trajectory arrives in metres
    with coordinates near 60, so a default initialisation starts the register about
    five times larger than the compressed targets it has to predict.  Tests that
    assert the register *depends* on the trajectory therefore have to open the path
    first, and ``test_trajectory_projection_starts_at_zero`` pins the closed state.
    """

    projection = model.phi["trajectory_projection"]
    with torch.no_grad():
        projection.weight.copy_(torch.randn(projection.weight.shape, generator=torch.Generator().manual_seed(0)) * 0.02)
    return model


def test_trajectory_projection_starts_at_zero():
    projection = make_model().phi["trajectory_projection"]
    assert not projection.weight.abs().any()
    assert not projection.bias.abs().any()
    assert tuple(projection.weight.shape) == (D_MODEL, 8 * TRAJECTORY_STATE_DIM)


def make_scorer(scene_tokens: int = SCENE_TOKENS):
    from nvs_v6_wan_e2e.scorer import ScratchDrivoRScorer

    torch.manual_seed(3)
    return ScratchDrivoRScorer(d_model=D_MODEL, heads=1, ffn_dim=64, scene_tokens=scene_tokens).eval()


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


# --------------------------------------------------------------------------- #
# v6 directed contract
# --------------------------------------------------------------------------- #


def test_strict_config_loader_rejects_direction_changes():
    # The e2e package has its own config: `config_v6_final.yaml` is the shared
    # nvs_v6/nvs_v6_wan spec file and still asserts the 66-token, 12-wide ego,
    # 3-way-detach contract those packages implement.
    config = load_v6_config(REPO_ROOT / "config" / "config_v6_wan_e2e.yaml")
    assert config["model"]["register_reads_action"] is False
    assert config["model"]["action_reads_register"] is True
    # The e2e Stage 1 conditions the Register on the metric trajectory, which the
    # shared v6 spec withdrew.  The directed attention contract is unchanged: the
    # trajectory enters R as an initialisation feature, never as an attention key.
    assert config["model"]["physical_trajectory_to_register_readout"] is True
    assert config["model"]["register_conditioning_path"] == "initialisation_feature_not_attention_key"
    assert config["model"]["register_init_inputs"] == [
        "learned_time_queries",
        "metric_source_trajectory",
        "immutable_mode_code",
    ]
    assert config["model"]["register_stage1_mode_code"] == "omitted_to_prevent_row_index_shortcut"
    assert config["model"]["stage1_pairing"] == "identity_one_register_row_per_gt"
    assert config["model"]["ego_tokens"] == 1
    assert config["model"]["ego_observable_dim"] == 11
    assert config["model"]["trajectory_state_dim"] == TRAJECTORY_STATE_DIM
    assert config["model"]["trajectory_rescaling"] == "none"
    assert config["model"]["action_codec"] == "drivor_metric_xy_heading"
    assert config["vision"]["num_scene_tokens"] == 16
    assert config["vision"]["scene_tokens"] == 81
    assert config["vision"]["drivor_image_hw"] == [672, 1148]
    assert config["vision"]["right_zero_pad_px"] == 4
    assert config["vision"]["image_hw"] == [672, 1152]
    assert config["vision"]["student_resampling"] == "forbidden_zero_pad_only"
    assert config["vision"]["position_encoding"] == "dinov3_rope_analytic_no_interpolation"
    assert config["future_target"]["teacher_image_hw"] == [464, 832]
    assert config["future_target"]["teacher_grid_hw"] == [29, 52]
    assert config["future_target"]["teacher_resize_filter"] == "none_native_resolution_no_interpolation"
    assert config["future_target"]["rope_normalizes_coords_per_axis"] is True
    assert config["future_target"]["teacher_checkpoint"] == config["vision"]["checkpoint"]
    assert config["scorer"]["scene_kv_detached"] is False
    assert config["scorer"]["input_detach"] == ["trajectory", "predicted_register"]


def test_strict_config_rejects_resampling_the_future_frames(tmp_path):
    import yaml

    source = REPO_ROOT / "config" / "config_v6_wan_e2e.yaml"
    config = yaml.safe_load(source.read_text())
    # Upscaling the 464x832 futures to the Student's size would invent ~1.4x more
    # pixels than the generator rendered.  The grids may differ because DINOv3's
    # RoPE normalizes coordinates per axis, so resampling buys nothing.
    config["future_target"]["teacher_image_hw"] = [672, 1152]
    config["future_target"]["teacher_grid_hw"] = [42, 72]
    config["future_target"]["teacher_resize_filter"] = "pil_bicubic_shared_with_history_frames"
    mutated = tmp_path / "config_v6_wan_e2e.yaml"
    mutated.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="native"):
        load_v6_config(mutated)


def test_strict_config_rejects_resampling_instead_of_zero_padding(tmp_path):
    import yaml

    source = REPO_ROOT / "config" / "config_v6_wan_e2e.yaml"
    config = yaml.safe_load(source.read_text())
    # 1148 is DrivoR's official width and is 14*82 for its patch14 DINOv2, but not
    # a multiple of 16.  The remainder must be closed with zero columns, not by
    # stretching the frame, so DrivoR's own pixels stay byte-for-byte.
    config["vision"]["drivor_image_hw"] = [672, 1152]
    config["vision"]["right_zero_pad_px"] = 0
    mutated = tmp_path / "config_v6_wan_e2e.yaml"
    mutated.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="1148|zero columns"):
        load_v6_config(mutated)


def test_group_allow_and_expansion_match_v6():
    assert GROUP_ALLOW == ((1, 0, 0), (1, 1, 1), (1, 0, 1))
    mask = build_group_mask(2, 3, 4)
    assert mask[:2, 2:5].sum() == 0
    assert mask[2:5, 5:].all()
    assert mask[5:, 2:5].sum() == 0
    assert mask[5:, 5:].all()


def test_r_only_equals_full_r_path_and_the_trajectory_conditions_r():
    model = _open_trajectory_path(make_model())
    observation = make_observation()
    modes = ImmutableModeBuffer().expand_for_batch(1)
    source_a = torch.randn(1, 64, 8, TRAJECTORY_STATE_DIM)
    source_b = source_a + torch.randn_like(source_a)
    full_a = model.forward_pairs(observation, source_a, modes)
    full_b = model.forward_pairs(observation, source_b, modes)
    only = model.forward_register_only(observation, source_a, modes)
    # The Action stream still cannot reach R, so for one trajectory the R-only path
    # and the full directed path agree exactly.
    assert torch.allclose(full_a.predicted_register, only, atol=1e-6, rtol=1e-6)
    # The trajectory is R's conditioning input, so a different trajectory must give
    # a different future, while Scene stays a function of the observation alone.
    assert not torch.allclose(full_a.predicted_register, full_b.predicted_register, atol=1e-6, rtol=1e-6)
    assert torch.allclose(full_a.scene_context, full_b.scene_context, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(full_a.trajectory, full_b.trajectory)


def test_register_rows_follow_the_trajectory_not_the_row_index():
    """The Stage-1 anti-shortcut guarantee.

    With the mode code omitted, permuting the source rows must permute the register
    rows identically: what a row predicts is decided by the trajectory in it, not by
    where it sits.  Without this, `register[k]` could answer "which GT am I" from the
    row index alone and never read the trajectory, which is how the anchor-only
    Stage 1 ended up at a 0.4% shuffle margin.
    """

    model = _open_trajectory_path(make_model())
    observation = make_observation()
    source = torch.randn(1, 4, 8, TRAJECTORY_STATE_DIM)
    permutation = torch.tensor([2, 0, 3, 1])
    base = model.forward_register_only(observation, source)
    permuted = model.forward_register_only(observation, source[:, permutation])
    assert torch.allclose(base[:, permutation], permuted, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(base, permuted, atol=1e-6, rtol=1e-6)


def test_r_reads_scene_but_not_action_and_action_reads_r():
    model = make_model()
    observation = make_observation()
    modes = ImmutableModeBuffer().expand_for_batch(1)[:, :4]
    source = torch.randn(1, 4, 8, TRAJECTORY_STATE_DIM)
    clean_ego = model.eta.encode_clean_ego(observation)
    scene = model.eta(observation, clean_ego=clean_ego)
    action = model.theta.encode_action(source, modes, clean_ego)
    condition = model.theta.encode_condition(modes, clean_ego)
    register = model._initialize_register(modes, source)
    scene_a, register_a, action_a = model._directed_pass(scene, action, register, condition)
    scene_b, register_b, action_b = model._directed_pass(scene, action, register + 0.2, condition)
    assert torch.allclose(scene_a, scene_b, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(register_a, register_b, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(action_a, action_b, atol=1e-6, rtol=1e-6)


def test_source_trajectory_changes_neither_scene_nor_the_directed_mask():
    model = make_model()
    observation = make_observation()
    modes = ImmutableModeBuffer().expand_for_batch(1)[:, :8]
    source_a = torch.randn(1, 8, 8, TRAJECTORY_STATE_DIM)
    source_b = source_a + 0.3 * torch.randn_like(source_a)
    output_a = model.forward_pairs(observation, source_a, modes)
    output_b = model.forward_pairs(observation, source_b, modes)
    assert torch.allclose(output_a.scene_context, output_b.scene_context, atol=1e-6, rtol=1e-6)


def test_model_forward_signature_excludes_future_answers():
    assert set(inspect.signature(DirectedPairDiT.forward_pairs).parameters) == {
        "self",
        "observation",
        "source",
        "modes",
    }


def test_forward_flow_pins_sigma_zero_to_the_data_and_sigma_one_to_the_anchor():
    """The two endpoints the whole design rests on, checked through the real model.

    ``sigma = 0`` is Stage 1's noise-free diverse action and must come back as the
    GT exactly (the inverse ``x0 = x_sigma - sigma*v`` annihilates the head's
    velocity there).  ``sigma = 1`` is the deployed single forward and must stay
    near the anchor at init, which is what keeps the oracle-best PDMS bar.
    """
    model = make_model()
    observation = make_observation(batch=2)
    modes = ImmutableModeBuffer().subset(6).expand_for_batch(2, torch.device("cpu"))
    source = StructuredSourceInitializer(jitter_std=0.0).clean(observation.ego_history, modes)
    data = source + torch.randn_like(source) * 3.0

    zero = torch.zeros(2)
    clean = model.forward_flow(observation, source, modes, data, zero)
    assert isinstance(clean, FlowOutputs)
    assert torch.allclose(clean.pairs.endpoint, data, atol=1e-3)
    assert torch.allclose(clean.pairs.trajectory, data, atol=1e-3)
    assert torch.equal(clean.timestep, zero)
    assert torch.allclose(clean.velocity_target, norm_odo(source) - norm_odo(data), atol=1e-5)
    assert torch.allclose(clean.noisy_action, data, atol=1e-3)

    full = torch.full((2,), INFERENCE_SIGMA * model.flow.num_train_timesteps)
    deployed = model.forward_flow(observation, source, modes, data, full)
    assert torch.allclose(deployed.noisy_action, source, atol=1e-3)
    spread = float((source.amax(dim=(0, 1, 2)) - source.amin(dim=(0, 1, 2))).max())
    deviation = float((deployed.pairs.endpoint - source).abs().max())
    assert deviation < 0.05 * spread, f"endpoint moved {deviation:.4f} m of a {spread:.1f} m anchor spread"

    # forward_pairs is the sigma = 1 path with no data argument at all.
    pairs = model.forward_pairs(observation, source, modes)
    assert torch.allclose(pairs.endpoint, deployed.pairs.endpoint, atol=1e-5)

    with pytest.raises(ValueError):
        model.forward_flow(observation, source, modes, data, torch.zeros(3))
    with pytest.raises(ValueError):
        model.forward_flow(observation, source, modes, data[:, :3], zero)
    with pytest.raises(ValueError):
        model.forward_flow(observation, source, modes, data[..., :2], zero)


def test_flow_register_depends_on_sigma_once_the_trajectory_path_is_open():
    model = _open_trajectory_path(make_model())
    observation = make_observation(batch=2)
    modes = ImmutableModeBuffer().subset(6).expand_for_batch(2, torch.device("cpu"))
    source = StructuredSourceInitializer(jitter_std=0.0).clean(observation.ego_history, modes)
    data = source + torch.randn_like(source) * 3.0
    clean = model.forward_flow(observation, source, modes, data, torch.zeros(2))
    deployed = model.forward_flow(
        observation, source, modes, data, torch.full((2,), float(model.flow.num_train_timesteps))
    )
    assert tuple(clean.pairs.predicted_register.shape) == (2, 6, 4, D_MODEL)
    assert not torch.allclose(clean.pairs.predicted_register, deployed.pairs.predicted_register, atol=1e-3)
    rows = clean.pairs.predicted_register.reshape(2, 6, -1)
    assert not torch.allclose(rows[:, 0], rows[:, 1], atol=1e-3)


def test_parameter_groups_are_disjoint_and_complete():
    model = make_model()
    groups = parameter_groups(model)
    ids = {name: set(map(id, group)) for name, group in groups.items()}
    assert set(ids) == {"eta", "theta", "phi"}
    assert not (ids["eta"] & ids["theta"] or ids["eta"] & ids["phi"] or ids["theta"] & ids["phi"])
    assert all(groups.values())
    assert id(model.phi_time_queries) in ids["phi"]
    assert id(model.theta.time_queries) in ids["theta"]
    assert set().union(*ids.values()) == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }


def test_unassigned_parameter_is_rejected():
    model = make_model()
    model.stray = nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError, match="stray"):
        parameter_groups(model)


# --------------------------------------------------------------------------- #
# DrivoR scene perception: ego token, scene token budget, metric trajectories
# --------------------------------------------------------------------------- #


def test_ego_observable_is_eleven_wide_single_token():
    assert (EGO_OBSERVABLE_DIM, EGO_STATUS_DIM, TRAJECTORY_STATE_DIM) == (11, 7, 3)
    model = make_model()
    assert isinstance(model.eta.hist_encoding, nn.Linear)
    assert (model.eta.hist_encoding.in_features, model.eta.hist_encoding.out_features) == (11, D_MODEL)
    assert model.eta.scene_tokens == SCENE_TOKENS


def test_scene_emits_frame_queries_plus_exactly_one_ego_token():
    model = make_model()
    observation = make_observation(batch=2)
    scene = model.eta(observation)
    assert scene.shape == (2, SCENE_TOKENS, D_MODEL)
    clean_ego = model.eta.encode_clean_ego(observation)
    assert clean_ego.shape == (2, D_MODEL)
    assert torch.allclose(scene[:, -1], clean_ego, atol=1e-6, rtol=1e-6)


def test_clean_ego_reads_only_current_frame_like_drivor():
    model = make_model()
    observation = make_observation()
    mutated = Observation(**{**observation.__dict__, "ego_history": observation.ego_history.clone()})
    mutated.ego_history[:, :-1] += 5.0
    assert torch.allclose(
        model.eta.encode_clean_ego(observation), model.eta.encode_clean_ego(mutated), atol=1e-6
    )
    moved = Observation(**{**observation.__dict__, "ego_history": observation.ego_history.clone()})
    moved.ego_history[:, -1] += 5.0
    assert not torch.allclose(
        model.eta.encode_clean_ego(observation), model.eta.encode_clean_ego(moved), atol=1e-6
    )


def test_navsim_ego_and_command_match_drivor_order_without_padding():
    pose = torch.tensor([1.0, 2.0, 3.0])
    dynamic = torch.tensor([4.0, 5.0, 6.0, 7.0])
    command = drivor_route_command([0, 1, 0, 0])
    ego = drivor_observable_ego_feature(pose, dynamic)
    assert ego.shape == (7,)
    assert torch.equal(ego, torch.cat((pose, dynamic)))
    observation = make_observation()
    observation.ego_history[:, -1] = ego
    observation.route_command[:] = command
    model = make_model()
    captured = {}
    handle = model.eta.hist_encoding.register_forward_pre_hook(
        lambda _module, inputs: captured.setdefault("ego", inputs[0].detach().clone())
    )
    try:
        model.eta.encode_clean_ego(observation)
    finally:
        handle.remove()
    assert torch.equal(captured["ego"], torch.cat((pose, dynamic, command)).unsqueeze(0))
    assert captured["ego"].shape[-1] == 11
    with pytest.raises(ValueError):
        drivor_route_command([1, 0, 0])


def test_time_embedding_separates_frames_frame_major():
    model = make_model()
    # Zero the student so the frame embedding is the only signal reaching a token.
    model.eta.student.neck.weight.data.zero_()
    model.eta.student.neck.bias.data.zero_()
    model.eta.time_embedding.data.zero_()
    model.eta.time_embedding.data[:, 0] = torch.arange(FRAMES, dtype=torch.float32)
    scene = model.eta(make_observation())
    visual = scene[:, :-1, 0].reshape(1, FRAMES, NUM_SCENE_TOKENS)
    assert torch.allclose(visual, torch.arange(FRAMES, dtype=torch.float32).view(1, FRAMES, 1).expand_as(visual))


def test_trajectories_are_metric_with_no_rescaling():
    model = make_model()
    observation = make_observation()
    modes = ImmutableModeBuffer().expand_for_batch(1)[:, :3]
    source = torch.randn(1, 3, 8, TRAJECTORY_STATE_DIM)
    output = model.forward_pairs(observation, source, modes)
    assert output.trajectory.shape == (1, 3, 8, TRAJECTORY_STATE_DIM)
    assert torch.allclose(output.trajectory, output.endpoint)
    assert torch.allclose(output.endpoint - source, output.endpoint - source)
    with pytest.raises(ValueError):
        model.forward_pairs(observation, torch.randn(1, 3, 8, 4), modes)
    assert not hasattr(model, "xy_scale")
    assert "decode_trajectory" not in dir(sys.modules["nvs_v6_wan_e2e.data"])


def test_structured_source_is_metric_and_target_free():
    modes = ImmutableModeBuffer().expand_for_batch(1)[:, :5]
    ego_history = torch.zeros(1, FRAMES, EGO_STATUS_DIM)
    ego_history[:, -1, 3] = 10.0
    source = StructuredSourceInitializer().clean(ego_history, modes)
    assert source.shape == (1, 5, 8, TRAJECTORY_STATE_DIM)
    times = torch.arange(1, 9).float() * 0.5
    speed = (ego_history[0, -1, 3] + modes[0, :, 0]).clamp_min(0.0)
    yaw_rate = modes[0, :, 1]
    assert torch.allclose(source[0, :, :, 0], speed[:, None] * times[None, :])
    assert torch.allclose(source[0, :, :, 1], 0.5 * (speed * yaw_rate)[:, None] * times[None, :].square())
    assert torch.allclose(source[0, :, :, 2], yaw_rate[:, None] * times[None, :])
    # Metres, not a normalized codec: a 10 m/s cruise covers tens of metres.
    assert source[..., 0].abs().max() > 5.0
    assert "gt" not in inspect.signature(StructuredSourceInitializer.clean).parameters
    ego_history[:, -1, 3] = -3.0
    assert StructuredSourceInitializer().clean(ego_history, modes)[..., 0].min() >= 0.0


def test_mode_buffer_is_immutable_and_subsets_stay_inside_the_codebook():
    modes = ImmutableModeBuffer()
    assert modes.count == 64
    assert len(modes.codebook_hash) == 64
    subset = modes.subset(8)
    assert subset.count == 8
    assert torch.equal(subset.expand_for_batch(2), modes.expand_for_batch(2)[:, :8])
    with pytest.raises(ValueError):
        modes.subset(65)
    with pytest.raises(ValueError):
        ImmutableModeBuffer(count=8)


def test_pair_targets_support_up_to_four_gt_and_masked_short_scenes():
    make_targets(gt=4).validate()
    make_targets(gt=3).validate()
    with pytest.raises(ValueError):
        make_targets(gt=5).validate()
    predicted = torch.randn(1, 64, 8, TRAJECTORY_STATE_DIM)
    targets = make_targets(gt=4)
    assignment = hungarian_xy(predicted, targets.gt_trajectory, targets.gt_valid)[0]
    assert len(assignment.as_pairs()) == 4


@pytest.mark.parametrize("candidate_count", [8, 64])
def test_candidate_shapes(candidate_count):
    model = make_model()
    observation = make_observation()
    modes = ImmutableModeBuffer().expand_for_batch(1)[:, :candidate_count]
    source = torch.randn(1, candidate_count, 8, TRAJECTORY_STATE_DIM)
    output = model.forward_pairs(observation, source, modes)
    assert output.scene_context.shape == (1, SCENE_TOKENS, D_MODEL)
    assert output.trajectory.shape == (1, candidate_count, 8, TRAJECTORY_STATE_DIM)
    assert output.predicted_register.shape == (1, candidate_count, 4, D_MODEL)


def test_scene_and_directed_kv_token_lengths_are_stable():
    model = make_model(depth=1)
    observation = make_observation()
    modes = ImmutableModeBuffer().expand_for_batch(1)[:, :3]
    source = torch.randn(1, 3, 8, TRAJECTORY_STATE_DIM)
    lengths = {}

    def capture(name, index):
        def hook(_module, inputs):
            lengths[name] = int(inputs[index].shape[1])

        return hook

    handles = [
        model.phi["blocks"][0].attn.register_forward_pre_hook(capture("register_kv", 1)),
        model.theta.blocks[0].self_attn.register_forward_pre_hook(capture("action_self", 0)),
        model.theta.blocks[0].cross_attn.register_forward_pre_hook(capture("action_cross", 1)),
    ]
    try:
        model.forward_pairs(observation, source, modes)
    finally:
        for handle in handles:
            handle.remove()
    # Register K/V is [scene | register]; the Action stream reads
    # [scene | stop-gradient register] per layer, and self-attends its own 8 tokens.
    assert lengths == {
        "register_kv": SCENE_TOKENS + 4,
        "action_self": 8,
        "action_cross": SCENE_TOKENS + 4,
    }


def test_action_context_is_detached_so_action_cannot_shape_the_register():
    model = make_model(depth=1)
    observation = make_observation()
    modes = ImmutableModeBuffer().expand_for_batch(1)[:, :2]
    source = torch.randn(1, 2, 8, TRAJECTORY_STATE_DIM)
    captured = []
    handle = model.theta.text_embedding.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0])
    )
    try:
        outputs = model.forward_pairs(observation, source, modes)
    finally:
        handle.remove()
    # The Action stream reads `[S, sg(R)]`: Scene tokens first, then the Register.
    assert captured[0].shape[-2] == SCENE_TOKENS + model.register_tokens
    phi = [parameter for name, parameter in model.named_parameters() if name.startswith("phi")]
    eta = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith(("eta.", "scene_blocks.")) and parameter.requires_grad
    ]
    gradients = torch.autograd.grad(outputs.trajectory.sum(), phi + eta, allow_unused=True)
    assert all(gradient is None for gradient in gradients[: len(phi)])
    assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients[len(phi) :])


# --------------------------------------------------------------------------- #
# Wan2.2 ActionDiT geometry and initialization
# --------------------------------------------------------------------------- #


def test_wan_blocks_use_wide_qk_rmsnorm_and_six_segment_modulation():
    dit = NvsActionDiT(hidden_dim=D_MODEL, ffn_dim=64, text_dim=D_MODEL, num_heads=4, attn_head_dim=8, freq_dim=D_MODEL, num_layers=2)
    block = dit.blocks[0]
    assert isinstance(block, DiTBlock)
    assert isinstance(dit.head, Head)
    attn_hidden = 4 * 8
    assert block.self_attn.attn_hidden_dim == attn_hidden
    assert isinstance(block.self_attn.norm_q, RMSNorm)
    assert tuple(block.self_attn.norm_q.weight.shape) == (attn_hidden,)
    assert tuple(block.self_attn.q.weight.shape) == (attn_hidden, D_MODEL)
    assert tuple(block.self_attn.o.weight.shape) == (D_MODEL, attn_hidden)
    assert tuple(block.modulation.shape) == (1, 6, D_MODEL)
    assert tuple(dit.head.modulation.shape) == (1, 2, D_MODEL)
    assert tuple(dit.time_projection[1].weight.shape) == (6 * D_MODEL, D_MODEL)
    assert tuple(dit.head.proj.weight.shape) == (TRAJECTORY_STATE_DIM, D_MODEL)
    assert dit.action_dim == TRAJECTORY_STATE_DIM


def test_wan_cross_attention_has_no_rope_but_self_attention_does():
    dit = NvsActionDiT(hidden_dim=D_MODEL, ffn_dim=64, text_dim=D_MODEL, num_heads=4, attn_head_dim=8, freq_dim=D_MODEL, num_layers=1)
    block = dit.blocks[0]
    assert "freqs" in inspect.signature(block.self_attn.forward).parameters
    assert "freqs" not in inspect.signature(block.cross_attn.forward).parameters
    freqs = dit.rope_for(8, torch.device("cpu"))
    assert freqs.shape[0] == 8
    with pytest.raises(ValueError):
        dit.rope_for(dit.freqs.shape[0] + 1, torch.device("cpu"))


def test_wan_modulate_formula_matches_source():
    x = torch.randn(2, 3, 4)
    shift = torch.randn(2, 1, 4)
    scale = torch.randn(2, 1, 4)
    assert torch.allclose(modulate(x, shift, scale), x * (1 + scale) + shift)


def test_wan_condition_and_action_inputs_are_explicit():
    dit = NvsActionDiT(
        hidden_dim=D_MODEL,
        ffn_dim=64,
        text_dim=D_MODEL,
        num_heads=4,
        attn_head_dim=8,
        freq_dim=D_MODEL,
        num_layers=1,
        ego_dim=D_MODEL,
    )
    modes = torch.randn(2, 3, 2)
    clean_ego = torch.randn(2, D_MODEL)
    source = torch.randn(2, 3, 8, TRAJECTORY_STATE_DIM)

    # The anchor code enters at a bounded scale; the ego enters RMS-clamped.  Both
    # are spelled out here so a silent rescale change breaks the contract.
    expected_mode = dit.mode_projection(modes / dit.mode_scale)
    expected_ego = rms_unit(dit.ego_to_action_projection(clean_ego), dit.eps, WAN_CONDITION_RMS)
    assert torch.allclose(dit.mode_scale, torch.tensor(MODE_SCALE))

    condition = dit.encode_condition(modes, clean_ego)
    assert condition.shape == (2, 3, D_MODEL)
    expected_condition = dit.time_embedding(
        rms_unit(expected_mode + expected_ego.unsqueeze(1), dit.eps, WAN_CONDITION_RMS)
    )
    assert torch.allclose(condition, expected_condition, atol=1e-6)

    # With a flow timestep the condition additionally carries Wan's own sinusoid,
    # which is what the transferred time_embedding/time_projection were trained on.
    timestep = torch.tensor([0.0, 731.5])
    timed = dit.encode_condition(modes, clean_ego, timestep)
    sinusoid = sinusoidal_embedding_1d(D_MODEL, timestep.reshape(2, 1).expand(2, 3).reshape(-1))
    expected_timed = dit.time_embedding(
        rms_unit(
            (expected_mode + expected_ego.unsqueeze(1)).reshape(-1, D_MODEL) + sinusoid,
            dit.eps,
            WAN_CONDITION_RMS,
        ).reshape(2, 3, D_MODEL)
    )
    assert torch.allclose(timed, expected_timed, atol=1e-6)
    assert not torch.allclose(timed, condition, atol=1e-3)
    # A per-scene timestep must not be broadcast into a per-candidate one.
    assert torch.allclose(
        dit.encode_condition(modes, clean_ego, timestep.reshape(2, 1).expand(2, 3)), timed, atol=1e-6
    )
    with pytest.raises(ValueError):
        dit.encode_condition(modes, clean_ego, torch.randn(2, 3, 2))
    with pytest.raises(ValueError):
        dit.encode_condition(modes, clean_ego, torch.randn(5))
    with pytest.raises(ValueError):
        dit.encode_condition(torch.randn(2, 3), clean_ego)

    action = dit.encode_action(source, modes, clean_ego)
    assert action.shape == (2, 3, 8, D_MODEL)
    expected_action = (
        dit.action_encoder(norm_odo(source))
        + expected_ego.view(2, 1, 1, D_MODEL)
        + dit.time_queries.view(1, 1, 8, D_MODEL)
        + expected_mode.unsqueeze(2)
    )
    assert torch.allclose(action, expected_action, atol=1e-6)
    # The metric action must be normalized before the encoder: metres and radians
    # have to contribute comparably, and the sum must not be RMS-normalized after.
    assert not torch.allclose(
        dit.action_encoder(source), dit.action_encoder(norm_odo(source)), atol=1e-3
    )
    with pytest.raises(ValueError):
        dit.encode_action(torch.randn(2, 3, 8, 4), modes, clean_ego)
    with pytest.raises(ValueError):
        dit.encode_action(source, modes, torch.randn(2, D_MODEL + 1))


def test_norm_odo_is_simwam_min_max_and_round_trips():
    trajectory = torch.tensor([[-1.57, -19.68, -1.67], [65.17, 22.32, 1.86], [0.0, 0.0, 0.0]])
    normalized = norm_odo(trajectory)
    assert torch.allclose(normalized[0], torch.tensor([-1.0, -1.0, -1.0]), atol=1e-6)
    assert torch.allclose(normalized[1], torch.tensor([1.0, 1.0, 1.0]), atol=1e-6)
    assert torch.allclose(denorm_odo(normalized), trajectory, atol=1e-4)
    random = torch.randn(4, 8, 3) * torch.tensor([30.0, 15.0, 1.0])
    assert torch.allclose(denorm_odo(norm_odo(random)), random, atol=1e-4)
    # The v_loss lives in normalized units; `recover` converts back with this constant.
    scale = velocity_to_metric_scale(torch.ones(3))
    assert torch.allclose(scale, torch.tensor([value / 2.0 for value in ODO_NORM_RANGE]))
    assert ODO_METRIC_JACOBIAN_MAX == max(ODO_NORM_RANGE) / 2.0


def test_wan_scheduler_matches_simwam_source():
    from simwam.models.wan22.schedulers.scheduler_continuous import (
        WanContinuousFlowMatchScheduler as SimwamScheduler,
    )

    scheduler = WanContinuousFlowMatchScheduler()
    reference = SimwamScheduler()
    assert (scheduler.num_train_timesteps, scheduler.shift) == (
        reference.num_train_timesteps,
        reference.shift,
    )
    torch.manual_seed(0)
    t = scheduler.sample_training_t(64, torch.device("cpu"), torch.float32)
    assert t.shape == (64,) and bool((t >= 0).all()) and bool((t <= 1000).all())
    data, noise = torch.randn(4, 8, 3), torch.randn(4, 8, 3)
    assert torch.allclose(scheduler.add_noise(data, noise, t[:4]), reference.add_noise(data, noise, t[:4]))
    assert torch.allclose(scheduler.training_weight(t), reference.training_weight(t))
    assert torch.allclose(scheduler.training_target(data, noise, t[:4]), noise - data)
    # The bell weight is exactly zero at both endpoints and peaks mid-schedule.
    endpoints = scheduler.training_weight(torch.tensor([0.0, 1000.0]))
    assert torch.allclose(endpoints, torch.zeros(2), atol=1e-6)
    assert float(scheduler.training_weight(torch.tensor([500.0]))) > float(endpoints.max()) + 1.0
    steps, deltas = scheduler.build_inference_schedule(10, torch.device("cpu"), torch.float32)
    reference_steps, reference_deltas = reference.build_inference_schedule(10, torch.device("cpu"), torch.float32)
    assert torch.allclose(steps, reference_steps) and torch.allclose(deltas, reference_deltas)
    assert torch.allclose(
        scheduler.step(torch.ones(2, 3), deltas[0], torch.zeros(2, 3)),
        reference.step(torch.ones(2, 3), reference_deltas[0], torch.zeros(2, 3)),
    )
    with pytest.raises(ValueError):
        WanContinuousFlowMatchScheduler(shift=0.0)
    with pytest.raises(ValueError):
        scheduler.sample_training_t(0, torch.device("cpu"), torch.float32)


def test_action_flow_uses_the_anchor_as_the_sigma_one_prior():
    """NVS substitutes the 64-anchor grid for Gaussian epsilon.

    ``sigma = 1`` is the clean anchor (the deployed single forward) and
    ``sigma = 0`` is the clean matched GT (Stage 1's noise-free input), so the
    acceptance bar -- every candidate starts at a high-PDMS anchor -- holds at
    step 0 and the same algebra inverts to the GT at ``sigma = 0``.
    """
    flow = ActionFlow()
    anchor = torch.randn(3, 8, 8, 3) * torch.tensor([20.0, 10.0, 0.5])
    data = anchor + torch.randn(3, 8, 8, 3) * 2.0
    zero = torch.zeros(3)
    full = torch.full((3,), float(flow.num_train_timesteps))

    x_zero, v_zero = flow.noisy_action(anchor, data, zero)
    assert torch.allclose(x_zero, data, atol=1e-3)
    x_one, v_one = flow.noisy_action(anchor, data, full)
    assert torch.allclose(x_one, anchor, atol=1e-3)
    assert torch.allclose(v_zero, v_one, atol=1e-6)
    assert torch.allclose(v_one, norm_odo(anchor) - norm_odo(data), atol=1e-5)

    # gt_hat = x_sigma - sigma * v is an exact inverse at every sigma, so one
    # forward yields a trajectory with no rollout (action_sampling_loops: 0).
    for t in (0.0, 137.0, 500.0, 999.0, 1000.0):
        timestep = torch.full((3,), t)
        x_sigma, v_target = flow.noisy_action(anchor, data, timestep)
        recovered = flow.recover(x_sigma, v_target, timestep)
        assert torch.allclose(recovered, data, atol=1e-3), f"sigma={t / 1000}"
    assert INFERENCE_SIGMA == 1.0
    # A zero velocity must leave the anchor untouched at the deployed sigma.
    assert torch.allclose(flow.recover(anchor, torch.zeros_like(anchor), full), anchor, atol=1e-6)
    # The bell weight makes Stage 1's (sigma = 0) v_loss an automatic no-op.
    assert torch.allclose(flow.weight(zero), torch.zeros(3), atol=1e-6)
    assert float(flow.weight(torch.tensor([500.0]))) > 1.0
    # A per-scene timestep broadcasts over the candidates without a reshape.
    per_scene = flow.recover(*flow.noisy_action(anchor, data, torch.tensor([500.0]))[:2], torch.tensor([500.0]))
    assert torch.allclose(per_scene, data, atol=1e-3)
    with pytest.raises(ValueError):
        flow.noisy_action(anchor, data[:, :4], zero)


def test_wan_backbone_skip_prefixes_keep_nvs_only_projections_random():
    assert NvsActionDiT.BACKBONE_SKIP_PREFIXES == (
        "action_encoder.",
        "head.",
        "mode_projection.",
        "ego_to_action_projection.",
        "time_queries",
    )
    keys = {"blocks.0.self_attn.q.weight", "action_encoder.weight", "head.proj.weight", "time_queries"}
    assert NvsActionDiT.backbone_key_set(keys) == {"blocks.0.self_attn.q.weight"}


def test_select_source_layers_is_uniform_and_never_expands():
    assert select_source_layers(30, 8) == [0, 3, 7, 11, 15, 18, 22, 26]
    assert select_source_layers(8, 8) == list(range(8))
    with pytest.raises(ValueError):
        select_source_layers(8, 9)


def test_wan_payload_loads_strictly_and_overrides_random_init():
    if not WAN_PAYLOAD.is_file():
        pytest.skip("Action DiT Wan payload has not been generated")
    torch.manual_seed(11)
    loaded_a = NvsActionDiT()
    torch.manual_seed(12)
    loaded_b = NvsActionDiT()
    torch.manual_seed(13)
    untouched = NvsActionDiT()
    assert not torch.equal(loaded_a.blocks[0].self_attn.q.weight, loaded_b.blocks[0].self_attn.q.weight)
    info_a = load_action_dit_backbone(loaded_a, WAN_PAYLOAD)
    info_b = load_action_dit_backbone(loaded_b, WAN_PAYLOAD)
    assert info_a["loaded_keys"] == info_b["loaded_keys"] == 226
    assert info_a["policy"]["layer_map"] == [0, 3, 7, 11, 15, 18, 22, 26]
    assert torch.equal(loaded_a.blocks[0].self_attn.q.weight, loaded_b.blocks[0].self_attn.q.weight)
    assert torch.equal(loaded_a.text_embedding[0].weight, untouched.text_embedding[0].weight) is False
    assert not torch.equal(loaded_a.action_encoder.weight, untouched.action_encoder.weight)
    with pytest.raises(ValueError):
        load_action_dit_backbone(NvsActionDiT(hidden_dim=64, ffn_dim=128, text_dim=64, freq_dim=64), WAN_PAYLOAD)


def test_directed_pair_dit_records_the_wan_backbone_it_loaded():
    if not WAN_PAYLOAD.is_file():
        pytest.skip("Action DiT Wan payload has not been generated")
    model = DirectedPairDiT(
        d_model=256,
        heads=4,
        ffn_dim=1024,
        depth=8,
        num_scene_tokens=NUM_SCENE_TOKENS,
        frames=FRAMES,
        freq_dim=256,
        wan_backbone_path=str(WAN_PAYLOAD),
    )
    assert model.wan_backbone_info["loaded"] is True
    assert model.wan_backbone_info["loaded_keys"] == 226
    assert model.wan_backbone_path == str(WAN_PAYLOAD)


# --------------------------------------------------------------------------- #
# DrivoR scene queries, LoRA and RoPE prefix
# --------------------------------------------------------------------------- #


def test_lora_wraps_only_q_and_v_with_zeroed_b():
    qkv = nn.Linear(16, 48)
    adapter = LoRAQKVtimm(qkv, rank=4)
    assert adapter.linear_a_q.weight.shape == (4, 16)
    assert adapter.linear_b_q.weight.shape == (16, 4)
    assert adapter.linear_b_q.weight.abs().max() == 0.0
    assert adapter.linear_b_v.weight.abs().max() == 0.0
    assert not hasattr(adapter, "linear_a_k")
    x = torch.randn(2, 3, 16)
    assert torch.allclose(adapter(x), qkv(x), atol=1e-6)
    with pytest.raises(ValueError):
        LoRAQKVtimm(nn.Linear(16, 32), rank=4)


def test_lora_residual_reaches_q_and_v_slabs_only():
    qkv = nn.Linear(8, 24)
    adapter = LoRAQKVtimm(qkv, rank=2)
    adapter.linear_b_q.weight.data.normal_()
    adapter.linear_b_v.weight.data.normal_()
    x = torch.randn(1, 2, 8)
    delta = adapter(x) - qkv(x)
    assert delta[..., :8].abs().max() > 0
    assert delta[..., 8:16].abs().max() == 0
    assert delta[..., 16:].abs().max() > 0


def test_drivor_student_constants_match_the_official_config():
    assert DRIVOR_NUM_SCENE_TOKENS == 16
    assert DRIVOR_LORA_RANK == 32
    # DrivoR's official image_size is [1148, 672] (W, H) -- 1148 = 14*82 for its
    # patch14 DINOv2.  Under patch16 that width leaves a 12 px remainder, which is
    # closed with zero columns rather than by resampling, giving the 1152 the
    # backbone is actually built for.
    assert DRIVOR_OFFICIAL_IMAGE_HW == (672, 1148)
    assert DRIVOR_IMAGE_HW == (672, 1152)
    assert all(side % 16 == 0 for side in DRIVOR_IMAGE_HW)
    assert DRIVOR_IMAGE_HW[1] - DRIVOR_OFFICIAL_IMAGE_HW[1] == DRIVOR_RIGHT_ZERO_PAD_PX == 4
    padded = pad_to_patch_multiple(torch.zeros(1, 3, *DRIVOR_OFFICIAL_IMAGE_HW))
    assert tuple(padded.shape) == (1, 3, *DRIVOR_IMAGE_HW)
    assert bool((padded[..., -4:] == 0).all())
    # The native future resolution already tiles exactly, so the Teacher pads nothing.
    assert tuple(pad_to_patch_multiple(torch.zeros(1, 3, 464, 832)).shape) == (1, 3, 464, 832)


def test_real_dinov3_student_prepends_scene_queries_and_widens_the_rope_prefix():
    if not DINOV3_CHECKPOINT.is_file():
        pytest.skip("configured real DINOv3 checkpoint is unavailable")
    from nvs_v6_wan_e2e.vision import DrivoRDinoV3Student

    torch.manual_seed(0)
    student = DrivoRDinoV3Student(checkpoint=str(DINOV3_CHECKPOINT), num_frames=2, lora_rank=4)
    layout = student.token_layout()
    assert layout["patch_grid"] == [42, 72]
    assert layout["image_patches"] == 3024
    assert layout["native_prefix_tokens"] == 5
    assert layout["backbone_qkv_tokens"] == 5 + 16 + 3024
    assert layout["output_tokens"] == 32
    assert layout["position_encoding"] == "dinov3_rope_analytic_no_interpolation"
    # The scene queries are prefix tokens, so every attention must skip them when
    # applying RoPE; the offset is permanent so checkpointed recomputation agrees.
    assert {int(block.attn.num_prefix_tokens) for block in student.backbone.blocks} == {21}
    assert student.rope_prefix_tokens == 21
    assert student.backbone.pos_embed is None
    trainable = [name for name, parameter in student.backbone.named_parameters() if parameter.requires_grad]
    assert trainable and all("linear_a_" in name or "linear_b_" in name for name in trainable)
    with torch.no_grad():
        tokens = student(torch.rand(1, 2, 3, *DRIVOR_OFFICIAL_IMAGE_HW))
    assert tokens.shape == (1, 32, 256)
    assert torch.isfinite(tokens).all()
    with pytest.raises(ValueError):
        student(torch.rand(1, 3, 3, *DRIVOR_OFFICIAL_IMAGE_HW))
    # The Student zero-pads internally.  Feeding it the already-padded width would
    # mean a loader had resampled, so that shape is refused rather than re-padded.
    with pytest.raises(ValueError, match="does not resample"):
        student(torch.rand(1, 2, 3, *DRIVOR_IMAGE_HW))


def test_real_dinov3_grad_checkpointing_matches_the_eager_backward():
    if not DINOV3_CHECKPOINT.is_file():
        pytest.skip("configured real DINOv3 checkpoint is unavailable")
    if not torch.cuda.is_available():
        pytest.skip("checkpointed DINOv3 backward requires CUDA")
    from nvs_v6_wan_e2e.vision import DrivoRDinoV3Student

    results = {}
    for checkpointed in (False, True):
        torch.manual_seed(0)
        student = DrivoRDinoV3Student(
            checkpoint=str(DINOV3_CHECKPOINT), num_frames=2, lora_rank=4, grad_checkpointing=checkpointed
        ).cuda()
        torch.manual_seed(1234)
        images = torch.rand(1, 2, 3, *DRIVOR_OFFICIAL_IMAGE_HW, device="cuda")
        tokens = student(images)
        tokens.square().mean().backward()
        results[checkpointed] = (
            tokens.detach().clone(),
            {name: parameter.grad.detach().clone() for name, parameter in student.named_parameters() if parameter.grad is not None},
        )
        del student
        torch.cuda.empty_cache()
    assert torch.equal(results[False][0], results[True][0])
    eager, checkpointed = results[False][1], results[True][1]
    assert set(eager) == set(checkpointed) and eager
    for name in eager:
        assert torch.allclose(eager[name], checkpointed[name], atol=1e-5, rtol=1e-4), name


# --------------------------------------------------------------------------- #
# Stage freezing and optimizer coverage
# --------------------------------------------------------------------------- #


@requires_drivor
def test_stage_freeze_and_optimizer_coverage():
    model = make_model()
    scorer = make_scorer()
    theta_ids = {id(parameter) for parameter in model.theta.parameters()}

    stage1_generator, stage1_scorer = build_optimizers(model, scorer, stage=1)
    stage1_ids = {id(parameter) for group in stage1_generator.param_groups for parameter in group["params"]}
    assert not (theta_ids & stage1_ids)
    assert all(not parameter.requires_grad for parameter in model.theta.parameters())
    assert all(not parameter.requires_grad for parameter in scorer.parameters())
    assert stage1_ids == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    stage1_scorer_ids = [id(parameter) for group in stage1_scorer.param_groups for parameter in group["params"]]
    assert len(stage1_scorer_ids) == len(set(stage1_scorer_ids))
    assert set(stage1_scorer_ids) == {id(parameter) for parameter in scorer.parameters()}
    assert not (stage1_ids & set(stage1_scorer_ids))

    stage2_generator, stage2_scorer = build_optimizers(model, scorer, stage=2)
    stage2_ids = [id(parameter) for group in stage2_generator.param_groups for parameter in group["params"]]
    assert theta_ids <= set(stage2_ids)
    assert all(parameter.requires_grad for parameter in model.theta.parameters())
    assert all(parameter.requires_grad for parameter in scorer.parameters())
    assert len(stage2_ids) == len(set(stage2_ids))
    assert set(stage2_ids) == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert set(id(parameter) for group in stage2_scorer.param_groups for parameter in group["params"]) == {
        id(parameter) for parameter in scorer.parameters()
    }


@requires_drivor
def test_stage1_never_reaches_the_dinov3_base_or_the_action_dit():
    model = make_model()
    scorer = make_scorer()
    optimizer, _ = build_optimizers(model, scorer, stage=1)
    targets = make_targets(gt=2)
    result = stage1_step(make_observation(), targets, model, optimizer)
    assert torch.isfinite(result.loss_future)
    # Identity pairing supervises every valid GT row, not only Hungarian matches.
    assert result.supervised_rows == int(targets.gt_valid.sum()) == 2
    assert all(parameter.grad is None for parameter in model.theta.parameters())
    # The trajectory projection belongs to phi and is the whole point of Stage 1.
    assert model.phi["trajectory_projection"].weight.grad is not None
    frozen = [
        parameter
        for name, parameter in model.eta.student.named_parameters()
        if name.startswith("patch.")
    ]
    assert frozen and all(parameter.grad is None for parameter in frozen)


def test_dinov3_base_stays_frozen_while_lora_and_neck_train():
    model = make_model()
    model.eta.student = nn.Module()
    model.eta.student.backbone = nn.Module()
    model.eta.student.backbone.base = nn.Parameter(torch.zeros(2))
    model.eta.student.backbone.linear_a_q = nn.Parameter(torch.zeros(2))
    from nvs_v6_wan_e2e.vision import freeze_backbone_keep_lora

    freeze_backbone_keep_lora(model.eta.student.backbone)
    assert model.eta.student.backbone.base.requires_grad is False
    assert model.eta.student.backbone.linear_a_q.requires_grad is True


def test_student_train_mode_keeps_the_frozen_backbone_in_eval():
    model = make_model()
    model.train()
    assert model.eta.student.patch.training is False
    assert model.phi["blocks"][0].training is True


# --------------------------------------------------------------------------- #
# Losses, oracle labels and selection
# --------------------------------------------------------------------------- #


def test_hungarian_is_one_assignment_shared_by_action_and_future():
    predicted = torch.zeros(1, 4, 8, TRAJECTORY_STATE_DIM)
    predicted[:, 0, :, 0] = 1.0
    predicted[:, 1, :, 0] = 10.0
    ground_truth = torch.zeros(1, 2, 8, TRAJECTORY_STATE_DIM)
    ground_truth[:, 0, :, 0] = 10.0
    ground_truth[:, 1, :, 0] = 1.0
    match = hungarian_xy(predicted, ground_truth, torch.ones(1, 2, dtype=torch.bool))[0]
    assert match.as_pairs() == [(0, 1), (1, 0)]


def test_empty_gt_yields_an_empty_but_finite_assignment():
    targets = make_targets(gt=0)
    targets.gt_valid[:] = False
    endpoint = torch.randn(1, 8, 8, TRAJECTORY_STATE_DIM, requires_grad=True)
    source = torch.randn(1, 8, 8, TRAJECTORY_STATE_DIM)
    match = hungarian_xy(torch.randn(1, 8, 8, TRAJECTORY_STATE_DIM), targets.gt_trajectory, targets.gt_valid)[0]
    assert match.as_pairs() == []
    action_loss = paired_residual_velocity_mse(endpoint, source, targets.gt_trajectory, [match])
    future_loss = paired_future_l1(torch.randn(1, 8, 4, D_MODEL, requires_grad=True), targets, [match])
    assert torch.isfinite(action_loss) and torch.isfinite(future_loss)
    (action_loss + future_loss).backward()
    assert endpoint.grad is not None


def test_paired_residual_velocity_loss_is_a_metric_delta_mse():
    source = torch.zeros(1, 1, 8, TRAJECTORY_STATE_DIM)
    endpoint = torch.full((1, 1, 8, TRAJECTORY_STATE_DIM), 2.0)
    ground_truth = torch.full((1, 1, 8, TRAJECTORY_STATE_DIM), 5.0)
    match = hungarian_xy(endpoint, ground_truth, torch.ones(1, 1, dtype=torch.bool))
    loss = paired_residual_velocity_mse(endpoint, source, ground_truth, match)
    assert torch.allclose(loss, torch.tensor(9.0))


def test_only_valid_matched_gt_supervise_candidate_future_registers():
    predicted_trajectory = torch.zeros(1, 64, 8, TRAJECTORY_STATE_DIM)
    gt_trajectory = torch.zeros(1, 4, 8, TRAJECTORY_STATE_DIM)
    for index in range(4):
        predicted_trajectory[:, index, :, 0] = float(index)
        gt_trajectory[:, index, :, 0] = float(index)
    valid = torch.tensor([[True, True, True, False]])
    targets = PairTargets(
        gt_trajectory=gt_trajectory,
        gt_valid=valid,
        future_register=torch.ones(1, 4, 4, D_MODEL),
        future_chunk_valid=valid.unsqueeze(-1).expand(-1, -1, 4),
        future_quality=torch.ones(1, 4, 4),
        pair_ids=[["a", "b", "c", "padding"]],
    )
    prediction = torch.zeros(1, 64, 4, D_MODEL, requires_grad=True)
    assignment = hungarian_xy(predicted_trajectory, gt_trajectory, valid)
    paired_future_l1(prediction, targets, assignment).backward()
    supervised = prediction.grad.abs().sum(dim=(-1, -2)).ne(0)
    assert assignment[0].as_pairs() == [(0, 0), (1, 1), (2, 2)]
    assert supervised.sum().item() == 3
    assert not supervised[0, 3:].any()


def test_identity_future_loss_supervises_every_valid_gt_row():
    """Stage-1 pairing is given by the dataset, so no matching decides it.

    Row ``k`` is conditioned on ``gt_trajectory[k]`` and supervised by
    ``future_register[k]``; padded rows carry no gradient.  Distinct per-row
    targets make a silent row shift detectable, which Hungarian matching would
    have hidden.
    """

    valid = torch.tensor([[True, True, True, False]])
    # Every row's target is non-zero, so a zero prediction receives gradient on all
    # three valid rows rather than silently passing row 0 by matching a zero target.
    future_register = torch.arange(1, 5, dtype=torch.float32).view(1, 4, 1, 1).expand(1, 4, 4, D_MODEL).clone()
    targets = PairTargets(
        gt_trajectory=torch.zeros(1, 4, 8, TRAJECTORY_STATE_DIM),
        gt_valid=valid,
        future_register=future_register,
        future_chunk_valid=valid.unsqueeze(-1).expand(-1, -1, 4).clone(),
        future_quality=torch.ones(1, 4, 4),
        pair_ids=[["a", "b", "c", "padding"]],
    )
    prediction = future_register.clone().requires_grad_(True)
    assert torch.allclose(identity_future_l1(prediction, targets), torch.tensor(0.0))
    # Shifting the predictions by one row must raise the loss: the pairing is positional.
    shifted = torch.roll(future_register, 1, dims=1).requires_grad_(True)
    assert identity_future_l1(shifted, targets) > 0.0
    zeroed = torch.zeros_like(future_register).requires_grad_(True)
    identity_future_l1(zeroed, targets).backward()
    supervised = zeroed.grad.abs().sum(dim=(-1, -2)).ne(0)
    assert supervised[0].tolist() == [True, True, True, False]
    with pytest.raises(ValueError):
        identity_future_l1(torch.zeros(1, 64, 4, D_MODEL), targets)


def _stage1_script():
    """Load ``scripts/run_stage1_multiview.py`` so its reporting maths is testable.

    The verdict that script prints is the only evidence Stage 1 learned a
    trajectory-conditional future, so a sign error or a threshold that can never be
    met would be indistinguishable from a training failure.
    """

    import importlib.util

    path = REPO_ROOT / "scripts" / "run_stage1_multiview.py"
    spec = importlib.util.spec_from_file_location("run_stage1_multiview_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trajectory_spread_is_mean_pairwise_ade_in_metres():
    script = _stage1_script()
    assert script._trajectory_spread(torch.zeros(3, 8, TRAJECTORY_STATE_DIM)) == 0.0
    # Two straight lines held 1 m apart in y at every step have an ADE of exactly 1 m.
    apart = torch.zeros(2, 8, TRAJECTORY_STATE_DIM)
    apart[1, :, 1] = 1.0
    assert script._trajectory_spread(apart) == pytest.approx(1.0)
    assert math.isnan(script._trajectory_spread(apart[:1]))


def test_register_verdict_claims_skill_only_when_the_trajectory_earned_it():
    script = _stage1_script()

    def metrics(paired: float, **overrides) -> dict[str, object]:
        values: dict[str, object] = {
            "mean_paired_future_l1": paired,
            "mean_scene_mean_target_l1": 0.50,
            "mean_constant_target_l1": 0.60,
            "mean_scene_blind_l1": 0.70,
            "mean_shuffled_pair_l1": 0.55,
            "mean_trajectory_blind_l1": 0.52,
            "mean_distance_to_mean_target_l1": 0.30,
            "mean_trajectory_conditioning_fidelity": 0.80,
            "mean_trajectory_response_alignment": 0.70,
        }
        values.update(overrides)
        return values

    learned = script._register_verdict(metrics(0.40), None)
    assert learned["train_beats_trajectory_blind_baseline"] is True
    assert learned["train_learned_trajectory_conditional_future"] is True
    assert learned["train_gain_over_scene_mean_target"] == pytest.approx(0.2)
    assert learned["train_collapse_toward_mean"] == pytest.approx(0.75)

    # Matching the trajectory-blind scene reader exactly is not skill, however well
    # the same number looks against the weaker dataset-mean baseline.
    flat = script._register_verdict(metrics(0.50), None)
    assert flat["train_gain_over_constant_target"] == pytest.approx(1 / 6)
    assert flat["train_beats_trajectory_blind_baseline"] is False
    assert flat["train_learned_trajectory_conditional_future"] is False

    # Shrinking toward the scene average also lowers the paired L1; only the
    # scale-free pair shows that the register moved the wrong way.
    collapsed = script._register_verdict(
        metrics(0.40, mean_trajectory_conditioning_fidelity=0.05, mean_trajectory_response_alignment=-0.1), None
    )
    assert collapsed["train_beats_trajectory_blind_baseline"] is True
    assert collapsed["train_learned_trajectory_conditional_future"] is False

    generalized = script._register_verdict(metrics(0.40), metrics(0.48))
    assert generalized["eval_learned_trajectory_conditional_future"] is True
    assert generalized["generalizes_beyond_fitted_scenes"] is True
    assert "train_gain_over_scene_mean_target" in generalized


def test_future_loss_respects_chunk_mask_and_detached_target():
    prediction = torch.zeros(1, 4, 4, D_MODEL, requires_grad=True)
    targets = make_targets(gt=1)
    targets.future_register.zero_()
    targets.future_register[..., 0, :] = 1.0
    targets.future_chunk_valid[..., 1:] = False
    assignment = hungarian_xy(torch.zeros(1, 4, 8, TRAJECTORY_STATE_DIM), targets.gt_trajectory, targets.gt_valid)
    paired_future_l1(prediction, targets, assignment).backward()
    assert targets.future_register.grad is None
    assert prediction.grad.abs().sum(dim=(0, 1, 3)).tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-6)


def test_ttc_sentinel_masks_only_ttc_head():
    logits = torch.zeros(1, 2, 6, requires_grad=True)
    targets = torch.ones(1, 2, 6)
    valid = torch.ones(1, 2, 6, dtype=torch.bool)
    valid[..., 3] = False
    masked_soft_bce(logits, targets, valid).backward()
    assert torch.isfinite(logits.grad).all()
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
    pred = {name: logits[..., index] for index, name in enumerate(NAMED_ORDER_TO_HEAD)}
    source = DrivoRLoss().score_loss(pred, None, None, None, raw.double(), None, None, None, None)[1]
    named = raw[..., list(HEAD_INDICES)]
    valid = torch.ones_like(named, dtype=torch.bool)
    valid[..., 3] = named[..., 3].ne(2.0)
    assert torch.allclose(source, masked_soft_bce(logits, named, valid), atol=1e-6, rtol=1e-6)
    assert torch.equal(raw, original)


def test_selection_matches_drivor_model_source_formula():
    logits = torch.tensor([[[0.1, 0.2, 10.0, 0.3, 0.4, 0.5], [0.6, 0.7, -10.0, 0.8, 0.9, 1.0]]])
    p = logits.sigmoid()
    expected = (
        DRIVOR_WEIGHTS["NC"] * p[..., 0].log()
        + DRIVOR_WEIGHTS["DAC"] * p[..., 1].log()
        + DRIVOR_WEIGHTS["DDC"] * p[..., 2].log()
        + (DRIVOR_WEIGHTS["TTC"] * p[..., 3] + DRIVOR_WEIGHTS["EP"] * p[..., 4] + DRIVOR_WEIGHTS["C"] * p[..., 5]).log()
    )
    assert torch.allclose(aggregate_score_logits(logits), expected)


def test_score_labels_are_per_candidate_not_candidate_by_gt():
    labels = FakeOracle().evaluate(["scene"], torch.randn(1, 64, 8, TRAJECTORY_STATE_DIM))
    assert labels.raw7.shape == (1, 64, 7)
    assert labels.named_six.shape == (1, 64, 6)
    assert (make_targets(gt=4).gt_trajectory.shape[1], 64) == (4, 64)


# --------------------------------------------------------------------------- #
# Scratch DrivoR scorer on the NVS Scene K/V
# --------------------------------------------------------------------------- #


@requires_drivor
def test_scratch_scorer_is_official_shape_and_scratch_initialized():
    scorer = make_scorer()
    assert len(scorer.scorer_attention.layers) == 4
    assert tuple(scorer.trajectory_projection[0].weight.shape) == (64, 24)
    assert tuple(scorer.trajectory_projection[2].weight.shape) == (D_MODEL, 64)
    assert len(scorer.scorer.pred_score) == 6
    assert not hasattr(scorer, "future_attention")
    assert scorer.initialization == "scratch_random"
    assert scorer.feature_source == "nvs_dinov3_scene_context"
    assert scorer.checkpoint_loaded is False
    assert scorer.native_feature_cache is False
    assert scorer.strict_drivor_features is False
    assert scorer.feature_audit["scene_kv_detached"] is False
    assert scorer.feature_audit["ego_pool"] == "scene_context[:, -1:, :]"
    assert scorer.feature_audit["predicted_register_input"] is False
    assert "checkpoint" not in inspect.signature(type(scorer)).parameters
    torch.manual_seed(201)
    other = type(scorer)(d_model=D_MODEL, heads=1, ffn_dim=64, scene_tokens=SCENE_TOKENS)
    assert not torch.equal(scorer.trajectory_projection[0].weight, other.trajectory_projection[0].weight)


@requires_drivor
def test_scratch_scorer_reads_the_nvs_scene_end_to_end_and_detaches_only_proposals():
    scorer = make_scorer()
    scene = torch.randn(1, SCENE_TOKENS, D_MODEL, requires_grad=True)
    trajectory = torch.randn(1, 8, 8, TRAJECTORY_STATE_DIM, requires_grad=True)
    logits = scorer(scene, trajectory)
    assert logits.shape == (1, 8, 6)
    logits.sum().backward()
    assert scene.grad is not None
    assert trajectory.grad is None
    assert any(parameter.grad is not None for parameter in scorer.parameters())


@requires_drivor
def test_scratch_scorer_logits_depend_on_the_scene_kv():
    scorer = make_scorer()
    trajectory = torch.randn(1, 8, 8, TRAJECTORY_STATE_DIM)
    scene_a = torch.randn(1, SCENE_TOKENS, D_MODEL, requires_grad=True)
    scene_b = scene_a.detach().clone()
    scene_b[:, :-1] += 3.0 * torch.randn_like(scene_b[:, :-1])
    assert not torch.allclose(scorer(scene_a, trajectory), scorer(scene_b, trajectory))


@requires_drivor
def test_scratch_scorer_ego_token_is_the_last_scene_slot():
    scorer = make_scorer()
    trajectory = torch.randn(1, 4, 8, TRAJECTORY_STATE_DIM)
    scene = torch.randn(1, SCENE_TOKENS, D_MODEL)
    moved = scene.clone()
    moved[:, -1] += 10.0
    assert not torch.allclose(scorer(scene, trajectory), scorer(moved, trajectory))
    # The decoder's K/V LayerNorm is shift-invariant, so a constant offset can
    # only reach the logits through the raw ego pool.  Invariance here plus the
    # sensitivity above pins the ego token to the last Scene slot.
    shifted = scene.clone()
    shifted[:, :-1] += 10.0
    assert torch.allclose(scorer(scene, trajectory), scorer(shifted, trajectory))
    perturbed = scene.clone()
    perturbed[:, :-1] += torch.arange(D_MODEL, dtype=torch.float32).view(1, 1, -1)
    assert not torch.allclose(scorer(scene, trajectory), scorer(perturbed, trajectory))


@requires_drivor
def test_scratch_scorer_rejects_the_wrong_scene_length_and_register_input():
    scorer = make_scorer()
    trajectory = torch.randn(1, 4, 8, TRAJECTORY_STATE_DIM)
    with pytest.raises(ValueError):
        scorer(torch.randn(1, SCENE_TOKENS + 1, D_MODEL), trajectory)
    with pytest.raises(ValueError):
        scorer(torch.randn(1, SCENE_TOKENS, D_MODEL), torch.randn(1, 4, 8, 4))
    with pytest.raises(ValueError):
        scorer(torch.randn(2, SCENE_TOKENS, D_MODEL), trajectory)


@requires_drivor
def test_scratch_scorer_selection_and_source_loss_reject_failed_oracles():
    scorer = make_scorer()
    logits = torch.randn(1, 8, 6)
    selected, utility = scorer.select(logits)
    assert selected.shape == (1,) and utility.shape == (1, 8)
    assert 0 <= int(selected[0]) < 8
    assert torch.equal(selected, utility.argmax(dim=1))
    # drivoR.yaml weights DDC at 0.0, so it cannot move the utility or the pick.
    ddc_blind = logits.clone()
    ddc_blind[..., 2] += 7.0
    _, blinded_utility = scorer.select(ddc_blind)
    assert torch.allclose(utility, blinded_utility)
    assert int(scorer.select(ddc_blind)[0][0]) == int(selected[0])
    labels = FakeOracle().evaluate(["scene"], torch.randn(1, 8, 8, TRAJECTORY_STATE_DIM))
    assert torch.isfinite(scorer.source_loss(logits, labels))
    broken = OracleLabels(
        raw7=labels.raw7,
        named_six=labels.named_six,
        head_valid=labels.head_valid,
        final_score=labels.final_score,
        final_valid=torch.zeros_like(labels.final_valid),
        scene_hashes=labels.scene_hashes,
        trajectory_hashes=labels.trajectory_hashes,
        evaluator_version=labels.evaluator_version,
        failure_diagnostics=[{"scene_id": "scene"}],
    )
    with pytest.raises(RuntimeError):
        scorer.source_loss(logits, broken)


# --------------------------------------------------------------------------- #
# Real Multi-GT data contract
# --------------------------------------------------------------------------- #


def test_multiview_future_pose_indices_are_the_drivor_half_second_grid():
    from nvs_v6_wan_e2e.multiview_data import DENSE_FRAMES, FUTURE_POSE_INDICES, MAX_GT

    assert (DENSE_FRAMES, MAX_GT) == (41, 4)
    assert FUTURE_POSE_INDICES == (5, 10, 15, 20, 25, 30, 35, 40)
    assert len(FUTURE_POSE_INDICES) == 8
    assert [index * 0.1 for index in FUTURE_POSE_INDICES] == pytest.approx([0.5 * step for step in range(1, 9)])


def test_scene_labels_reads_multigt_labels_without_decoding_video(tmp_path):
    from nvs_v6_wan_e2e.multiview_data import scene_labels

    for index, label in enumerate(("human_log", "stop", "human_log")):
        trajectory = tmp_path / f"trajectory_{index}"
        trajectory.mkdir()
        (trajectory / "metadata.json").write_text('{"trajectory_label": "%s"}' % label)
    assert scene_labels(tmp_path) == ["human_log", "stop", "human_log"]
