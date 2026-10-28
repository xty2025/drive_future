"""Contract tests for the DrivoR-scene + Wan2.2-ActionDiT v6 generator.

Every assertion here is a contract the training scripts rely on: the future
representation as a DiT *output* over the read-only Scene context, DrivoR's
11-wide single ego token, metric 3-wide trajectories with no rescaling, the Wan
block geometry, and the official DrivoR scorer/loss/selection formulas.
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
    FUTURE_FRAMES,
    REGISTER_QUERIES_PER_FRAME,
    REGISTER_TOKENS,
    Observation,
    PairTargets,
    drivor_observable_ego_feature,
    drivor_route_command,
)
from nvs_v6_wan_e2e.losses import (
    dense_future_l1,
    hungarian_xy,
    identity_future_l1,
    masked_soft_bce,
    paired_residual_velocity_mse,
)
from nvs_v6_wan_e2e.model import (
    EGO_OBSERVABLE_DIM,
    EGO_STATUS_DIM,
    TRAJECTORY_STATE_DIM,
    DirectedPairDiT,
    FlowOutputs,
    parameter_groups,
)
from nvs_v6_wan_e2e.modes import ImmutableModeBuffer, StructuredSourceInitializer
from nvs_v6_wan_e2e.oracle import HEAD_INDICES, NAMED_ORDER, RAW_ORDER, OracleLabels
from nvs_v6_wan_e2e.selection import DRIVOR_WEIGHTS, aggregate_score_logits
from nvs_v6_wan_e2e.stages import build_optimizers, stage1_step
from nvs_v6_wan_e2e.target_cache import (
    MIN_GT_SPREAD_M,
    PAYLOAD_FORMAT,
    STANDARDIZER_NAME,
    atomic_torch_save,
    filter_usable_tokens,
    is_trajectory_degenerate,
    load_standardizer,
    multi_gt_spread_metres,
    save_standardizer,
    standardizer_path,
)
from nvs_v6_wan_e2e.teacher import (
    MAX_QUERY_ATTENTION_ENTROPY,
    QUERY_HEADS,
    QUERY_TEMPERATURE,
    TARGET_STD_FLOOR,
    FutureQueryExtractor,
    TargetStandardizer,
    normalized_attention_entropy,
    query_future_register,
)
from nvs_v6_wan_e2e.vision import (
    DRIVOR_IMAGE_HW,
    DRIVOR_LORA_RANK,
    DRIVOR_NUM_SCENE_TOKENS,
    DRIVOR_OFFICIAL_IMAGE_HW,
    DRIVOR_WIDTH_RESAMPLE_PX,
    LoRAQKVtimm,
    pad_to_patch_multiple,
    resample_to_patch_multiple,
)
from nvs_v6_wan_e2e.wan_dit import (
    MODE_SCALE,
    RMSNorm,
    SHAPE_RANGE,
    WAN_CONDITION_RMS,
    DiTBlock,
    Head,
    NvsActionDiT,
    detrend_trajectory,
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
#: The register target's width is deliberately NOT ``D_MODEL``.  In production it is
#: the Teacher's own feature width (1024 against a 256 d_model), and the fixtures keep
#: the two unequal so that any code assuming the future head emits ``d_model`` fails
#: here rather than in a run that silently rebuilt the head at the wrong width.
REGISTER_TARGET_DIM = 64
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
        register_target_dim=REGISTER_TARGET_DIM,
    )
    kwargs.update(overrides)
    return DirectedPairDiT(**kwargs).eval()


def make_targets(batch: int = 1, gt: int = 2, d: int = REGISTER_TARGET_DIM) -> PairTargets:
    return PairTargets(
        gt_trajectory=torch.randn(batch, gt, 8, TRAJECTORY_STATE_DIM),
        gt_valid=torch.ones(batch, gt, dtype=torch.bool),
        future_register=torch.randn(batch, gt, REGISTER_TOKENS, d),
        future_chunk_valid=torch.ones(batch, gt, REGISTER_TOKENS, dtype=torch.bool),
        future_quality=torch.ones(batch, gt, REGISTER_TOKENS),
        pair_ids=[[{"gt_id": str(index)} for index in range(gt)] for _ in range(batch)],
    )


RETIRED_BYPASS_PREFIXES = ("future_fuse", "ema_theta", "_ema_theta")


def assert_no_retired_bypass(model: DirectedPairDiT) -> None:
    """The mask is only a contract if nothing routes around it outside self-attention.

    ``future_fuse`` was exactly such a route: a ``Register -> Action`` residual added
    after the block stack, which no attention mask can see.  It is gone, and this is
    what keeps it gone -- a reintroduced module under any of these prefixes fails here
    rather than silently re-opening the arrow :data:`SEGMENT_VISIBILITY` closes.
    """

    offenders = [
        name for name, _ in model.named_parameters() if name.startswith(RETIRED_BYPASS_PREFIXES)
    ]
    offenders += [name for name in dir(model) if name.startswith(RETIRED_BYPASS_PREFIXES)]
    offenders += [name for name, _ in model.named_modules() if name.startswith(RETIRED_BYPASS_PREFIXES)]
    assert not offenders, f"a retired Register -> Action bypass is back: {sorted(set(offenders))}"
    assert not hasattr(model, "update_ema") and not hasattr(model, "enable_ema")


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
    # `config_v6_wan_e2e.yaml` is this package's spec file.  The loader is strict,
    # so a flag that contradicts the token contract fails here rather than silently
    # changing what the model computes.
    config = load_v6_config(REPO_ROOT / "config" / "config_v6_wan_e2e.yaml")
    model = config["model"]
    # One self-attention sequence, [scene | action | register], under ONE mask that
    # every stage and every entry point shares -- FastWAM's layout for the sequence,
    # block lower-triangular.  The register reads the action beside it; the action
    # never reads the register, so the predicted future cannot reach the velocity
    # head.  Scene reads only itself, which is what keeps the scorer's K/V exactly
    # DrivoR's.
    assert model["token_order"] == ["scene", "action", "register"]
    assert model["token_counts"] == [81, 8, REGISTER_TOKENS]
    assert model["self_attention_mask"] == "single_segment_block"
    assert model["segment_visibility"] == [list(r) for r in NvsActionDiT.SEGMENT_VISIBILITY]
    assert model["segment_visibility"] == [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
    # The invariant, read straight off the table rather than off a comment: row 1 is
    # the Action query and column 2 is the Register key.
    assert NvsActionDiT.SEGMENT_VISIBILITY[1][2] == 0
    assert NvsActionDiT.SEGMENT_VISIBILITY[2][1] == 1
    assert model["register_tokens"] == REGISTER_TOKENS
    assert config["future_target"]["registers_per_gt"] == REGISTER_TOKENS
    assert model["register_frames"] == FUTURE_FRAMES
    assert REGISTER_TOKENS == FUTURE_FRAMES * REGISTER_QUERIES_PER_FRAME
    assert model["register_rope_phase"] == "frame_index_repeated_per_query"
    assert model["future_reads_action"] is True
    assert model["action_reads_future"] is False
    assert model["future_self_causal"] is False
    assert model["scene_in_self_attention"] is True
    assert model["scene_written_by_action_dit"] is False
    assert model["scene_returned_to_scorer_is_unfused"] is True
    assert model["cross_attention_context"] == "scene"
    assert model["scene_embedded_once_used_for_self_and_cross_attention"] is True
    assert model["rope"] == "per_segment_restart_at_zero"
    assert model["segment_type_embeddings"] == [
        "scene_type_embedding",
        "action_type_embedding",
        "register_type_embedding",
    ]
    assert model["future_is_auxiliary_readout_only"] is True
    # Every one of these named either a per-stage mask selector or the removed
    # ``Register -> Action`` fusion.  Their absence is the contract now: a yaml edit
    # that re-declares one would describe an information flow no code runs.
    for retired in (
        "attention_stage_of",
        "stage1_segment_visibility",
        "stage2_segment_visibility",
        "future_reads_action_in_stage1",
        "future_reads_action_in_stage2",
        "action_reads_future_in_stage1",
        "action_reads_future_in_stage2",
        "future_condition_assists_action",
        "future_fuse",
        "future_fuse_stages",
        "future_fuse_out_proj_init",
        "ema_teacher",
        "ema_decay",
        "ema_update",
        "ema_teacher_input",
    ):
        assert retired not in model, f"{retired} must stay retired"
    for stage in ("stage1", "stage2"):
        assert "attention_mask" not in config[stage], f"{stage} must not declare a mask"
        for retired in ("future_fuse", "ema_theta"):
            assert retired not in config[stage]["train"]
            assert retired not in config[stage]["freeze"]
    assert model["future_query_init_inputs"] == ["learned_future_queries", "projected_clean_ego"]
    # Load-bearing now that a frame has four queries sharing one RoPE phase: an exact
    # zero would make them bitwise identical for the whole of step 0 and they would
    # never separate.
    assert model["action_dit"]["future_query_init"] == "randn_scaled_0.5_not_exact_zero"
    assert model["action_dit"]["future_tokens"] == REGISTER_TOKENS
    # The future head emits the register target's width, so the Action branch and the
    # target definition cannot drift apart.
    assert model["action_dit"]["register_target_dim"] == config["future_target"]["register_target_dim"]
    # Stage 1 must not be able to identify a GT from its row index, so the mode code
    # is zeros rather than the per-row immutable code the flow path uses.
    assert model["register_stage1_mode_code"] == "zeroed_to_prevent_row_index_shortcut"
    assert model["stage1_pairing"] == "identity_one_register_row_per_gt"
    assert model["ego_tokens"] == 1
    assert model["ego_observable_dim"] == 11
    assert model["trajectory_state_dim"] == TRAJECTORY_STATE_DIM
    assert model["trajectory_rescaling"] == "none"
    assert model["action_codec"] == "drivor_metric_xy_heading"
    assert config["vision"]["num_scene_tokens"] == 16
    assert config["vision"]["scene_tokens"] == 81
    assert config["vision"]["drivor_image_hw"] == [672, 1148]
    assert config["vision"]["right_resample_px"] == 4
    assert config["vision"]["right_zero_pad_px"] == 0
    assert config["vision"]["image_hw"] == [672, 1152]
    assert config["vision"]["student_resampling"] == "bicubic_to_patch_multiple"
    assert config["vision"]["position_encoding"] == "dinov3_rope_analytic_no_interpolation"
    assert config["future_target"]["teacher_image_hw"] == [464, 832]
    assert config["future_target"]["teacher_grid_hw"] == [29, 52]
    assert config["future_target"]["teacher_resize_filter"] == "none_native_resolution_no_interpolation"
    assert config["future_target"]["rope_normalizes_coords_per_axis"] is True
    assert config["future_target"]["teacher_checkpoint"] == config["vision"]["checkpoint"]
    # The target definition itself, written out as literals rather than re-read from
    # the constants ``load_v6_config`` already compared them to.  The first three are
    # what ``TargetStandardizer.fingerprint`` hashes, so two caches agreeing on a
    # fingerprint are comparable; this is the assertion that catches someone changing
    # the constant and the yaml together, which the strict loader cannot by
    # construction.  The token count is not in the fingerprint -- it is caught by the
    # ``(gt_count, REGISTER_TOKENS, dim)`` shape check in ``targets_from_payload``.
    assert config["future_target"]["spatial_readout"] == (
        "dinov3_register_query_cosine_attention_over_patch_grid"
    )
    assert config["future_target"]["query_is_parameter_free"] is True
    assert config["future_target"]["query_temperature"] == 0.005
    assert config["future_target"]["query_heads"] == 8
    # The ceiling was 0.5 when one CLS query read the grid.  The four DINOv3
    # registers are *broader* reads -- measured selective points sit at 0.594-0.623 of
    # uniform while a degenerate mean-pool sits at 0.936+, with nothing in between --
    # so the ceiling moved to the only gap in that distribution.
    assert config["future_target"]["max_query_attention_entropy"] == 0.75
    assert float(config["future_target"]["max_query_attention_entropy"]) == MAX_QUERY_ATTENTION_ENTROPY
    assert config["future_target"]["compression"] == (
        "dinov3_register_query_cosine_attention_then_train_only_frozen_diagonal_standardization"
    )
    assert config["future_target"]["target_std_floor"] == 0.01
    # Four queries per frame, and they are DINOv3's own four prefix registers rather
    # than a tuned count: the probe put CLS-only at 0.211 separation and the registers
    # read verbatim at 0.194, while the same registers used as queries over the patch
    # grid reach 0.458 with a 0.771 Pearson correlation against GT distance.
    assert config["future_target"]["registers_per_frame"] == 4
    assert config["future_target"]["temporal_pooling"] == "none_registers_per_future_frame"
    assert config["future_target"]["registers_per_gt"] == 8 * 4
    # The Teacher's own feature width, and deliberately not d_model: the future head
    # projects 256 -> 1024, so folding the two back into one key would widen the whole
    # DiT and then fail Wan's num_heads * attn_head_dim == hidden_dim check.
    assert config["future_target"]["register_target_dim"] == 1024
    assert model["action_dit"]["register_target_dim"] == 1024
    assert model["action_dit"]["hidden_dim"] == 256
    for retired in ("target_dim", "whiten_std_floor", "teacher_quadrant_split"):
        assert retired not in config["future_target"], retired
    for stage in ("stage1", "stage2"):
        assert "target_standardizer" in config[stage]["freeze"]
        assert "target_compressor" not in config[stage]["freeze"]
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


def test_strict_config_rejects_zero_padding_instead_of_resampling(tmp_path):
    import yaml

    source = REPO_ROOT / "config" / "config_v6_wan_e2e.yaml"
    config = yaml.safe_load(source.read_text())
    # 1148 is DrivoR's official width and is 14*82 for its patch14 DINOv2, but not
    # a multiple of 16.  The remainder must be closed by resampling the width up to
    # the backbone's native 1152 and building the position encoding for that grid;
    # zero columns would fabricate patch content, so a padding config is refused.
    config["vision"]["right_resample_px"] = 0
    config["vision"]["right_zero_pad_px"] = 4
    config["vision"]["student_resampling"] = "forbidden_zero_pad_only"
    mutated = tmp_path / "config_v6_wan_e2e.yaml"
    mutated.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="zero-pad"):
        load_v6_config(mutated)


def test_dit_sequence_is_scene_action_register():
    """The whole token contract in one place: 81 scene, then 8 action, then 32 register.

    All three segments sit in one self-attention sequence under one mask.  Scene is
    *additionally* still the cross-attention context, which keeps Wan's pretrained
    ``cross_attn`` on the input distribution it was trained on.  Asserting the
    concatenation order here is what makes ``x[:, scene:]`` /
    ``x[:, scene:scene+action]`` / ``x[:, scene+action:]`` in ``NvsActionDiT.forward``
    a contract instead of a convention -- the segment table is indexed by exactly that
    order.
    """

    model = make_model()
    dit = model.theta
    assert dit.action_tokens == 8
    assert dit.future_tokens == REGISTER_TOKENS
    assert tuple(dit.future_queries.shape) == (REGISTER_TOKENS, D_MODEL)
    observation = make_observation()
    modes = ImmutableModeBuffer().subset(4).expand_for_batch(1)
    source = torch.randn(1, 4, 8, TRAJECTORY_STATE_DIM)
    clean_ego = model.eta.encode_clean_ego(observation)
    action = dit.encode_action(source, modes, clean_ego)
    future = dit.encode_future(action, clean_ego)
    condition = dit.encode_condition(modes, clean_ego)
    context = model._refine_scene(model.eta(observation, clean_ego=clean_ego)).expand(4, SCENE_TOKENS, D_MODEL)

    seen: dict[str, tuple[int, ...]] = {}
    rope_positions: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []

    def record_self(module, args, kwargs, output):
        seen["self"] = (tuple(args[0].shape)[1], tuple(args[1].shape)[0])
        rope_positions.append(args[1])
        masks.append(kwargs["self_attn_mask"])

    def record_cross(module, args, kwargs, output):
        seen["cross"] = (tuple(args[0].shape)[1], tuple(args[1].shape)[1])
        # The cross-attention context is the whole Scene; nothing masks it.
        assert kwargs.get("ctx_mask") is None

    handles = [
        dit.blocks[0].self_attn.register_forward_hook(record_self, with_kwargs=True),
        dit.blocks[0].cross_attn.register_forward_hook(record_cross, with_kwargs=True),
    ]
    try:
        velocity, register = dit(
            action.reshape(4, dit.action_tokens, D_MODEL),
            future.reshape(4, dit.future_tokens, D_MODEL),
            condition.reshape(4, D_MODEL),
            context,
        )
    finally:
        for handle in handles:
            handle.remove()
    sequence = SCENE_TOKENS + dit.action_tokens + dit.future_tokens
    assert seen["self"] == (sequence, sequence)
    assert seen["cross"] == (sequence, SCENE_TOKENS)
    assert tuple(velocity.shape) == (4, dit.action_tokens, TRAJECTORY_STATE_DIM)
    # The future head emits the *target's* width, not the trunk's -- which is why this
    # is not D_MODEL, and why the register segment stays 256-wide inside the trunk.
    assert tuple(register.shape) == (4, dit.future_tokens, REGISTER_TARGET_DIM)

    # A mask really is passed, it is boolean (True = may read), and it is 2-D so it
    # broadcasts over batch and heads.
    for mask in masks:
        assert mask is not None
        assert mask.dtype is torch.bool
        assert tuple(mask.shape) == (sequence, sequence)
        expected = dit.segment_attention_mask([SCENE_TOKENS, dit.action_tokens, dit.future_tokens], mask.device)
        assert torch.equal(mask, expected)
        # Every query row keeps at least its own segment visible, so no softmax row is
        # fully masked -- the one way adding a mask produces a silent NaN.
        assert bool(mask.any(dim=-1).all())

    # RoPE restarts per segment.  Action keeps Wan's own 0..7 despite Scene being
    # prepended -- ``head``, ``time_queries`` and all eight transferred blocks were
    # pretrained with the action tokens at exactly those positions -- and the register
    # segment repeats that same 0..7, once per frame: its four queries are four reads
    # of ONE instant, so they must share one phase.  Counting 0..31 instead would
    # place a frame's queries at four different times and teach the trunk that query
    # identity is a timestamp.
    freqs = rope_positions[0]
    assert tuple(freqs.shape)[0] == sequence
    action_slice = slice(SCENE_TOKENS, SCENE_TOKENS + dit.action_tokens)
    register_slice = slice(SCENE_TOKENS + dit.action_tokens, sequence)
    assert torch.equal(freqs[:SCENE_TOKENS], dit.rope_for(SCENE_TOKENS, freqs.device))
    assert torch.equal(freqs[action_slice], dit.rope_for(dit.action_tokens, freqs.device))
    assert torch.equal(freqs[register_slice], dit.register_rope(dit.future_tokens, freqs.device))
    # ``rope_for`` returns ``[L, 1, freq_dim]``, so the query axis is spliced in
    # ahead of those trailing dims rather than flattening them into the reshape.
    per_frame = freqs[register_slice].reshape(FUTURE_FRAMES, REGISTER_QUERIES_PER_FRAME, *freqs.shape[1:])
    assert torch.equal(per_frame, per_frame[:, :1].expand_as(per_frame))
    assert torch.equal(per_frame[:, 0], freqs[action_slice])


def test_the_segment_mask_implements_the_one_information_flow():
    """The mask table is the spec, read literally: rows are queries, columns keys.

    There is one table for every stage and every entry point.  It is lower triangular
    over ``(Scene, Action, Register)``, so the invariant it states is that Action never
    reads Register -- the velocity head cannot see a predicted future, in training or
    in deployment, through self-attention.  ``Action -> Register`` stays open because
    that is the only thing distinguishing the K Multi-GT rows of a scene: ``encode_future``
    supplies no trajectory and no mode code, so a register that could not read the action
    would be bitwise identical across them.  Scene reads only itself, which is what keeps
    the scorer's K/V untouched by the ActionDiT.
    """

    dit = make_model().theta
    lengths = [SCENE_TOKENS, dit.action_tokens, dit.future_tokens]
    scene, action, register = (slice(0, SCENE_TOKENS),
                               slice(SCENE_TOKENS, SCENE_TOKENS + dit.action_tokens),
                               slice(SCENE_TOKENS + dit.action_tokens, sum(lengths)))

    mask = dit.segment_attention_mask(lengths, torch.device("cpu"))
    assert [list(row) for row in dit.SEGMENT_VISIBILITY] == [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
    assert not bool(dit.SEGMENT_VISIBILITY[1][2]), "Action must never read Register"
    assert bool(dit.SEGMENT_VISIBILITY[2][1]), "Register must read Action"
    assert mask[action, register].logical_not().all(), "Action -> Register must be closed"
    assert mask[register, action].all(), "Register -> Action must be open"
    assert mask[register, scene].all()
    assert mask[action, scene].all()
    assert mask[scene, action].logical_not().all()
    assert mask[scene, register].logical_not().all()
    assert mask[scene, scene].all()

    # ``predict_future=False`` truncates the table to its top-left 2x2, which is exactly
    # the mask a two-segment sequence needs -- so there is no special case to maintain.
    two_segments = sum(lengths[:2])
    expected = torch.zeros(two_segments, two_segments, dtype=torch.bool)
    expected[:SCENE_TOKENS, :SCENE_TOKENS] = True
    expected[SCENE_TOKENS:, :SCENE_TOKENS] = True
    expected[SCENE_TOKENS:, SCENE_TOKENS:] = True
    assert torch.equal(dit.segment_attention_mask(lengths[:2], torch.device("cpu")), expected)

    with pytest.raises(ValueError):
        dit.segment_attention_mask([SCENE_TOKENS], torch.device("cpu"))
    with pytest.raises(ValueError):
        dit.segment_attention_mask([*lengths, dit.future_tokens], torch.device("cpu"))


def _dit_block_inputs(dit, candidates: int = 2):
    """Independent leaves for the block stack, one per segment.

    Built as raw tensors rather than through ``encode_action``/``encode_future`` on
    purpose: ``encode_future`` consumes the action encoding, so perturbing the action
    there would move the register input too and the probe could not tell a mask from a
    shared encoder.
    """

    torch.manual_seed(23)
    return (
        torch.randn(candidates, dit.action_tokens, D_MODEL),
        torch.randn(candidates, dit.future_tokens, D_MODEL),
        torch.randn(candidates, D_MODEL),
        torch.randn(candidates, SCENE_TOKENS, D_MODEL),
    )


def test_attention_keeps_the_token_axis_intact_across_the_head_split():
    """``_split`` must *transpose* the head axis into place, not reshape it.

    ``q``/``k``/``v`` emit ``[B, L, H*D]`` in (token, head, dim) order.  Reading that
    buffer as ``[B, H, L, D]`` with a plain ``view`` hands SDPA a strided mix of every
    token, so each "query position" already contains other tokens' features before any
    mask is applied -- an attention no ``attn_mask`` can restrict, with the RoPE phases
    ``rope_apply`` just wrote attached to the wrong positions.  Every shape check still
    passes, which is why this is asserted behaviourally: under a mask that lets query
    ``i`` read only key ``i``, output row ``i`` must be bitwise independent of the other
    rows' inputs.  With ``num_heads == 1`` the buggy and correct forms coincide, so the
    fixture is required to use more.
    """

    dit = make_model().theta
    block = dit.blocks[0]
    torch.manual_seed(5)
    length = 6
    half = length // 2
    identity = torch.eye(length, dtype=torch.bool)
    freqs = dit.rope_for(length, torch.device("cpu"))
    queries = torch.randn(2, length, dit.hidden_dim)
    moved = queries.clone()
    moved[:, half:] += 7.0
    assert block.self_attn.num_heads > 1

    for name, call in (
        # Self-attention uses one tensor for q/k/v, so the moved tensor *is* the input.
        ("self_attn", lambda query, key: block.self_attn(key, freqs, self_attn_mask=identity)),
        ("cross_attn", lambda query, key: block.cross_attn(query, key, ctx_mask=identity)),
    ):
        with torch.no_grad():
            base = call(queries, queries)
            after = call(queries, moved)
        assert torch.equal(base[:, :half], after[:, :half]), name
        assert not torch.equal(base[:, half:], after[:, half:]), name


def test_the_mask_blocks_the_leak_in_both_directions_of_autograd():
    """The table is only a contract if the forward and the backward obey it.

    ``test_segment_mask_implements_the_information_flow`` pins the table; this pins
    what it *does*.  The leak it exists to rule out is not cosmetic: the register is
    supervised by ``||R_pred - R^GT||_1``, so if the register could reach the action
    stream the trajectory head would be trained by the future it is supposed to
    predict, and the deployed forward -- which has no GT future at all -- would be
    evaluating a different function from the one training fitted.

    ``head.proj`` is lifted off its deliberately near-zero init first, otherwise the
    velocity is O(3e-5) and the "this arrow is open" half of the test would pass or
    fail on float noise rather than on the mask.
    """

    dit = make_model().theta
    with torch.no_grad():
        dit.head.proj.weight.normal_(0.0, 0.05)
    action, future, condition, context = _dit_block_inputs(dit)
    action_delta = torch.randn_like(action)
    future_delta = torch.randn_like(future)

    with torch.no_grad():
        velocity, register = dit(action, future, condition, context)
        velocity_after_register, register_after_register = dit(action, future + future_delta, condition, context)
        register_after_action, _ = dit(action + action_delta, future, condition, context)
        velocity_after_action, _ = dit(action + action_delta, future, condition, context)

    # Sanity: each perturbation moves its own segment's read-out, so a blocked arrow
    # below is the mask and not a dead input.
    assert not torch.equal(register_after_register, register)
    assert not torch.equal(velocity_after_action, velocity)

    # The invariant, forward: perturbing the register segment cannot move the velocity.
    assert torch.equal(velocity_after_register, velocity), "Register -> Action must be closed"
    # ...and the arrow that must stay open really is open.
    assert not torch.equal(register_after_action, register), "Action -> Register must be open"

    # The same two arrows, read as gradients -- which is the form the leak actually
    # takes during training.  A blocked arrow shows up as an exactly-zero ``.grad``
    # rather than ``None``: SDPA still runs a backward over the register keys, it just
    # contributes nothing, so the test asserts on the value and not on presence.
    def gradient_scale(loss_source: str) -> float:
        probe_action = action.clone().requires_grad_(True)
        probe_future = future.clone().requires_grad_(True)
        velocity_out, register_out = dit(probe_action, probe_future, condition, context)
        (velocity_out if loss_source == "velocity" else register_out).sum().backward()
        wanted, other = (
            (probe_future, probe_action) if loss_source == "velocity" else (probe_action, probe_future)
        )
        # The unblocked direction must actually carry signal, or a zero in the blocked
        # direction would prove nothing.
        assert float(other.grad.abs().max()) > 0.0
        return float(wanted.grad.abs().max())

    assert gradient_scale("velocity") == 0.0, "the velocity loss must not reach the register segment"
    assert gradient_scale("register") > 0.0, "the register loss must reach the action segment"

    # Scene reads only itself, so the rows DrivoR's scorer takes as K/V are a function
    # of the scene alone: perturbing the action and the future together must leave them
    # bitwise identical after the whole block stack.
    scene_rows: list[torch.Tensor] = []
    handle = dit.blocks[-1].register_forward_hook(lambda module, args, output: scene_rows.append(output[:, :SCENE_TOKENS]))
    try:
        with torch.no_grad():
            dit(action, future, condition, context)
            dit(action + action_delta, future + future_delta, condition, context)
    finally:
        handle.remove()
    assert torch.equal(scene_rows[0], scene_rows[1]), "Scene must not read Action or Register"


def test_segment_type_embeddings_break_the_overlapping_rope_tie():
    """Scene and Action share RoPE positions, so only the type embeddings separate them.

    With ``scene_type_embedding``/``action_type_embedding``/``register_type_embedding``
    all equal, a block could not tell scene token 5 from action token 5, and the
    read-out slice ``x[:, scene:]`` would be reading a permutation-invariant soup.
    """

    dit = make_model().theta
    for name in ("scene_type_embedding", "action_type_embedding", "register_type_embedding"):
        parameter = getattr(dit, name)
        assert tuple(parameter.shape) == (1, 1, D_MODEL)
        assert float(parameter.std()) > 0.0
    assert not torch.allclose(dit.scene_type_embedding, dit.action_type_embedding)
    assert not torch.allclose(dit.action_type_embedding, dit.register_type_embedding)
    # Same 0.02 scale as time_queries/future_queries: far below the residual stream, so
    # the Wan-transferred blocks still see essentially the tokens they were trained on.
    assert float(dit.scene_type_embedding.std()) < 0.05


def test_future_register_reads_the_trajectory_and_the_noise_level():
    """Stage 1 reads the clean GT, Stage 2 reads the noised one -- through the DiT.

    The future queries have no trajectory input of their own, so the only way the
    register can depend on ``source`` is by attending to the action tokens beside
    them.  That dependency, plus the timestep dependence through Wan's modulation,
    is what makes one trunk serve both stages.
    """

    model = make_model()
    observation = make_observation()
    modes = ImmutableModeBuffer().subset(4).expand_for_batch(1)
    source_a = torch.randn(1, 4, 8, TRAJECTORY_STATE_DIM)
    source_b = source_a + torch.randn_like(source_a)
    clean = model.forward_register_only(observation, source_a, modes)
    other = model.forward_register_only(observation, source_b, modes)
    deployed = model.forward_pairs(observation, source_a, modes)
    assert not torch.allclose(clean, other, atol=1e-6, rtol=1e-6)
    # Same action tokens, different sigma: the register still moves, because the
    # condition feeds Wan's six-segment modulation.
    assert not torch.allclose(clean, deployed.predicted_register, atol=1e-6, rtol=1e-6)
    # Distinct candidates must stay distinct, or the eight rows collapse onto one
    # future and the per-frame targets become unreachable.
    assert not torch.allclose(clean[:, 0], clean[:, 1], atol=1e-6, rtol=1e-6)
    clean_ego = model.eta.encode_clean_ego(observation)
    assert torch.allclose(
        model._refine_scene(model.eta(observation, clean_ego=clean_ego)),
        deployed.scene_context,
        atol=1e-6,
        rtol=1e-6,
    )


def test_predict_future_false_drops_the_register_segment():
    """The no-target-cache path must leave the register tokens out entirely.

    ``ScratchDrivoRScorer`` deletes ``predicted_register``, so a joint run without a
    Teacher cache would otherwise hand DDP an identically-unused ``future_head``;
    eight unsupervised tokens would also perturb a pretrained action stream for no
    benefit.  Scene stays prepended in both cases -- it is the conditioning, not an
    auxiliary output.  Parameters stay allocated so a Stage 1 checkpoint still loads
    strict.
    """

    model = make_model(predict_future=False)
    assert model.theta.future_tokens == REGISTER_TOKENS
    assert sum(parameter.requires_grad for parameter in model.theta.future_head.parameters()) == 2
    observation = make_observation()
    modes = ImmutableModeBuffer().subset(4).expand_for_batch(1)
    source = torch.randn(1, 4, 8, TRAJECTORY_STATE_DIM)

    lengths: list[int] = []
    handle = model.theta.blocks[0].self_attn.register_forward_hook(
        lambda module, inputs, output: lengths.append(tuple(inputs[0].shape)[1])
    )
    try:
        pairs = model.forward_pairs(observation, source, modes)
    finally:
        handle.remove()
    assert pairs.predicted_register is None
    assert lengths == [SCENE_TOKENS + model.theta.action_tokens]
    with pytest.raises(ValueError, match="predict_future"):
        model.forward_register_only(observation, source, modes)
    # Without a register segment there is nothing to fuse, and with the fusion gone
    # there is nothing to freeze -- but a ``Register -> Action`` residual bolted back
    # on would run here on a zero-length register, so the retirement is checked in the
    # one configuration where such a module would be hardest to notice.
    assert_no_retired_bypass(model)


def test_register_rows_follow_the_trajectory_not_the_row_index():
    """The Stage-1 anti-shortcut guarantee.

    With a zeroed mode code, permuting the source rows must permute the register rows
    identically: what a row predicts is decided by the trajectory in it, not by where
    it sits.  Without this, ``register[k]`` could answer "which GT am I" from the row
    index alone and never read the action tokens, which is how the anchor-only
    Stage 1 ended up at a 0.4% shuffle margin.
    """

    model = make_model()
    observation = make_observation()
    source = torch.randn(1, 4, 8, TRAJECTORY_STATE_DIM)
    zeros = torch.zeros(1, 4, 2)
    permutation = torch.tensor([2, 0, 3, 1])
    base = model.forward_register_only(observation, source, zeros)
    permuted = model.forward_register_only(observation, source[:, permutation], zeros)
    assert torch.allclose(base[:, permutation], permuted, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(base, permuted, atol=1e-6, rtol=1e-6)


def test_each_gt_gets_its_own_register_group_and_the_groups_do_not_mix():
    """Multi-GT is expanded along the **batch** axis, so every GT owns its own 32 tokens.

    ``_flow_pass`` reshapes ``[B,M,...]`` to ``[B*M,...]`` and replicates Scene per
    row, which means the sequence length is ``scene + action + register`` no matter
    how many GT a scene carries -- M buys M *groups* of ``REGISTER_TOKENS`` register
    tokens, not ``M * REGISTER_TOKENS`` slots in one sequence.  Two things follow and both are asserted here,
    because the alternative layouts are easy to write by accident and equally easy to
    read as correct:

    * a scene with 1 GT and a scene with 4 run through the same code and produce
      ``[B,M,8,D]``, so the 19% of the corpus with fewer than four distinct labels is
      not a special case;
    * register group ``m`` depends on ``source[:, m]`` **only**.  That is the whole
      content of "one group per GT": if the groups could see each other, row ``k``
      could read GT ``k'``'s trajectory and the identity pairing in
      ``identity_future_l1`` would supervise the wrong future.
    """

    model = make_model()
    observation = make_observation(batch=2)
    lengths = [model.scene_tokens, model.action_tokens, model.register_tokens]
    device = next(model.parameters()).device
    mask = model.theta.segment_attention_mask(lengths, device)
    assert tuple(mask.shape) == (sum(lengths), sum(lengths))

    source = torch.randn(2, 4, 8, TRAJECTORY_STATE_DIM)
    zeros = torch.zeros(2, 4, 2)
    full = model.forward_register_only(observation, source, zeros)
    assert tuple(full.shape) == (2, 4, REGISTER_TOKENS, REGISTER_TARGET_DIM)

    for gt in (1, 2, 3):
        narrow = model.forward_register_only(observation, source[:, :gt], zeros[:, :gt])
        assert tuple(narrow.shape) == (2, gt, REGISTER_TOKENS, REGISTER_TARGET_DIM)
        # Same scene, same trunk, same mask width: the first ``gt`` groups of the
        # 4-GT forward and the ``gt``-GT forward are the same computation.
        assert torch.allclose(narrow, full[:, :gt], atol=2e-5, rtol=2e-5)

    # Group independence, one row at a time: perturbing GT 1's trajectory may only
    # move register group 1.
    perturbed = source.clone()
    perturbed[:, 1] += 1.0
    moved = model.forward_register_only(observation, perturbed, zeros)
    for gt in range(4):
        changed = not torch.allclose(moved[:, gt], full[:, gt], atol=1e-6, rtol=1e-6)
        assert changed == (gt == 1)

    # The loss supervises every group, at every M.
    for gt in (1, 2, 3, 4):
        targets = make_targets(batch=2, gt=gt)
        predicted = model.forward_register_only(observation, targets.gt_trajectory, torch.zeros(2, gt, 2))
        assert float(identity_future_l1(predicted, targets)) > 0.0

    # A padded scene: rows past ``gt_valid`` must contribute nothing, so corrupting
    # their targets cannot move the loss.  This is what lets one batch mix a 2-GT
    # scene with a 4-GT one.
    padded = make_targets(batch=2, gt=4)
    padded.gt_valid[1, 2:] = False
    predicted = model.forward_register_only(observation, padded.gt_trajectory, zeros)
    reference = float(identity_future_l1(predicted, padded))
    assert reference > 0.0
    padded.future_register[1, 2:] = 999.0
    assert float(identity_future_l1(predicted, padded)) == pytest.approx(reference, rel=1e-6)


def test_future_queries_are_distinct_at_initialization():
    """Symmetry breaking: the eight queries must differ before any training step.

    V-JEPA initializes its mask tokens to exactly zero and relies on RoPE phase to
    tell them apart.  Here each query owns one future frame, so a shared value would
    make all eight rows of the register identical at step 0 and leave the per-frame
    targets unreachable until the trunk had already collapsed them together.
    """

    queries = make_model().theta.future_queries
    assert float(queries.std()) > 0.0
    assert not torch.allclose(queries[0], queries[1])
    assert tuple(queries.shape) == (REGISTER_TOKENS, D_MODEL)


def test_source_trajectory_changes_neither_scene_nor_the_scene_token_count():
    model = make_model()
    observation = make_observation()
    modes = ImmutableModeBuffer().expand_for_batch(1)[:, :8]
    source_a = torch.randn(1, 8, 8, TRAJECTORY_STATE_DIM)
    source_b = source_a + 0.3 * torch.randn_like(source_a)
    output_a = model.forward_pairs(observation, source_a, modes)
    output_b = model.forward_pairs(observation, source_b, modes)
    assert torch.allclose(output_a.scene_context, output_b.scene_context, atol=1e-6, rtol=1e-6)
    assert tuple(output_a.scene_context.shape) == (1, SCENE_TOKENS, D_MODEL)


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


def test_flow_register_depends_on_sigma():
    model = make_model()
    observation = make_observation(batch=2)
    modes = ImmutableModeBuffer().subset(6).expand_for_batch(2, torch.device("cpu"))
    source = StructuredSourceInitializer(jitter_std=0.0).clean(observation.ego_history, modes)
    data = source + torch.randn_like(source) * 3.0
    clean = model.forward_flow(observation, source, modes, data, torch.zeros(2))
    deployed = model.forward_flow(
        observation, source, modes, data, torch.full((2,), float(model.flow.num_train_timesteps))
    )
    assert tuple(clean.pairs.predicted_register.shape) == (2, 6, REGISTER_TOKENS, REGISTER_TARGET_DIM)
    assert not torch.allclose(clean.pairs.predicted_register, deployed.pairs.predicted_register, atol=1e-3)
    rows = clean.pairs.predicted_register.reshape(2, 6, -1)
    assert not torch.allclose(rows[:, 0], rows[:, 1], atol=1e-3)


def test_parameter_groups_are_disjoint_and_complete():
    model = make_model()
    groups = parameter_groups(model)
    ids = {name: set(map(id, group)) for name, group in groups.items()}
    # The future queries and their read-out head live inside theta now, so the two
    # optimizer groups are the two streams and nothing else.
    assert set(ids) == {"eta", "theta"}
    assert not (ids["eta"] & ids["theta"])
    assert all(groups.values())
    assert id(model.theta.time_queries) in ids["theta"]
    assert id(model.theta.future_queries) in ids["theta"]
    assert all(id(parameter) in ids["theta"] for parameter in model.theta.future_head.parameters())
    # Nothing generator-side may live outside the two streams: a fusion residual
    # bolted onto the model would be routed to ``theta`` by name prefix, or rejected
    # outright, and either way the ``retired`` prefixes must not reappear.
    assert_no_retired_bypass(model)
    # The Scene refinement stack is eta's: it is what Stage 2 freezes under plan A so
    # a score loss cannot rewrite perception through the shared trunk.
    assert all(id(parameter) in ids["eta"] for parameter in model.scene_blocks.parameters())
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
    assert output.predicted_register.shape == (1, candidate_count, REGISTER_TOKENS, REGISTER_TARGET_DIM)


def test_scene_and_joint_kv_token_lengths_are_stable():
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
        model.theta.blocks[0].self_attn.register_forward_pre_hook(capture("dit_self", 0)),
        model.theta.blocks[0].cross_attn.register_forward_pre_hook(capture("dit_cross", 1)),
        model.scene_blocks[0].attn.register_forward_pre_hook(capture("scene_self", 0)),
    ]
    try:
        model.forward_pairs(observation, source, modes)
    finally:
        for handle in handles:
            handle.remove()
    # One joint sequence of [scene | action | register] self-attends, and the same
    # embedded Scene is *also* the cross-attention context at its own 81-token length.
    # The separate refinement stack still never sees a candidate.
    assert lengths == {
        "dit_self": SCENE_TOKENS + 8 + REGISTER_TOKENS,
        "dit_cross": SCENE_TOKENS,
        "scene_self": SCENE_TOKENS,
    }


def test_future_tokens_are_in_the_trunk_but_not_in_the_action_readout():
    """The future queries shape the trunk; the trunk never emits a waypoint from them.

    Both halves matter and they pull in opposite directions.  The register must be part
    of the same self-attention sequence -- a side branch could not shape the shared
    representation, which is the entire reason Stage 1 exists -- yet ``Head`` reads
    ``x[:, :8]`` and the mask closes ``Register -> Action``, so the recovered trajectory
    must be **bit-for-bit independent** of the future tokens.  A gradient reaching
    ``future_queries`` from the trajectory loss would mean the deployed forward is not
    the function Stage 1 fitted.
    """

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
    assert captured[0].shape[-2] == SCENE_TOKENS

    eta = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith(("eta.", "scene_blocks.")) and parameter.requires_grad
    ]
    gradients = torch.autograd.grad(
        outputs.trajectory.sum(), [*eta, model.theta.future_queries], allow_unused=True, retain_graph=True
    )
    # scene_kv_detached is False, so the candidate loss still shapes perception.
    assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients[: len(eta)])
    # Zero rather than None: SDPA still runs a backward over the register keys, it just
    # contributes nothing, so the assertion is on the value and not on presence.
    future_gradient = gradients[-1]
    assert future_gradient is not None
    assert float(future_gradient.abs().max()) == 0.0

    # The zero above proves nothing unless the same parameter carries signal for the
    # loss that is supposed to read it.
    register_gradient = torch.autograd.grad(outputs.predicted_register.sum(), model.theta.future_queries)[0]
    assert float(register_gradient.abs().max()) > 0.0


# --------------------------------------------------------------------------- #
# What Stage 1 freezes, and what the scorer is handed
# --------------------------------------------------------------------------- #


def test_stage1_freezes_the_velocity_head_and_stages_agree_on_the_trunk():
    """Stage 1 freezes only the read-out it does not supervise.

    ``theta.head.*`` is the velocity head: nothing in the Stage 1 loss reads it, and
    leaving it trainable would put its near-zero anchor-preserving initialization at the
    mercy of whatever optimizer replaces AdamW (which skips grad-less parameters).  The
    trunk stays trainable in *both* stages -- the register queries live inside theta, so
    "learn the future representation" is not "freeze theta".  That the two stages differ
    only in the head is what makes Stage 1's product transferable to Stage 2.
    """

    from nvs_v6_wan_e2e.stages import _set_generator_trainability

    model = make_model(depth=1)
    _set_generator_trainability(model, 1)
    assert all(not parameter.requires_grad for parameter in model.theta.head.parameters())
    assert model.theta.future_queries.requires_grad
    assert all(parameter.requires_grad for parameter in model.theta.future_head.parameters())
    assert_no_retired_bypass(model)

    _set_generator_trainability(model, 2)
    assert all(parameter.requires_grad for parameter in model.theta.head.parameters())
    assert_no_retired_bypass(model)


def test_the_scorer_reads_the_unfused_scene():
    """DrivoR's scorer takes Scene tokens as K/V, so it must see eta's output.

    If the Scene handed to the scorer were anything other than ``_refine_scene(eta(...))``
    -- a future-augmented copy, a theta-side rewrite -- the score loss would rewrite
    perception through a channel the register also uses, and the scorer's K/V would stop
    being a function of the observation alone.  ``SEGMENT_VISIBILITY[0] == (1, 0, 0)``
    states this inside the DiT; this pins it on the model side, where the tokens are
    actually produced.
    """

    model = make_model(depth=1)
    observation = make_observation()
    modes = ImmutableModeBuffer().subset(2).expand_for_batch(1)
    source = torch.randn(1, 2, 8, TRAJECTORY_STATE_DIM)
    outputs = model.forward_pairs(observation, source, modes)
    clean_ego = model.eta.encode_clean_ego(observation)
    assert torch.allclose(
        model._refine_scene(model.eta(observation, clean_ego=clean_ego)),
        outputs.scene_context,
        atol=1e-6,
        rtol=1e-6,
    )


# --------------------------------------------------------------------------- #
# Teacher target: the prefix-query readout and the diagonal standardizer
# --------------------------------------------------------------------------- #


def _query_fixture(batch: int = 2, frames: int = 8, patches: int = 11, dim: int = 16, seed: int = 7):
    torch.manual_seed(seed)
    return torch.randn(batch, frames, 1, dim), torch.randn(batch, frames, patches, dim)


def _reference_readout(cls_token: torch.Tensor, patch_tokens: torch.Tensor, extractor) -> torch.Tensor:
    """The readout written out per (frame, query, head) with explicit slices.

    Recomputing it from the definition rather than from ``_split`` is what makes the
    head-split convention testable: ``view(B,L,H,Dh).transpose(1,2)`` gives head ``h``
    the contiguous slice ``[h*Dh, (h+1)*Dh)``, while a plain ``view(B,H,L,Dh)`` would
    reinterpret the same buffer and hand every head a strided mix of all the others.
    Both produce a tensor of the right shape, so only a hand-written reference
    distinguishes them.  ``cls_token`` may carry R > 1 queries; R = 1 is the CLS
    variant and R = 4 the production register variant, and one reference covers both
    because the two paths must be the same computation.
    """

    batch, frames, queries = cls_token.shape[:3]
    dim, head_dim = extractor.feature_dim, extractor.head_dim
    rows = []
    for b in range(batch):
        for f in range(frames):
            per_query = []
            for q in range(queries):
                per_head = []
                for h in range(extractor.heads):
                    query = cls_token[b, f, q, h * head_dim : (h + 1) * head_dim]
                    keys = patch_tokens[b, f, :, h * head_dim : (h + 1) * head_dim]
                    logits = (
                        nn.functional.normalize(query, dim=-1)
                        @ nn.functional.normalize(keys, dim=-1).transpose(-1, -2)
                        / extractor.temperature
                    )
                    per_head.append(logits.softmax(dim=-1) @ keys)
                per_query.append(torch.cat(per_head, dim=-1))
            rows.append(torch.stack(per_query))
    return torch.stack(rows).reshape(batch, frames, queries, dim)


def test_the_query_readout_is_per_frame_cosine_and_a_convex_combination():
    """The CLS-query variant: one token per future frame from a parameter-free cosine attention.

    Kept even though production reads with DINOv3's four registers instead
    (:func:`test_the_register_readout_queries_with_the_prefix_tokens`): the probe's
    ``cls_query_attn`` row is this path, and it is the reference the register path is
    checked against, so a change that moves one must be seen to move the other.

    Three properties are load-bearing and none is implied by the others: the readout
    is per frame (pooling across time is what the retired quadrant target did wrong),
    it matches the definition written out by hand (the head-split convention), and its
    logits are cosine while its *values* are not normalized (which is what lets one
    fixed temperature transfer across feature scales).
    """

    extractor = FutureQueryExtractor(feature_dim=16, temperature=0.005, heads=4)
    cls_token, patch_tokens = _query_fixture()
    out = extractor(cls_token, patch_tokens)
    assert out.shape == (2, 8, 16)
    # The hand-written reference keeps the query axis, so R = 1 shows up as a length-1
    # dim there and not here.
    assert torch.allclose(out, _reference_readout(cls_token, patch_tokens, extractor).squeeze(2), atol=1e-5)

    # The weights are a distribution and the output a convex combination, so it stays
    # inside the patch tokens' per-dimension range: nothing is invented.
    weights = extractor.attention_weights(cls_token, patch_tokens)
    assert weights.shape == (16, 4, 11)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(16, 4), atol=1e-6)
    assert bool((weights >= 0).all())
    # At the production temperature the readout is close to a hard selection, which is
    # the point: a flat distribution is the mean pool it replaced.  Some weights
    # underflow to exactly zero in float32, so "positive" is not assertable, and a
    # single frame can have two nearly equidistant patches, so the *mean* is the honest
    # statistic -- stated as the gated quantity rather than as a weight.
    assert float(normalized_attention_entropy(weights).mean()) < MAX_QUERY_ATTENTION_ENTROPY
    flat = patch_tokens.reshape(16, 11, 16)
    assert bool((out.reshape(16, 16) >= flat.min(dim=1).values - 1e-5).all())
    assert bool((out.reshape(16, 16) <= flat.max(dim=1).values + 1e-5).all())

    # Pooling across time would be the failure this guards: +0.5s and +1.0s must stay
    # separable, so perturbing one frame may only move that frame's register row.
    perturbed = patch_tokens.clone()
    perturbed[:, 3] += 100.0
    moved = (extractor(cls_token, perturbed) != out).any(dim=-1)
    assert bool(moved[:, 3].all())
    assert not bool(moved[:, [0, 1, 2, 4, 5, 6, 7]].any())

    # Cosine logits are invariant to the query's magnitude; a raw dot product is not.
    # This is why one temperature works at every feature scale, and the measured cost
    # of getting it wrong is a flat softmax: entropy 0.9987 of log2(P) and separation
    # 0.1414, worse than the mean pooling the readout replaced.
    assert torch.allclose(
        extractor.attention_weights(cls_token * 100.0, patch_tokens), weights, atol=1e-6
    )

    # The *values* stay un-normalized on purpose: rescaling every patch to unit length
    # would discard the magnitude the target is supposed to carry.  With one patch per
    # frame the softmax is exactly 1.0, so the readout IS that patch and scaling it has
    # to scale the readout by the same factor bitwise -- no tolerance involved, and a
    # value-normalized readout would return the same unit vector twice and fail here.
    single = patch_tokens[:, :, :1]
    assert torch.equal(extractor(cls_token, single * 3.0), extractor(cls_token, single) * 3.0)

    # The key side is cosine, so scaling every patch leaves the distribution alone.  Only
    # the weights are compared, and only at 1e-5: at tau=0.005 the softmax is near-hard,
    # so float32 rounding in the normalized key moves individual weights by ~1e-5, which
    # against values of order ten is ~1e-3 of readout.  Scaling the readout itself is
    # therefore not assertable at a tight tolerance, which is what the exact single-patch
    # case above is for.
    amplified = patch_tokens * 3.0
    assert torch.allclose(extractor.attention_weights(cls_token, amplified), weights, atol=1e-5)

    for bad, reason in (
        ((torch.zeros(8, 1, 16), torch.zeros(8, 11, 16)), r"cls_token must be \[B,F,1,D\]"),
        ((torch.zeros(2, 8, 2, 16), torch.zeros(2, 8, 11, 16)), r"cls_token must be \[B,F,1,D\]"),
        ((torch.zeros(2, 8, 1, 16), torch.zeros(2, 8, 11)), r"patch_tokens must be \[B,F,P,D\]"),
        ((torch.zeros(2, 8, 1, 16), torch.zeros(2, 7, 11, 16)), "do not align with cls_token"),
        ((torch.zeros(2, 8, 1, 16), torch.zeros(2, 8, 11, 15)), "do not align with cls_token"),
    ):
        with pytest.raises(ValueError, match=reason):
            extractor(*bad)
    with pytest.raises(ValueError, match="not divisible"):
        FutureQueryExtractor(feature_dim=15, heads=4)
    with pytest.raises(ValueError, match="must be positive"):
        FutureQueryExtractor(feature_dim=16, temperature=0.0, heads=4)


def test_the_register_readout_queries_with_the_prefix_tokens():
    """The production readout: DINOv3's own four registers as queries over the patch grid.

    Four properties, none implied by the others:

    * it is the *same* cosine attention as the CLS path, just with R queries instead
      of one -- checked against the hand-written reference, which covers both;
    * the flattened token axis is frame-major, so token ``R*f + q`` is query ``q`` of
      frame ``f``.  This is the layout the DiT's frame-repeated RoPE phase and
      ``query_future_register``'s reshape both assume, and a transpose here would put
      query 0 of every frame first -- every downstream number would stay finite;
    * the four queries of one frame actually differ.  If they collapsed to one
      another the segment would be 4x wider for nothing, which is the failure the
      probe measured against reading the registers verbatim;
    * the entropies come back per query rather than pre-averaged, so one dead query
      is visible in the cache audit instead of hiding behind three good ones.
    """

    torch.manual_seed(11)
    batch, frames, queries, patches, dim = 2, 8, REGISTER_QUERIES_PER_FRAME, 11, 16
    extractor = FutureQueryExtractor(feature_dim=dim, temperature=0.005, heads=4)
    registers = torch.randn(batch, frames, queries, dim)
    patch_tokens = torch.randn(batch, frames, patches, dim)

    tokens, entropy = extractor.forward_registers(registers, patch_tokens)
    assert tokens.shape == (batch, frames, queries, dim)
    assert entropy.shape == (batch * frames * queries, 4)
    assert torch.allclose(tokens, _reference_readout(registers, patch_tokens, extractor), atol=1e-5)

    # Frame-major, asserted by index rather than by trusting the reshape: token
    # ``queries * f + q`` must be query ``q`` of frame ``f``.
    flat = tokens.reshape(batch, queries * frames, dim)
    for f in range(frames):
        for q in range(queries):
            assert torch.equal(flat[:, queries * f + q], tokens[:, f, q])

    # One query per frame would make these four rows identical; the whole reason the
    # segment is 32 wide rather than 8 is that they are not.
    spread = (tokens - tokens.mean(dim=2, keepdim=True)).abs().mean()
    assert float(spread) > 0.0
    assert not torch.allclose(tokens[:, :, 0], tokens[:, :, 1], atol=1e-4)

    # Per-query entropies, not one average per frame: a single dead query has to be
    # countable on its own.
    per_query = entropy.reshape(batch, frames, queries, 4)
    assert not torch.allclose(per_query[:, :, 0], per_query[:, :, 1], atol=1e-6)
    assert bool((per_query >= 0).all()) and bool((per_query <= 1 + 1e-6).all())

    # Pooling across time is still the failure this guards: perturbing one frame may
    # only move that frame's queries.
    perturbed = patch_tokens.clone()
    perturbed[:, 3] += 100.0
    moved_tokens, _ = extractor.forward_registers(registers, perturbed)
    moved = (moved_tokens != tokens).any(dim=-1)
    assert bool(moved[:, 3].all())
    assert not bool(moved[:, [0, 1, 2, 4, 5, 6, 7]].any())

    for bad, reason in (
        # The CLS path's shape, and the easy mistake when the two readouts sit side by
        # side: a rank error, not an unpack error.
        ((torch.zeros(2, 8, 16), torch.zeros(2, 8, 11, 16)), r"register_tokens must be \[B,F,R,D\]"),
        ((torch.zeros(2, 8, 4, 16), torch.zeros(2, 8, 11)), "do not align with register_tokens"),
        ((torch.zeros(2, 8, 4, 16), torch.zeros(2, 7, 11, 16)), "do not align with register_tokens"),
        ((torch.zeros(2, 8, 4, 16), torch.zeros(2, 8, 11, 15)), "do not align with register_tokens"),
        ((torch.zeros(2, 8, 4, 15), torch.zeros(2, 8, 11, 15)), "do not align with register_tokens"),
    ):
        with pytest.raises(ValueError, match=reason):
            extractor.forward_registers(*bad)
    with pytest.raises(ValueError, match="at least one query per frame"):
        extractor.forward_registers(torch.zeros(2, 8, 0, 16), torch.zeros(2, 8, 11, 16))


class _RecordingTeacher:
    """Stand-in for ``TimmDinoV3Teacher`` that records the shape it was handed.

    The real Teacher needs DINOv3 weights and native 464x832 frames; the contract
    under test here is the rank and axis order at the loader/Teacher seam, so a stub
    that echoes a fixture is sufficient.  The spatial extent is kept small for the
    same reason -- resolution is ``_check_shape``'s business and is tested there.

    It answers ``_families`` rather than ``encode_tokens`` because that is what the
    register readout calls: the CLS-query path needed only patches, the
    register-query path needs the backbone's own prefix tokens too.  The register
    count is ``REGISTER_QUERIES_PER_FRAME`` unless overridden, which is how the
    wrong-prefix-count guard is exercised without a second stub.
    """

    def __init__(self, dim: int = 16, patches: int = 11, registers: int = REGISTER_QUERIES_PER_FRAME) -> None:
        self.dim, self.patches, self.registers, self.seen = dim, patches, registers, []

    def _families(self, images: torch.Tensor):
        batch, frames = int(images.shape[0]), int(images.shape[1])
        self.seen.append(tuple(int(value) for value in images.shape))
        cls_token, patch_tokens = _query_fixture(
            batch=batch, frames=frames, patches=self.patches, dim=self.dim
        )
        return (
            cls_token,
            torch.randn(batch, frames, self.registers, self.dim),
            patch_tokens,
        )

    def encode_tokens(self, images: torch.Tensor):
        cls_token, _, patch_tokens = self._families(images)
        return cls_token, patch_tokens


def test_query_future_register_matches_the_loaders_layout():
    """The seam between ``load_multiview_sample`` and the Teacher.

    ``multiview_data`` builds ``future_rgb`` as
    ``torch.stack(future_videos).unsqueeze(0).unsqueeze(3)``, i.e. **seven**
    dimensions ``[1,K,8,V,3,H,W]``.  The guard on this function demanded six, so
    every real-data caller -- Stage 1, Stage 0 and the separability diagnostic, all
    three of which go through that loader -- raised on its first scene.  It survived
    84 passing tests because nothing here exercised the function at all: the
    extractor was tested, the loader was tested, and the call that connects them was
    not.  Asserting the rank alone would not have caught it either, since the same
    off-by-one is expressible as a wrong *axis*, which is what the ``seen`` list
    pins down.
    """

    gt, views, height, width = 4, 1, 32, 48
    extractor = FutureQueryExtractor(feature_dim=16, heads=4)
    teacher = _RecordingTeacher()
    block, entropies = query_future_register(
        teacher, extractor, torch.zeros(1, gt, 8, views, 3, height, width), torch.device("cpu")
    )
    assert block.shape == (gt, REGISTER_TOKENS, 16)
    assert entropies.shape == (gt * REGISTER_TOKENS, 4)
    # The K loop must peel dim 1 and leave the Teacher the [B,F,V,3,H,W] that
    # ``_check_shape`` validates -- once per GT, never the whole scene at once.
    assert teacher.seen == [(1, 8, views, 3, height, width)] * gt

    for bad in (
        torch.zeros(1, 8, views, 3, height, width),  # the loader's per-GT slice, not a scene
        torch.zeros(1, gt, 4, views, 3, height, width),  # right rank, wrong frame axis
        torch.zeros(1, gt, 8, 3, height, width),  # views folded away
    ):
        with pytest.raises(ValueError, match=r"must be \[1,K,8,V,3,H,W\]"):
            query_future_register(teacher, extractor, bad, torch.device("cpu"))

    # A checkpoint with a different prefix count would otherwise write a target whose
    # token axis silently disagrees with the DiT's query count.  Both directions: too
    # many registers and too few.
    future = torch.zeros(1, gt, 8, views, 3, height, width)
    for registers in (3, 5):
        with pytest.raises(ValueError, match="the target axis must be 32 tokens"):
            query_future_register(
                _RecordingTeacher(registers=registers), extractor, future, torch.device("cpu")
            )


def _fake_multiview_scene(root: Path, token: str = "0f1e2d3c4b5a6978", gt_count: int = 4) -> tuple[Path, Path, Path]:
    """A minimal Multi-GT scene that ``load_multiview_sample`` accepts.

    Structurally real, content-wise empty: the log frames carry the poses and ego
    states the loader reads, the trajectory folders carry the dense ``[41,3]`` pose
    and the ``COMPLETE`` metadata it validates against, and the sensor blobs are
    deliberately absent so the loader takes its ``initial_front.jpg`` fallback
    instead of needing a 1920x1080 JPEG.  ``future.mp4`` is absent for the same
    reason, so callers must pass ``load_future_rgb=False`` -- this fixture pins the
    target *shape* contract, not the video decode, which has its own test.
    """

    import json
    import pickle

    import numpy as np
    from PIL import Image

    from nvs_v6_wan_e2e.multiview_data import DENSE_FRAMES, metric_cache_path

    log_name, frame_index = "fake_log", 6
    scene_token, timestamp = f"scene_{token[:8]}", 1_700_000_000_000_000
    frames = []
    for index in range(frame_index + 1):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = float(index)
        frames.append(
            {
                "token": token if index == frame_index else f"{token}_{index}",
                "scene_token": scene_token,
                "timestamp": timestamp if index == frame_index else timestamp + index,
                "ego2global": pose,
                "ego_dynamic_state": np.array([1.0, 0.0, 0.1, 0.0], dtype=np.float32),
                "cams": {"CAM_F0": {"data_path": f"absent/{index}.jpg"}},
                "driving_command": [1.0, 0.0, 0.0, 0.0],
            }
        )
    log_root = root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    with (log_root / f"{log_name}.pkl").open("wb") as handle:
        pickle.dump(frames, handle)

    scene_dir = root / token
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "scene_metadata.json").write_text(
        json.dumps(
            {
                "token": token,
                "log_name": log_name,
                "navsim_frame_index": frame_index,
                "scene_token": scene_token,
                "t0_timestamp_us": timestamp,
            }
        )
    )
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(scene_dir / "initial_front.jpg")

    dense = np.zeros((DENSE_FRAMES, 3), dtype=np.float32)
    dense[:, 0] = np.linspace(0.0, 10.0, DENSE_FRAMES)
    for gt in range(gt_count):
        trajectory = scene_dir / f"trajectory_{gt}"
        trajectory.mkdir(exist_ok=True)
        np.save(trajectory / "ego_pose.npy", dense)
        (trajectory / "metadata.json").write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "valid": True,
                    "pose_coordinate_frame": "t0 local rear; x forward/y left/yaw CCW; meters/radians",
                    "source_selected_candidate_index": gt,
                    "trajectory_label": f"label_{gt}",
                    "behavior": "straight",
                    "is_human_log": gt == 0,
                    "PDMS": 90.0 - gt,
                    "output_artifact_sha256": {"future.mp4": "0" * 64},
                }
            )
        )

    cache_root = root / "metric_cache"
    cache_path = metric_cache_path(scene_dir, cache_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"the loader only checks that this file exists")
    return scene_dir, log_root, cache_root


def test_the_loaders_register_placeholder_is_the_teacher_width_not_d_model(tmp_path):
    """The width of ``future_register`` at the loader/standardizer seam.

    ``load_multiview_sample`` runs *before* the Teacher is constructed, so it cannot
    read ``feature_dim`` off it and has to allocate its placeholder at a default that
    already equals the Teacher's 1024.  The parameter used to be ``target_dim = 256``
    -- d_model, a width the register target has not had since the query readout
    replaced the pooled-then-PCA one -- so ``with_future_register`` rejected the real
    ``[1,K,8,1024]`` target on the first scene, after the whole future video had been
    decoded *and* encoded.  Its message named only the two shapes, which read as a
    Teacher fault rather than a mis-sized placeholder.

    The default alone would not be enough to pin: a hardcoded 1024 at the allocation
    site satisfies it while ignoring the parameter, so the explicit override is
    asserted too.
    """

    from nvs_v6_wan_e2e.config import EXPECTED_REGISTER_TARGET_DIM
    from nvs_v6_wan_e2e.multiview_data import load_multiview_sample, with_future_register

    gt_count = 4
    scene_dir, log_root, cache_root = _fake_multiview_scene(tmp_path, gt_count=gt_count)
    kwargs = dict(
        log_root=log_root,
        sensor_root=tmp_path / "sensor_blobs",
        cache_root=cache_root,
        image_hw=(32, 48),
        load_future_rgb=False,
    )
    sample = load_multiview_sample(scene_dir, **kwargs)
    wanted = (1, gt_count, REGISTER_TOKENS, EXPECTED_REGISTER_TARGET_DIM)
    assert tuple(sample.targets.future_register.shape) == wanted
    # The conflation this guards against: d_model is 256 in production and has never
    # been the register width, so a placeholder of that width is the old bug.
    assert wanted[-1] != 256
    assert load_multiview_sample(scene_dir, register_target_dim=64, **kwargs).targets.future_register.shape[-1] == 64

    register = torch.randn(*wanted)
    updated = with_future_register(sample.targets, register)
    assert torch.equal(updated.future_register, register)
    assert updated.gt_trajectory.shape == sample.targets.gt_trajectory.shape
    assert updated.future_chunk_valid.all()

    # The reported failure, reproduced through the loader rather than by hand: the old
    # default sized the placeholder at d_model and the Teacher's real target no longer
    # fitted it.  The message has to name the parameter, because "must be (1,4,32,256),
    # got (1,4,32,1024)" reads as a Teacher fault and sends the reader to the wrong file.
    mis_sized = load_multiview_sample(scene_dir, register_target_dim=256, **kwargs)
    with pytest.raises(ValueError, match=r"register_target_dim=256"):
        with_future_register(mis_sized.targets, register)
    # The width hint is conditional: a wrong token count is a different mistake and
    # must not be explained in terms of the Teacher's feature_dim.  Spelled from the
    # constants so widening the register segment does not silently turn this into a
    # regex that no longer matches anything.
    short = r"must be \(1, 4, %d, 1024\), got \(1, 4, %d, 1024\)$" % (REGISTER_TOKENS, REGISTER_TOKENS - 1)
    with pytest.raises(ValueError, match=short) as error:
        with_future_register(sample.targets, torch.randn(1, gt_count, REGISTER_TOKENS - 1, wanted[-1]))
    assert "feature_dim" not in str(error.value)


def test_the_target_cache_payload_round_trips_through_the_loaders_history_paths(tmp_path):
    """Stage 0's write path, end to end: loader -> payload -> disk -> streaming dataset.

    ``build_scene_payload`` had no test at all, which is why the third seam bug
    survived 86 green ones: it read ``metadata["history_paths"]``, but the loader
    resolves the five JPEGs while constructing the *observation* and records them in
    ``observation.camera_metadata``.  The key is genuinely absent from ``metadata`` --
    asserted below rather than assumed -- so every scene raised ``KeyError``, the
    builder's per-scene ``except`` collected 8/8 failures, and the run still printed
    ``done`` and exited 0.  Only running Stage 0 on real data showed it.

    The round trip is asserted rather than the payload dict, because what has to hold
    is that a *later process* can decode the history from the cached paths.
    """

    from dataclasses import replace

    from nvs_v6_wan_e2e.config import EXPECTED_REGISTER_TARGET_DIM
    from nvs_v6_wan_e2e.multiview_data import load_multiview_sample
    from nvs_v6_wan_e2e.target_cache import (
        HISTORY_FRAMES,
        atomic_torch_save,
        build_scene_payload,
        load_history_uint8,
        read_scene_payload,
        scene_target_path,
        targets_from_payload,
    )

    gt_count = 4
    image_hw = (32, 48)
    scene_dir, log_root, cache_root = _fake_multiview_scene(tmp_path, gt_count=gt_count)
    sample = load_multiview_sample(
        scene_dir,
        log_root=log_root,
        sensor_root=tmp_path / "sensor_blobs",
        cache_root=cache_root,
        image_hw=image_hw,
        load_future_rgb=False,
    )
    # The structural fact the old line contradicted: the loader puts the resolved paths
    # in the observation, not in metadata.  A revert to metadata["history_paths"]
    # therefore cannot be rescued by the fixture accidentally carrying both.
    assert "history_paths" not in sample.metadata
    assert sample.metadata["history_policy"] == sample.observation.camera_metadata["history_policy"]

    register = torch.randn(gt_count, REGISTER_TOKENS, EXPECTED_REGISTER_TARGET_DIM)
    payload = build_scene_payload(
        future_register=register,
        future_chunk_valid=sample.targets.future_chunk_valid[0],
        future_quality=sample.targets.future_quality[0],
        gt_trajectory=sample.targets.gt_trajectory[0],
        gt_valid=sample.targets.gt_valid[0],
        observation=sample.observation,
        pair_ids=list(sample.targets.pair_ids[0]),
        metadata=sample.metadata,
        target_fingerprint="test-fingerprint",
    )
    assert payload["history_paths"] == [str(path) for path in sample.observation.camera_metadata["history_paths"]]
    assert len(payload["history_paths"]) == HISTORY_FRAMES
    assert payload["format"] == PAYLOAD_FORMAT
    assert payload["image_hw"] == list(image_hw)

    target = scene_target_path(tmp_path / "cache", sample.scene_id)
    atomic_torch_save(payload, target)
    # weights_only=True is the point of flattening to tensors and JSON primitives; a
    # payload that needed a real unpickle would fail here rather than in the trainer.
    reloaded = read_scene_payload(target)
    assert reloaded["history_paths"] == payload["history_paths"]
    assert torch.equal(reloaded["future_register"], register)
    assert len(reloaded["pair_ids"]) == gt_count

    targets = targets_from_payload(reloaded, EXPECTED_REGISTER_TARGET_DIM)
    assert tuple(targets.future_register.shape) == (1, gt_count, REGISTER_TOKENS, EXPECTED_REGISTER_TARGET_DIM)
    assert torch.equal(targets.gt_trajectory[0], sample.targets.gt_trajectory[0])
    assert torch.equal(reloaded["history_valid"], sample.observation.history_valid[0, :, 0])

    history = load_history_uint8(reloaded["history_paths"], image_hw)
    assert tuple(history.shape) == (HISTORY_FRAMES, 3, *image_hw)
    assert history.dtype == torch.uint8

    # The guard has to name the loader, because the failure otherwise reads as a
    # corrupt cache entry rather than as an observation that did not come from it.
    bare = replace(sample.observation, camera_metadata={"history_policy": "real_CAM_F0_sensor_blob"})
    with pytest.raises(ValueError, match=r"only load_multiview_sample populates it"):
        build_scene_payload(
            future_register=register,
            future_chunk_valid=sample.targets.future_chunk_valid[0],
            future_quality=sample.targets.future_quality[0],
            gt_trajectory=sample.targets.gt_trajectory[0],
            gt_valid=sample.targets.gt_valid[0],
            observation=bare,
            pair_ids=list(sample.targets.pair_ids[0]),
            metadata=sample.metadata,
            target_fingerprint="test-fingerprint",
        )


def test_a_target_cache_build_that_writes_nothing_reports_failure():
    """The exit-code half of the same bug: total wipeout used to look like success.

    Tolerance has to stay per scene -- one unreadable scene out of 103075 must not
    condemn the build, and a resumed run legitimately builds zero *new* payloads while
    reusing thousands.  What must not be tolerated is a systematic fault, which by
    definition hits every scene: the ``KeyError`` build printed ``done``, wrote a
    manifest beside a ``targets/`` that did not exist, and exited 0.
    """

    import importlib.util

    path = REPO_ROOT / "scripts" / "build_future_target_cache.py"
    spec = importlib.util.spec_from_file_location("build_future_target_cache_under_test", path)
    builder = importlib.util.module_from_spec(spec)
    # Registered before execution, unlike ``_stage1_script``: this module declares
    # ``@dataclass`` under ``from __future__ import annotations``, and resolving those
    # string annotations looks the declaring module up in ``sys.modules``, which is
    # empty for a module built by hand.
    sys.modules[spec.name] = builder
    try:
        spec.loader.exec_module(builder)
    finally:
        sys.modules.pop(spec.name, None)

    def shard(built: int, reused: int, failed: int, scenes: int) -> dict:
        return {"shard_scenes": scenes, "built": built, "reused": reused, "failed_count": failed}

    assert builder.shard_produced_nothing(shard(0, 0, 8, 8), 0) is True  # the observed run
    assert builder.shard_produced_nothing(shard(0, 0, 0, 8), 0) is True  # nothing written, nothing failed
    assert builder.shard_produced_nothing(shard(7, 0, 1, 8), 7) is False  # one bad scene is survivable
    assert builder.shard_produced_nothing(shard(0, 8, 0, 8), 8) is False  # a resume that reused everything
    assert builder.shard_produced_nothing(shard(0, 0, 0, 0), 4) is False  # an empty shard slice, world > scenes
    # Rank 0 alone scans the destination, so only it can apply the global clause; the
    # other ranks pass None and are judged on their own shard.
    assert builder.shard_produced_nothing(shard(0, 0, 0, 0), None) is False
    assert builder.shard_produced_nothing(shard(3, 0, 0, 3), 0) is True  # rank 0 built, but nothing landed


def test_attention_entropy_flags_a_readout_that_has_become_a_mean_pool():
    """The gate that replaced ``compressor_full_rank``, and the arithmetic behind it.

    A diagonal standardizer has no subspace and therefore no rank, so the PCA target's
    failure -- dimensions past the fitted rank bitwise zero, where "emit zeros" beat
    learning -- cannot recur.  The query readout has its own silent failure of the same
    shape: uniform attention makes it *exactly* the mean pool it replaced, every
    downstream number stays finite, and 81% of the GT-discriminating variance is gone.
    """

    uniform = torch.full((3, 4, 8), 1.0 / 8)
    assert torch.allclose(normalized_attention_entropy(uniform), torch.ones(3, 4), atol=1e-6)

    one_hot = torch.zeros(3, 4, 8)
    one_hot[..., 5] = 1.0
    assert torch.allclose(normalized_attention_entropy(one_hot), torch.zeros(3, 4), atol=1e-6)

    # Two equally weighted patches out of eight: log2(2)/log2(8) = 1/3, independent of
    # which two, which is what makes the number a fraction of uniform rather than a
    # raw entropy that grows with the patch count.
    two = torch.zeros(2, 2, 8)
    two[..., 0] = two[..., 3] = 0.5
    assert torch.allclose(normalized_attention_entropy(two), torch.full((2, 2), 1.0 / 3.0), atol=1e-6)
    with pytest.raises(ValueError, match="at least two patches"):
        normalized_attention_entropy(torch.ones(2, 2, 1))

    # Both sides of the ceiling are reachable from one input by moving tau alone, which
    # is what makes it a gate rather than a formality: raising the temperature can only
    # flatten a softmax, and a flat readout *is* the mean pool.  At the production
    # temperature the same patches come out close to a hard selection.
    torch.manual_seed(5)
    dim = 16
    cls_token = torch.randn(1, 1, 1, dim)
    patch_tokens = torch.randn(1, 1, 12, dim)
    temperatures = (0.005, 0.05, 0.5, 5.0, 50.0)
    entropies = [
        float(FutureQueryExtractor(dim, tau, heads=2).attention_entropy(cls_token, patch_tokens).mean())
        for tau in temperatures
    ]
    assert entropies == sorted(entropies), (temperatures, entropies)
    assert entropies[0] < 0.05
    assert entropies[-1] > MAX_QUERY_ATTENTION_ENTROPY
    assert FutureQueryExtractor(dim, QUERY_TEMPERATURE, heads=2).attention_entropy(
        cls_token, patch_tokens
    ).shape == (1, 2)


def test_the_target_standardizer_is_diagonal_and_has_no_rank():
    """Fewer fit rows than dimensions must NOT zero any dimension.

    This is the regression the whole replacement exists for.  The PCA target's rank
    was ``min(target_dim, fit_rows)`` and every dimension past it was bitwise zero, so
    on a 4-scene fit -- rank 128 of 256 -- "emit zeros" scored L1 0.335 against 0.353
    for 20 real training steps, and the loss fell smoothly while nothing was learned.
    A diagonal mean/std has no subspace, so the same fit is merely a noisier estimate.
    """

    torch.manual_seed(3)
    dim = 64
    # 1 GT x 32 register tokens = 32 rows for 64 dimensions: the case that halved the
    # PCA target.  Spelled as one GT rather than four because the register segment
    # quadrupled -- at 4 GT the fit would see 128 rows for 64 dimensions and the
    # assertion below would pass vacuously, without the rank deficiency it exists for.
    targets = torch.randn(1, 1, REGISTER_TOKENS, dim)
    standardizer = TargetStandardizer(dim)
    standardizer.fit(targets, source_hash="scenes-a", split="train")
    out = standardizer.transform(targets).reshape(-1, dim)
    assert out.shape == (32, dim)
    assert not bool((out == 0).all(dim=0).any()), "a dimension is identically zero"
    assert bool((out.std(dim=0) > 0).all())
    assert standardizer.state.clamped_components == 0

    flat = targets.reshape(-1, dim)
    assert torch.allclose(out.mean(dim=0), torch.zeros(dim), atol=1e-5)
    assert torch.allclose(out.std(dim=0), torch.ones(dim), atol=1e-2)
    assert (standardizer.state.fit_split, standardizer.state.source_hash) == ("train", "scenes-a")
    assert standardizer.state.mean.shape == (dim,)
    assert standardizer.state.scale.shape == (dim,)

    # The std floor, and the count of dimensions it rescued.  Full standardization
    # would lift a genuinely degenerate dimension to the same weight as the dominant
    # one, and the L1 loss would then be mostly amplified noise.
    torch.manual_seed(13)
    lopsided = torch.randn(2, 4, REGISTER_TOKENS, 6) * torch.tensor([40.0, 40.0, 40.0, 40.0, 40.0, 1e-3])
    floored = TargetStandardizer(6)
    floored.fit(lopsided, source_hash="scenes-c")
    assert floored.state.clamped_components == 1
    floor = TARGET_STD_FLOOR * float(floored.state.scale.max())
    assert float(floored.state.scale[5]) == pytest.approx(floor, rel=1e-5)
    rescued = floored.transform(lopsided).reshape(-1, 6)
    assert torch.allclose(rescued[:, :5].std(dim=0), torch.ones(5), atol=1e-2)
    # The clamped dimension is deliberately *not* at unit variance.
    assert float(rescued[:, 5].std()) < 0.05

    # The fingerprint is the cache key, so it has to move with everything that changes
    # the target's *values* -- including the query readout's identity, which is not
    # recoverable from mean and scale.  A cache built at another temperature holds
    # different tensors of exactly the same width and would pass every shape check.
    twin = TargetStandardizer(dim)
    twin.fit(targets, source_hash="scenes-a", split="train")
    assert twin.fingerprint() == standardizer.fingerprint()
    other_rows = TargetStandardizer(dim)
    other_rows.fit(torch.randn(1, 4, REGISTER_TOKENS, dim), source_hash="scenes-z")
    assert other_rows.fingerprint() != standardizer.fingerprint()
    for temperature, heads in ((0.01, QUERY_HEADS), (QUERY_TEMPERATURE, 4)):
        variant = TargetStandardizer(dim, temperature, heads)
        variant.fit(targets, source_hash="scenes-a")
        assert variant.fingerprint() != standardizer.fingerprint()

    with pytest.raises(ValueError, match="no target scale exists"):
        TargetStandardizer(4).fit(torch.zeros(2, 4, REGISTER_TOKENS, 4), source_hash="flat")
    with pytest.raises(ValueError, match="train split"):
        TargetStandardizer(dim).fit(targets, source_hash="scenes-a", split="eval")
    with pytest.raises(ValueError, match="standardizer expects"):
        TargetStandardizer(dim).fit(torch.zeros(4, 8, dim + 1), source_hash="scenes-a")
    with pytest.raises(ValueError, match="feature dimension"):
        TargetStandardizer(dim).fit(torch.zeros(8), source_hash="scenes-a")

    unfit = TargetStandardizer(dim)
    with pytest.raises(RuntimeError, match="not fit"):
        unfit.fingerprint()
    with pytest.raises(RuntimeError, match="before transform"):
        unfit.transform(targets)
    with pytest.raises(ValueError, match="does not match the fitted"):
        standardizer.transform(torch.zeros(1, 4, REGISTER_TOKENS, dim + 1))


def test_the_persisted_standardizer_round_trips_and_rejects_a_pre_query_payload(tmp_path):
    """Every rank loads one standardizer; none of them may fit its own.

    A diagonal mean/std has no basis ambiguity -- unlike the SVD projector it replaced,
    two fits on different subsets differ by estimation noise rather than by an
    arbitrary column order and sign -- but they still differ, and the difference is a
    per-dimension rescaling of every cached target.  Rank 3's targets would then not be
    comparable to rank 7's and the all-reduced gradient would average them anyway.
    """

    assert STANDARDIZER_NAME == "standardizer.pt"
    torch.manual_seed(19)
    targets = torch.randn(3, 4, REGISTER_TOKENS, 12) * torch.tensor(
        [9.0, 3.0, 1.0, 0.3, 0.1, 0.03, 9.0, 3.0, 1.0, 0.3, 0.1, 0.03]
    )
    standardizer = TargetStandardizer(12, QUERY_TEMPERATURE, QUERY_HEADS)
    standardizer.fit(targets, source_hash="scenes-d", split="train")
    audit = {"teacher_image_hw": [464, 832], "standardizer_fit_rows": 96}
    save_standardizer(tmp_path, standardizer, audit)
    assert standardizer_path(tmp_path).exists()

    restored, loaded_audit = load_standardizer(tmp_path)
    assert loaded_audit == audit
    assert (restored.feature_dim, restored.temperature, restored.heads) == (12, QUERY_TEMPERATURE, QUERY_HEADS)
    assert restored.state.source_hash == "scenes-d"
    assert restored.state.clamped_components == standardizer.state.clamped_components
    assert restored.fingerprint() == standardizer.fingerprint()
    assert torch.equal(restored.transform(targets), standardizer.transform(targets))

    with pytest.raises(RuntimeError, match="unfit"):
        save_standardizer(tmp_path / "never", TargetStandardizer(12), {})
    with pytest.raises(FileNotFoundError, match="build_future_target_cache"):
        load_standardizer(tmp_path / "missing")

    # The retired payload is refused by name rather than loaded with defaults: it held
    # a projection matrix and a whitening scale, and interpreting those as a diagonal
    # mean/std would produce finite, plausible, wrong targets.
    stale = tmp_path / "stale"
    stale.mkdir()
    payload = torch.load(standardizer_path(tmp_path), map_location="cpu", weights_only=True)
    del payload["feature_dim"]
    atomic_torch_save(payload, standardizer_path(stale))
    with pytest.raises(ValueError, match="predates the query-extracted target"):
        load_standardizer(stale)

    # A fingerprint that does not match the tensors it was recorded over is a corrupted
    # or hand-edited cache, and loading it would silently train against the wrong target.
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    payload = torch.load(standardizer_path(tmp_path), map_location="cpu", weights_only=True)
    payload["scale"] = payload["scale"] * 2.0
    atomic_torch_save(payload, standardizer_path(tampered))
    with pytest.raises(ValueError, match="fingerprint"):
        load_standardizer(tampered)


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
    shape = detrend_trajectory(source) / dit.shape_range
    expected_action = (
        dit.action_encoder(torch.cat([norm_odo(source), shape], dim=-1))
        + expected_ego.view(2, 1, 1, D_MODEL)
        + dit.time_queries.view(1, 1, 8, D_MODEL)
        + expected_mode.unsqueeze(2)
    )
    assert torch.allclose(action, expected_action, atol=1e-6)
    # The metric action must be normalized before the encoder: metres and radians
    # have to contribute comparably, and the sum must not be RMS-normalized after.
    assert not torch.allclose(
        dit.action_encoder(torch.cat([source, shape], dim=-1)),
        dit.action_encoder(torch.cat([norm_odo(source), shape], dim=-1)),
        atol=1e-3,
    )
    with pytest.raises(ValueError):
        dit.encode_action(torch.randn(2, 3, 8, 4), modes, clean_ego)
    with pytest.raises(ValueError):
        dit.encode_action(source, modes, torch.randn(2, D_MODEL + 1))


def test_the_detrended_action_channel_is_per_sample_and_carries_the_manoeuvre():
    """The shape channel may not read its siblings, and must not restate the trend.

    Multi-GT is K independent ``(scene, trajectory) -> future`` samples.  Stage 1 feeds
    K ground truths and Stage 2 feeds 64 sampled candidates through the *same*
    ``encode_action``, so a shape channel normalised against the other rows of its own
    scene would make one trajectory mean something different in the two stages and a
    ``--stage1-checkpoint`` restore would silently change the model's behaviour.
    """

    assert SHAPE_RANGE == (1.31, 0.414, 0.0811), (
        "recalibrate against real GT trajectories, do not eyeball it: these three divisors "
        "are what put each detrended channel at norm_odo's overall RMS of 0.413"
    )
    source = torch.randn(2, 6, 8, 3) * 8.0

    # Per-sample: bitwise identical alone and beside five other candidates.  A tolerance
    # here would hide exactly the batch coupling this test exists to forbid.
    assert torch.equal(detrend_trajectory(source[:, :1]), detrend_trajectory(source)[:, :1])

    # A constant-velocity trajectory is pure trend, so it has no shape at all.
    t = torch.linspace(-1.0, 1.0, 8).view(8, 1)
    straight = torch.tensor([4.0, -1.0, 0.2]) + torch.tensor([9.0, 3.0, 0.5]) * t
    assert detrend_trajectory(straight).abs().max() < 1e-5

    # The residual is orthogonal to both basis vectors, so the channel adds information
    # rather than restating the norm_odo half it is concatenated with.
    residual = detrend_trajectory(source)
    assert residual.sum(dim=-2).abs().max() < 1e-4
    assert (residual * t).sum(dim=-2).abs().max() < 1e-4
    with pytest.raises(ValueError):
        detrend_trajectory(torch.randn(2, 1, 3))

    # Concatenating the two channels doubles the encoder's input width and nothing else:
    # the head still emits metric (x, y, heading), so denorm_odo stays the exact inverse
    # and the flow's coordinate system is untouched.
    dit = NvsActionDiT(hidden_dim=D_MODEL, text_dim=D_MODEL, action_dim=3, num_heads=4,
                       attn_head_dim=64, target_dim=8)
    assert dit.action_encoder.in_features == 6
    assert dit.action_encoder.out_features == D_MODEL
    assert [mod.out_features for mod in dit.head.modules() if isinstance(mod, nn.Linear)][-1] == 3
    assert tuple(dit.shape_range.shape) == (3,)
    from nvs_v6_wan_e2e.flow import denorm_odo

    metric = torch.randn(4, 8, 3) * 12.0
    assert torch.allclose(denorm_odo(norm_odo(metric)), metric, atol=1e-5)


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
    # vendor/ trees are gitignored except the force-added DrivoR sources, so SimWAM
    # is present on the training machine but not on a fresh clone.  This test diffs
    # our port against the upstream scheduler rather than against hard-coded
    # numbers, which is only meaningful when that source is actually there.
    scheduler_continuous = pytest.importorskip(
        "simwam.models.wan22.schedulers.scheduler_continuous",
        reason="vendor/SimWAM is not checked out; cannot diff against the upstream scheduler",
    )
    SimwamScheduler = scheduler_continuous.WanContinuousFlowMatchScheduler

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
        "future_queries",
        "future_head.",
        "trajectory_gates.",
        "scene_type_embedding",
        "action_type_embedding",
        "register_type_embedding",
    )
    keys = {
        "blocks.0.self_attn.q.weight",
        "action_encoder.weight",
        "head.proj.weight",
        "time_queries",
        "future_queries",
        "future_head.1.weight",
        "trajectory_gates.0.q.weight",
        "scene_type_embedding",
        "action_type_embedding",
        "register_type_embedding",
    }
    assert NvsActionDiT.backbone_key_set(keys) == {"blocks.0.self_attn.q.weight"}


def test_trajectory_gate_is_zero_at_init_one_way_and_reaches_the_register():
    """The gate must add trajectory signal without ever adding a leak.

    Measured on the production geometry, register rows give the eight action keys only
    8-10% of their self-attention mass while the register's own input is
    trajectory-independent, so the trajectory reaches the future readout through a
    ~9%-per-hop channel -- and a contrast loss that supervises the across-GT difference
    exactly still could not move ``residual_relative_l1`` off 1.0.  The gate is the
    structural fix, and it has three properties that must hold simultaneously: it
    contributes exactly zero at step 0 (so nothing pretrained is displaced), it writes
    only the register slice (so ``Action never reads Register`` survives), and once
    trained it is a real route from the action tokens into the register.
    """
    torch.manual_seed(4)
    dit = NvsActionDiT(hidden_dim=256, action_tokens=8, future_tokens=32, target_dim=1024,
                       num_heads=4, attn_head_dim=64, num_layers=8,
                       text_dim=256, freq_dim=256, ego_dim=256).eval()

    # (a) live at init: ``o`` keeps nn.Linear's default strength (a zero or 1e-2 gate
    #     measurably wasted the whole warmup on the overfit probe) and the bias is zero.
    for gate in dit.trajectory_gates:
        assert float(gate.o.weight.std()) > 1e-3
        assert bool((gate.o.bias == 0).all())
    register = torch.randn(2, 32, 256)
    action = torch.randn(2, 8, 256)
    with torch.no_grad():
        gate_out = dit.trajectory_gates[0](register, action)
    assert gate_out.abs().max() > 0
    # ...and same order as the register's own scale: a route, not a replacement.
    assert float(gate_out.abs().mean()) < 3.0 * float(register.abs().mean())

    # (b) one-way: perturbing the register input leaves the velocity bitwise identical
    #     even with the gates in the graph.
    ctx = torch.randn(8, 81, 256)
    cond = torch.randn(8, 256)
    act = torch.randn(8, 8, 256)
    fut = torch.randn(8, 32, 256)
    with torch.no_grad():
        v0, r0 = dit(act, fut, cond, ctx)
        v1, r1 = dit(act, fut + 50.0 * torch.randn_like(fut), cond, ctx)
    assert torch.equal(v0, v1)
    assert not torch.equal(r0, r1)

    # (c) a real route: with the output projection live, the register moves when the
    #     action changes, and the movement is exactly the gate's contribution on top of
    #     the (already trajectory-reading) self-attention path.
    for gate in dit.trajectory_gates:
        nn.init.normal_(gate.o.weight, std=1e-2)
    with torch.no_grad():
        r_gate_on = dit(act, fut, cond, ctx)[1]
        r_action_perturbed = dit(act + torch.randn_like(act), fut, cond, ctx)[1]
    assert (r_action_perturbed - r_gate_on).abs().max() > 0

    # (d) frame-aligned read: each register row has exactly one allowed action key, so the
    #     softmax weight is 1 by construction and the read is independent of the query --
    #     that is the alignment doing its job, not a bug.  What must change the output is
    #     the aligned action row, and it must change ONLY that frame's register rows.
    gate = dit.trajectory_gates[0]
    assert bool((gate.frame_mask.sum(dim=-1) == 1).all())
    with torch.no_grad():
        out_a = gate(register, action)
        out_query_shifted = gate(register + 7.0, action)
    assert torch.equal(out_a, out_query_shifted)
    with torch.no_grad():
        out_value_shifted = gate(register, action + 1.0)
    assert not torch.equal(out_a, out_value_shifted)
    action_moved = action.clone()
    action_moved[:, 3] += 5.0
    with torch.no_grad():
        out_frame_moved = gate(register, action_moved)
    delta = (out_frame_moved - out_a).abs().mean(dim=-1)[0]
    assert float(delta[12:16].max()) > 1e-4
    assert float(torch.cat([delta[:12], delta[16:]]).max()) < 1e-6


def test_future_queries_and_head_survive_the_wan_load_at_their_own_init():
    """Wan's payload has no future branch, so the load must not touch it.

    ``future_queries`` is the only thing that breaks the eight tokens' symmetry, and
    ``future_head`` is the whole Stage 1 read-out.  If either were overwritten -- or
    if a missing key turned the strict load into an error -- Stage 1 would start from
    a state that no longer matches the checkpoint the config claims was loaded.
    """

    if not WAN_PAYLOAD.is_file():
        pytest.skip("Action DiT Wan payload has not been generated")
    torch.manual_seed(21)
    dit = NvsActionDiT()
    queries_before = dit.future_queries.detach().clone()
    head_before = dit.future_head[1].weight.detach().clone()
    info = load_action_dit_backbone(dit, WAN_PAYLOAD)
    assert info["loaded_keys"] == 226
    assert torch.equal(dit.future_queries, queries_before)
    assert torch.equal(dit.future_head[1].weight, head_before)
    assert float(dit.future_queries.std()) > 0.0


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
    # closed by resampling the width up to the 1152 the backbone is built for, the
    # rotary position encoding then being built analytically for that grid.  Zero
    # padding is gone: it would fabricate patch content.
    assert DRIVOR_OFFICIAL_IMAGE_HW == (672, 1148)
    assert DRIVOR_IMAGE_HW == (672, 1152)
    assert all(side % 16 == 0 for side in DRIVOR_IMAGE_HW)
    assert DRIVOR_IMAGE_HW[1] - DRIVOR_OFFICIAL_IMAGE_HW[1] == DRIVOR_WIDTH_RESAMPLE_PX == 4
    resampled = resample_to_patch_multiple(torch.zeros(1, 3, *DRIVOR_OFFICIAL_IMAGE_HW), DRIVOR_IMAGE_HW)
    assert tuple(resampled.shape) == (1, 3, *DRIVOR_IMAGE_HW)
    assert resample_to_patch_multiple(torch.zeros(1, 3, *DRIVOR_IMAGE_HW), DRIVOR_IMAGE_HW).shape == resampled.shape
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
    # The Student resamples the patch16 remainder internally.  Feeding it the
    # already-resampled width would mean a loader stretched the frame, so that
    # shape is refused rather than resampled a second time.
    with pytest.raises(ValueError, match="only resamples the patch16 remainder"):
        student(torch.rand(1, 2, 3, *DRIVOR_IMAGE_HW))
    assert layout["right_zero_pad_px"] == 0 and layout["bottom_zero_pad_px"] == 0
    assert layout["right_resample_px"] == 4
    assert layout["student_resampling"] == "bicubic_to_patch_multiple"


def test_real_dinov3_grad_checkpointing_matches_the_eager_backward():
    if not DINOV3_CHECKPOINT.is_file():
        pytest.skip("configured real DINOv3 checkpoint is unavailable")
    if not torch.cuda.is_available():
        pytest.skip("checkpointed DINOv3 backward requires CUDA")
    from nvs_v6_wan_e2e.vision import DrivoRDinoV3Student

    results = {}
    peaks = {}
    for checkpointed in (False, True):
        torch.manual_seed(0)
        student = DrivoRDinoV3Student(
            checkpoint=str(DINOV3_CHECKPOINT), num_frames=2, lora_rank=4, grad_checkpointing=checkpointed
        ).cuda()
        assert student.grad_checkpointing is checkpointed
        assert student.token_layout()["grad_checkpointing"] is checkpointed
        torch.manual_seed(1234)
        images = torch.rand(1, 2, 3, *DRIVOR_OFFICIAL_IMAGE_HW, device="cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        resident = torch.cuda.memory_allocated()
        tokens = student(images)
        tokens.square().mean().backward()
        peaks[checkpointed] = torch.cuda.max_memory_allocated() - resident
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
    # Every assertion above is vacuous on its own: if the flag is inert both branches
    # run byte-identical code and agree trivially.  That is exactly how it shipped --
    # ``_forward_features`` is a hand-written RoPE loop that never calls timm's
    # ``forward_features``, so ``set_grad_checkpointing`` set a flag nothing read and
    # measured 4478 ms / 29.69 GiB either way.  A recomputing backward keeps one
    # block's activations instead of all of them, so the peak has to fall; this is
    # the one assertion here that can fail.
    assert peaks[True] < peaks[False] * 0.9, (
        f"grad checkpointing did not reduce peak activation memory "
        f"({peaks[True] / 2**20:.0f} MiB checkpointed vs {peaks[False] / 2**20:.0f} MiB eager); "
        "the flag is inert"
    )


# --------------------------------------------------------------------------- #
# Stage freezing and optimizer coverage
# --------------------------------------------------------------------------- #


@requires_drivor
def test_stage_freeze_and_optimizer_coverage():
    model = make_model()
    scorer = make_scorer()
    head_ids = {id(parameter) for name, parameter in model.theta.named_parameters() if name.startswith("head.")}
    trunk_ids = {id(parameter) for name, parameter in model.theta.named_parameters() if not name.startswith("head.")}
    assert head_ids and trunk_ids

    stage1_generator, stage1_scorer = build_optimizers(model, scorer, stage=1)
    stage1_ids = {id(parameter) for group in stage1_generator.param_groups for parameter in group["params"]}
    # Stage 1 trains the DiT trunk, because the future queries now live inside it.
    # Only the velocity head is frozen: nothing in the Stage 1 loss reads it, and
    # its near-zero anchor-preserving initialization must survive to Stage 2.
    assert trunk_ids <= stage1_ids
    assert not (head_ids & stage1_ids)
    assert model.theta.future_queries.requires_grad
    assert all(id(parameter) in stage1_ids for parameter in model.theta.future_head.parameters())
    assert all(not parameter.requires_grad for parameter in model.theta.head.parameters())
    assert all(not parameter.requires_grad for parameter in scorer.parameters())
    assert stage1_ids == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    stage1_scorer_ids = [id(parameter) for group in stage1_scorer.param_groups for parameter in group["params"]]
    assert len(stage1_scorer_ids) == len(set(stage1_scorer_ids))
    assert set(stage1_scorer_ids) == {id(parameter) for parameter in scorer.parameters()}
    assert not (stage1_ids & set(stage1_scorer_ids))

    stage2_generator, stage2_scorer = build_optimizers(model, scorer, stage=2)
    stage2_ids = [id(parameter) for group in stage2_generator.param_groups for parameter in group["params"]]
    assert (head_ids | trunk_ids) <= set(stage2_ids)
    assert all(parameter.requires_grad for parameter in model.theta.parameters())
    assert all(parameter.requires_grad for parameter in scorer.parameters())
    assert len(stage2_ids) == len(set(stage2_ids))
    assert set(stage2_ids) == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert set(id(parameter) for group in stage2_scorer.param_groups for parameter in group["params"]) == {
        id(parameter) for parameter in scorer.parameters()
    }


@requires_drivor
def test_stage1_trains_the_future_readout_but_never_the_dinov3_base_or_the_velocity_head():
    model = make_model()
    scorer = make_scorer()
    optimizer, _ = build_optimizers(model, scorer, stage=1)
    targets = make_targets(gt=2)
    head_before = model.theta.head.proj.weight.detach().clone()
    result = stage1_step(make_observation(), targets, model, optimizer)
    assert torch.isfinite(result.loss_future)
    # Identity pairing supervises every valid GT row, not only Hungarian matches.
    assert result.supervised_rows == int(targets.gt_valid.sum()) == 2
    # The whole point of Stage 1: the future queries and their read-out head moved.
    assert model.theta.future_queries.grad is not None
    assert model.theta.future_queries.grad.abs().sum() > 0
    assert model.theta.future_head[1].weight.grad is not None
    # The velocity head is frozen, so the anchor-preserving initialization survives
    # and Stage 2 still starts at the anchor grid's own oracle-best PDMS.
    assert model.theta.head.proj.weight.grad is None
    assert torch.equal(model.theta.head.proj.weight.detach(), head_before)
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
    assert model.theta.blocks[0].training is True


# --------------------------------------------------------------------------- #
# Losses, oracle labels and selection
# --------------------------------------------------------------------------- #


def test_hungarian_matches_by_distance_not_by_row_order():
    predicted = torch.zeros(1, 4, 8, TRAJECTORY_STATE_DIM)
    predicted[:, 0, :, 0] = 1.0
    predicted[:, 1, :, 0] = 10.0
    ground_truth = torch.zeros(1, 2, 8, TRAJECTORY_STATE_DIM)
    ground_truth[:, 0, :, 0] = 10.0
    ground_truth[:, 1, :, 0] = 1.0
    match = hungarian_xy(predicted, ground_truth, torch.ones(1, 2, dtype=torch.bool))[0]
    assert match.as_pairs() == [(0, 1), (1, 0)]


def test_empty_gt_yields_an_empty_but_finite_assignment():
    # Padded to four GT columns with none of them valid: the realistic "no usable
    # GT" batch.  The collate always pads the GT axis to at least one column, so an
    # all-invalid mask is what an empty assignment actually looks like.
    targets = make_targets(gt=4)
    targets.gt_valid[:] = False
    endpoint = torch.randn(1, 8, 8, TRAJECTORY_STATE_DIM, requires_grad=True)
    source = torch.randn(1, 8, 8, TRAJECTORY_STATE_DIM)
    match = hungarian_xy(torch.randn(1, 8, 8, TRAJECTORY_STATE_DIM), targets.gt_trajectory, targets.gt_valid)[0]
    assert match.as_pairs() == []
    action_loss = paired_residual_velocity_mse(endpoint, source, targets.gt_trajectory, [match])
    register = torch.randn(1, 8, REGISTER_TOKENS, REGISTER_TARGET_DIM, requires_grad=True)
    future_loss = dense_future_l1(register, targets, source)
    assert torch.isfinite(action_loss) and torch.isfinite(future_loss)
    (action_loss + future_loss).backward()
    assert endpoint.grad is not None
    assert register.grad is not None and torch.isfinite(register.grad).all()


def test_paired_residual_velocity_loss_is_a_metric_delta_mse():
    source = torch.zeros(1, 1, 8, TRAJECTORY_STATE_DIM)
    endpoint = torch.full((1, 1, 8, TRAJECTORY_STATE_DIM), 2.0)
    ground_truth = torch.full((1, 1, 8, TRAJECTORY_STATE_DIM), 5.0)
    match = hungarian_xy(endpoint, ground_truth, torch.ones(1, 1, dtype=torch.bool))
    loss = paired_residual_velocity_mse(endpoint, source, ground_truth, match)
    assert torch.allclose(loss, torch.tensor(9.0))


def test_dense_future_loss_supervises_every_candidate_from_its_nearest_valid_gt():
    """Stage 2's future pairing is dense, not Hungarian.

    A match-only pairing could reach just the K candidates it happened to pick, so
    60+ of the 64 register rows got zero future gradient even though Stage 1 had
    trained them -- and the future queries share the DiT trunk with the action
    tokens, so an untrained row is an untrained trajectory.  Each candidate is
    paired with its nearest *valid* GT instead, and the per-frame mask must be
    gathered with the same index so a candidate cannot inherit another GT's validity.
    """

    gt_trajectory = torch.zeros(2, 4, 8, TRAJECTORY_STATE_DIM)
    for index in range(4):
        gt_trajectory[:, index, :, 0] = 10.0 * index
    valid = torch.ones(2, 4, dtype=torch.bool)
    valid[1] = False
    future_register = (
        torch.arange(1, 5, dtype=torch.float32).view(1, 4, 1, 1).expand(2, 4, REGISTER_TOKENS, REGISTER_TARGET_DIM).clone()
    )
    chunk_valid = torch.ones(2, 4, REGISTER_TOKENS, dtype=torch.bool)
    chunk_valid[0, 0, 1:] = False
    targets = PairTargets(
        gt_trajectory=gt_trajectory,
        gt_valid=valid,
        future_register=future_register,
        future_chunk_valid=chunk_valid,
        future_quality=torch.ones(2, 4, REGISTER_TOKENS),
        pair_ids=[["a", "b", "c", "d"]] * 2,
    )
    proposals = torch.zeros(2, 64, 8, TRAJECTORY_STATE_DIM)
    for candidate in range(64):
        proposals[:, candidate, :, 0] = 10.0 * (candidate % 4)
    prediction = torch.zeros(2, 64, REGISTER_TOKENS, REGISTER_TARGET_DIM, requires_grad=True)
    dense_future_l1(prediction, targets, proposals).backward()

    supervised = prediction.grad.abs().sum(dim=(-1, -2)).ne(0)
    assert supervised[0].all()
    # A scene with no valid GT at all stays unsupervised rather than inheriting one.
    assert not supervised[1].any()
    frames = prediction.grad[0].abs().sum(dim=-1).ne(0)
    for candidate in range(64):
        expected = [candidate % 4 != 0 or frame == 0 for frame in range(REGISTER_TOKENS)]
        assert frames[candidate].tolist() == expected, candidate


def test_identity_future_loss_supervises_every_valid_gt_row():
    """Stage-1 pairing is given by the dataset, so no matching decides it.

    Row ``k`` is read out of the DiT that encoded ``gt_trajectory[k]`` and is
    supervised by ``future_register[k]``; padded rows carry no gradient.  Distinct
    per-row targets make a silent row shift detectable, which Hungarian matching
    would have hidden.
    """

    valid = torch.tensor([[True, True, True, False]])
    # Every row's target is non-zero, so a zero prediction receives gradient on all
    # three valid rows rather than silently passing row 0 by matching a zero target.
    future_register = (
        torch.arange(1, 5, dtype=torch.float32).view(1, 4, 1, 1).expand(1, 4, REGISTER_TOKENS, REGISTER_TARGET_DIM).clone()
    )
    targets = PairTargets(
        gt_trajectory=torch.zeros(1, 4, 8, TRAJECTORY_STATE_DIM),
        gt_valid=valid,
        future_register=future_register,
        future_chunk_valid=valid.unsqueeze(-1).expand(-1, -1, REGISTER_TOKENS).clone(),
        future_quality=torch.ones(1, 4, REGISTER_TOKENS),
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
        identity_future_l1(torch.zeros(1, 64, REGISTER_TOKENS, REGISTER_TARGET_DIM), targets)
    # A row-count mismatch is what breaks the pairing, but a width mismatch is the
    # likelier one in practice -- ``future_head`` built at ``d_model`` while the cache's
    # standardizer reports the Teacher's own feature width.  Both have to be refused
    # here: unchecked, they reach the elementwise subtract in
    # ``_quality_weighted_chunk_l1`` and surface as a broadcast ``RuntimeError`` naming
    # neither tensor nor axis, which reads like a data bug rather than a config bug.
    with pytest.raises(ValueError, match=r"\[B, K, register_tokens, register_target_dim\]"):
        identity_future_l1(torch.zeros(1, 4, REGISTER_TOKENS, REGISTER_TARGET_DIM // 2), targets)
    with pytest.raises(ValueError, match=r"\[B, K, register_tokens, register_target_dim\]"):
        identity_future_l1(torch.zeros(1, 4, REGISTER_TOKENS + 1, REGISTER_TARGET_DIM), targets)


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


def _stage1_cache_ddp_script():
    """Load ``scripts/run_stage1_cache_ddp.py``, the 16-rank Stage-1 trainer.

    Registered in ``sys.modules`` before execution for the same reason as the
    Stage-0 builder: it declares ``@dataclass`` under ``from __future__ import
    annotations``, and resolving those string annotations looks the declaring
    module up there.
    """

    import importlib.util

    path = REPO_ROOT / "scripts" / "run_stage1_cache_ddp.py"
    spec = importlib.util.spec_from_file_location("run_stage1_cache_ddp_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _synthetic_stage1_cache(
    tmp_path: Path,
    tokens: list[str],
    fingerprint: str = "test-fingerprint",
    eval_scenes: int = 2,
    seed: int = 17,
    image_hw: tuple[int, int] = (32, 48),
    gt_count: int = 2,
) -> Path:
    """A Stage-0 cache root that the streaming dataset and the split logic accept.

    One payload is produced by the real write path (``load_multiview_sample`` ->
    ``build_scene_payload``) and the rest are clones of it with a distinct
    ``scene_id`` and ``future_register``, so every entry carries the full field set
    the builder writes rather than a hand-rolled subset that could pass here while a
    real cache fails.  Cloning keeps this CPU-only and fast; the video decode and the
    Teacher have their own tests.
    """

    from nvs_v6_wan_e2e.multiview_data import load_multiview_sample
    from nvs_v6_wan_e2e.target_cache import (
        atomic_json_write,
        atomic_torch_save,
        build_scene_payload,
        manifest_path,
        ready_scenes_path,
        scene_target_path,
    )

    root = tmp_path / "cache"
    scene_dir, log_root, cache_root = _fake_multiview_scene(tmp_path / "corpus", token=tokens[0], gt_count=gt_count)
    sample = load_multiview_sample(
        scene_dir,
        log_root=log_root,
        sensor_root=tmp_path / "corpus" / "sensor_blobs",
        cache_root=cache_root,
        image_hw=image_hw,
        load_future_rgb=False,
    )
    generator = torch.Generator().manual_seed(0)
    base = build_scene_payload(
        future_register=torch.randn(gt_count, REGISTER_TOKENS, REGISTER_TARGET_DIM, generator=generator),
        future_chunk_valid=sample.targets.future_chunk_valid[0],
        future_quality=sample.targets.future_quality[0],
        gt_trajectory=sample.targets.gt_trajectory[0],
        gt_valid=sample.targets.gt_valid[0],
        observation=sample.observation,
        pair_ids=list(sample.targets.pair_ids[0]),
        metadata=sample.metadata,
        target_fingerprint=fingerprint,
    )
    for index, token in enumerate(tokens):
        payload = dict(base)
        payload["scene_id"] = token
        payload["future_register"] = torch.randn(
            gt_count, REGISTER_TOKENS, REGISTER_TARGET_DIM, generator=generator
        )
        atomic_torch_save(payload, scene_target_path(root, token))
    atomic_json_write(
        {
            "format": PAYLOAD_FORMAT,
            "image_hw": [int(value) for value in image_hw],
            "eval_scenes": eval_scenes,
            "seed": seed,
            "register_target_dim": REGISTER_TARGET_DIM,
            "target_fingerprint": fingerprint,
            "cached_scene_count": len(tokens),
        },
        manifest_path(root),
    )
    atomic_json_write({"_meta": {"limit": len(tokens)}, "tokens": list(tokens)}, ready_scenes_path(root))
    return root


def test_both_stage1_trainers_share_one_copy_of_the_verdict_maths():
    """Neither trainer may define its own metric or verdict.

    ``run_stage1_multiview.py`` decodes ``future.mp4`` inline for a handful of
    resident scenes; ``run_stage1_cache_ddp.py`` streams the Stage-0 cache across 16
    ranks.  A number printed by one has to mean the same thing in the other, and the
    part that cannot drift is the verdict rule -- it is what decides whether Stage 1
    learned anything.  So both import from ``stage1_eval`` and neither keeps a
    private copy: a second definition would silently become the one that runs.
    """

    from nvs_v6_wan_e2e import stage1_eval

    script = _stage1_script()
    ddp = _stage1_cache_ddp_script()
    shared = ("future_fit_metrics", "register_verdict", "trajectory_errors", "CONDITIONING_SUMMARY_KEYS")
    for name in shared:
        assert getattr(script, name) is getattr(stage1_eval, name), f"{name} is not stage1_eval's in the smoke trainer"
        assert getattr(ddp, name) is getattr(stage1_eval, name), f"{name} is not stage1_eval's in the DDP trainer"
    for module in (script, ddp):
        for private in ("_future_fit_metrics", "_register_verdict", "_trajectory_spread", "_trajectory_errors"):
            assert not hasattr(module, private), f"{module.__name__} kept a private copy of {private}"
    # The streaming per-step readout is only imported by the DDP trainer -- the smoke
    # trainer's step log predates it and prints no cosine -- but it must still be
    # stage1_eval's copy, since it reports the verdict's two numbers between passes.
    assert ddp.streaming_future_alignment is stage1_eval.streaming_future_alignment


def test_subset_index_below_default_eval_count_is_clamped_not_fatal():
    """``--limit`` subset builds must split, not die on the eval count.

    The builder advertises ``--limit`` as "useful for a first subset build", but the split
    refuses an eval count that leaves no training scene, so with the default
    ``--eval-scenes 256`` any limit at or below 256 raised ``ValueError`` before a single
    payload was written -- caught by the single-GPU pipeline verify at ``--limit 20``.
    The builder now clamps to ``len(tokens) - 1`` (a no-op on the full corpus, so the
    production split is untouched) and records the clamped value in the manifest, which is
    what the trainer re-derives its split from.  This pins both halves: the unclamped call
    is fatal, the clamped one splits cleanly.
    """
    from nvs_v6_wan_e2e.target_cache import split_scene_tokens

    tokens = [f"{index:016x}" for index in range(20)]
    with pytest.raises(ValueError, match="leaves no training scene"):
        split_scene_tokens(tokens, 256, 17)
    clamped = min(256, max(1, len(tokens) // 8))
    train, evaluation = split_scene_tokens(tokens, clamped, 17)
    # The cap must leave a train split large enough for a batched loader: clamping to
    # len-1 instead would give 19 eval / 1 train, and Stage 1's drop_last DataLoader
    # then dies with "rank 0 got no batch" -- the exact regression this pins.
    assert len(train) >= 8 and evaluation
    assert set(train).isdisjoint(evaluation)
    assert len(train) + len(evaluation) == len(tokens)
    # The clamp must be inert on a full-size index, or the production split would move.
    assert min(256, max(1, 103075 // 8)) == 256


def test_the_ddp_trainer_reproduces_the_builders_train_eval_split(tmp_path):
    """The trainer must inherit the split the standardizer was fit under.

    ``run_stage1_cache_ddp.py`` cannot take ``--eval-scenes``/``--seed`` and recompute
    the split, however natural that looks: the builder fit ``mean``/``std`` on *its*
    train scenes, so a scene this run labels "eval" may have contributed to the
    scaling its own targets were standardized by.  The held-out numbers would then be
    leak-free in name only, and nothing downstream would look wrong.  So the split is
    re-derived from the same ``_ready_scenes.json`` sidecar with the manifest's own
    ``eval_scenes``/``seed``, and the leak claim is asserted here against the
    persisted ``fit_scene_ids`` rather than left as a docstring.

    ``args.seed`` is deliberately *not* the manifest's: a trainer that reached for the
    CLI seed would produce a different split and this test would catch it.
    """

    import argparse

    from nvs_v6_wan_e2e.target_cache import (
        load_standardizer,
        ready_scenes_path,
        save_standardizer,
        split_scene_tokens,
    )
    from nvs_v6_wan_e2e.teacher import TargetStandardizer

    script = _stage1_cache_ddp_script()
    tokens = [f"{index:016x}" for index in range(8)]
    root = _synthetic_stage1_cache(tmp_path, tokens, eval_scenes=2, seed=17)

    builder_train, builder_eval = split_scene_tokens(tokens, 2, 17)
    fit_ids = builder_train[:3]
    standardizer = TargetStandardizer(feature_dim=REGISTER_TARGET_DIM)
    standardizer.fit(torch.randn(len(fit_ids), 2, REGISTER_TOKENS, REGISTER_TARGET_DIM), "+".join(fit_ids))
    save_standardizer(root, standardizer, {"fit_scene_ids": fit_ids})
    _, audit = load_standardizer(root)

    args = argparse.Namespace(
        target_cache_root=root, image_hw=None, skip_degenerate=False, seed=999,
        min_gt_spread_m=MIN_GT_SPREAD_M,
    )
    dctx = script.Dist(rank=0, world=1, local_rank=0, device=torch.device("cpu"), enabled=False)
    resolved = script.resolve_scene_sets(dctx, args, "test-fingerprint")

    assert resolved["provenance"]["source"] == "ready_scenes_sidecar"
    assert sorted(resolved["train"]) == sorted(builder_train)
    assert sorted(resolved["eval"]) == sorted(builder_eval)
    assert not set(resolved["train"]) & set(resolved["eval"])
    assert set(resolved["train"]) | set(resolved["eval"]) == set(tokens)
    # The claim the docstring makes, executable: nothing the standardizer saw is
    # reported as held out.
    assert not set(audit["fit_scene_ids"]) & set(resolved["eval"])
    # image_hw comes from the manifest, because a payload cached at another size is
    # rejected by Stage1SceneDataset and the trainer should not invent one.
    assert tuple(resolved["image_hw"]) == (32, 48)

    # Without the sidecar the builder's ordering is unrecoverable, so the fallback has
    # to say that it can no longer claim leak-freedom instead of silently guessing.
    ready_scenes_path(root).unlink()
    fallback = script.resolve_scene_sets(dctx, args, "test-fingerprint")
    assert fallback["provenance"]["source"] == "cached_tokens_fallback"
    assert "standardizer fit" in fallback["provenance"]["warning"]
    assert not set(fallback["train"]) & set(fallback["eval"])


def test_the_ddp_trainer_filters_by_the_standardizers_fingerprint_not_the_manifests(tmp_path):
    """Which fingerprint decides whether a cached payload may be used.

    The manifest is written *after* the shard finishes, so a build killed in between
    leaves a manifest whose ``target_fingerprint`` no longer describes the payloads
    sitting next to it.  Filtering by the manifest would then either reject a good
    cache or, worse, accept payloads standardized into a different space -- same
    width, same shapes, every downstream number finite and meaningless.  The
    standardizer beside them is the authority.
    """

    import argparse
    import json

    from nvs_v6_wan_e2e.target_cache import (
        atomic_torch_save,
        manifest_path,
        read_scene_payload,
        scene_target_path,
    )

    script = _stage1_cache_ddp_script()
    tokens = [f"{index:016x}" for index in range(4)]
    root = _synthetic_stage1_cache(tmp_path, tokens, fingerprint="payload-fingerprint", eval_scenes=1)
    manifest = json.loads(manifest_path(root).read_text())
    manifest["target_fingerprint"] = "a-stale-manifest-copy"
    manifest_path(root).write_text(json.dumps(manifest))

    args = argparse.Namespace(target_cache_root=root, image_hw=None, skip_degenerate=False, seed=17,
                              min_gt_spread_m=MIN_GT_SPREAD_M)
    dctx = script.Dist(rank=0, world=1, local_rank=0, device=torch.device("cpu"), enabled=False)
    resolved = script.resolve_scene_sets(dctx, args, "payload-fingerprint")
    assert resolved["provenance"]["fingerprint_filtered"] == "payload-fingerprint"
    assert resolved["provenance"]["rejected"] == 0
    assert len(resolved["train"]) + len(resolved["eval"]) == len(tokens)

    # The filter is real, not decorative: a payload built into another target space is
    # dropped rather than averaged in with the ones that match.
    foreign = _synthetic_stage1_cache(tmp_path / "other", ["f" * 16], fingerprint="another-space")
    moved = read_scene_payload(scene_target_path(foreign, "f" * 16))
    atomic_torch_save(moved, scene_target_path(root, tokens[-1]))
    resolved = script.resolve_scene_sets(dctx, args, "payload-fingerprint")
    assert tokens[-1] not in resolved["train"] + resolved["eval"]
    assert resolved["provenance"]["rejected"] == 1
    assert resolved["provenance"]["first_rejected"][0]["reason"] == "target_fingerprint_mismatch"


def test_a_scene_whose_gt_trajectories_are_centimetres_apart_is_degenerate(tmp_path):
    """``degenerate_multi_gt`` tests the labels, which is not the thing that matters.

    It is ``len(set(labels)) < len(labels)``, so a scene carrying four *distinct*
    labels whose trajectories differ by centimetres passes it -- and one such scene was
    measured in the smoke cache at 0.029 m mean pairwise spread over a 4 s horizon.
    That scene is actively harmful to Stage 1 rather than merely unhelpful: its four
    register targets differ while its four inputs are identical to within float noise,
    so the only way to reduce the loss on it is to stop reading the action tokens.  The
    gate has to be on the metric trajectories, not on the labels.
    """

    from nvs_v6_wan_e2e.target_cache import read_scene_payload, scene_target_path

    tokens = [f"{index:016x}" for index in range(4)]
    root = _synthetic_stage1_cache(tmp_path, tokens, gt_count=4)

    near_identical, single_gt = tokens[0], tokens[1]
    payload = read_scene_payload(scene_target_path(root, near_identical))
    assert payload["degenerate_multi_gt"] is False
    assert len(set(payload["labels"])) == len(payload["labels"])
    # 1 cm of jitter across all four rows and eight waypoints: four distinct futures
    # that are one manoeuvre.  The labels stay distinct, so the old test still passes.
    # The payload carries [K,8,3], not the collated [1,K,8,3] -- K is axis 0 here.
    trajectory = payload["gt_trajectory"].clone()
    trajectory[..., :2] = trajectory[..., :2].mean(dim=0, keepdim=True) + torch.arange(4).view(4, 1, 1) * 0.01
    payload["gt_trajectory"] = trajectory
    atomic_torch_save(payload, scene_target_path(root, near_identical))

    one_row = read_scene_payload(scene_target_path(root, single_gt))
    one_row["gt_valid"] = torch.zeros_like(one_row["gt_valid"])
    one_row["gt_valid"][0] = True
    atomic_torch_save(one_row, scene_target_path(root, single_gt))

    reloaded = read_scene_payload(scene_target_path(root, near_identical))
    spread = multi_gt_spread_metres(reloaded["gt_trajectory"], reloaded["gt_valid"])
    # The jitter lands on x and y alike, so consecutive rows sit 0.01*sqrt(2) m apart
    # and the mean over the six pairs is that times 10/6.  Asserted as the constructed
    # value rather than "small compared to the floor", so moving MIN_GT_SPREAD_M cannot
    # quietly turn this into a test that no longer reproduces the 0.028 m scene it is
    # standing in for.
    assert spread == pytest.approx(0.01 * math.sqrt(2) * 10 / 6, rel=1e-6), spread
    assert spread < MIN_GT_SPREAD_M
    assert is_trajectory_degenerate(reloaded["gt_trajectory"], reloaded["gt_valid"])
    assert reloaded["degenerate_multi_gt"] is False, "the label test this gate supplements cannot see it"

    # A single valid GT has no pair to contrast, so its spread is undefined rather than
    # zero.  Reporting zero would make every legitimate one-GT scene look degenerate to
    # a caller that compares against a floor, and one-GT scenes are valid Multi-GT.
    single = read_scene_payload(scene_target_path(root, single_gt))
    assert math.isnan(multi_gt_spread_metres(single["gt_trajectory"], single["gt_valid"]))
    assert not is_trajectory_degenerate(single["gt_trajectory"], single["gt_valid"])

    # The comparison has to point the other way too: a real, if modest, separation is
    # kept.  Without this the gate passes with ``>`` in place of ``<``, which would drop
    # exactly the scenes Stage 1 needs and keep only the contradictory ones.
    separated = tokens[2]
    wide = read_scene_payload(scene_target_path(root, separated))
    wide_trajectory = wide["gt_trajectory"].clone()
    wide_trajectory[..., 1] += torch.arange(wide_trajectory.shape[0]).view(-1, 1) * 1.0
    wide["gt_trajectory"] = wide_trajectory
    atomic_torch_save(wide, scene_target_path(root, separated))
    reloaded_wide = read_scene_payload(scene_target_path(root, separated))
    assert multi_gt_spread_metres(reloaded_wide["gt_trajectory"], reloaded_wide["gt_valid"]) > MIN_GT_SPREAD_M
    assert not is_trajectory_degenerate(reloaded_wide["gt_trajectory"], reloaded_wide["gt_valid"])


def test_the_split_filter_and_the_dataset_agree_on_which_scenes_are_degenerate(tmp_path):
    """Two code paths decide whether a scene may be used, and they must decide alike.

    ``filter_usable_tokens`` builds the train/eval split; ``Stage1SceneDataset``
    re-checks at load time so a payload changed since the split cannot slip through.
    If the second is stricter than the first, ``__getitem__`` walks forward to a token
    the split never approved -- silently training on an eval scene or scoring a train
    scene as held out, with the only symptom being a metric attributed to the wrong
    scene.  The agreement is asserted as a property over every token rather than
    checked on one, because a single-scene assertion passes with either path stricter.
    """

    from nvs_v6_wan_e2e.target_cache import Stage1SceneDataset, read_scene_payload, scene_target_path

    tokens = [f"{index:016x}" for index in range(6)]
    root = _synthetic_stage1_cache(tmp_path, tokens, gt_count=4)
    collapsed = tokens[2]
    payload = read_scene_payload(scene_target_path(root, collapsed))
    trajectory = payload["gt_trajectory"].clone()
    trajectory[..., :2] = trajectory[..., :2].mean(dim=0, keepdim=True)
    payload["gt_trajectory"] = trajectory
    atomic_torch_save(payload, scene_target_path(root, collapsed))

    for skip_degenerate in (False, True):
        kept, rejected = filter_usable_tokens(
            tokens, root, skip_degenerate=skip_degenerate, target_fingerprint="test-fingerprint",
        )
        dataset = Stage1SceneDataset(
            tokens, root, image_hw=(32, 48), skip_degenerate=skip_degenerate,
            target_fingerprint="test-fingerprint",
        )
        for token in tokens:
            loadable = True
            try:
                dataset.load_payload(token)
            except ValueError:
                loadable = False
            assert (token in kept) == loadable, (
                f"skip_degenerate={skip_degenerate}: the split and the dataset disagree about {token}"
            )
        assert (collapsed in kept) is (not skip_degenerate)

    _, rejected = filter_usable_tokens(
        tokens, root, skip_degenerate=True, target_fingerprint="test-fingerprint",
    )
    assert [entry["reason"] for entry in rejected if entry["scene"] == collapsed][0].startswith("gt_spread_")


def test_load_eval_samples_reports_the_scene_the_dataset_actually_returned(tmp_path):
    """The id attached to a metric row has to be the scene that was scored.

    ``Stage1SceneDataset.__getitem__`` walks forward to the next token when a decode
    fails, which is what keeps one unreadable scene from changing how many batches a
    rank produces -- and therefore from desyncing the DDP all-reduce.  The side effect
    is that ``dataset[index]`` is not necessarily ``scene_ids[index]``, so taking the
    id from the input list would label a metric with a scene that was never scored.
    A mislabelled eval row is worse than a missing one: it reads as a per-scene result.
    """

    from nvs_v6_wan_e2e.target_cache import HISTORY_FRAMES, Stage1SceneDataset, scene_target_path

    script = _stage1_cache_ddp_script()
    tokens = [f"{index:016x}" for index in range(4)]
    root = _synthetic_stage1_cache(tmp_path, tokens, gt_count=2)
    dataset = Stage1SceneDataset(
        tokens, root, image_hw=(32, 48), target_fingerprint="test-fingerprint"
    )

    samples, skipped = script.load_eval_samples(dataset, tokens, 2)
    assert skipped == []
    assert [sample.scene_id for sample in samples] == tokens[:2]
    for sample in samples:
        observation = sample.observation
        observation.validate()
        # uint8 on the host is the whole point of the streaming dataset: float32 here
        # would put 46.3 MB per eval scene next to a model that owns most of the card.
        assert observation.rgb_history.dtype == torch.uint8
        assert tuple(observation.rgb_history.shape) == (1, HISTORY_FRAMES, 1, 3, 32, 48)
        assert tuple(sample.targets.future_register.shape) == (1, 2, REGISTER_TOKENS, REGISTER_TARGET_DIM)
        assert observation.camera_metadata[0]["scene_id"] == sample.scene_id

    # The forward-walk case, reproduced rather than described: with the first payload
    # gone, index 0 walks to the second scene and index 1 returns it again.  The
    # duplicate is the tell -- reading the id off the input list would have reported
    # two distinct scenes, one of which was never scored, and a per-scene metric
    # attributed to the wrong scene is worse than one scene counted twice.
    scene_target_path(root, tokens[0]).unlink()
    walked, _ = script.load_eval_samples(dataset, tokens, 2)
    assert [sample.scene_id for sample in walked] == [tokens[1], tokens[1]]


def test_trajectory_spread_is_mean_pairwise_ade_in_metres():
    from nvs_v6_wan_e2e.stage1_eval import trajectory_spread

    assert trajectory_spread(torch.zeros(3, 8, TRAJECTORY_STATE_DIM)) == 0.0
    # Two straight lines held 1 m apart in y at every step have an ADE of exactly 1 m.
    apart = torch.zeros(2, 8, TRAJECTORY_STATE_DIM)
    apart[1, :, 1] = 1.0
    assert trajectory_spread(apart) == pytest.approx(1.0)
    assert math.isnan(trajectory_spread(apart[:1]))


def test_register_verdict_claims_skill_only_when_the_trajectory_earned_it():
    from nvs_v6_wan_e2e.stage1_eval import CONDITIONING_SUMMARY_KEYS, register_verdict

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
            "mean_future_cosine_similarity": 0.42,
            "mean_future_relative_l1": 0.85,
            "mean_residual_relative_l1": 0.70,
            "mean_residual_cos_future": 0.35,
        }
        values.update(overrides)
        return values

    learned = register_verdict(metrics(0.40), None)
    assert learned["train_beats_scene_mean_baseline"] is True
    assert learned["train_learned_trajectory_conditional_future"] is True
    assert learned["train_gain_over_scene_mean_target"] == pytest.approx(0.2)
    assert learned["train_collapse_toward_mean"] == pytest.approx(0.75)
    # The two direct "how far from the real future" readings are reported verbatim and
    # deliberately gate nothing.  relative_l1 still has its built-in 1.0 bar (the
    # all-zero predictor); "good enough" is anchored by scene_mean_target_l1 instead.
    assert learned["train_future_cosine_similarity"] == pytest.approx(0.42)
    assert learned["train_future_relative_l1"] == pytest.approx(0.85)
    # The background-removed pair is passed through verbatim too, and gates nothing:
    # it is the reading that stays honest when the raw cosine is flattered by the scene
    # mean, so it must survive into the verdict JSON rather than live only in the step log.
    assert learned["train_residual_relative_l1"] == pytest.approx(0.70)
    assert learned["train_residual_cos_future"] == pytest.approx(0.35)
    # Every key the before/after summary prints must be one future_fit_metrics emits,
    # or the report silently drops a metric.
    assert set(CONDITIONING_SUMMARY_KEYS) <= set(metrics(0.40))

    # Matching the trajectory-blind scene reader exactly is not skill, however well
    # the same number looks against the weaker dataset-mean baseline.
    flat = register_verdict(metrics(0.50), None)
    assert flat["train_gain_over_constant_target"] == pytest.approx(1 / 6)
    assert flat["train_beats_scene_mean_baseline"] is False
    assert flat["train_learned_trajectory_conditional_future"] is False

    # Shrinking toward the scene average also lowers the paired L1; only the
    # scale-free pair shows that the register moved the wrong way.  This is the case
    # the headline boolean's name has to survive: a register with fidelity 0.05 is
    # trajectory-blind by the scale-free reading, yet it does beat the scene-mean
    # baseline, so beats_scene_mean_baseline is True and the verdict is still False.
    collapsed = register_verdict(
        metrics(0.40, mean_trajectory_conditioning_fidelity=0.05, mean_trajectory_response_alignment=-0.1), None
    )
    assert collapsed["train_beats_scene_mean_baseline"] is True
    assert collapsed["train_learned_trajectory_conditional_future"] is False

    generalized = register_verdict(metrics(0.40), metrics(0.48))
    assert generalized["eval_learned_trajectory_conditional_future"] is True
    assert generalized["generalizes_beyond_fitted_scenes"] is True
    assert "train_gain_over_scene_mean_target" in generalized


def test_streaming_alignment_prints_the_verdict_own_two_readings():
    """The per-step log and the verdict must report the same two quantities.

    ``run_stage1_cache_ddp.py`` logs ``cos_future`` and ``relative_l1_future`` every
    ``--log-every`` steps so a 20-epoch run can be read while it is still going,
    instead of only at the next verdict pass.  Those numbers are only worth reading
    if they *are* ``future_fit_metrics``' ``future_cosine_similarity`` and
    ``future_relative_l1`` -- so both paths run over one batch here: a stub model
    hands the verdict function a canned register one scene at a time, exactly as
    production does, and the streaming version sees the same predictions stacked.

    The fixtures are shaped to break the cheap variants.  Weighting the cosine by
    ``future_quality`` (the L1's weight, not the cosine's) diverges as soon as
    quality is uneven; dropping the per-GT-row balance diverges when one row has far
    more valid tokens than its siblings; forgetting ``gt_valid`` diverges because the
    padded rows hold garbage.
    """

    from nvs_v6_wan_e2e.stage1_eval import future_fit_metrics, streaming_future_alignment

    class CannedRegister(nn.Module):
        """Stands in for ``DirectedPairDiT``: returns the fixture's register, per scene."""

        def __init__(self, predicted: torch.Tensor, trajectories: torch.Tensor) -> None:
            super().__init__()
            self.predicted = predicted
            self.trajectories = trajectories
            self.device_probe = nn.Parameter(torch.zeros(1))

        def forward_register_only(self, observation, source, modes):
            # Matched on GT 0's trajectory alone: the trajectory-blind control re-runs
            # the same scene with every row set to it, and both calls must land here.
            count = source.shape[1]
            for scene in range(self.predicted.shape[0]):
                if torch.equal(source[0, 0], self.trajectories[scene, 0]):
                    return (self.predicted[scene, :count],)
            raise AssertionError("stub was asked for a scene it does not have")

    class Scene:
        def __init__(self, scene_id: str, observation, targets) -> None:
            self.scene_id, self.observation, self.targets = scene_id, observation, targets

    def slice_targets(batched: PairTargets, scene: int) -> PairTargets:
        return PairTargets(
            gt_trajectory=batched.gt_trajectory[scene : scene + 1],
            gt_valid=batched.gt_valid[scene : scene + 1],
            future_register=batched.future_register[scene : scene + 1],
            future_chunk_valid=batched.future_chunk_valid[scene : scene + 1],
            future_quality=batched.future_quality[scene : scene + 1],
            pair_ids=[batched.pair_ids[scene]],
        )

    torch.manual_seed(3)
    counts = [4, 1, 3, 2]
    scenes, rows = len(counts), max(counts)
    gt_trajectory = torch.randn(scenes, rows, 8, TRAJECTORY_STATE_DIM)
    gt_valid = torch.zeros(scenes, rows, dtype=torch.bool)
    for scene, count in enumerate(counts):
        gt_valid[scene, :count] = True
    future_register = torch.randn(scenes, rows, REGISTER_TOKENS, REGISTER_TARGET_DIM)
    chunk_valid = torch.ones(scenes, rows, REGISTER_TOKENS, dtype=torch.bool)
    for scene in range(scenes):
        chunk_valid[scene, :, (scene + 1) * REGISTER_TOKENS // 5 :] = False
    quality = torch.rand(scenes, rows, REGISTER_TOKENS) + 0.5
    targets = PairTargets(
        gt_trajectory=gt_trajectory,
        gt_valid=gt_valid,
        future_register=future_register,
        future_chunk_valid=chunk_valid,
        future_quality=quality,
        pair_ids=[[{"gt_id": str(index)} for index in range(rows)] for _ in range(scenes)],
    )
    predicted = future_register + 0.7 * torch.randn(scenes, rows, REGISTER_TOKENS, REGISTER_TARGET_DIM)

    samples = [
        Scene(f"scene{scene}", make_observation(1), slice_targets(targets, scene)) for scene in range(scenes)
    ]
    summary = future_fit_metrics(samples, CannedRegister(predicted, gt_trajectory), None, None)
    streamed = streaming_future_alignment(predicted, targets)
    assert streamed["valid_scenes"] == scenes
    assert streamed["future_cosine_similarity"] == pytest.approx(
        summary["mean_future_cosine_similarity"], abs=1e-6
    )
    assert streamed["future_relative_l1"] == pytest.approx(summary["mean_future_relative_l1"], abs=1e-6)
    assert streamed["residual_relative_l1"] == pytest.approx(summary["mean_residual_relative_l1"], abs=1e-6)
    assert streamed["residual_cos_future"] == pytest.approx(summary["mean_residual_cos_future"], abs=1e-6)

    # A perfect register scores 1.0 / 0.0 on both readings.
    exact = streaming_future_alignment(future_register.clone(), targets)
    assert exact["future_cosine_similarity"] == pytest.approx(1.0, abs=1e-5)
    assert exact["future_relative_l1"] == pytest.approx(0.0, abs=1e-6)
    assert exact["residual_relative_l1"] == pytest.approx(0.0, abs=1e-6)
    assert exact["residual_cos_future"] == pytest.approx(1.0, abs=1e-5)

    # A batch with nothing valid returns zeros, never NaN: the trainer all-reduces
    # these across 16 ranks, where one NaN would erase every rank's reading.
    blank = PairTargets(
        gt_trajectory=torch.zeros(1, 1, 8, TRAJECTORY_STATE_DIM),
        gt_valid=torch.zeros(1, 1, dtype=torch.bool),
        future_register=torch.zeros(1, 1, REGISTER_TOKENS, REGISTER_TARGET_DIM),
        future_chunk_valid=torch.zeros(1, 1, REGISTER_TOKENS, dtype=torch.bool),
        future_quality=torch.zeros(1, 1, REGISTER_TOKENS),
        pair_ids=[[]],
    )
    empty = streaming_future_alignment(torch.zeros(1, 1, REGISTER_TOKENS, REGISTER_TARGET_DIM), blank)
    assert empty == {
        "future_cosine_similarity": 0.0,
        "future_relative_l1": 0.0,
        "residual_relative_l1": 0.0,
        "residual_cos_future": 0.0,
        "valid_scenes": 0,
    }

    with pytest.raises(ValueError, match="does not match target"):
        streaming_future_alignment(predicted[:, :, : REGISTER_TOKENS - 1], targets)


def test_residual_readings_exclude_the_scene_background():
    """The residual readings are blind to anything a per-scene constant can explain.

    Measured on 400 cached scenes the per-scene mean of the register target carries
    **59.8%** of the L1 objective (0.792 predicting zero -> 0.318 predicting the scene
    mean) and **79.3%** of its raw L2 energy (11.8% per-token global DC + 67.5%
    scene-specific), leaving 40.2% that only reading the trajectory can win.  So a
    register that reads the scene perfectly and the trajectory not at all still looks
    *good* on the raw cosine -- the retired 8-token run read cos 0.6555 while both
    gains were negative -- and the raw reading cannot carry the verdict on its own.
    ``residual_relative_l1`` subtracts each side's own scene mean, which pins that
    exact failure mode at 1.0 and makes anything below it real evidence.
    """
    from nvs_v6_wan_e2e.stage1_eval import streaming_future_alignment

    torch.manual_seed(11)
    scenes, rows = 6, 4
    background = torch.randn(scenes, 1, REGISTER_TOKENS, REGISTER_TARGET_DIM) * 3.0
    signal = torch.randn(scenes, rows, REGISTER_TOKENS, REGISTER_TARGET_DIM)
    future_register = background + signal

    def pair_targets() -> PairTargets:
        return PairTargets(
            gt_trajectory=torch.randn(scenes, rows, 8, TRAJECTORY_STATE_DIM),
            gt_valid=torch.ones(scenes, rows, dtype=torch.bool),
            future_register=future_register.clone(),
            future_chunk_valid=torch.ones(scenes, rows, REGISTER_TOKENS, dtype=torch.bool),
            future_quality=torch.ones(scenes, rows, REGISTER_TOKENS),
            pair_ids=[[{"gt_id": str(index)} for index in range(rows)] for _ in range(scenes)],
        )

    # (a) pure background reader: every GT row gets the scene's own mean future.  The
    #     raw cosine is high -- this is the false positive -- while the residual is 1.0.
    background_only = background.expand(scenes, rows, REGISTER_TOKENS, REGISTER_TARGET_DIM).clone()
    reading = streaming_future_alignment(background_only, pair_targets())
    assert reading["future_cosine_similarity"] > 0.9
    assert reading["residual_relative_l1"] == pytest.approx(1.0, abs=1e-5)
    assert reading["residual_cos_future"] == pytest.approx(0.0, abs=1e-3)

    # (b) half the trajectory signal recovered -> residual 0.5, i.e. the scale is linear
    #     in how much of the background-free part the model actually got.
    half = streaming_future_alignment(background + 0.5 * signal, pair_targets())
    assert half["residual_relative_l1"] == pytest.approx(0.5, abs=1e-5)

    # (c) adding an arbitrary per-scene constant must not move the residual reading at
    #     all -- that is the whole claim, and it is what "excludes background" means.
    shifted = streaming_future_alignment(background + 0.5 * signal + 7.0 * background, pair_targets())
    assert shifted["residual_relative_l1"] == pytest.approx(half["residual_relative_l1"], abs=1e-6)

    # (d) a wrong signal is worse than no signal: above 1.0, mirroring relative_l1's bar.
    wrong = streaming_future_alignment(background - signal, pair_targets())
    assert wrong["residual_relative_l1"] > 1.0

    # (e) perfect prediction -> 0.0 and cosine 1.0.
    perfect = streaming_future_alignment(future_register.clone(), pair_targets())
    assert perfect["residual_relative_l1"] == pytest.approx(0.0, abs=1e-6)
    assert perfect["residual_cos_future"] == pytest.approx(1.0, abs=1e-5)


def test_horizon_resolved_gain_separates_the_frames_that_carry_trajectory_signal():
    """Per-frame gains must exist for all eight horizons and be individually readable.

    The trajectory-attributable share of a frame rises monotonically with horizon --
    measured 30.6% of the L1 at +0.5s against 46.9% at +4.0s, and 34.5% against 70.6%
    of the frame's L2 energy -- because the near future is mostly the scene continuing
    while the far future is mostly where the trajectory went.  One pooled gain hides
    which end is moving, and the near end is exactly where a background reader scores
    best, so a pooled gain can look flat while the far frames are learning.
    """
    from nvs_v6_wan_e2e.data import FUTURE_FRAMES
    from nvs_v6_wan_e2e.stage1_eval import CONDITIONING_SUMMARY_KEYS, future_fit_metrics

    class Canned(nn.Module):
        def __init__(self, predicted: torch.Tensor, trajectories: torch.Tensor) -> None:
            super().__init__()
            self.predicted, self.trajectories = predicted, trajectories
            self.device_probe = nn.Parameter(torch.zeros(1))

        def forward_register_only(self, observation, source, modes):
            count = source.shape[1]
            for scene in range(self.predicted.shape[0]):
                if torch.equal(source[0, 0], self.trajectories[scene, 0]):
                    return (self.predicted[scene, :count],)
            raise AssertionError("stub was asked for a scene it does not have")

    class Sample:
        def __init__(self, scene_id: str, observation, targets) -> None:
            self.scene_id, self.observation, self.targets = scene_id, observation, targets

    torch.manual_seed(5)
    scenes, rows = 3, 4
    future_register = torch.randn(scenes, rows, REGISTER_TOKENS, REGISTER_TARGET_DIM)
    gt_trajectory = torch.randn(scenes, rows, 8, TRAJECTORY_STATE_DIM)
    targets = PairTargets(
        gt_trajectory=gt_trajectory,
        gt_valid=torch.ones(scenes, rows, dtype=torch.bool),
        future_register=future_register.clone(),
        future_chunk_valid=torch.ones(scenes, rows, REGISTER_TOKENS, dtype=torch.bool),
        future_quality=torch.ones(scenes, rows, REGISTER_TOKENS),
        pair_ids=[[{"gt_id": str(index)} for index in range(rows)] for _ in range(scenes)],
    )

    def slice_scene(batched: PairTargets, scene: int) -> PairTargets:
        return PairTargets(**{
            key: (value[scene : scene + 1] if isinstance(value, torch.Tensor) else [value[scene]])
            for key, value in batched.__dict__.items()
        })

    samples = [
        Sample(f"scene{scene}", make_observation(1), slice_scene(targets, scene)) for scene in range(scenes)
    ]

    summary = future_fit_metrics(samples, Canned(future_register.clone(), gt_trajectory), None, None)
    for frame in range(1, FUTURE_FRAMES + 1):
        key = f"mean_frame{frame}_gain_over_scene_mean"
        assert key in summary, f"the horizon-resolved gain is missing {key}"
        # A perfect register beats the scene-mean anchor by 100% at every horizon.
        assert summary[key] == pytest.approx(1.0, abs=1e-5)
        assert key not in CONDITIONING_SUMMARY_KEYS  # per-frame detail, not a headline
    assert summary["mean_residual_relative_l1"] == pytest.approx(0.0, abs=1e-6)


def test_paired_contrast_loss_is_invariant_to_the_scene_background():
    """The contrast term must see only what the trajectories differ in.

    Measured on 400 cached scenes the scene mean of the register target carries 59.8% of
    the L1 objective, and on the rendered pixels 79.5% of the future's variance is fixed
    by the scene before any trajectory is read.  ``identity_future_l1`` cannot avoid
    paying for that; the paired contrast term cancels it by construction, and this test
    pins the cancellation rather than the arithmetic: adding an arbitrary per-scene
    constant to every GT row of a prediction must leave the contrast loss bit-identical,
    while the plain identity loss must move.
    """
    from nvs_v6_wan_e2e.losses import identity_future_l1, paired_contrast_future_l1

    torch.manual_seed(21)
    targets = make_targets(batch=3, gt=4)
    predicted = targets.future_register + 0.3 * torch.randn_like(targets.future_register)
    base_contrast = paired_contrast_future_l1(predicted, targets)
    base_identity = identity_future_l1(predicted, targets)

    # (a) a per-scene constant added to every row: pure background, invisible to the
    #     contrast term, very visible to the identity term.
    offset = torch.randn(3, 1, REGISTER_TOKENS, REGISTER_TARGET_DIM) * 5.0
    shifted = paired_contrast_future_l1(predicted + offset, targets)
    assert shifted == pytest.approx(float(base_contrast), abs=1e-5)
    assert identity_future_l1(predicted + offset, targets) != pytest.approx(
        float(base_identity), abs=1e-3
    )

    # (b) a perfect predictor scores zero on the contrast term.
    assert paired_contrast_future_l1(targets.future_register.clone(), targets) == pytest.approx(0.0, abs=1e-6)

    # (c) a trajectory-blind predictor (one row repeated) scores the target's own spread,
    #     not zero: the term does not reward ignoring the trajectory.
    blind = targets.future_register.mean(dim=1, keepdim=True).expand_as(targets.future_register)
    blind_loss = paired_contrast_future_l1(blind, targets)
    assert float(blind_loss) > 0.1

    # (d) a single-GT scene has no pair: the zero must still carry the graph, so DDP's
    #     reducer keeps seeing every parameter with a gradient.
    single = make_targets(batch=1, gt=1)
    pred_single = single.future_register.clone().requires_grad_(True)
    zero = paired_contrast_future_l1(pred_single, single)
    assert float(zero) == 0.0
    zero.backward()
    assert pred_single.grad is not None and bool((pred_single.grad == 0).all())

    # (e) the target is detached: the contrast term must not back-prop into the cache.
    grad_target = make_targets(batch=1, gt=2)
    grad_target.future_register = grad_target.future_register.clone().requires_grad_(True)
    pred_grad = torch.randn_like(grad_target.future_register, requires_grad=True)
    paired_contrast_future_l1(pred_grad, grad_target).backward()
    assert grad_target.future_register.grad is None
    assert pred_grad.grad is not None


def test_future_loss_respects_frame_mask_and_detached_target():
    prediction = torch.zeros(1, 4, REGISTER_TOKENS, REGISTER_TARGET_DIM, requires_grad=True)
    targets = make_targets(gt=1)
    targets.future_register.zero_()
    targets.future_register[..., 0, :] = 1.0
    targets.future_chunk_valid[..., 1:] = False
    dense_future_l1(prediction, targets, torch.zeros(1, 4, 8, TRAJECTORY_STATE_DIM)).backward()
    assert targets.future_register.grad is None
    assert prediction.grad.abs().sum(dim=(0, 1, 3)).tolist() == pytest.approx(
        [1.0] + [0.0] * (REGISTER_TOKENS - 1), abs=1e-6
    )


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
def test_scratch_scorer_rejects_the_wrong_scene_length_and_ignores_the_register():
    scorer = make_scorer()
    trajectory = torch.randn(1, 4, 8, TRAJECTORY_STATE_DIM)
    scene = torch.randn(1, SCENE_TOKENS, D_MODEL)
    with pytest.raises(ValueError):
        scorer(torch.randn(1, SCENE_TOKENS + 1, D_MODEL), trajectory)
    with pytest.raises(ValueError):
        scorer(scene, torch.randn(1, 4, 8, 4))
    with pytest.raises(ValueError):
        scorer(torch.randn(2, SCENE_TOKENS, D_MODEL), trajectory)
    # DrivoR's score path reads scene + trajectory only.  The register is now a DiT
    # output rather than a scorer input, so passing one must not move the logits --
    # `predicted_register_is_input` is False and stays that way.
    register = torch.randn(1, 4, REGISTER_TOKENS, REGISTER_TARGET_DIM)
    assert torch.equal(scorer(scene, trajectory), scorer(scene, trajectory, register))


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


def _paired_pos_neg_caches(tmp_path: Path, tokens: list[str], pos_gt: int = 2, neg_gt: int = 3):
    """A positive and a negative Stage-0 cache sharing ONE standardizer.

    Sharing is the whole point: ``TargetStandardizer.fingerprint`` hashes the fitted
    mean/scale, so two caches fitted separately live in two target spaces and their
    rows cannot be concatenated behind one head.  Both roots are written with the
    real ``save_standardizer`` and both payload sets carry that fingerprint, which is
    what the dataset's construction-time guard compares.
    """

    from nvs_v6_wan_e2e.teacher import TargetStandardizer

    standardizer = TargetStandardizer(feature_dim=REGISTER_TARGET_DIM)
    standardizer.fit(torch.randn(3, 2, REGISTER_TOKENS, REGISTER_TARGET_DIM), "+".join(tokens[:3]))
    fingerprint = standardizer.fingerprint()
    positive = _synthetic_stage1_cache(tmp_path / "pos", tokens, fingerprint=fingerprint, gt_count=pos_gt)
    negative = _synthetic_stage1_cache(tmp_path / "neg", tokens, fingerprint=fingerprint, gt_count=neg_gt)
    for root in (positive, negative):
        save_standardizer(root, standardizer, {"fit_scene_ids": tokens[:3]})
    return positive, negative, fingerprint


def test_negative_cache_rows_append_after_positives_without_reaching_the_model(tmp_path):
    """Mined negatives join the *training* rows only, positives first and unchanged.

    Three contracts:
    (a) with ``negative_target_cache_root=None`` the item is exactly what it was
        before the flag existed, so a positive-only run does not move;
    (b) a merged item is the positive rows followed by the negative cache's own,
        each bit-identical to its source, so a row's provenance is its position;
    (c) row kind never becomes a tensor.  ``negative_rows`` exists only in
        ``camera_metadata``; every supervised tensor keeps the positive-only
        trailing shape.  That is what stops a merged batch from handing the network
        a per-row "am I a negative" code -- the shortcut ``stage1_step`` zeroes the
        mode code to make structurally unavailable.
    """

    from nvs_v6_wan_e2e.target_cache import Stage1SceneDataset, collate_stage1_batch, scene_target_path

    tokens = [f"{index:016x}" for index in range(8)]
    positive, negative, fingerprint = _paired_pos_neg_caches(tmp_path, tokens)
    kwargs = dict(image_hw=(32, 48), target_fingerprint=fingerprint)
    plain = Stage1SceneDataset(tokens, positive, **kwargs)
    merged = Stage1SceneDataset(tokens, positive, negative_target_cache_root=negative, **kwargs)
    negatives_only = Stage1SceneDataset(tokens, negative, **kwargs)

    plain_item, merged_item = plain[0], merged[0]
    assert plain_item["gt_valid"].shape[0] == 2
    assert plain_item["negative_rows"] == 0
    assert merged_item["gt_valid"].shape[0] == 5, "2 positive + 3 negative rows"
    assert merged_item["negative_rows"] == 3

    # (b) positives first and untouched, negatives exactly as their own cache holds them
    row_keys = Stage1SceneDataset.NEGATIVE_ROW_KEYS
    for key in row_keys:
        assert torch.equal(merged_item[key][:2], plain_item[key]), key
        assert torch.equal(merged_item[key][2:], negatives_only[0][key]), key
    assert merged_item["labels"][:2] == plain_item["labels"]
    assert all(label.startswith("neg:") for label in merged_item["labels"][2:])

    # (a) the observation side is not part of the merge at all
    for key in ("rgb_history", "history_valid", "ego_history", "route_command",
                "relative_ego_poses", "scene_id", "degenerate_multi_gt"):
        assert torch.equal(merged_item[key], plain_item[key]) if torch.is_tensor(plain_item[key]) else (
            merged_item[key] == plain_item[key]
        ), key

    # (c) mixed K across a batch pads to the widest scene and masks the padding
    plain_dataset_second = plain[1]
    negative_payload = scene_target_path(negative, tokens[1])
    negative_payload.unlink()
    assert merged[1]["gt_valid"].shape[0] == 2, "a scene with no negative payload keeps its positives"
    assert merged[1]["negative_rows"] == 0
    observation, targets, scene_ids = collate_stage1_batch([merged_item, merged[1]])
    assert targets.gt_trajectory.shape == (2, 5, 8, 3)
    assert targets.gt_valid.tolist() == [[True] * 5, [True, True, False, False, False]]
    assert not targets.future_chunk_valid[1, 2:].any(), "padded rows are masked, not zero-supervised"
    assert [entry["negative_rows"] for entry in observation.camera_metadata] == [3, 0]
    for tensor in (targets.gt_trajectory, targets.future_register, targets.future_quality):
        assert tensor.shape[1] == 5 and tensor.shape[2:] != (5,), tensor.shape
    assert scene_ids == [tokens[0], tokens[1]]
    del plain_dataset_second


def test_negative_cache_rejects_a_different_target_space(tmp_path):
    """A negative cache fitted separately is a hard error, not a silent mixture.

    Checked at construction rather than per scene because ``__getitem__`` retries the
    next token on any exception: a per-scene raise would surface as "could not load
    any scene near index" and hide the real cause.
    """

    from nvs_v6_wan_e2e.teacher import TargetStandardizer
    from nvs_v6_wan_e2e.target_cache import Stage1SceneDataset

    tokens = [f"{index:016x}" for index in range(8)]
    positive, negative, fingerprint = _paired_pos_neg_caches(tmp_path, tokens)
    kwargs = dict(image_hw=(32, 48), target_fingerprint=fingerprint)
    # The pair that shares one standardizer constructs *and* merges.
    assert Stage1SceneDataset(tokens, positive, negative_target_cache_root=negative, **kwargs)[0]["negative_rows"] == 3

    other = TargetStandardizer(feature_dim=REGISTER_TARGET_DIM)
    other.fit(torch.randn(3, 2, REGISTER_TOKENS, REGISTER_TARGET_DIM) * 7.0 + 3.0, "+".join(tokens[:3]))
    foreign = _synthetic_stage1_cache(tmp_path / "foreign", tokens, fingerprint=other.fingerprint(), gt_count=2)
    save_standardizer(foreign, other, {"fit_scene_ids": tokens[:3]})
    assert other.fingerprint() != fingerprint

    with pytest.raises(ValueError, match="standardized as"):
        Stage1SceneDataset(tokens, positive, negative_target_cache_root=foreign, **kwargs)
