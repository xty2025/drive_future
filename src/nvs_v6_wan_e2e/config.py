"""Strict v6 configuration loader; conflicting legacy direction flags fail fast."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .data import FUTURE_FRAMES, REGISTER_QUERIES_PER_FRAME
from .teacher import MAX_QUERY_ATTENTION_ENTROPY, QUERY_HEADS, QUERY_TEMPERATURE, TARGET_STD_FLOOR
from .wan_dit import DEFAULT_FUTURE_TOKENS, FUTURE_QUERY_SCALE, NvsActionDiT


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


EXPECTED_NUM_SCENE_TOKENS = 16
EXPECTED_FRAMES = 5
EXPECTED_LORA_RANK = 32
#: One register token per (future frame, DINOv3 prefix query), frame-major.  Aliased to
#: the DiT's own default rather than recomputed, so the number the yaml must declare and
#: the number ``NvsActionDiT`` builds cannot drift apart; both trace back to the loader's
#: ``FUTURE_FRAMES`` and the Teacher readout's ``REGISTER_QUERIES_PER_FRAME``.
EXPECTED_REGISTER_TOKENS = DEFAULT_FUTURE_TOKENS
#: Distinct *times* in the register segment.  The segment is longer than this because
#: each frame is read by several queries; the RoPE phase and the trajectory alignment
#: are per frame, not per token.
EXPECTED_REGISTER_FRAMES = FUTURE_FRAMES
# DrivoR's scene perception: patch16 DINOv3 at its own training resolution, with
# the prepended scene queries counted as RoPE prefix tokens.
EXPECTED_PATCH_SIZE = 16
# DrivoR's official image_size is [1148, 672] (W, H).  1148 = 14*82 for its
# patch14 DINOv2, but patch16 leaves a 12 px remainder, which is closed by
# resampling the width up to the backbone's native 1152 -- never by zero columns.
EXPECTED_DRIVOR_IMAGE_HW = [672, 1148]
EXPECTED_RIGHT_RESAMPLE_PX = 4
EXPECTED_IMAGE_HW = [672, 1152]
EXPECTED_PATCH_GRID = [42, 72]
# The generated future videos are natively 464x832, which patch16 tiles exactly,
# so the Teacher needs no resize at all.
EXPECTED_TEACHER_IMAGE_HW = [464, 832]
EXPECTED_TEACHER_GRID_HW = [29, 52]
# The register target's width is the Teacher backbone's own ``embed_dim``, *per token*:
# the query readout is a convex combination of patch tokens and standardization is
# diagonal, so nothing between the frozen backbone and the cache changes it.  It is
# *not* d_model, and pinning it here is what stops a yaml edit from quietly making the
# future head a different width than the cached targets.  Widening the target means
# raising ``registers_per_frame``, which multiplies the token count and leaves this
# number alone -- the head stays ``Linear(d_model, 1024)`` either way.
EXPECTED_REGISTER_TARGET_DIM = 1024
EXPECTED_NATIVE_PREFIX_TOKENS = 5
# The four queries are DINOv3's own four register tokens (prefix 5 = CLS + 4), not a
# count anyone chose.  Reading the frame with more than four would mean inventing
# queries the frozen backbone never trained, which is exactly the randomly initialised
# learnable query that measured as a mean pool in an attention costume.
_require(
    REGISTER_QUERIES_PER_FRAME == EXPECTED_NATIVE_PREFIX_TOKENS - 1,
    f"the register readout queries with DINOv3's {EXPECTED_NATIVE_PREFIX_TOKENS - 1} prefix "
    f"registers, got REGISTER_QUERIES_PER_FRAME={REGISTER_QUERIES_PER_FRAME}",
)


def load_v6_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    # The id still says "R_to_A" but the direction was reversed: the future register
    # is now a DiT *output* read by nobody upstream.  Kept as a frozen identifier
    # because recorded artifacts quote it; ``token_order`` below is what asserts the
    # real layout.
    _require(config.get("spec_id") == "navsim_future_register_R_to_A_v6", "only navsim_future_register_R_to_A_v6 is supported")

    vision = config["vision"]
    _require(vision.get("image_hw") == EXPECTED_IMAGE_HW, f"DrivoR scene input must be {EXPECTED_IMAGE_HW}")
    _require(vision.get("patch_grid") == EXPECTED_PATCH_GRID, f"patch16 DINOv3 grid must be {EXPECTED_PATCH_GRID}")
    _require(
        vision.get("position_encoding") == "dinov3_rope_analytic_no_interpolation",
        "DINOv3 RoPE must be built analytically; no position-table interpolation",
    )
    _require(vision.get("native_prefix_tokens") == EXPECTED_NATIVE_PREFIX_TOKENS, "DINOv3 has cls + 4 register prefix tokens")
    _require(vision.get("num_scene_tokens") == EXPECTED_NUM_SCENE_TOKENS, "DrivoR num_scene_tokens must be 16")
    _require(vision.get("rope_prefix_tokens") == EXPECTED_NATIVE_PREFIX_TOKENS + EXPECTED_NUM_SCENE_TOKENS, "scene queries are RoPE prefix tokens")
    _require(vision.get("base_trainable") is False, "the DINOv3 base stays frozen")
    _require(vision.get("frames") == EXPECTED_FRAMES, "v6 requires current plus four history frames")
    lora = vision.get("student_lora") or {}
    _require(lora.get("enabled") is True and lora.get("rank") == EXPECTED_LORA_RANK, f"DrivoR LoRA must be enabled at rank {EXPECTED_LORA_RANK}")
    _require(lora.get("qv_only") is True, "DrivoR's _LoRA_qkv_timm adapts q and v only")
    _require(
        vision.get("scene_tokens") == EXPECTED_FRAMES * EXPECTED_NUM_SCENE_TOKENS + vision.get("ego_tokens", 1),
        "scene_tokens must be frames * num_scene_tokens plus the single ego token",
    )

    # DrivoR resizes 1920x1080 down to its official 1148x672 and stops there.
    # patch16 cannot tile 1148, so the remainder is closed by resampling the width
    # to the backbone's native 1152 -- DrivoR's own methodology, with the position
    # encoding then built for the resulting grid.  Zero columns are refused: they
    # would put fabricated patch content into the scene encoder.
    _require(
        vision.get("drivor_image_hw") == EXPECTED_DRIVOR_IMAGE_HW,
        f"history/current frames must be DrivoR's official {EXPECTED_DRIVOR_IMAGE_HW}, "
        f"got {vision.get('drivor_image_hw')}",
    )
    _require(
        vision.get("right_zero_pad_px") == 0 and vision.get("bottom_zero_pad_px") == 0,
        "the Student must not zero-pad the image; patch16 divisibility comes from resampling",
    )
    _require(
        vision.get("right_resample_px") == EXPECTED_RIGHT_RESAMPLE_PX and vision.get("bottom_resample_px") == 0,
        f"patch16 closes DrivoR's 1148 width by resampling exactly {EXPECTED_RIGHT_RESAMPLE_PX} px, "
        f"got {vision.get('right_resample_px')}",
    )
    _require(
        EXPECTED_DRIVOR_IMAGE_HW[1] + EXPECTED_RIGHT_RESAMPLE_PX == EXPECTED_IMAGE_HW[1]
        and EXPECTED_IMAGE_HW[1] % EXPECTED_PATCH_SIZE == 0,
        "the resampled width must equal the patch16 size the backbone is built for",
    )
    _require(
        vision.get("student_resampling") == "bicubic_to_patch_multiple",
        "the Student closes the patch16 remainder by bicubic resample, never by zero padding",
    )

    # The Teacher runs on the futures at the resolution they were generated in.
    # 464x832 tiles exactly under patch16, so no pixel is invented.  Its 29x52 grid
    # is deliberately coarser than the Student's 42x72: DINOv3's RoPE divides
    # coordinates by H and W separately (normalize_coords='separate') and maps them
    # to [-1, 1], so a token's rotation depends on where it sits in the frame rather
    # than on how many tokens the frame was cut into.  That is what keeps target and
    # scene tokens commensurable without upscaling the futures by ~1.4x.
    future_target = config["future_target"]
    _require(
        future_target.get("teacher_image_hw") == EXPECTED_TEACHER_IMAGE_HW,
        f"the Teacher must encode futures at their native {EXPECTED_TEACHER_IMAGE_HW}, "
        f"got {future_target.get('teacher_image_hw')}",
    )
    _require(
        future_target.get("teacher_grid_hw") == EXPECTED_TEACHER_GRID_HW,
        f"native 464x832 tiles into {EXPECTED_TEACHER_GRID_HW} under patch16, "
        f"got {future_target.get('teacher_grid_hw')}",
    )
    _require(
        all(side % EXPECTED_PATCH_SIZE == 0 for side in EXPECTED_TEACHER_IMAGE_HW),
        "the native future resolution must tile exactly under patch16, else padding would be required",
    )
    _require(
        future_target.get("teacher_resize_filter") == "none_native_resolution_no_interpolation",
        "the Teacher must not resample the generated futures",
    )
    _require(
        future_target.get("rope_normalizes_coords_per_axis") is True,
        "cross-grid comparability relies on DINOv3 RoPE normalizing coordinates per axis",
    )
    _require(
        future_target.get("teacher_checkpoint") == vision.get("checkpoint"),
        "Teacher and Student must share one DINOv3 checkpoint for the target to be comparable",
    )
    # The readout that replaces the ordered-2x2 quadrant pool, and the reason it has
    # to be pinned: it is parameter-free by design, so the target is a constant
    # function of the frozen backbone.  A learned extractor would move the target
    # while the Student chases it, which is how a predict/target pair collapses.
    _require(
        future_target.get("spatial_readout") == "dinov3_register_query_cosine_attention_over_patch_grid",
        "the register target must be read out by DINOv3's own prefix queries over the patch grid",
    )
    _require(
        future_target.get("query_is_parameter_free") is True,
        "the query readout must have no parameters, or the target moves while the Student chases it",
    )
    # The per-frame query count is the number that made the readout work.  CLS alone
    # reaches a standardized separation of 0.211 and the four registers read verbatim
    # 0.194, while the same four used as *queries* in the cosine attention reach 0.458
    # and lift the target-distance/trajectory-distance correlation from 0.684 to 0.771.
    # Four is DINOv3's own prefix count at this resolution, so "more tokens" means the
    # Teacher's queries rather than an invented split of the grid.
    _require(
        future_target.get("registers_per_frame") == REGISTER_QUERIES_PER_FRAME,
        f"each future frame is read by DINOv3's {REGISTER_QUERIES_PER_FRAME} prefix queries, "
        f"got {future_target.get('registers_per_frame')}",
    )
    _require(
        future_target.get("registers_per_gt") == EXPECTED_REGISTER_TOKENS,
        f"a GT's register target is {EXPECTED_REGISTER_TOKENS} tokens "
        f"({FUTURE_FRAMES} frames x {REGISTER_QUERIES_PER_FRAME} queries), "
        f"got {future_target.get('registers_per_gt')}",
    )
    _require(
        future_target.get("temporal_pooling") == "none_registers_per_future_frame",
        "future targets are per frame; nothing is pooled across time",
    )
    _require(
        future_target.get("query_temperature") == QUERY_TEMPERATURE,
        f"the query cosine temperature is {QUERY_TEMPERATURE}; it is part of the target's fingerprint "
        f"and cannot be changed without rebuilding the cache",
    )
    _require(
        future_target.get("query_heads") == QUERY_HEADS,
        f"the query readout uses {QUERY_HEADS} heads, which measured better than 1 at every temperature",
    )
    # Uniform attention makes the readout the mean pool it replaced: every downstream
    # number stays finite and 81% of the GT-discriminating variance is gone.  This is
    # the gate that replaced ``compressor_full_rank``, which policed the PCA target's
    # own silent failure -- dimensions past the fitted rank were bitwise zero, so
    # "emit zeros" beat learning.
    _require(
        future_target.get("max_query_attention_entropy") == MAX_QUERY_ATTENTION_ENTROPY,
        f"the cache builder refuses an attention entropy above {MAX_QUERY_ATTENTION_ENTROPY} of uniform",
    )
    _require(
        future_target.get("register_target_dim") == EXPECTED_REGISTER_TARGET_DIM,
        f"the register target is the Teacher's own {EXPECTED_REGISTER_TARGET_DIM}-d feature width, "
        f"got {future_target.get('register_target_dim')}",
    )
    # A plain L1 over raw DINOv3 coordinates is almost entirely the highest-variance
    # dimensions, and those are scene-level rather than trajectory-level, so the loss
    # would spend capacity on the part the Scene tokens already provide.  Per-dimension
    # standardization is what makes the loss reach the whole target; the std floor
    # keeps genuinely degenerate dimensions from being lifted to equal weight.
    _require(
        future_target.get("compression")
        == "dinov3_register_query_cosine_attention_then_train_only_frozen_diagonal_standardization",
        "the target must be standardized per dimension, on the train split only, after the query readout",
    )
    _require(
        future_target.get("target_std_floor") == TARGET_STD_FLOOR,
        f"the standardization divisor is floored at {TARGET_STD_FLOOR} of the largest per-dimension std",
    )
    for legacy in ("target_dim", "whiten_std_floor", "teacher_quadrant_split", "compressor_rank_note"):
        _require(legacy not in future_target, f"{legacy} belonged to the retired quadrant-pool/PCA target")

    model = config["model"]
    _require(
        model.get("token_order") == ["scene", "action", "register"],
        "the DiT self-attention sequence is [scene, action, register], FastWAM's layout",
    )
    _require(
        model.get("token_counts") == [81, 8, EXPECTED_REGISTER_TOKENS],
        f"81 Scene tokens, 8 action tokens, {EXPECTED_REGISTER_TOKENS} register tokens",
    )
    _require(model.get("ego_tokens") == 1, "DrivoR emits exactly one ego token")
    _require(model.get("ego_observable_dim") == 11, "DrivoR's ego_status is pose3 + velocity2 + acceleration2 + command4")
    _require(model.get("trajectory_state_dim") == 3, "DrivoR trajectories are metric [x, y, heading]")
    _require(model.get("trajectory_rescaling") == "none", "DrivoR trajectories are used in metres and radians")
    _require("xy_scale_m" not in model, "the legacy /20 xy rescaling is removed")
    _require(model.get("action_codec") == "drivor_metric_xy_heading", "Action A must be DrivoR-native metric xy/heading")
    # One DiT future query token per cached register target.  The two counts are the
    # same contract read from either end: a mismatch would silently align +0.5s against
    # +1.0s, or query 1 of a frame against query 2's target.
    _require(
        model.get("register_tokens") == future_target.get("registers_per_gt") == EXPECTED_REGISTER_TOKENS,
        f"register_tokens and registers_per_gt must both be {EXPECTED_REGISTER_TOKENS}",
    )
    _require(
        model.get("register_frames") == EXPECTED_REGISTER_FRAMES,
        f"the {EXPECTED_REGISTER_TOKENS} register tokens cover {EXPECTED_REGISTER_FRAMES} future frames, "
        f"got {model.get('register_frames')}",
    )
    # One sequence, one mask.  The future representation is an auxiliary OUTPUT of
    # the Action DiT -- learnable query tokens in the same self-attention sequence,
    # read out of its last layer, which is V-JEPA's predictor shape -- and it is never
    # fed back.  The register reads the action (that is what makes it a function of
    # the trajectory rather than of the scene alone); the action never reads the
    # register, so the predicted future cannot reach the velocity head.  Scene reads
    # only itself, which is what keeps the scorer's K/V untouched by the DiT.
    directed = {
        "future_reads_action": True,
        "action_reads_future": False,
        "future_self_causal": False,
        "scene_in_self_attention": True,
        "scene_written_by_action_dit": False,
        "scene_returned_to_scorer_is_unfused": True,
        "future_is_auxiliary_readout_only": True,
        "scene_embedded_once_used_for_self_and_cross_attention": True,
    }
    for key, expected in directed.items():
        _require(model.get(key) is expected, f"conflicting v6 model flag: {key}={model.get(key)!r}")
    _require(model.get("cross_attention_context") == "scene", "Wan's pretrained cross_attn keeps Scene as its K/V")
    _require(
        model.get("self_attention_mask") == "single_segment_block",
        "one mask for every stage and entry point; a per-stage selector would let one of them silently run a different information flow",
    )
    # Pinned against NvsActionDiT.SEGMENT_VISIBILITY, rows = queries, cols = keys.
    _require(
        model.get("segment_visibility") == [list(row) for row in NvsActionDiT.SEGMENT_VISIBILITY] == [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
        "block lower-triangular: the register reads the action, the action never reads the register",
    )
    _require(
        model.get("rope") == "per_segment_restart_at_zero", "Action keeps Wan's own RoPE 0..7 even with Scene prepended"
    )
    # The register segment is longer than the action segment now, but its *phases* are
    # not: register token ``q + R*f`` is a read of the frame at ``(f+1)*0.5s``, the same
    # instant action token ``f`` predicts, so the phase repeats R times per frame.
    # Counting 0..31 would put one frame's four queries at four different times and
    # stretch the segment past every action position it is supposed to line up with.
    _require(
        model.get("register_rope_phase") == "frame_index_repeated_per_query",
        "the register's RoPE phase must be the frame index, repeated across that frame's queries",
    )
    _require(
        model.get("segment_type_embeddings")
        == ["scene_type_embedding", "action_type_embedding", "register_type_embedding"],
        "overlapping RoPE ranges need a type embedding per segment",
    )
    # Every key below is retired.  The ``*_in_stage{1,2}`` flags and
    # ``attention_stage_of`` existed only to select between two masks; ``future_fuse``
    # was an explicit channel from the predicted future into the action stream and the
    # EMA teacher only ever existed to feed it.  Asserting their absence is what stops
    # a yaml edit from re-declaring an information flow no code runs.
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
        _require(retired not in model, f"{retired} belonged to the removed per-stage mask / Register -> Action fusion")
    _require("group_allow" not in model, "the legacy 3-segment GROUP_ALLOW table is replaced by the single segment table")
    # theta is a rectified-flow denoiser, so training must draw a random timestep
    # per scene per forward.  The deployed path is still a single forward at
    # sigma = 1 (action_sampling_loops: 0) -- the random time is a training-time
    # property only, and the two flags together pin that split.
    _require(
        model.get("random_flow_time_training") is True,
        "Stage 2 must sample a random flow timestep per forward",
    )
    _require(
        model.get("action_sampling_loops") == 0,
        "one forward must yield a metric trajectory; no rollout at train or eval time",
    )
    _require(
        model.get("future_query_init_inputs") == ["learned_future_queries", "projected_clean_ego"],
        "F0 is the learned queries plus the clean Ego encoding; the trajectory enters by attention, not by projection",
    )
    _require(
        model.get("register_stage1_mode_code") == "zeroed_to_prevent_row_index_shortcut",
        "Stage 1 must not feed a per-row-constant mode code, or the row index identifies the GT",
    )
    _require(
        model.get("stage1_pairing") == "identity_one_register_row_per_gt",
        "Stage 1 pairs register row k with GT row k; the dataset already provides the pairing",
    )
    expected_action_inputs = ["source", "immutable_mode_code", "action_time_embedding", "projected_clean_ego"]
    _require(model.get("action_init_inputs") == expected_action_inputs, "Action A0 must include the projected clean Ego encoding exactly once")
    _require(
        model.get("clean_ego_shared_before_action_projection") is True,
        "Scene and Action must share the clean Ego encoding before Action projection",
    )
    _require(
        model.get("clean_ego_encoding") == "single_linear_11_to_d_model",
        "DrivoR's hist_encoding is a single Linear(11, tf_d_model)",
    )

    action_dit = model.get("action_dit") or {}
    _require(bool(action_dit), "the Action branch must declare its Wan ActionDiT geometry")
    _require(action_dit.get("hidden_dim") == model.get("d_model"), "the ActionDiT hidden_dim is the NVS d_model")
    _require(action_dit.get("action_dim") == 3, "the ActionDiT predicts the metric residual [dx, dy, dheading]")
    _require(action_dit.get("freq_dim") == model.get("d_model"), "the Wan time_embedding input is the NVS condition vector")
    _require(action_dit.get("qk_rmsnorm_before_rope") is True, "Wan normalizes q/k with RMSNorm before RoPE")
    _require(action_dit.get("cross_attention_rope") is False, "Wan cross-attention carries no RoPE")
    _require(action_dit.get("modulation_segments_per_block") == 6, "Wan blocks modulate six segments")
    _require(
        action_dit.get("future_tokens") == EXPECTED_REGISTER_TOKENS,
        "the DiT carries one future query token per register target",
    )
    _require(
        action_dit.get("register_target_dim") == future_target.get("register_target_dim"),
        "the future head must emit the register target's width, which is what the cache holds",
    )
    # Deliberately NOT equal, and the assertion is here because the two used to be the
    # same number: someone "simplifying" this back into one key would widen the whole
    # DiT to 1024, which then fails Wan's own num_heads * attn_head_dim == hidden_dim
    # check (4 * 64 == 256) only after the payload has been loaded.
    _require(
        action_dit.get("register_target_dim") != action_dit.get("hidden_dim"),
        "the future head's output is the Teacher's feature width, not hidden_dim; "
        "if these are equal the register target has silently been folded back into d_model",
    )
    _require(
        action_dit.get("num_heads") * action_dit.get("attn_head_dim") == action_dit.get("hidden_dim"),
        "Wan's payload requires num_heads * attn_head_dim == hidden_dim",
    )
    _require(
        action_dit.get("future_head") == ["layernorm_no_affine", "linear_hidden_to_target"],
        "the future head normalizes then projects, so its output is comparable to the standardized target",
    )
    # This one is asserted rather than merely documented because it silently disagreed
    # with the code for a whole revision: the yaml still said 0.02 after
    # FUTURE_QUERY_SCALE went to 0.5, and nothing read the key.  It matters now that the
    # register holds several queries per frame sharing one RoPE phase -- that init is the
    # only thing separating them at step 0.
    _require(
        action_dit.get("future_query_init") == f"randn_scaled_{FUTURE_QUERY_SCALE}_not_exact_zero",
        f"future_queries are initialised at wan_dit.FUTURE_QUERY_SCALE = {FUTURE_QUERY_SCALE}, "
        f"got {action_dit.get('future_query_init')!r}",
    )
    wan = model.get("wan_backbone") or {}
    _require(wan.get("interpolation") == "sequential_1d_linear_align_corners_true", "SimWAM interpolates the last dim with align_corners=True")
    _require(wan.get("alpha_scaling") is True, "SimWAM scales interpolated weights by sqrt(src_last/target_last)")
    skip_prefixes = wan.get("skip_prefixes") or []
    for required in ("action_encoder.", "head.", "future_queries", "future_head."):
        _require(required in skip_prefixes, f"{required} has no Wan counterpart and must stay randomly initialized")

    scorer = config["scorer"]
    _require(
        scorer.get("input_detach") == ["trajectory", "predicted_register"],
        "DrivoR detaches only the proposals before scoring (drivor_model.py:247)",
    )
    _require(scorer.get("scene_kv_detached") is False, "DrivoR feeds scene_features in attached, so the score loss shapes the Scene")
    _require(scorer.get("ego_pool") == "last_scene_token", "DrivoR adds its single ego token to the scorer output")
    _require(scorer.get("predicted_register_is_input") is False, "strict DrivoR score path cannot consume predicted future registers")
    expected_score_gradients = {
        "updates_generator": False,
        "updates_trajectory_branch": False,
        "updates_register": False,
        "updates_scene_lora": True,
    }
    for key, expected in expected_score_gradients.items():
        _require(scorer.get(key) is expected, f"DrivoR-compatible scorer gradient flag mismatch: {key}")

    # Stage 2's future term is dense: every candidate is supervised by the GT it is
    # nearest to, not only by the K a Hungarian match happened to pick.  A matched-only
    # future loss left 60+ of the 64 register rows with zero gradient, and those rows
    # now share the DiT trunk with the trajectory, so an untrained row is an
    # untrained trajectory too.
    _require(
        config["loss"].get("stage2_future_pairing") == "nearest_gt_dense_all_candidates",
        "Stage 2's future loss must reach every candidate, not just the matched ones",
    )
    _require(config["loss"].get("supervise_unmatched_future") is True, "the dense future term is what supervises the unmatched candidates")
    _require(
        config["future_target"].get("times_s") == [0.5 * step for step in range(1, 9)],
        "Future Register targets sit on DrivoR's 0.5s eight-pose grid",
    )
    # Both stages run the same mask, so neither declares one; what they must not do
    # is name the removed fusion in a train list, which would silently imply a
    # Register -> Action channel the model no longer has.
    for stage in ("stage1", "stage2"):
        _require("attention_mask" not in config[stage], f"{stage} must not declare an attention_mask; there is exactly one")
        for retired in ("future_fuse", "ema_theta"):
            _require(retired not in config[stage].get("train", []), f"{stage} still trains {retired}, which was removed")
            _require(retired not in config[stage].get("freeze", []), f"{stage} still freezes {retired}, which no longer exists")
    # The target definition is frozen in both stages and has to be *named* as frozen:
    # the query readout has no parameters and the standardizer is fit once on the train
    # split, so a trainable target would mean the Student chases a moving goalpost --
    # the collapse the parameter-free design exists to prevent.  Asserted rather than
    # left to the yaml because the key was renamed from ``target_compressor``, and a
    # list nobody reads would not notice the old name surviving the rename.
    for stage in ("stage1", "stage2"):
        frozen = config[stage].get("freeze", [])
        _require(
            "target_standardizer" in frozen,
            f"{stage} must freeze the target standardizer; the target is a fixed function of the frozen Teacher",
        )
        _require(
            "target_compressor" not in frozen and "target_compressor" not in config[stage].get("train", []),
            f"{stage} still names target_compressor, which belonged to the retired PCA target",
        )
    return config
