"""Strict v6 configuration loader; conflicting legacy direction flags fail fast."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


EXPECTED_GROUP_ALLOW = [[1, 0, 0], [1, 1, 1], [1, 0, 1]]
# DrivoR's scene perception: patch16 DINOv3 at its own training resolution, with
# the prepended scene queries counted as RoPE prefix tokens.
EXPECTED_PATCH_SIZE = 16
# DrivoR's official image_size is [1148, 672] (W, H).  1148 = 14*82 for its
# patch14 DINOv2, but patch16 leaves a 12 px remainder, which is closed with four
# zero columns rather than by resampling the width up.
EXPECTED_DRIVOR_IMAGE_HW = [672, 1148]
EXPECTED_RIGHT_ZERO_PAD_PX = 4
EXPECTED_IMAGE_HW = [672, 1152]
EXPECTED_PATCH_GRID = [42, 72]
# The generated future videos are natively 464x832, which patch16 tiles exactly,
# so the Teacher needs no resize at all.
EXPECTED_TEACHER_IMAGE_HW = [464, 832]
EXPECTED_TEACHER_GRID_HW = [29, 52]
EXPECTED_NATIVE_PREFIX_TOKENS = 5
EXPECTED_NUM_SCENE_TOKENS = 16
EXPECTED_FRAMES = 5
EXPECTED_LORA_RANK = 32


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_v6_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
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
    # patch16 cannot tile 1148, so the remainder becomes zero columns: F.pad
    # synthesizes no pixel value, which is the whole difference from resampling.
    _require(
        vision.get("drivor_image_hw") == EXPECTED_DRIVOR_IMAGE_HW,
        f"history/current frames must be DrivoR's official {EXPECTED_DRIVOR_IMAGE_HW}, "
        f"got {vision.get('drivor_image_hw')}",
    )
    _require(
        vision.get("right_zero_pad_px") == EXPECTED_RIGHT_ZERO_PAD_PX and vision.get("bottom_zero_pad_px") == 0,
        f"patch16 closes DrivoR's 1148 width with exactly {EXPECTED_RIGHT_ZERO_PAD_PX} zero columns, "
        f"got {vision.get('right_zero_pad_px')}",
    )
    _require(
        EXPECTED_DRIVOR_IMAGE_HW[1] + EXPECTED_RIGHT_ZERO_PAD_PX == EXPECTED_IMAGE_HW[1]
        and EXPECTED_IMAGE_HW[1] % EXPECTED_PATCH_SIZE == 0,
        "the zero-padded width must equal the patch16 size the backbone is built for",
    )
    _require(
        vision.get("student_resampling") == "forbidden_zero_pad_only",
        "the Student must not interpolate; DrivoR's resize happens in the loader and the remainder is zero-padded",
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

    model = config["model"]
    _require(model.get("token_order") == ["scene", "action", "register"], "v6 token order must be [scene, action, register]")
    _require(model.get("group_allow") == EXPECTED_GROUP_ALLOW, "v6 group mask is [[1,0,0],[1,1,1],[1,0,1]]")
    _require(model.get("ego_tokens") == 1, "DrivoR emits exactly one ego token")
    _require(model.get("ego_observable_dim") == 11, "DrivoR's ego_status is pose3 + velocity2 + acceleration2 + command4")
    _require(model.get("trajectory_state_dim") == 3, "DrivoR trajectories are metric [x, y, heading]")
    _require(model.get("trajectory_rescaling") == "none", "DrivoR trajectories are used in metres and radians")
    _require("xy_scale_m" not in model, "the legacy /20 xy rescaling is removed")
    _require(model.get("action_codec") == "drivor_metric_xy_heading", "Action A must be DrivoR-native metric xy/heading")
    forbidden = {
        # The e2e Stage 1 conditions the Register on the metric trajectory, which
        # the shared v6 spec explicitly withdrew (its "attack 5").  That withdrawal
        # no longer binds anything: this loader governs nvs_v6_wan_e2e only, whose
        # Stage 1 is a trajectory-conditioned future representation and cannot work
        # without the trajectory as input.
        "physical_trajectory_to_register_readout": True,
        # The directed attention contract is untouched: R still never reads A.
        "register_reads_action": False,
        "action_reads_register": True,
        "register_self_causal": False,
    }
    for key, expected in forbidden.items():
        _require(model.get(key) == expected, f"conflicting v6 model flag: {key}={model.get(key)!r}")
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
        model.get("register_init_inputs") == ["learned_time_queries", "metric_source_trajectory", "immutable_mode_code"],
        "R0 must carry the metric trajectory; the mode code is optional and is dropped at Stage 1",
    )
    _require(
        model.get("register_stage1_mode_code") == "omitted_to_prevent_row_index_shortcut",
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
    wan = model.get("wan_backbone") or {}
    _require(wan.get("interpolation") == "sequential_1d_linear_align_corners_true", "SimWAM interpolates the last dim with align_corners=True")
    _require(wan.get("alpha_scaling") is True, "SimWAM scales interpolated weights by sqrt(src_last/target_last)")
    _require("action_encoder." in (wan.get("skip_prefixes") or []) and "head." in (wan.get("skip_prefixes") or []), "SimWAM's skip prefixes must stay randomly initialized")

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

    _require(config["loss"].get("independent_future_matching") is False, "Action and Future must share one Hungarian assignment")
    _require(
        config["future_target"].get("times_s") == [0.5 * step for step in range(1, 9)],
        "Future Register targets sit on DrivoR's 0.5s eight-pose grid",
    )
    return config
