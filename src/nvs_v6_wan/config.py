"""Strict v6 configuration loader; conflicting legacy direction flags fail fast."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


EXPECTED_GROUP_ALLOW = [[1, 0, 0], [1, 1, 1], [1, 0, 1]]


def load_v6_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    if config.get("spec_id") != "navsim_future_register_R_to_A_v6":
        raise ValueError("only navsim_future_register_R_to_A_v6 is supported")
    model = config["model"]
    if model.get("token_order") != ["scene", "action", "register"]:
        raise ValueError("v6 token order must be [scene, action, register]")
    if model.get("group_allow") != EXPECTED_GROUP_ALLOW:
        raise ValueError("v6 group mask is [[1,0,0],[1,1,1],[1,0,1]]")
    forbidden = {
        "physical_trajectory_to_register_readout": False,
        "register_reads_action": False,
        "action_reads_register": True,
        "register_self_causal": False,
        "random_flow_time_training": False,
    }
    for key, expected in forbidden.items():
        if model.get(key) != expected:
            raise ValueError(f"conflicting v6 model flag: {key}={model.get(key)!r}")
    expected_action_inputs = ["source", "immutable_mode_code", "action_time_embedding", "projected_clean_ego"]
    if model.get("action_init_inputs") != expected_action_inputs:
        raise ValueError("Action A0 must include the projected clean Ego encoding exactly once")
    if model.get("clean_ego_shared_before_action_projection") is not True:
        raise ValueError("Scene and Action must share the clean Ego encoding before Action projection")
    scorer = config["scorer"]
    if scorer.get("input_detach") != ["scene", "trajectory", "predicted_register"]:
        raise ValueError("score supervision must detach every generator-side input")
    if scorer.get("predicted_register_is_input") is not False:
        raise ValueError("strict DrivoR score path cannot consume predicted future registers")
    expected_score_gradients = {
        "updates_generator": False,
        "updates_trajectory_branch": False,
        "updates_register": False,
        "updates_scene_lora": False,
    }
    for key, expected in expected_score_gradients.items():
        if scorer.get(key) is not expected:
            raise ValueError(f"DrivoR-compatible scorer gradient flag mismatch: {key}")
    if config["loss"].get("independent_future_matching") is not False:
        raise ValueError("Action and Future must share one Hungarian assignment")
    return config
