"""Half-train selection, untouched validation, real loader/geometry seams, launch routing."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import pickle
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from hydra import compose, initialize_config_dir
from hydra._internal.utils import get_args_parser
from omegaconf import OmegaConf
from pyquaternion import Quaternion
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_navtrain_half", ROOT / "tools/build_navtrain_half.py")
half = importlib.util.module_from_spec(spec)
spec.loader.exec_module(half)


def test_per_log_alternation_preserves_rare_logs_exact_total_and_temporal_spacing():
    logs = np.array([f"log{number}" for number in range(1, 8) for _ in range(number)])
    timestamps = np.array([time for number in range(1, 8) for time in range(number - 1, -1, -1)])
    selected = half.log_time_alternating(logs, timestamps)
    assert selected.sum() == (len(logs) + 1) // 2
    assert selected[0]  # The singleton is never erased.
    assert np.array_equal(selected, half.log_time_alternating(logs, timestamps))
    for log in np.unique(logs):
        indices = np.flatnonzero(logs == log)
        picked_times = np.sort(timestamps[indices[selected[indices]]])
        assert len(picked_times) in {len(indices) // 2, (len(indices) + 1) // 2}
        assert len(picked_times) > 0
        assert np.all(np.diff(picked_times) == 2)


def test_writing_filter_only_changes_tokens_and_retains_header(tmp_path):
    source = tmp_path / "source.yaml"
    source.write_text("num_history_frames: 4\n# exact existing list\nlog_names:\n  - 'a'\n  - 'b'\ntokens:\n  - 'a0'\n  - 'a1'\n  - 'b0'\n")
    output = tmp_path / "half.yaml"
    half.write_filter(source, output, ["a0", "b0"])
    prefix = source.read_text().split("tokens:\n")[0]
    assert output.read_text().startswith(prefix)
    split = {"train_logs": ["a"], "val_logs": ["b"]}
    train, val = half.split_tokens(half.read_yaml(output), split, {"a0": "a", "a1": "a", "b0": "b"})
    assert train == ["a0"] and val == ["b0"]


def source_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ImportFrom))
                 and (isinstance(node, ast.FunctionDef) and node.name in names
                      or isinstance(node, ast.ImportFrom) and node.module == "__future__")]
    # Deferred annotations allow the actual loader/geometry bodies to run without
    # constructing maps/images or importing the optional training/GPU stack.
    if not any(isinstance(node, ast.ImportFrom) and node.module == "__future__" for node in functions):
        functions.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(path), "exec"), namespace)


def test_raw_log_geometry_and_scene_ids_match_production_functions(tmp_path):
    raw = []
    for index in range(20):
        q = Quaternion(axis=[0, 0, 1], angle=3.05 + index * .06) * Quaternion(axis=[0, 1, 0], angle=.08)
        raw.append({"token": f"t{index}", "log_name": "logA", "roadblock_ids": [1],
                    "ego2global_translation": np.array([650000.0 + index * 2.1, 4000000.0 + index * .2, 0.0]),
                    "ego2global_rotation": q.elements, "ego_dynamic_state": [4.2, .4, 0, 0],
                    "timestamp": 1000000 + index * 500000, "map_location": "test_city", "vehicle_name": "test_car",
                    "driving_command": [0, 0, 0, 1], "anns": {"gt_names": ["vehicle"]}})
    with (tmp_path / "logA.pkl").open("wb") as handle:
        pickle.dump(raw, handle)
    cfg = {"num_history_frames": 4, "num_future_frames": 10, "frame_interval": 1,
           "has_route": True, "log_names": ["logA"], "tokens": ["t3", "t5", "t6"], "max_scenes": None}
    data = half.load_human_features(tmp_path, cfg, ["t3", "t5"], ["t6"], {token: "logA" for token in cfg["tokens"]})
    namespace = {"pickle": pickle, "tqdm": lambda values, **kw: values}
    source_functions(ROOT / "navsim/common/dataloader.py", {"filter_scenes"}, namespace)
    loader_filter = SimpleNamespace(**cfg, num_frames=14)
    assert set(namespace["filter_scenes"](tmp_path, loader_filter)) == set(cfg["tokens"])
    geometry = {"np": np, "SE2Index": range(3)}
    source_functions(ROOT / "navsim/planning/simulation/planner/pdm_planner/utils/pdm_geometry_utils.py",
                     {"normalize_angle", "convert_absolute_to_relative_se2_array"}, geometry)
    for row, current in enumerate([3, 5]):
        poses = np.array([[*frame["ego2global_translation"][:2], Quaternion(*frame["ego2global_rotation"]).yaw_pitch_roll[0]]
                          for frame in raw[current:current+9]])
        origin = SimpleNamespace(x=poses[0, 0], y=poses[0, 1], heading=poses[0, 2])
        expected = geometry["convert_absolute_to_relative_se2_array"](origin, poses[1:])
        np.testing.assert_allclose(data["trajectories"][row], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(half.quaternion_yaw(np.array([frame["ego2global_rotation"] for frame in raw])),
                               [Quaternion(*frame["ego2global_rotation"]).yaw_pitch_roll[0] for frame in raw], atol=1e-12)


def test_generated_half_scope_and_distribution_pass_for_real_data():
    source = half.read_yaml(ROOT / "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml")
    generated = half.read_yaml(ROOT / "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain_half.yaml")
    split = half.read_yaml(ROOT / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml")
    token_logs = half.read_token_logs(ROOT.parent / "data/metric_cache_navtrain/metadata/metric_cache_navtrain_metadata_node_0.csv")
    train, val = half.split_tokens(source, split, token_logs)
    new_train, new_val = half.split_tokens(generated, split, token_logs)
    assert len(train) == 85109 and len(new_train) == 42555
    assert len(val) == 18179 and new_val == val
    assert {key: value for key, value in source.items() if key != "tokens"} == {key: value for key, value in generated.items() if key != "tokens"}
    assert {token_logs[token] for token in train} == {token_logs[token] for token in new_train}
    audit = json.loads((ROOT / "exp/navtrain_half/distribution_summary.json").read_text())
    assert all(audit["checks"].values())


def test_half_primary_config_uses_token_filter_and_actual_scheduler_size():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "navsim/planning/script/config/training")):
        cfg = compose(config_name="half_training", overrides=["train_test_split_synthetic/scene_filter=all_scenes"])
    assert cfg.cache_path is None and cfg.use_cache_without_dataset is False
    assert cfg.runtime.scheduler_dataset_size_from_data
    assert cfg.agent.scheduler_args.dataset_size == 42555
    assert len(cfg.train_test_split.scene_filter.tokens) == 42555 + 18179
    assert cfg.runtime.get("training_data_scope", "train_only") != "trainval"
    split = half.read_yaml(ROOT / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml")
    assert list(cfg.train_logs) == split["train_logs"] and list(cfg.val_logs) == split["val_logs"]


@pytest.mark.parametrize("launcher,agent", [
    ("run_drivor_training.sh", "drivoR"),
    ("run_drivor_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_training.sh", "drivoR_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register"),
    ("run_drivor_query_single_gt_training.sh", "drivoR_query_dinov3_vitl_single_gt"),
    ("run_drivor_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_human_only_training.sh", "drivoR_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_human_only"),
    ("run_drivor_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_scorer_human_only_training.sh", "drivoR_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_scorer_human_only"),
])
def test_half_launcher_preserves_model_but_forces_half_train_split(tmp_path, launcher, agent):
    fake = tmp_path / "python"
    fake.write_text("#!/usr/bin/python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    fake.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if not key.startswith("DRIVOR_")}
    env.update(DRIVOR_PYTHON=str(fake), DRIVOR_HALF_LAUNCHER=launcher)
    result = subprocess.run(["bash", str(ROOT / "scripts/training/run_drivor_navtrain_half_training.sh"),
                             "train_test_split=navtrain", "use_cache_without_dataset=true",
                             "++runtime.scheduler_dataset_size_from_data=false", "--cfg", "job"],
                            capture_output=True, text=True, env=env, timeout=15)
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout)
    assert str(ROOT / "navsim/planning/script/run_training.py") in args
    assert "--config-name=half_training" in args
    assert [arg for arg in args if arg.startswith("agent=")][-1] == "agent=" + agent
    assert [arg for arg in args if arg.startswith("train_test_split=")][-1] == "train_test_split=navtrain_half"
    assert [arg for arg in args if arg.startswith("use_cache_without_dataset=")][-1] == "use_cache_without_dataset=false"
    assert "++runtime.scheduler_dataset_size_from_data=true" in args
    assert "agent.scheduler_args.dataset_size=42555" in args
    assert "agent.num_gpus=64" in args
    # This checks the real Hydra parser, including an option after an existing
    # option with the same name; merely observing argv misses parse failures.
    entry_index = args.index(str(ROOT / "navsim/planning/script/run_training.py"))
    parsed = get_args_parser().parse_args(args[entry_index+1:])
    assert parsed.config_name == "half_training" and parsed.cfg == "job"
    assert [arg for arg in parsed.overrides if arg.startswith("train_test_split=")][-1] == "train_test_split=navtrain_half"
    assert parsed.overrides[-1] == "agent.scheduler_args.dataset_size=42555"
    assert "++runtime.scheduler_dataset_size_from_data=false" not in parsed.overrides
    if "human_only" in launcher:
        assert "--config-name=default_training" in args  # Inner wrapper resets env.
        assert args[-1] == "--config-name=half_training"  # Final CLI option wins.


@pytest.mark.parametrize("launcher", [
    "run_drivor_query_single_gt_full_training.sh",
    "run_drivor_query_single_gt_train_eval.sh",
])
def test_half_launcher_rejects_full_and_train_eval_before_starting_a_model(tmp_path, launcher):
    fake = tmp_path / "python"
    fake.write_text("#!/usr/bin/python3\nraise RuntimeError('model launcher must not execute')\n")
    fake.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if not key.startswith("DRIVOR_")}
    env.update(DRIVOR_PYTHON=str(fake), DRIVOR_HALF_LAUNCHER=launcher)
    result = subprocess.run(["bash", str(ROOT / "scripts/training/run_drivor_navtrain_half_training.sh")],
                            capture_output=True, text=True, env=env, timeout=15)
    assert result.returncode == 2
    assert "not full or train_eval" in result.stderr and result.stdout == ""
