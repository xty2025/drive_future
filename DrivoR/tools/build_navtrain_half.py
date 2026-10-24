#!/usr/bin/env python3
"""Build a train-only half split and audit real human-trajectory distributions.

Reads one trusted local NAVSIM log pickle at a time; no images, metric payloads,
models, CUDA, network, or DLC. The metric-cache CSV is only a token -> log index.
The validation tokens and both official log lists are preserved in full.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "DrivoR/navsim/planning/script/config"
DEFAULT_OUTPUT = PROJECT_ROOT / "DrivoR/exp/navtrain_half"
FEATURE_NAMES = (
    "endpoint_x_m", "endpoint_y_m", "heading_change_deg", "path_length_m",
    "mean_speed_mps", "initial_speed_mps", "max_lateral_offset_m", "actor_count",
)
CATEGORY_NAMES = ("stationary", "straight", "left", "right")


def read_yaml(path: Path) -> dict:
    with path.open() as handle:
        return yaml.load(handle, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))


def read_token_logs(path: Path) -> dict[str, str]:
    result = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            item = PurePosixPath(row["file_name"])
            token, log = item.parent.name, item.parents[2].name
            if token in result and result[token] != log:
                raise ValueError(f"Token {token} appears in two logs")
            result[token] = log
    return result


def split_tokens(config: dict, split: dict, token_logs: dict[str, str]):
    tokens = config["tokens"]
    if len(tokens) != len(set(tokens)):
        raise ValueError("The source scene filter contains duplicate tokens")
    train_logs, val_logs = set(split["train_logs"]), set(split["val_logs"])
    if train_logs & val_logs:
        raise ValueError("Training and validation log lists overlap")
    allowed_logs = set(config["log_names"])
    train, val = [], []
    for token in tokens:
        log = token_logs[token]
        if log not in allowed_logs:
            raise ValueError(f"Token {token} is outside scene_filter.log_names")
        if log in train_logs:
            train.append(token)
        elif log in val_logs:
            val.append(token)
        else:
            raise ValueError(f"Token {token} is in neither the train nor val logs")
    return train, val


def quaternion_yaw(quaternion: np.ndarray) -> np.ndarray:
    """Exactly pyquaternion.Quaternion(...).yaw_pitch_roll[0], including normalization."""
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2 * (w * z - x * y), 1 - 2 * (y * y + z * z))


def relative_trajectories(global_poses: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Scene.get_future_trajectory(8): current rear axle -> +0.5,...,+4.0 s."""
    origin = global_poses[indices]
    future = global_poses[indices[:, None] + np.arange(1, 9)]
    difference = future - origin[:, None]
    angle = origin[:, 2, None]
    cosine, sine = np.cos(angle), np.sin(angle)
    output = np.empty_like(difference)
    output[..., 0] = difference[..., 0] * cosine + difference[..., 1] * sine
    output[..., 1] = -difference[..., 0] * sine + difference[..., 1] * cosine
    output[..., 2] = np.arctan2(np.sin(difference[..., 2]), np.cos(difference[..., 2]))
    return output


def log_time_alternating(log_names: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Keep every other scene *within each log*, with exact ceil(N/2) total.

    Quotas use only log counts, never future trajectory values. Each quota is
    floor(n/2) or ceil(n/2); singleton logs are retained. Odd-count round-ups are
    spread deterministically over sorted log names, rather than all going to
    the first city/day. Even-count logs alternate their phase between logs.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for index, log in enumerate(log_names.tolist()):
        groups[log].append(index)
    ordered_logs = sorted(groups)
    target = (len(log_names) + 1) // 2
    floor_total = sum(len(indices) // 2 for indices in groups.values())
    round_up = {log for log, indices in groups.items() if len(indices) == 1}
    needed = target - floor_total - len(round_up)
    odd_pool = [log for log in ordered_logs if len(groups[log]) > 1 and len(groups[log]) % 2]
    if not 0 <= needed <= len(odd_pool):
        raise ValueError("Cannot preserve every log with an exactly half-sized split")
    if needed:
        positions = np.floor((np.arange(needed) + 0.5) * len(odd_pool) / needed).astype(int)
        round_up.update(odd_pool[position] for position in positions)
    selected = np.zeros(len(log_names), dtype=bool)
    for log_index, log in enumerate(ordered_logs):
        indices = np.array(groups[log], dtype=np.int64)
        indices = indices[np.argsort(timestamps[indices], kind="stable")]
        phase = (0 if log in round_up else 1) if len(indices) % 2 else log_index % 2
        selected[indices[phase::2]] = True
    if int(selected.sum()) != target:
        raise AssertionError("Half split count does not match its target")
    return selected


def load_human_features(log_dir: Path, config: dict, train: list[str], val: list[str], token_logs: dict[str, str]):
    """Stream one raw log at a time, reproducing filter_scenes geometry and filters."""
    rows_by_token = {token: index for index, token in enumerate(train)}
    wanted = set(train) | set(val)
    per_log: dict[str, set[str]] = defaultdict(set)
    for token in wanted:
        per_log[token_logs[token]].add(token)
    available = set(path[:-4] for path in __import__("os").listdir(log_dir) if path.endswith(".pkl"))
    missing_logs = set(per_log) - available
    if missing_logs:
        raise FileNotFoundError(f"Missing NAVSIM log files: {sorted(missing_logs)[:5]}")
    n = len(train)
    trajectories = np.empty((n, 8, 3), dtype=np.float32)
    features = np.empty((n, len(FEATURE_NAMES)), dtype=np.float64)
    timestamps = np.empty(n, dtype=np.int64)
    cities = np.empty(n, dtype="U40")
    commands = np.empty(n, dtype="U16")
    vehicles = np.empty(n, dtype="U16")
    categories = np.empty(n, dtype="U16")
    found_train: set[str] = set()
    found_val: set[str] = set()
    val_tokens = set(val)
    history, future, interval = config["num_history_frames"], config["num_future_frames"], config["frame_interval"]
    if future < 8:
        raise ValueError("Need at least eight future poses for the human 4-second target")
    before = time.monotonic()
    for log_number, log_name in enumerate(sorted(per_log), start=1):
        # Trusted local NAVSIM dict/NumPy metadata, not a model checkpoint.
        with (log_dir / f"{log_name}.pkl").open("rb") as handle:
            raw = pickle.load(handle)
        indices, tokens = [], []
        for start in range(0, len(raw), interval):
            if start + history + future > len(raw):
                continue
            current = start + history - 1
            frame = raw[current]
            token = frame["token"]
            if token not in per_log[log_name]:
                continue
            if config["has_route"] and len(frame["roadblock_ids"]) == 0:
                continue
            if frame["log_name"] != log_name:
                raise ValueError(f"Unexpected log name at {token}")
            if token in val_tokens:
                found_val.add(token)
            else:
                if token in found_train:
                    raise ValueError(f"Duplicate train token {token} in raw data")
                found_train.add(token)
                indices.append(current)
                tokens.append(token)
        if indices:
            indices_array = np.array(indices, dtype=np.int64)
            translation = np.array([frame["ego2global_translation"][:2] for frame in raw], dtype=np.float64)
            quaternion = np.array([frame["ego2global_rotation"] for frame in raw], dtype=np.float64)
            poses = np.column_stack((translation, quaternion_yaw(quaternion)))
            relative = relative_trajectories(poses, indices_array)
            dest = np.array([rows_by_token[token] for token in tokens], dtype=np.int64)
            trajectories[dest] = relative
            with_current = np.concatenate((np.zeros((len(dest), 1, 2)), relative[..., :2]), axis=1)
            path_length = np.linalg.norm(np.diff(with_current, axis=1), axis=-1).sum(axis=-1)
            initial_speed = np.array([np.linalg.norm(raw[index]["ego_dynamic_state"][:2]) for index in indices])
            actor_count = np.array([len(raw[index]["anns"]["gt_names"]) for index in indices])
            heading_deg = np.rad2deg(relative[:, -1, 2])
            features[dest] = np.column_stack((relative[:, -1, 0], relative[:, -1, 1], heading_deg,
                                              path_length, path_length / 4.0, initial_speed,
                                              np.abs(relative[..., 1]).max(axis=-1), actor_count))
            categories[dest] = np.where(path_length < 1.0, "stationary",
                                        np.where(heading_deg > 15.0, "left", np.where(heading_deg < -15.0, "right", "straight")))
            for destination, index in zip(dest, indices):
                frame = raw[index]
                timestamps[destination] = int(frame["timestamp"])
                cities[destination] = str(frame["map_location"])
                command = np.asarray(frame["driving_command"]).astype(int)
                commands[destination] = "".join(str(int(value)) for value in command)
                vehicles[destination] = str(frame["vehicle_name"])
        del raw
        if log_number % 25 == 0 or log_number == len(per_log):
            print(json.dumps({"event": "logs", "done": log_number, "total": len(per_log),
                              "train_seen": len(found_train), "val_seen": len(found_val),
                              "seconds": round(time.monotonic() - before, 1)}), flush=True)
    if found_train != set(train) or found_val != set(val):
        raise ValueError(f"Raw SceneFilter mismatch: missing train={len(set(train) - found_train)}, val={len(set(val) - found_val)}")
    if not np.isfinite(features).all() or not np.isfinite(trajectories).all():
        raise ValueError("Non-finite human trajectory features")
    return {"tokens": np.array(train), "log_names": np.array([token_logs[token] for token in train]),
            "timestamps": timestamps, "cities": cities, "commands": commands, "vehicles": vehicles,
            "categories": categories, "trajectories": trajectories, "features": features}


def ks_distance(full: np.ndarray, subset: np.ndarray) -> float:
    """Descriptive empirical CDF distance; no independent-sample p-value claim."""
    a, b = np.sort(full), np.sort(subset)
    points = np.unique(a)
    return float(np.max(np.abs(np.searchsorted(a, points, side="right") / len(a)
                               - np.searchsorted(b, points, side="right") / len(b))))


def categorical_comparison(full: np.ndarray, selected: np.ndarray) -> dict:
    counts, half = Counter(full.tolist()), Counter(full[selected].tolist())
    rows = [{"name": key, "full_count": counts[key], "half_count": half[key],
             "full_percent": 100 * counts[key] / len(full),
             "half_percent": 100 * half[key] / int(selected.sum()),
             "delta_percentage_points": 100 * (half[key] / int(selected.sum()) - counts[key] / len(full))}
            for key in sorted(counts)]
    return {"rows": rows, "total_variation": sum(abs(row["delta_percentage_points"]) for row in rows) / 200,
            "max_abs_percentage_points": max(abs(row["delta_percentage_points"]) for row in rows),
            "full_categories": len(counts), "half_categories": len(half),
            "missing_categories": sorted(set(counts) - set(half))}


def summarize(data: dict[str, np.ndarray], selected: np.ndarray) -> dict:
    scalars = {}
    for index, name in enumerate(FEATURE_NAMES):
        full, half = data["features"][:, index], data["features"][selected, index]
        edges = np.unique(np.quantile(full, np.linspace(0, 1, 21)))
        if len(edges) == 1:
            edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
        full_hist = np.histogram(full, bins=edges)[0] / len(full)
        half_hist = np.histogram(half, bins=edges)[0] / len(half)
        scalars[name] = {"full_mean": float(np.mean(full)), "half_mean": float(np.mean(half)),
                         "full_std": float(np.std(full)), "half_std": float(np.std(half)),
                         "full_quantiles_0_1_25_50_75_99_100": np.quantile(full, [0, .01, .25, .5, .75, .99, 1]).tolist(),
                         "half_quantiles_0_1_25_50_75_99_100": np.quantile(half, [0, .01, .25, .5, .75, .99, 1]).tolist(),
                         "ks_distance": ks_distance(full, half),
                         "histogram_total_variation": float(np.abs(full_hist - half_hist).sum() / 2),
                         "histogram_edges": edges.tolist(), "full_histogram": full_hist.tolist(), "half_histogram": half_hist.tolist()}
    categories = {key: categorical_comparison(data[key], selected)
                  for key in ["log_names", "cities", "commands", "vehicles", "categories"]}
    return {"n_full": len(selected), "n_half": int(selected.sum()), "scalars": scalars, "categorical": categories,
            "max_scalar_ks": max(value["ks_distance"] for value in scalars.values()),
            "max_scalar_histogram_tv": max(value["histogram_total_variation"] for value in scalars.values())}


def write_filter(source: Path, target: Path, keep_tokens: list[str]) -> None:
    """Keep the original header/log_names text byte-for-byte; replace tokens only."""
    lines = source.read_text().splitlines(keepends=True)
    token_line = next(index for index, line in enumerate(lines) if line.startswith("tokens:"))
    prefix = "".join(lines[:token_line])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(prefix + "# Half of training scenes; all validation scenes retained.\n"
                      + "# Rebuild and audit: DrivoR/tools/build_navtrain_half.py\n"
                      + "tokens:\n" + "".join(f"  - '{token}'\n" for token in keep_tokens))


def write_csvs(output: Path, data: dict, half: np.ndarray, yaml_half: np.ndarray, summary: dict) -> None:
    with (output / "scene_statistics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["token", "log_name", "timestamp_us", "selected_half", "yaml_order_half", "city", "command", "vehicle", "trajectory_category", *FEATURE_NAMES])
        for index, token in enumerate(data["tokens"]):
            writer.writerow([token, data["log_names"][index], data["timestamps"][index], int(half[index]), int(yaml_half[index]),
                             data["cities"][index], data["commands"][index], data["vehicles"][index], data["categories"][index],
                             *data["features"][index].tolist()])
    with (output / "log_distribution.csv").open("w", newline="") as handle:
        rows = summary["categorical"]["log_names"]["rows"]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trajectory_plotter():
    import importlib.util
    path = Path(__file__).with_name('plot_navtrain_half_trajectories.py')
    spec = importlib.util.spec_from_file_location('navtrain_half_trajectory_plotter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_figures(output: Path, data: dict, half: np.ndarray, summary: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    full_color, half_color = "#2878B5", "#E87935"
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    labels = {"path_length_m": "4-second path length (m)", "heading_change_deg": "Final heading change (degrees)",
              "endpoint_y_m": "Final lateral displacement (m)", "initial_speed_mps": "Current speed (m/s)"}
    for ax, name in zip(axes.flat[:4], labels):
        i = FEATURE_NAMES.index(name)
        values = data["features"][:, i]
        left, right = np.quantile(values, [.001, .999])
        edges = np.linspace(left, right, 61)
        for selected, color, label in [(np.ones(len(half), bool), full_color, "All train (85,109)"),
                                        (half, half_color, "Half train (42,555)")]:
            hist = np.histogram(values[selected], bins=edges)[0] / int(selected.sum())
            ax.stairs(hist * 100, edges, color=color, label=label, linewidth=1.6)
        ax.set(xlabel=labels[name], ylabel="Scenes per bin (%)",
               title=f"KS = {summary['scalars'][name]['ks_distance']:.4f}; display: central 99.8%")
        ax.grid(alpha=.2)
    rows = {row["name"]: row for row in summary["categorical"]["categories"]["rows"]}
    ax = axes.flat[4]
    x = np.arange(len(CATEGORY_NAMES))
    ax.bar(x-.18, [rows[k]["full_percent"] for k in CATEGORY_NAMES], .36, label="All train", color=full_color)
    ax.bar(x+.18, [rows[k]["half_percent"] for k in CATEGORY_NAMES], .36, label="Half train", color=half_color)
    ax.set(xticks=x, xticklabels=CATEGORY_NAMES, ylabel="Scenes (%)", title="Human trajectory categories")
    ax.grid(axis="y", alpha=.2)
    ax = axes.flat[5]
    logs = summary["categorical"]["log_names"]["rows"]
    full_p, half_p = np.array([r["full_percent"] for r in logs]), np.array([r["half_percent"] for r in logs])
    ax.scatter(full_p, half_p, s=9, alpha=.5, color=full_color)
    lim = max(full_p.max(), half_p.max()) * 1.02
    ax.plot([0, lim], [0, lim], color=half_color, linewidth=1)
    ax.set(xlabel="All train: scenes per log (%)", ylabel="Half train: scenes per log (%)",
           title=f"All {len(logs)} train logs kept; TV={summary['categorical']['log_names']['total_variation']:.4f}")
    ax.grid(alpha=.2)
    axes.flat[0].legend(frameon=False)
    fig.suptitle("NAVSIM train-only half split: real human GT, 8 poses at 2 Hz", fontsize=15)
    for extension in ["png", "svg"]:
        fig.savefig(figure_dir / f"distribution_comparison.{extension}", dpi=160)
    plt.close(fig)

    xy = data["trajectories"][..., :2]
    x_edges, y_edges = np.linspace(-5, 85, 91), np.linspace(-45, 45, 91)
    full_hist = np.histogram2d(xy[..., 0].ravel(), xy[..., 1].ravel(), bins=[x_edges, y_edges])[0] / xy.shape[0] / 8
    half_hist = np.histogram2d(xy[half, :, 0].ravel(), xy[half, :, 1].ravel(), bins=[x_edges, y_edges])[0] / int(half.sum()) / 8
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    vmax = max(full_hist.max(), half_hist.max())
    for ax, hist, title in zip(axes[:2], [full_hist, half_hist], ["All train: 85,109 human trajectories", "Half train: 42,555 human trajectories"]):
        im = ax.pcolormesh(x_edges, y_edges, np.maximum(hist.T, 1e-8), cmap="viridis", norm=LogNorm(vmin=1e-5, vmax=vmax))
        ax.set(title=title, xlabel="Forward (m)", ylabel="Left (m)", aspect="equal")
    fig.colorbar(im, ax=list(axes[:2]), label="Probability per 1 m grid (all 8 future points)", shrink=.8)
    difference = (half_hist - full_hist) * 100
    limit = max(float(np.abs(difference).max()), 1e-8)
    im = axes[2].pcolormesh(x_edges, y_edges, difference.T, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    axes[2].set(title="Half minus full (percentage points)", xlabel="Forward (m)", ylabel="Left (m)", aspect="equal")
    fig.colorbar(im, ax=axes[2], label="Difference (percentage points)", shrink=.8)
    fig.suptitle("Trajectory occupancy: same bins, normalized by each split's own size", fontsize=14)
    for extension in ["png", "svg"]:
        fig.savefig(figure_dir / f"trajectory_occupancy.{extension}", dpi=160)
    plt.close(fig)
    direct = trajectory_plotter().render_trajectory_comparisons(data, half, output)
    return ["figures/distribution_comparison.png", "figures/distribution_comparison.svg",
            "figures/trajectory_occupancy.png", "figures/trajectory_occupancy.svg", *direct['figures']]


def write_report(output: Path, audit: dict) -> None:
    comparison, baseline = audit["selected_distribution"], audit["yaml_order_half_distribution"]
    selected_log, baseline_log = comparison["categorical"]["log_names"], baseline["categorical"]["log_names"]
    lines = ["# NAVSIM 半量训练集：抽样与分布检查", "", f"生成时间：{audit['created_at']}。只读取真实 NAVSIM 元数据，未启动模型训练或 DLC。", "",
             "## 数据范围", "", "| 范围 | 原数量 | 新数量 |", "|---|---:|---:|",
             f"| train，参与梯度更新 | {comparison['n_full']:,} | {comparison['n_half']:,} |",
             f"| val，仅验证 | {audit['counts']['val_original']:,} | {audit['counts']['val_retained']:,} |",
             f"| shared scene-filter tokens | {audit['counts']['source_tokens']:,} | {audit['counts']['output_tokens']:,} |", "",
             "只改 scene-filter 的 `tokens`；`log_names`、`train_logs`、`val_logs`、历史/未来帧数、frame_interval 均保持原值。"
             "`run_training.py` 会把 shared tokens 分别与 train/val 日志相交，所以 YAML 中同时保留所有 val tokens；验证集没有减半。", "",
             "## 抽样规则", "", "先按原 train YAML 顺序直接 `[::2]`，再比较按 log 时序隔点。原 tokens 并非按时间排列。"
             "最终采用每个 log 内按真实时间排序后每隔一个场景取一个；偶数场景日志交替取 0/1 起点，奇数日志按 floor/ceil 分配，"
             "单场景日志优先保留，其他 ceil 名额沿排序后的日志均匀分布，精确得到 42,555 条。写回 YAML 时保持原 token 相对顺序。"
             "抽样只使用日志归属和时间，不使用未来轨迹数值调参或筛选。", "",
             "| 检查 | 原 tokens 直接隔项 | 每 log 时序隔项（最终） |", "|---|---:|---:|",
             f"| 保留训练日志数 | {baseline_log['half_categories']}/{baseline_log['full_categories']} | {selected_log['half_categories']}/{selected_log['full_categories']} |",
             f"| 日志分布 TV 距离 | {baseline_log['total_variation']:.6f} | {selected_log['total_variation']:.6f} |",
             f"| 最大连续特征 KS 距离 | {baseline['max_scalar_ks']:.6f} | {comparison['max_scalar_ks']:.6f} |",
             f"| 最大轨迹类别比例变化（百分点） | {baseline['categorical']['categories']['max_abs_percentage_points']:.4f} | {comparison['categorical']['categories']['max_abs_percentage_points']:.4f} |", "",
             "## 真实 human GT 分布", "", "对完整 85,109 个 train 场景逐一读取 4 秒、2 Hz、8 点的原始 human 轨迹。"
             "与 `Scene.get_future_trajectory(8)` 一致：当前帧后轴为原点，先用 pyquaternion 的 yaw 定义得到全局 heading，再做 SE(2) 相对坐标转换和角度归一化。"
             "不是 D8、pseudo-expert 或 metric-cache 候选轨迹。路程包含当前原点至第一个未来点，平均速度=路程/4秒。", "",
             "| 特征 | 全量均值 | 半量均值 | 全量标准差 | 半量标准差 | KS 距离 | 20 分位桶 TV |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in comparison["scalars"].items():
        lines.append(f"| {name} | {row['full_mean']:.4f} | {row['half_mean']:.4f} | {row['full_std']:.4f} | {row['half_std']:.4f} | {row['ks_distance']:.6f} | {row['histogram_total_variation']:.6f} |")
    lines += ["", "KS 是经验累积分布的最大差；TV 是归一化直方图概率差的绝对值之和的一半，均越小越接近。"
              "半量本身是全量的子集，不对这种嵌套样本报告独立两样本检验的 p 值。", "",
              "轨迹粗类别：4 秒路程<1米为 stationary；其他场景按最终 heading >15°/<−15°/其余定义 left/right/straight。"
              "这是用于分布检查的几何分组，不声称它等同于数据集语义标签。", "",
              "| 轨迹类别 | 全量数量 | 半量数量 | 全量占比 | 半量占比 | 差（百分点） |", "|---|---:|---:|---:|---:|---:|"]
    for row in comparison["categorical"]["categories"]["rows"]:
        lines.append(f"| {row['name']} | {row['full_count']} | {row['half_count']} | {row['full_percent']:.4f}% | {row['half_percent']:.4f}% | {row['delta_percentage_points']:+.4f} |")
    lines += ["", "| 场景维度 | 全量类别数 | 半量类别数 | TV 距离 | 最大比例变化（百分点） |", "|---|---:|---:|---:|---:|"]
    for key in ["log_names", "cities", "commands", "vehicles"]:
        row = comparison["categorical"][key]
        lines.append(f"| {key} | {row['full_categories']} | {row['half_categories']} | {row['total_variation']:.6f} | {row['max_abs_percentage_points']:.4f} |")
    lines += ["", "## 可视化", "", "![分布对照](figures/distribution_comparison.png)", "", "![轨迹位置分布](figures/trajectory_occupancy.png)", "",
              "曲线和轨迹热图分别按各自场景数归一化。曲线只展示中心 99.8% 区间，表内统计与 KS 使用所有场景，未删异常尾部。"
              "热图为所有 8 个未来点在同一米制网格的概率分布，不仅是抽几条轨迹。", "",
              "## 检查结论与范围", "", f"**{'通过' if audit['checks']['passed'] else '未通过'}预先设置的分布门槛**：连续特征最大 KS≤0.01，日志 TV≤0.01，"
              "场景城市/指令/车辆与轨迹类别占比最大变化≤0.5个百分点，所有原训练日志至少保留一条，训练精确减半且验证完整保留。"
              "这证明已检查的真实轨迹几何、运动状态和可读场景属性接近；不保证未统计的天气、罕见交互或所有模型最终精度完全不变。", "",
              "每个场景的完整历史和未来序列不变，只减少训练样本起点。因此它不会把一条轨迹内部的 2 Hz 改成 1 Hz。"
              "每个 epoch 的优化器更新次数约减半，25 epoch 总训练预算也约减半；若比较相同更新次数，应单独说明或调 epoch，不能把半量 25 epoch 说成同训练预算。", "",
              "## 复现与启动", "", "从项目根目录执行：", "", "```bash",
              "python DrivoR/tools/build_navtrain_half.py", "",
              "# 原版 DrivoR，半量 train + 完整 val；run_training.py",
              "bash DrivoR/scripts/training/run_drivor_navtrain_half_training.sh", "",
              "# 对已有模型脚本只替换数据范围，例如 A27（仍保留 human + D8 + register）",
              "DRIVOR_HALF_LAUNCHER=run_drivor_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_training.sh \\",
              "  bash DrivoR/scripts/training/run_drivor_navtrain_half_training.sh", "```", "",
              "新入口使用 `run_training.py --config-name=half_training`，显式 `train_test_split=navtrain_half`、"
              "`cache_path=null`、`use_cache_without_dataset=false`，因此不会走忽略 tokens 的 cache-only 分支。"
              "`runtime.scheduler_dataset_size_from_data=true` 让 scheduler 使用实际训练长度 42,555；"
              "不修改全量旧启动命令，不修改 full/trainval 实验。默认仍为 8机×8卡、总 batch64、25轮、BF16训练和每轮FP32验证。", "",
              "文件：`distribution_summary.json`（完整统计和检查）、`scene_statistics.csv`（每个 train token）、"
              "`log_distribution.csv`、`human_trajectory_features.npz`（可重复画图的轻量数组）、`selected_train_tokens.txt`、"
              "`retained_val_tokens.txt`。配置位于 `DrivoR/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain_half.yaml`。"]
    content = "\n".join(lines) + "\n"
    if (output / 'TRAJECTORY_VISUALIZATIONS.md').exists():
        content = content.replace('## 检查结论与范围',
                                  trajectory_plotter().report_section() + '## 检查结论与范围', 1)
    (output / "REPORT.md").write_text(content)


def run(args: argparse.Namespace) -> dict:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config, split = read_yaml(args.source_filter), read_yaml(args.log_split)
    token_logs = read_token_logs(args.metadata)
    train, val = split_tokens(config, split, token_logs)
    print(json.dumps({"event": "scope", "train": len(train), "val": len(val)}), flush=True)
    cache_path = output / "human_trajectory_features.npz"
    if args.reuse_features:
        with np.load(cache_path, allow_pickle=False) as saved:
            data = {key: saved[key] for key in saved.files}
        if data["tokens"].tolist() != train or data["log_names"].tolist() != [token_logs[t] for t in train]:
            raise ValueError("Saved feature tokens/logs differ; rebuild from raw logs")
    else:
        data = load_human_features(args.log_dir, config, train, val, token_logs)
        np.savez_compressed(cache_path, **data)
    half = log_time_alternating(data["log_names"], data["timestamps"])
    yaml_half = np.zeros(len(train), dtype=bool)
    yaml_half[::2] = True
    selected = set(data["tokens"][half].tolist())
    val_set = set(val)
    output_tokens = [token for token in config["tokens"] if token in selected or token in val_set]
    write_filter(args.source_filter, args.output_filter, output_tokens)
    generated = read_yaml(args.output_filter)
    new_train, new_val = split_tokens(generated, split, token_logs)
    unchanged = {key: value for key, value in config.items() if key != "tokens"} == {key: value for key, value in generated.items() if key != "tokens"}
    selected_summary, baseline_summary = summarize(data, half), summarize(data, yaml_half)
    checks = {
        "non_token_config_fields_identical": unchanged,
        "train_count_exact_half_rounded_up": len(new_train) == (len(train) + 1) // 2,
        "val_tokens_identical_in_order": new_val == val,
        "train_val_disjoint": not (set(new_train) & set(new_val)),
        "all_train_logs_retained": not selected_summary["categorical"]["log_names"]["missing_categories"],
        "max_scalar_ks_le_0p01": selected_summary["max_scalar_ks"] <= .01,
        "log_tv_le_0p01": selected_summary["categorical"]["log_names"]["total_variation"] <= .01,
        "semantic_category_max_delta_le_0p5pp": all(selected_summary["categorical"][key]["max_abs_percentage_points"] <= .5
                                                   for key in ["categories", "cities", "commands", "vehicles"]),
    }
    checks["passed"] = all(checks.values())
    audit = {"created_at": datetime.now().astimezone().isoformat(), "method": "log_time_alternating_exact_half",
             "source_filter": str(args.source_filter), "output_filter": str(args.output_filter),
             "log_split": str(args.log_split), "metadata_index": str(args.metadata), "raw_log_dir": str(args.log_dir),
             "counts": {"source_tokens": len(config["tokens"]), "train_original": len(train), "train_retained": len(new_train),
                        "val_original": len(val), "val_retained": len(new_val), "output_tokens": len(output_tokens)},
             "features_reused": args.reuse_features,
             "selected_distribution": selected_summary, "yaml_order_half_distribution": baseline_summary, "checks": checks}
    (output / "selected_train_tokens.txt").write_text("\n".join(new_train) + "\n")
    (output / "retained_val_tokens.txt").write_text("\n".join(new_val) + "\n")
    write_csvs(output, data, half, yaml_half, selected_summary)
    audit["figures"] = make_figures(output, data, half, selected_summary)
    (output / "distribution_summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    write_report(output, audit)
    print(json.dumps({"event": "done", "counts": audit["counts"], "checks": checks,
                      "max_scalar_ks": selected_summary["max_scalar_ks"], "report": str(output / "REPORT.md")}), flush=True)
    if not checks["passed"]:
        raise RuntimeError("Half split failed its distribution checks; see REPORT.md")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-filter", type=Path, default=CONFIG_ROOT / "common/train_test_split/scene_filter/navtrain.yaml")
    parser.add_argument("--log-split", type=Path, default=CONFIG_ROOT / "training/default_train_val_test_log_split.yaml")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data/metric_cache_navtrain/metadata/metric_cache_navtrain_metadata_node_0.csv")
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "data/navsim/navsim_logs/trainval")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-filter", type=Path, default=CONFIG_ROOT / "common/train_test_split/scene_filter/navtrain_half.yaml")
    parser.add_argument("--reuse-features", action="store_true", help="Reuse this output directory's compact feature NPZ to redraw/recheck")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
