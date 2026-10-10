"""GT-overlay visualization for the PASP-DrivoR navtest PDMS evaluation.

For a handful of navtest scenes, plots the 64 dynamic proposals (grey), the GT
ego future (black) and the learned-scorer selected trajectory (blue) in the same
ego-local frame and style as the training-time diagnostic figure, together with
the native PDMS of the selected trajectory. Uses exactly the evaluation forward
path (run_pdm_score.py overrides), so train/test stays consistent.

Usage:
  PYTHONPATH=... python scripts/evaluation/viz_pasp_navtest_gt.py [num_scenes] [out_dir]
"""
import json
import lzma
import pickle
import sys
import time
from pathlib import Path

import hydra
import matplotlib
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from navsim.agents.drivoR.drivor_features import DrivoRTargetBuilder
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "navsim/planning/script/config/pdm_scoring"
CKPT = REPO / "outputs/pasp_drivor_dpo_8gpu_50ep_fixed/lightning_logs/version_1/checkpoints/last.ckpt"
METRIC_CACHE = Path("/mnt/zhaozc_workspace/project/FastWAM/data/metric_cache_navtest")
OVERRIDES = [
    "train_test_split=navtest",
    f"agent.checkpoint_path={CKPT}",
    "agent.scheduler_args.num_epochs=1",
    "agent.batch_size=1",
    f"metric_cache_path={METRIC_CACHE}",
    "navsim_log_path=/mnt/data/navsim/test_navsim_logs/test",
    "sensor_blobs_path=/mnt/data/navsim/test_sensor_blobs/test",
    "output_dir=/tmp/pasp_viz",
]


def plot_scene(ax, lateral_ax, proposals, gt, selected, sel_idx, title):
    axes = [ax] if lateral_ax is ax else [ax, lateral_ax]
    for traj in proposals:
        for target_ax in axes:
            target_ax.plot(traj[:, 0], traj[:, 1], color="#9ca3af", alpha=0.22, linewidth=0.8)
    for target_ax in axes:
        # Labels only on the primary axis so a shared axis cannot duplicate handles.
        primary = target_ax is ax
        target_ax.plot(gt[:, 0], gt[:, 1], "o-", color="#111827",
                       label="GT" if primary else None, linewidth=2.6, markersize=4)
        target_ax.plot(selected[:, 0], selected[:, 1], "o-", color="#2563eb",
                       label=f"selected (slot {sel_idx})" if primary else None,
                       linewidth=2.2, markersize=4)
        target_ax.scatter([0.0], [0.0], marker="^", s=80, color="#059669",
                          label="ego origin" if primary else None, zorder=5)
        target_ax.axhline(0.0, color="0.4", linewidth=0.6)
        target_ax.axvline(0.0, color="0.4", linewidth=0.6)
        target_ax.set_xlabel("longitudinal x [m]")
        target_ax.set_ylabel("lateral y [m]")
        target_ax.grid(alpha=0.2)
    ax.axis("equal")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="best", frameon=False, fontsize=7)


def main() -> None:
    num_scenes = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "outputs/pasp_navtest_pdms_step22150/viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg: DictConfig = hydra.compose(config_name="default_run_pdm_score", overrides=OVERRIDES)

    agent = instantiate(cfg.agent)
    agent.initialize()
    agent.eval()  # deterministic proposals/selection (dropout off) for the figures
    device = next(agent.parameters()).device
    simulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=agent.get_sensor_config(),
    )
    target_builder = DrivoRTargetBuilder(config=agent._config)
    feature_builders = agent.get_feature_builders()

    cached = set(metric_cache_loader.tokens)
    tokens = [t for t in scene_loader.tokens if t in cached]
    stride = max(1, len(tokens) // num_scenes)
    picked = tokens[::stride][:num_scenes]

    summary = []
    for token in picked:
        t0 = time.time()
        scene = scene_loader.get_scene_from_token(token)
        agent_input = scene_loader.get_agent_input_from_token(token)
        features = {}
        for builder in feature_builders:
            features.update(builder.compute_features(agent_input))
        features = {k: v.unsqueeze(0).to(device) for k, v in features.items()}
        with torch.no_grad():
            pred = agent.forward(features)
        proposals = pred["proposals"][0].detach().cpu()
        scores = pred["pdm_score"][0].detach().cpu()
        sel_idx = int(scores.argmax())
        selected = proposals[sel_idx]
        gt = target_builder.compute_targets(scene)["trajectory"].detach().cpu()

        with lzma.open(metric_cache_loader.metric_cache_paths[token], "rb") as f:
            metric_cache = pickle.load(f)
        result = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=Trajectory(selected.numpy()),
            future_sampling=simulator.proposal_sampling,
            simulator=simulator,
            scorer=scorer,
        )

        fig, (ax, lateral_ax) = plt.subplots(1, 2, figsize=(12.0, 5.2), facecolor="white")
        plot_scene(ax, lateral_ax, proposals, gt, selected, sel_idx,
                   f"metric geometry | PDMS={result.score:.3f}")
        lateral_ax.set_title("lateral detail", fontsize=9)
        fig.suptitle(f"navtest {token} | selected slot {sel_idx} | PDMS={result.score:.3f} "
                     f"(NC={result.no_at_fault_collisions:.2f} DAC={result.drivable_area_compliance:.2f} "
                     f"EP={result.ego_progress:.2f} TTC={result.time_to_collision_within_bound:.2f})")
        fig.tight_layout()
        fig.savefig(out_dir / f"navtest_{token}.png", dpi=150)
        plt.close(fig)
        summary.append({"token": token, "pdms": float(result.score), "selected_slot": sel_idx,
                        "seconds": round(time.time() - t0, 1)})
        print(f"[viz] {token} PDMS={result.score:.3f} slot={sel_idx} ({time.time() - t0:.1f}s)", flush=True)

    # montage: one equal-aspect panel per scene
    n = len(summary)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 5.0 * rows), facecolor="white", squeeze=False)
    for i, row in enumerate(summary):
        token = row["token"]
        scene = scene_loader.get_scene_from_token(token)
        agent_input = scene_loader.get_agent_input_from_token(token)
        features = {}
        for builder in feature_builders:
            features.update(builder.compute_features(agent_input))
        features = {k: v.unsqueeze(0).to(device) for k, v in features.items()}
        with torch.no_grad():
            pred = agent.forward(features)
        proposals = pred["proposals"][0].detach().cpu()
        sel_idx = int(pred["pdm_score"][0].argmax())
        gt = target_builder.compute_targets(scene)["trajectory"].detach().cpu()
        ax = axes[i // cols][i % cols]
        plot_scene(ax, ax, proposals, gt, proposals[sel_idx], sel_idx,
                   f"{token[:12]}… PDMS={row['pdms']:.3f}")
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("PASP-DrivoR step22150 (last.ckpt) | navtest GT vs selected | grey = 64 proposals")
    fig.tight_layout()
    fig.savefig(out_dir / "navtest_montage.png", dpi=150)
    plt.close(fig)

    (out_dir / "viz_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[viz] wrote {n} scene figures + montage to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
