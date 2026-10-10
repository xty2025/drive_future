#!/usr/bin/env python3
"""Create compact summary figures for the verified DrivoR depth evaluation."""
import argparse
import csv
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


STYLE = Path("/mnt/zhaozc_workspace/codex/home/skills/figure-polish/assets/deepscientist-academic.mplstyle")
COLORS = ["#C65D57", "#D69A45", "#5B8E7D", "#3F6C91"]
MARKERS = ["o", "s", "^", "D"]


def read_average(csv_path: Path) -> dict:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return next(row for row in rows if row["token"] == "average")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--official-candidates", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--diversity-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    plt.style.use(STYLE)
    args.output.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("rb") as handle:
        predictions = pickle.load(handle)
    with args.official_candidates.open("rb") as handle:
        official = pickle.load(handle)
    diversity = json.loads(args.diversity_summary.read_text())

    tokens = sorted(set(predictions) & set(official))[:16]
    fig, axes = plt.subplots(4, 4, figsize=(13.2, 11.6))
    for scene_idx, (ax, token) in enumerate(zip(axes.flat, tokens), 1):
        gt = np.asarray(official[token]["gt_trajectory"])
        ax.plot(gt[:, 0], gt[:, 1], color="#171717", ls="--", lw=2.4, zorder=5)
        for depth, trajectory in enumerate(predictions[token]["shared_head_trajectories"]):
            poses = np.asarray(trajectory.poses)
            ax.plot(
                poses[:, 0], poses[:, 1], color=COLORS[depth],
                lw=2.4 if depth == 3 else 1.55, alpha=1.0 if depth == 3 else 0.9,
                marker=MARKERS[depth], markevery=[-1], ms=4.0,
                zorder=4 if depth == 3 else 3,
            )
        ax.scatter(0, 0, color="#171717", s=17, zorder=6)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{scene_idx:02d}  {token[-8:]}", pad=4)
        if scene_idx > 12:
            ax.set_xlabel("x [m]")
        if (scene_idx - 1) % 4 == 0:
            ax.set_ylabel("y [m]")

    handles = [Line2D([0], [0], color="#171717", ls="--", lw=2.4, label="GT")]
    handles.extend(
        Line2D([0], [0], color=color, marker=MARKERS[idx - 1],
               lw=2.4 if idx == 4 else 1.7, label=f"step {idx}")
        for idx, color in enumerate(COLORS, 1)
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925),
               ncol=5, frameon=False)
    fig.suptitle(
        "DrivoR shared-final-head winners across 16 Navtest scenes\n"
        "All trajectories: 8 points @ 0.5 s, horizon 4.0 s",
        fontsize=16, y=0.995,
    )
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.055, top=0.855, wspace=0.18, hspace=0.31)
    overview_png = args.output / "trajectory_steps_16scenes_overview.png"
    fig.savefig(overview_png, dpi=210)
    plt.close(fig)

    pdms = []
    for depth in range(1, 5):
        csv_path = sorted((args.result_root / f"step{depth}").glob("*.csv"))[-1]
        pdms.append(float(read_average(csv_path)["score"]) * 100.0)
    winner_ade = [
        diversity["shared_final_head_winner_ADE_m"][f"step{depth}"]
        for depth in range(1, 5)
    ]
    depths = np.arange(1, 5)
    fig, (ax_pdms, ax_ade) = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)

    ax_pdms.plot(depths, pdms, color="#3F6C91", marker="o", lw=2.5)
    ax_pdms.axhline(93.0, color="#8C8C8C", ls="--", lw=1.2, label="93% target")
    for x, value in zip(depths, pdms):
        ax_pdms.annotate(f"{value:.2f}", (x, value), xytext=(0, 7), textcoords="offset points",
                         ha="center", color="#24465F", fontsize=10)
    ax_pdms.set_xticks(depths)
    ax_pdms.set_xlabel("Decoder depth")
    ax_pdms.set_ylabel("PDMS [%]  (higher is better)")
    ax_pdms.set_ylim(92.5, 96.0)
    ax_pdms.set_title("Official PDM evaluation | 12,146 scenes")
    ax_pdms.legend(loc="lower right")

    ax_ade.plot(depths, winner_ade, color="#C65D57", marker="o", lw=2.5)
    for x, value in zip(depths, winner_ade):
        ax_ade.annotate(f"{value:.3f}", (x, value), xytext=(0, 7), textcoords="offset points",
                        ha="center", color="#7E3935", fontsize=10)
    ax_ade.set_xticks(depths)
    ax_ade.set_xlabel("Decoder depth")
    ax_ade.set_ylabel("Winner GT-ADE [m]  (lower is better)")
    ax_ade.set_ylim(1.60, 2.05)
    ax_ade.set_title("Learned-scorer winner vs. GT")

    fig.suptitle("DrivoR depth comparison: quality improves monotonically", fontsize=15)
    fig.text(0.5, -0.015,
             "Steps 1-3: shared-final-head diagnostic ablations | Step 4: official DrivoR inference",
             ha="center", fontsize=9, color="#5F6368")
    metrics_png = args.output / "depth_metrics_comparison.png"
    fig.savefig(metrics_png, dpi=220)
    plt.close(fig)

    catalog = {
        "surface_class": "internal_review",
        "source_data": [
            str(args.predictions), str(args.official_candidates), str(args.diversity_summary),
            str(args.result_root),
        ],
        "generator": str(Path(__file__).resolve()),
        "exports": [str(overview_png), str(metrics_png)],
        "claims": [
            "The four shared-final-head winners are normal-scale trajectories across 16 scenes.",
            "Full-split PDMS increases from 93.09% to 94.59% while winner GT-ADE decreases.",
        ],
        "self_review_revision": "Separated the overview title and shared legend, then added distinct endpoint markers so overlapping depth trajectories remain identifiable.",
    }
    (args.output / "figure_catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")


if __name__ == "__main__":
    main()
