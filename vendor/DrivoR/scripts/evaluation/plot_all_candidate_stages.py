#!/usr/bin/env python3
"""Render every proposal at every DrivoR refinement stage with GT overlay."""
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--scenes", type=int, default=16)
    args = ap.parse_args()
    with args.pkl.open("rb") as f:
        data = pickle.load(f)
    args.output.mkdir(parents=True, exist_ok=True)
    tokens = sorted(data)[: args.scenes]
    stage_colors = ["#7f7f7f", "#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
    for token in tokens:
        item = data[token]
        stages = [np.asarray(x)[..., :2] for x in item["all_proposals"]]
        gt = np.asarray(item["gt_trajectory"])[..., :2]
        final_scores = np.asarray(item["learned_scores"])
        all_xy = np.concatenate([gt.reshape(-1, 2), *[x.reshape(-1, 2) for x in stages]])
        lo, hi = all_xy.min(0), all_xy.max(0)
        span = max(float((hi - lo).max()), 1.0)
        pad = 0.08 * span
        fig, axes = plt.subplots(1, len(stages), figsize=(4.8 * len(stages), 4.5), sharex=True, sharey=True)
        axes = np.atleast_1d(axes)
        for i, (ax, stage) in enumerate(zip(axes, stages)):
            for candidate in stage:
                ax.plot(candidate[:, 0], candidate[:, 1], color=stage_colors[i], alpha=0.18, lw=0.8)
            ax.plot(gt[:, 0], gt[:, 1], "k--", lw=2.0, label="GT")
            if i == len(stages) - 1:
                winner = int(final_scores.argmax())
                ax.plot(stage[winner, :, 0], stage[winner, :, 1], color="#111111", lw=2.2, label=f"scorer winner {winner}")
            ax.scatter([0], [0], c="black", s=18, zorder=3)
            ax.set_xlim(lo[0] - pad, hi[0] + pad)
            ax.set_ylim(lo[1] - pad, hi[1] + pad)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)
            ax.set_title(f"stage {i} | all {len(stage)} candidates")
            ax.set_xlabel("x [m]")
        axes[0].set_ylabel("y [m]")
        axes[-1].legend(loc="best", fontsize=8)
        fig.suptitle(f"DrivoR proposal-set refinement with GT | {token}")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(args.output / f"{token}_all_candidates.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {len(tokens)} scenes to {args.output}")


if __name__ == "__main__":
    main()
