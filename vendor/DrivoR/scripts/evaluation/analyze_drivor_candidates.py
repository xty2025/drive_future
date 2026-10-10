#!/usr/bin/env python3
"""Analyze and render candidate trajectories recorded during official DrivoR eval."""
import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def diversity(x):
    xy = x[..., :2]
    d = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    iu = np.triu_indices(len(x), 1)
    return float(d.mean(-1)[iu].mean()), float(d[..., -1][iu].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pickle", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-plots", type=int, default=12)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    with a.pickle.open("rb") as f:
        data = pickle.load(f)
    rows = []
    for plot_i, (token, item) in enumerate(data.items()):
        stages = item["all_proposals"]
        final = np.asarray(item["final_proposals"])
        gt = np.asarray(item["gt_trajectory"])
        scores = np.asarray(item["learned_scores"])
        winner = int(scores.argmax())
        ade = np.linalg.norm(final[..., :2] - gt[None, ..., :2], axis=-1).mean(-1)
        fde = np.linalg.norm(final[:, -1, :2] - gt[-1, :2], axis=-1)
        p_ade, p_fde = diversity(final)
        row = dict(token=token, winner=winner, candidate_count=len(final),
                   pairwise_ADE=p_ade, endpoint_dispersion=p_fde,
                   winner_ADE=float(ade[winner]), winner_FDE=float(fde[winner]),
                   oracle_ADE=float(ade.min()), oracle_FDE=float(fde.min()),
                   score_std=float(scores.std()))
        for i, stage in enumerate(stages):
            row[f"refine_{i}_pairwise_ADE"], row[f"refine_{i}_endpoint_dispersion"] = diversity(np.asarray(stage))
        rows.append(row)
        if plot_i < a.max_plots:
            n = len(stages)
            fig, axes = plt.subplots(1, n, figsize=(4*n, 4), squeeze=False)
            for i, (ax, stage) in enumerate(zip(axes[0], stages)):
                stage = np.asarray(stage)
                ax.plot(stage[..., 0].T, stage[..., 1].T, color="#4c78a8", alpha=.18, lw=.8)
                ax.plot(gt[:, 0], gt[:, 1], "k--", lw=2.4, label="GT")
                if i == n - 1:
                    ax.plot(stage[winner, :, 0], stage[winner, :, 1], color="#e45756", lw=2.5, label="winner")
                ax.scatter([0], [0], c="black", s=22)
                ax.set_title(f"refinement {i}")
                ax.axis("equal"); ax.grid(alpha=.2); ax.legend()
            fig.suptitle(f"DrivoR official candidates + GT | {token}")
            fig.tight_layout()
            fig.savefig(a.output / f"{token}.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
    summary = {"num_scenes": len(rows)}
    for k in rows[0]:
        if k not in ("token", "winner"):
            vals = [r[k] for r in rows if isinstance(r[k], (int, float))]
            summary[k] = float(np.mean(vals))
    (a.output / "per_scene.json").write_text(json.dumps(rows, indent=2))
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
