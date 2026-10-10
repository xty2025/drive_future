#!/usr/bin/env python3
"""Plot official DrivoR stages and one full stage-4 candidate-set example."""
import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, type=Path)
    ap.add_argument("--diversity-pkl", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--scenes", type=int, default=16)
    args = ap.parse_args()
    with args.pkl.open("rb") as f:
        data = pickle.load(f)
    diversity_data = data
    if args.diversity_pkl:
        with args.diversity_pkl.open("rb") as f:
            diversity_data = pickle.load(f)
    args.output.mkdir(parents=True, exist_ok=True)
    tokens = sorted(set(data) & set(diversity_data))[: args.scenes]
    stage_colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
    for token in tokens:
        item = data[token]
        if "shared_head_trajectories" in item:
            stages = [np.asarray(x.poses)[None] for x in item["shared_head_trajectories"]]
            stage_scores = [np.asarray(x) for x in item["shared_head_scores"]]
        elif "shared_head_proposals" in item:
            stages = [np.asarray(x) for x in item["shared_head_proposals"]]
            stage_scores = [np.asarray(x) for x in item["shared_head_scores"]]
        else:
            stages = [np.asarray(x) for x in item["all_proposals"][1:]]
            stage_scores = [np.asarray(item["learned_scores"])] * len(stages)
        gt = np.asarray(item.get("gt_trajectory", diversity_data[token]["gt_trajectory"]))
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        ax.plot(gt[:, 0], gt[:, 1], "k--", lw=2.3, label="GT")
        winners = []
        for stage_idx, (stage, scores) in enumerate(zip(stages, stage_scores), 1):
            candidate_idx = int(scores.argmax())
            trajectory_idx = 0 if len(stage) == 1 else candidate_idx
            winners.append(trajectory_idx)
            traj = stage[trajectory_idx]
            ax.plot(traj[:, 0], traj[:, 1], color=stage_colors[stage_idx - 1], lw=2.0,
                    label=f"step {stage_idx} (candidate {candidate_idx})")
        ax.scatter([0], [0], c="black", s=22, zorder=4)
        ax.scatter([stages[-1][winners[-1], -1, 0]], [stages[-1][winners[-1], -1, 1]], c="#1f77b4", s=28, zorder=4)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_title(f"DrivoR shared final trajectory head | {token}\n8 points @ 0.5 s, horizon 4.0 s")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(args.output / f"{token}_stages.png", dpi=180)
        plt.close(fig)

    def endpoint_spread(candidate_item):
        candidate_xy = np.asarray(candidate_item["final_proposals"])[:, -1, :2]
        center = np.median(candidate_xy, axis=0)
        return float(np.percentile(np.linalg.norm(candidate_xy - center, axis=1), 90))

    token = max(diversity_data, key=lambda key: endpoint_spread(diversity_data[key]))
    item = diversity_data[token]
    final = np.asarray(item.get("shared_head_proposals", [item["final_proposals"]])[-1])
    gt = np.asarray(item["gt_trajectory"])
    scores = np.asarray(item.get("shared_head_scores", [item["learned_scores"]])[-1])
    winner = int(scores.argmax())
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for idx, traj in enumerate(final):
        ax.plot(traj[:, 0], traj[:, 1], color="#4c78a8", alpha=0.22, lw=0.9)
    ax.scatter(final[:, -1, 0], final[:, -1, 1], color="#4c78a8", alpha=0.35, s=10,
               label="64 candidate endpoints")
    ax.plot(gt[:, 0], gt[:, 1], "k--", lw=2.4, label="GT")
    ax.plot(final[winner, :, 0], final[winner, :, 1], color="#e45756", lw=2.8,
            label=f"scorer winner {winner}")
    ax.scatter([0], [0], c="black", s=24, zorder=4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    spread = endpoint_spread(item)
    ax.set_title(f"DrivoR official stage 4: all 64 candidates + GT | {token}\n"
                 f"8 points @ 0.5 s, horizon 4.0 s | endpoint spread P90={spread:.1f} m")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(args.output / "diversity_4step_example.png", dpi=200)
    plt.close(fig)

    pairwise_ade = []
    endpoint_dispersion = []
    winner_ade = []
    winner_fde = []
    oracle_ade = []
    oracle_fde = []
    score_std = []
    stage_winner_ade = [[] for _ in range(4)]
    upper = np.triu_indices(64, 1)
    for key in sorted(set(data) & set(diversity_data)):
        full_item = diversity_data[key]
        candidates = np.asarray(full_item["final_proposals"])[..., :2]
        ground_truth = np.asarray(full_item["gt_trajectory"])[..., :2]
        learned_scores = np.asarray(full_item["learned_scores"])
        selected = int(learned_scores.argmax())
        distances = np.linalg.norm(candidates - ground_truth[None], axis=-1)
        candidate_ade = distances.mean(axis=-1)
        candidate_fde = distances[:, -1]
        pairwise = np.linalg.norm(candidates[:, None] - candidates[None], axis=-1)
        pairwise_ade.append(float(pairwise.mean(axis=-1)[upper].mean()))
        endpoint_dispersion.append(float(pairwise[..., -1][upper].mean()))
        winner_ade.append(float(candidate_ade[selected]))
        winner_fde.append(float(candidate_fde[selected]))
        oracle_ade.append(float(candidate_ade.min()))
        oracle_fde.append(float(candidate_fde.min()))
        score_std.append(float(learned_scores.std()))
        for stage_idx, stage_trajectory in enumerate(data[key]["shared_head_trajectories"]):
            stage_xy = np.asarray(stage_trajectory.poses)[..., :2]
            stage_winner_ade[stage_idx].append(
                float(np.linalg.norm(stage_xy - ground_truth, axis=-1).mean())
            )

    summary = {
        "num_scenes": len(data),
        "num_candidates_per_scene": 64,
        "trajectory_sampling": {"num_poses": 8, "interval_s": 0.5, "horizon_s": 4.0},
        "step4_pairwise_ADE_m": float(np.mean(pairwise_ade)),
        "step4_endpoint_dispersion_m": float(np.mean(endpoint_dispersion)),
        "step4_learned_score_std": float(np.mean(score_std)),
        "step4_scorer_winner_ADE_m": float(np.mean(winner_ade)),
        "step4_scorer_winner_FDE_m": float(np.mean(winner_fde)),
        "step4_oracle_ADE_m": float(np.mean(oracle_ade)),
        "step4_oracle_FDE_m": float(np.mean(oracle_fde)),
        "shared_final_head_winner_ADE_m": {
            f"step{idx + 1}": float(np.mean(values))
            for idx, values in enumerate(stage_winner_ade)
        },
        "diversity_figure_token": token,
        "diversity_figure_endpoint_spread_P90_m": spread,
    }
    (args.output / "diversity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {len(tokens)} stage plots and one diversity plot to {args.output}")


if __name__ == "__main__":
    main()
