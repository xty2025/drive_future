#!/usr/bin/env python3
"""Prepare CLOVER Stage-1 experts once, instead of loading a 4-GB PKL per rank.

All source candidates are eligible (no Bank-64 truncation or negative sampling).
Human GT inclusion stays in the target builder, exactly as in CLOVER.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from build_world4scorer_bank import read_scenes
from navsim.agents.drivoR_pseudo_expert.selection import select_candidates


def build(source, output, top_k=8, score_thr=0.8, preview=None):
    print(f"Loading CLOVER source: {source}", flush=True)
    source_scenes = read_scenes(source, preview)
    scenes = [scene for scene in source_scenes if scene.get("valid", False)
              and len(scene.get("trajectories_relative", [])) and len(scene.get("scores", []))]
    if not scenes:
        raise ValueError("CLOVER source contains no valid nonempty scenes")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").unlink(missing_ok=True)
    arrays = {
        "tokens": np.lib.format.open_memmap(output / "tokens.npy", mode="w+", dtype="S64", shape=(len(scenes),)),
        "pseudo_experts": np.lib.format.open_memmap(output / "pseudo_experts.npy", mode="w+", dtype=np.float32,
                                                   shape=(len(scenes), top_k, 8, 3)),
        "pseudo_expert_mask": np.lib.format.open_memmap(output / "pseudo_expert_mask.npy", mode="w+", dtype=bool,
                                                       shape=(len(scenes), top_k)),
    }
    counts = np.zeros(top_k + 1, dtype=np.int64)
    source_candidates = 0
    for row, scene in enumerate(scenes):
        trajectories = scene["trajectories_relative"]
        scores = [score["pdm_score"] for score in scene["scores"]]
        experts, mask = select_candidates(trajectories, scores, top_k, score_thr)
        arrays["tokens"][row] = scene["token"].encode()
        arrays["pseudo_experts"][row] = experts
        arrays["pseudo_expert_mask"][row] = mask
        counts[int(mask.sum())] += 1
        source_candidates += len(trajectories)
        if (row + 1) % 10000 == 0:
            print(f"Selected {row + 1}/{len(scenes)} scenes", flush=True)
    for array in arrays.values():
        array.flush()
    report = dict(source=str(Path(source).resolve()), source_repository="https://github.com/WilliamXuanYu/CLOVER",
                  selection="score_filter_then_greedy_xy_fps", top_k=top_k, score_thr=score_thr,
                  diversity_min_distance=0.05, human_coverage_distance=0.5, human_inclusion="online",
                  num_tokens=len(scenes), source_candidates=source_candidates,
                  scored_expert_count_histogram=counts.tolist(), preview=preview is not None)
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--score-thr", type=float, default=0.8)
    parser.add_argument("--preview-scenes", type=int)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.top_k, args.score_thr, args.preview_scenes)), flush=True)
