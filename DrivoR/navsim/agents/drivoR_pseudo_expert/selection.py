"""CLOVER Stage-1 selection, split into offline FPS and online human-GT inclusion.

Adapted from CLOVER/navsim/agents/clover/pseudo_expert_utils.py (Apache-2.0).
The thresholds, stable score ordering and FPS tie-breaking follow that reference.
"""

import numpy as np


def select_candidates(trajectories, scores, top_k=8, score_thr=0.8):
    """Select up to K high-scoring diverse trajectories; do not add human GT yet."""
    if top_k < 1:
        raise ValueError("pseudo-expert top_k must be positive")
    trajectories = np.asarray(trajectories, dtype=np.float32)
    scores = np.asarray(scores)
    experts = np.zeros((top_k, 8, 3), dtype=np.float32)
    mask = np.zeros(top_k, dtype=bool)
    if not len(trajectories):
        return experts, mask
    if trajectories.shape != (len(scores), 8, 3):
        raise ValueError(f"Expected scored trajectories [N,8,3], got {trajectories.shape}")
    keep = np.isfinite(trajectories).all((1, 2)) & np.isfinite(scores) & (scores >= score_thr)
    candidates = trajectories[keep][np.argsort(-scores[keep], kind="stable")]
    if not len(candidates):
        return experts, mask
    selected = [0]
    distance = np.abs(candidates[:, :, :2] - candidates[0, :, :2]).mean((1, 2))
    distance[0] = -1
    for _ in range(min(top_k, len(candidates)) - 1):
        index = int(distance.argmax())
        if distance[index] < 0.05:
            break
        selected.append(index)
        distance = np.minimum(distance, np.abs(candidates[:, :, :2] - candidates[index, :, :2]).mean((1, 2)))
        distance[selected] = -1
    experts[:len(selected)] = candidates[selected]
    mask[:len(selected)] = True
    return experts, mask


def include_human_expert(experts, mask, original_expert):
    """Match CLOVER's fallback/append rule without displacing a full expert set."""
    experts, mask = experts.copy(), mask.copy()
    original_expert = np.asarray(original_expert, dtype=np.float32)
    selected = experts[mask]
    covered = len(selected) and (np.abs(selected[:, :, :2] - original_expert[:, :2]).mean((1, 2)) < 0.5).any()
    if not covered and len(selected) < len(experts):
        index = int(np.flatnonzero(~mask)[0])
        experts[index], mask[index] = original_expert, True
    return experts, mask
