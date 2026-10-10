#!/usr/bin/env python3
"""Run an actual two-stage local optimizer smoke on one CPU/GPU process.

This executes real forward/backward/optimizer steps through the v6 training code.
The local oracle is deterministic and trajectory-based because NAVSIM/DrivoR is not
importable in the default environment; the report must not be read as NAVSIM parity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nvs_v6.data import Observation, PairTargets, encode_trajectory
from nvs_v6.model import DirectedPairDiT, parameter_groups
from nvs_v6.modes import ImmutableModeBuffer, StructuredSourceInitializer
from nvs_v6.oracle import HEAD_INDICES, OracleLabels
from nvs_v6.scorer import PairedFutureScorer
from nvs_v6.selection import select_best_candidate
from nvs_v6.stages import stage1_step, stage2_step


class TrajectorySmokeOracle:
    """Score the current trajectory only; it does not read future targets or cache."""

    def evaluate(self, scene_ids, trajectories, batch_id=""):
        xy = trajectories[..., :2]
        yaw = trajectories[..., 2]
        displacement = xy.square().sum(dim=-1).sqrt().mean(dim=-1)
        smoothness = yaw.diff(dim=-1).abs().mean(dim=-1)
        quality = torch.exp(-0.08 * displacement - 0.25 * smoothness).clamp(0.02, 0.98)
        raw = torch.stack(
            (quality, quality * 0.97, quality * 0.94, quality * 0.91, quality * 0.88, quality * 0.85, quality), dim=-1
        )
        valid = torch.ones_like(raw, dtype=torch.bool)
        return OracleLabels(
            raw7=raw,
            named_six=raw[..., list(HEAD_INDICES)],
            head_valid=valid[..., list(HEAD_INDICES)],
            final_score=raw[..., 6],
            final_valid=valid[..., 6],
            scene_hashes=["local_trajectory_smoke"] * len(scene_ids),
            trajectory_hashes=[[f"{batch_id}-{candidate}" for candidate in range(trajectories.shape[1])] for _ in scene_ids],
            evaluator_version="local_trajectory_smoke_v1",
            failure_diagnostics=[],
        )


def make_batch(d_model: int, gt_count: int = 2) -> tuple[Observation, PairTargets]:
    observation = Observation(
        rgb_history=torch.randn(1, 5, 1, 3, 16, 32),
        history_valid=torch.ones(1, 5, 1, dtype=torch.bool),
        ego_history=torch.randn(1, 5, 8),
        route_command=torch.randn(1, 4),
        relative_ego_poses=torch.randn(1, 5, 4, 4),
    )
    gt = torch.randn(1, gt_count, 8, 3)
    targets = PairTargets(
        gt_trajectory=gt,
        gt_encoded=encode_trajectory(gt),
        gt_valid=torch.ones(1, gt_count, dtype=torch.bool),
        future_register=torch.randn(1, gt_count, 4, d_model),
        future_chunk_valid=torch.ones(1, gt_count, 4, dtype=torch.bool),
        future_quality=torch.ones(1, gt_count, 4),
        pair_ids=[[f"local-smoke-{index}" for index in range(gt_count)]],
    )
    return observation, targets


def parameter_norm(parameters) -> float:
    return float(torch.sqrt(sum(parameter.detach().square().sum() for parameter in parameters)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--output", default="artifacts/two_stage_smoke_10steps.json")
    args = parser.parse_args()
    if args.candidates not in (8, 64):
        raise SystemExit("local smoke candidate count must be 8 or 64")
    torch.manual_seed(20261008)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model, heads = 32, 4
    model = DirectedPairDiT(d_model=d_model, heads=heads, ffn_dim=64, depth=2).to(device)
    scorer = PairedFutureScorer(d_model=d_model, heads=1).to(device)
    groups = parameter_groups(model)
    optimizer_stage1 = torch.optim.AdamW(groups["eta"] + groups["phi"], lr=2e-4)
    optimizer_generator = torch.optim.AdamW(groups["eta"] + groups["theta"] + groups["phi"], lr=2e-4)
    optimizer_scorer = torch.optim.AdamW(scorer.parameters(), lr=2e-4)
    modes = ImmutableModeBuffer().subset(args.candidates)
    initializer = StructuredSourceInitializer(jitter_std=0.02)
    oracle = TrajectorySmokeOracle()
    stage1_logs, stage2_logs = [], []
    eta_before = parameter_norm(groups["eta"])
    phi_before = parameter_norm(groups["phi"])
    theta_before = parameter_norm(groups["theta"])
    psi_before = parameter_norm(scorer.parameters())

    for step in range(args.steps):
        observation, targets = make_batch(d_model)
        observation = Observation(*(value.to(device) if isinstance(value, torch.Tensor) else value for value in observation.__dict__.values()))
        targets = PairTargets(*(value.to(device) if isinstance(value, torch.Tensor) else value for value in targets.__dict__.values()))
        result = stage1_step(observation, targets, model, optimizer_stage1, modes.to(device), initializer)
        stage1_logs.append({"step": step, "loss_future": float(result.loss_future), "gradient_update": True})

    for step in range(args.steps):
        observation, targets = make_batch(d_model)
        observation = Observation(*(value.to(device) if isinstance(value, torch.Tensor) else value for value in observation.__dict__.values()))
        targets = PairTargets(*(value.to(device) if isinstance(value, torch.Tensor) else value for value in targets.__dict__.values()))
        result = stage2_step(observation, targets, ["local-smoke-scene"], model, scorer, oracle, optimizer_generator, optimizer_scorer, modes.to(device), initializer, batch_id=str(step))
        stage2_logs.append({"step": step, "loss_action": float(result.loss_action), "loss_future": float(result.loss_future), "loss_score": float(result.loss_score), "gradient_update": True})

    observation, _ = make_batch(d_model)
    observation = Observation(*(value.to(device) if isinstance(value, torch.Tensor) else value for value in observation.__dict__.values()))
    with torch.no_grad():
        mode_codes = modes.expand_for_batch(1, device)
        source = StructuredSourceInitializer(jitter_std=0.0).clean(observation.ego_history, mode_codes)
        outputs = model.forward_pairs(observation, source, mode_codes)
        logits = scorer(outputs.scene_context, outputs.trajectory, outputs.predicted_register)
        selected, utility = select_best_candidate(logits)
    eta_after = parameter_norm(groups["eta"])
    phi_after = parameter_norm(groups["phi"])
    theta_after = parameter_norm(groups["theta"])
    psi_after = parameter_norm(scorer.parameters())
    result = {
        "device": str(device),
        "steps_per_stage": args.steps,
        "candidate_count": args.candidates,
        "synthetic_local_oracle": True,
        "stage1": stage1_logs,
        "stage2": stage2_logs,
        "finite_losses": all(torch.isfinite(torch.tensor(item["loss_future"])).item() for item in stage1_logs) and all(torch.isfinite(torch.tensor(item[key])).item() for item in stage2_logs for key in ("loss_action", "loss_future", "loss_score")),
        "parameter_norms": {"eta_before": eta_before, "eta_after": eta_after, "phi_before": phi_before, "phi_after": phi_after, "theta_before": theta_before, "theta_after": theta_after, "psi_before": psi_before, "psi_after": psi_after},
        "parameter_updates": {"eta": eta_after != eta_before, "phi": phi_after != phi_before, "theta": theta_after != theta_before, "psi": psi_after != psi_before},
        "score_shape": list(logits.shape),
        "selected_candidate": int(selected.item()),
        "selected_utility": float(utility[0, selected.item()]),
        "all_utilities": utility[0].detach().cpu().tolist(),
        "selection_is_finite": bool(torch.isfinite(utility).all()),
        "real_navsim_oracle": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
