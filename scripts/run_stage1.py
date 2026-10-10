#!/usr/bin/env python3
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
from nvs_v6.scorer import PairedFutureScorer
from nvs_v6.stages import stage1_step


def synthetic_batch(d_model: int, candidates: int = 8):
    obs = Observation(
        torch.randn(1, 5, 1, 3, 16, 32), torch.ones(1, 5, 1, dtype=torch.bool),
        torch.randn(1, 5, 8), torch.randn(1, 4), torch.randn(1, 5, 4, 4)
    )
    gt = torch.randn(1, 2, 8, 3)
    targets = PairTargets(
        gt, encode_trajectory(gt), torch.ones(1, 2, dtype=torch.bool),
        torch.randn(1, 2, 4, d_model), torch.ones(1, 2, 4, dtype=torch.bool),
        torch.ones(1, 2, 4), [["synthetic-0", "synthetic-1"]]
    )
    return obs, targets


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v6 Stage1; --synthetic is an explicit smoke path.")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output", default="artifacts/stage1_smoke.json")
    args = parser.parse_args()
    if not args.synthetic:
        raise SystemExit("real Stage1 requires audited Observation/PairTargets and fixed Teacher target cache; use --synthetic for the contract smoke path")
    torch.manual_seed(17)
    d_model, heads = 32, 4
    model = DirectedPairDiT(d_model=d_model, heads=heads, ffn_dim=64, depth=2)
    scorer = PairedFutureScorer(d_model=d_model, heads=1)
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    groups = parameter_groups(model)
    optimizer = torch.optim.AdamW(groups["eta"] + groups["phi"], lr=2e-4)
    modes = ImmutableModeBuffer().to(torch.device("cpu"))
    initializer = StructuredSourceInitializer()
    logs = []
    for step in range(args.steps):
        observation, targets = synthetic_batch(d_model, args.candidates)
        result = stage1_step(observation, targets, model, optimizer, modes, initializer)
        logs.append({"step": step, "loss_future": float(result.loss_future), "assignment": [m.as_pairs() for m in result.assignment]})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"synthetic_only": True, "logs": logs}, indent=2) + "\n")
    print(json.dumps(logs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
