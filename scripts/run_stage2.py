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
from nvs_v6.oracle import HEAD_INDICES, OracleLabels
from nvs_v6.scorer import PairedFutureScorer
from nvs_v6.stages import stage2_step


class SyntheticOracle:
    def evaluate(self, scene_ids, trajectories, batch_id=""):
        b, m = trajectories.shape[:2]
        raw = torch.sigmoid(torch.zeros(b, m, 7))
        valid = torch.ones(b, m, 7, dtype=torch.bool)
        raw[..., 3] = 2.0
        valid[..., 3] = False
        return OracleLabels(raw, raw[..., list(HEAD_INDICES)], valid[..., list(HEAD_INDICES)], raw[..., 6], valid[..., 6], ["synthetic"] * b, [[str(i) for i in range(m)] for _ in range(b)], "synthetic", [])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v6 Stage2; --synthetic is an explicit smoke path.")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output", default="artifacts/stage2_smoke.json")
    args = parser.parse_args()
    if not args.synthetic:
        raise SystemExit("real Stage2 requires audited NAVSIM OnlineOracle; use --synthetic for the contract smoke path")
    if args.candidates not in (8, 64):
        raise SystemExit("v6 smoke candidates must be 8 or 64")
    torch.manual_seed(17)
    d_model, heads = 32, 4
    model = DirectedPairDiT(d_model=d_model, heads=heads, ffn_dim=64, depth=2)
    scorer = PairedFutureScorer(d_model=d_model, heads=1)
    groups = parameter_groups(model)
    opt_generator = torch.optim.AdamW(groups["eta"] + groups["theta"] + groups["phi"], lr=2e-4)
    opt_scorer = torch.optim.AdamW(scorer.parameters(), lr=2e-4)
    modes = ImmutableModeBuffer().subset(args.candidates)
    initializer = StructuredSourceInitializer()
    oracle = SyntheticOracle()
    logs = []
    for step in range(args.steps):
        observation = Observation(torch.randn(1, 5, 1, 3, 16, 32), torch.ones(1, 5, 1, dtype=torch.bool), torch.randn(1, 5, 8), torch.randn(1, 4), torch.randn(1, 5, 4, 4))
        gt = torch.randn(1, 2, 8, 3)
        targets = PairTargets(gt, encode_trajectory(gt), torch.ones(1, 2, dtype=torch.bool), torch.randn(1, 2, 4, d_model), torch.ones(1, 2, 4, dtype=torch.bool), torch.ones(1, 2, 4), [["synthetic-0", "synthetic-1"]])
        result = stage2_step(observation, targets, ["synthetic-scene"], model, scorer, oracle, opt_generator, opt_scorer, modes, initializer, batch_id=str(step))
        logs.append({"step": step, "loss_action": float(result.loss_action), "loss_future": float(result.loss_future), "loss_score": float(result.loss_score), "assignment": [m.as_pairs() for m in result.assignment]})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"synthetic_only": True, "candidate_count": args.candidates, "logs": logs}, indent=2) + "\n")
    print(json.dumps(logs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
