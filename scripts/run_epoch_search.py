"""Descending two-stage epoch search for the two-scene overfit contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _passes(result: dict, pdms_threshold: float, ade_threshold: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not result.get("finite_losses", False):
        reasons.append("non-finite loss")
    audit = result.get("stage1_future_audit", {})
    if not audit.get("loss_decreased", False):
        reasons.append("Stage1 future loss did not decrease")
    if not audit.get("paired_better_than_shuffled", False):
        reasons.append("Stage1 paired future is not better than shuffled")
    evaluations = result.get("evaluation", [])
    if len(evaluations) != 2:
        reasons.append(f"expected two evaluation scenes, got {len(evaluations)}")
    for row in evaluations:
        scene = row.get("scene_id", "unknown")
        if row.get("matched_gt_max_ade_m", float("inf")) > ade_threshold:
            reasons.append(f"{scene}: matched GT max ADE exceeds {ade_threshold}")
        if row.get("learned_scorer_candidate_pdms", -1.0) < pdms_threshold:
            reasons.append(f"{scene}: learned scorer PDMS below {pdms_threshold}")
        if row.get("oracle_best_pdms", -1.0) < pdms_threshold:
            reasons.append(f"{scene}: no generated candidate reaches PDMS threshold")
    return not reasons, reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="artifacts/epoch_search_scratch_nvs_scene_50x50")
    parser.add_argument("--steps-per-epoch", type=int, default=10)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--pdms-threshold", type=float, default=0.999999)
    parser.add_argument("--ade-threshold", type=float, default=0.5)
    args = parser.parse_args()

    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "search_status.json"
    trials: list[dict] = []

    def save_status(state: str, selected: dict | None = None) -> None:
        payload = {
            "state": state,
            "updated_unix": time.time(),
            "search_policy": "scratch NVS-scene scorer; start 50/50; descend Stage2, then Stage1",
            "thresholds": {"pdms": args.pdms_threshold, "matched_gt_max_ade_m": args.ade_threshold},
            "trials": trials,
            "minimum_passing_on_tested_grid": selected,
        }
        status_path.write_text(json.dumps(payload, indent=2) + "\n")

    def run(stage1_epochs: int, stage2_epochs: int) -> dict:
        name = f"s1_{stage1_epochs:03d}_s2_{stage2_epochs:03d}"
        output = root / name
        result_path = output / "result.json"
        log_path = root / f"{name}.log"
        if not result_path.exists():
            output.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-u",
                "scripts/run_two_scene_overfit.py",
                "--stage1-epochs",
                str(stage1_epochs),
                "--stage2-epochs",
                str(stage2_epochs),
                "--steps-per-epoch",
                str(args.steps_per_epoch),
                "--depth",
                str(args.depth),
                "--output-dir",
                str(output),
            ]
            with log_path.open("w") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            if completed.returncode != 0:
                trial = {
                    "stage1_epochs": stage1_epochs,
                    "stage2_epochs": stage2_epochs,
                    "passed": False,
                    "returncode": completed.returncode,
                    "reasons": ["training process failed"],
                    "log": str(log_path),
                }
                trials.append(trial)
                save_status("failed")
                raise RuntimeError(f"epoch trial failed: {name}; see {log_path}")
        result = json.loads(result_path.read_text())
        passed, reasons = _passes(result, args.pdms_threshold, args.ade_threshold)
        trial = {
            "stage1_epochs": stage1_epochs,
            "stage2_epochs": stage2_epochs,
            "passed": passed,
            "reasons": reasons,
            "result": str(result_path),
            "log": str(log_path),
            "selected_pdms": [row["learned_scorer_candidate_pdms"] for row in result.get("evaluation", [])],
            "matched_gt_max_ade_m": [row.get("matched_gt_max_ade_m") for row in result.get("evaluation", [])],
        }
        trials.append(trial)
        save_status("running")
        return trial

    baseline = run(50, 50)
    if not baseline["passed"]:
        save_status("baseline_did_not_overfit")
        return 2

    minimum = baseline
    minimum_stage2 = 50
    for stage2_epochs in (40, 30, 20, 10):
        trial = run(50, stage2_epochs)
        if not trial["passed"]:
            break
        minimum = trial
        minimum_stage2 = stage2_epochs

    for stage1_epochs in (40, 30, 20, 15, 10, 5):
        trial = run(stage1_epochs, minimum_stage2)
        if not trial["passed"]:
            break
        minimum = trial

    save_status("complete", minimum)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
