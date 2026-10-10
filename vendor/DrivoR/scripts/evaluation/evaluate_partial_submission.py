#!/usr/bin/env python3
"""Run the official NavSIM PDM scorer on only tokens present in a submission.

This keeps the canonical simulator/scorer/metric definitions unchanged while
supporting smoke or crossover files that contain a subset of the metric cache.
"""

import argparse
import pickle
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--metric-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.submission.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "predictions" in payload:
        predictions = payload["predictions"]
        if len(predictions) != 1:
            raise ValueError("Only one prediction seed is supported")
        predictions = predictions[0]
    else:
        predictions = payload

    # Crossover reranking files retain the full ChainFlow item per token;
    # official submission files instead map tokens directly to Trajectory.
    if predictions and isinstance(next(iter(predictions.values())), dict):
        predictions = {
            token: item["trajectory"] if "trajectory" in item else item
            for token, item in predictions.items()
        }

    # Use the same default objects as run_pdm_score_from_submission.py.
    config_dir = Path(__file__).resolve().parents[3] / "ChainFlow-VLA/navsim/planning/script/config/pdm_scoring"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="default_run_pdm_score_from_submission",
            overrides=[f"metric_cache_path={args.metric_cache}"],
        )
    simulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    if simulator.proposal_sampling != scorer.proposal_sampling:
        raise ValueError("Simulator and scorer proposal sampling differ")

    loader = MetricCacheLoader(args.metric_cache)
    rows = []
    for index, token in enumerate(sorted(predictions)):
        row = {"token": token, "valid": True}
        try:
            result = pdm_score(
                metric_cache=loader.get_from_token(token),
                model_trajectory=predictions[token],
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            row.update(asdict(result))
        except Exception:
            row["valid"] = False
            row["error"] = traceback.format_exc()
        rows.append(row)
        print(f"[{index + 1}/{len(predictions)}] {token} valid={row['valid']}", flush=True)

    frame = pd.DataFrame(rows)
    metric_columns = [column for column in frame.columns if column not in {"token", "valid", "error"}]
    average = frame.loc[frame["valid"], metric_columns].mean()
    average_row = {column: average.get(column, float("nan")) for column in frame.columns}
    average_row["token"] = "average"
    average_row["valid"] = bool(frame["valid"].all())
    frame = pd.concat([frame, pd.DataFrame([average_row])], ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    csv_path = args.output_dir / f"{timestamp}.csv"
    frame.to_csv(csv_path, index=False)
    summary = {
        "csv": str(csv_path),
        "num_predictions": len(predictions),
        "num_valid": int(frame.iloc[:-1]["valid"].sum()),
        "pdms": float(average.get("score", float("nan"))),
    }
    (args.output_dir / f"{timestamp}.json").write_text(__import__("json").dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
