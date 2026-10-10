#!/usr/bin/env python3
import argparse
import csv
import lzma
import pickle
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--metric-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scoring-config", type=Path, default=Path("navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml"))
    args = parser.parse_args()
    predictions = pickle.load(args.predictions.open("rb"))
    cache = MetricCacheLoader(args.metric_cache)
    cfg = OmegaConf.load(args.scoring_config)
    simulator, scorer = instantiate(cfg.simulator), instantiate(cfg.scorer)
    rows = []
    for i, token in enumerate(sorted(predictions)):
        metric = cache.get_from_token(token)
        result = pdm_score(metric, predictions[token]["trajectory"], simulator.proposal_sampling, simulator, scorer)
        rows.append({"token": token, "valid": True, **asdict(result)})
        print(f"[{i+1:02d}/{len(predictions)}] {token} PDMS={rows[-1]['score']:.6f}", flush=True)
    frame = pd.DataFrame(rows)
    average = frame.drop(columns=["token", "valid"]).mean().to_dict()
    frame.loc[len(frame)] = {"token": "average", "valid": True, **average}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"average PDMS={average['score']:.6f}; saved {args.output}")


if __name__ == "__main__":
    main()
