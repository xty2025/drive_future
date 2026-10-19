"""Action-DiT proposals ranked by DrivoR's scorer: how much does the scorer alone change the navtest PDMS?

Each Action-DiT checkpoint samples its 64 proposals per navtest scene (BF16, 10 Euler steps; per-scene noise shared by
all models, proposal_analysis.flow_noise). Every proposal set is ranked by the model's own scorer and by the scorer of
each DrivoR checkpoint, which needs that checkpoint's own image backbone and ego encoding (proposal_analysis.
drivor_score: DrivoRModel.forward's scoring branch on foreign proposals, checked against DrivoR's own scores on every
batch). The DrivoR checkpoints' own outputs are kept for reference. Every proposal is scored with the official
NAVSIM-v1 PDMS (navsim.evaluate.pairwise_pdm_score), so each ranking's PDMS, its sub-metrics and the best of 64 come
from the same proposals.

Stage 1 runs one process per GPU on a token shard (--shard i --num-shards n; run_cross_scorer_analysis.sh starts one
per visible GPU) and writes shard_<i>.npz. Stage 2 (--report) merges the shards, scores every proposal and writes
summary.md, summary.json and pdm.npz; with pdm.npz present it only re-reports.
"""

import argparse
import json
import lzma
import multiprocessing
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

DRIVOR_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = DRIVOR_ROOT.parent
sys.path.insert(0, str(DRIVOR_ROOT / "scripts/viz"))
from visualize_action_dit_proposals import MODELS  # noqa: E402  checkpoints, eval dirs and eval overrides

ACTION_DIT = [key for key, spec in MODELS.items() if spec["family"] == "Action-DiT"]
DRIVOR = [key for key, spec in MODELS.items() if spec["family"] == "DrivoR"]
METRICS = (("NC", "no_at_fault_collisions"), ("DAC", "drivable_area_compliance"),
           ("TTC", "time_to_collision_within_bound"), ("C", "comfort"), ("EP", "ego_progress"))

_SIMULATOR = None
_SCORER = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generators", nargs="+", default=ACTION_DIT[:4], choices=ACTION_DIT,
                        help="Action-DiT checkpoints whose proposals are ranked")
    parser.add_argument("--scorers", nargs="+", default=DRIVOR[:2], choices=DRIVOR, help="DrivoR checkpoints that rank them")
    parser.add_argument("--max-scenes", type=int, default=2000, help="random navtest scenes; 0 = all")
    parser.add_argument("--metric-cache", type=Path, default=PROJECT_ROOT / "data/metric_cache_navtest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--report", action="store_true", help="stage 2: merge shards, score, summarise")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--loader-workers", type=int, default=6)
    parser.add_argument("--score-workers", type=int, default=min(32, len(os.sched_getaffinity(0))))
    parser.add_argument("--seed", type=int, default=0, help="scene subset and the flow noise of every scene")
    return parser.parse_args()


def label(key):
    return f"{MODELS[key]['family']} {MODELS[key]['variant']}"


# ----------------------------------------------------------------------------- stage 1: proposals and rankings

def compute_shard(args):
    import torch
    from torch.utils.data import DataLoader

    from navsim.agents.drivoR.proposal_analysis import (
        SPLITS, TokenDataset, compose_eval_config, drivor_scene_context, drivor_score, load_agent, log_of,
        make_scene_loader, model_forward, sensor_key, union_sensor_config)
    from navsim.common.dataloader import MetricCacheLoader

    device = torch.device("cuda")
    sources = args.generators + args.scorers  # proposal sets: Action-DiT samples, then DrivoR's own proposals
    agents = {key: load_agent(compose_eval_config(MODELS[key]["overrides"]), MODELS[key]["checkpoint"], device)
              for key in sources}
    cache_paths = MetricCacheLoader(args.metric_cache).metric_cache_paths
    tokens = sorted(cache_paths)
    if args.max_scenes:
        rng = np.random.default_rng(args.seed)
        tokens = sorted(rng.choice(tokens, size=min(args.max_scenes, len(tokens)), replace=False).tolist())
    tokens = tokens[args.shard::args.num_shards]
    os.environ["NAVSIM_OSS_SENSOR_BLOBS_PATH"] = SPLITS["test"]["oss"]
    loader = make_scene_loader("test", [log_of(cache_paths, t) for t in tokens], tokens, union_sensor_config(agents))
    tokens = [t for t in tokens if t in loader.scene_frames_dicts]
    index = {token: i for i, token in enumerate(tokens)}
    n, g, s = len(tokens), len(args.generators), len(args.scorers)
    print(f"[shard {args.shard}] {n} navtest scenes", flush=True)

    proposals = np.zeros((n, len(sources), 64, 8, 3), dtype=np.float32)
    own_score = np.zeros((n, len(sources), 64), dtype=np.float32)
    cross_score = np.zeros((n, s, g, 64), dtype=np.float32)  # DrivoR scorer s ranking Action-DiT set
    human = np.zeros((n, 8, 3), dtype=np.float32)
    keys = {key: sensor_key(agent.get_sensor_config()) for key, agent in agents.items()}
    start, done = time.time(), 0
    data = DataLoader(TokenDataset(loader, tokens, agents), batch_size=args.batch_size, num_workers=args.loader_workers)
    with torch.no_grad():
        for features, targets, batch_tokens in data:
            features = {key: {name: value.to(device) for name, value in group.items() if torch.is_tensor(value)}
                        for key, group in features.items()}
            rows = [index[t] for t in batch_tokens]
            human[rows] = targets["trajectory"].float().numpy()
            outputs = {key: model_forward(agent, features[keys[key]], batch_tokens, args.seed)
                       for key, agent in agents.items()}
            for j, key in enumerate(sources):
                proposals[rows, j] = outputs[key]["proposals"].float().cpu().numpy()
                own_score[rows, j] = outputs[key]["pdm_score"].float().cpu().numpy()
            for i, key in enumerate(args.scorers):
                model = agents[key]._drivor_model
                context = drivor_scene_context(model, features[keys[key]])
                check = drivor_score(model, outputs[key]["proposals"].float(), *context)
                if not torch.allclose(check, outputs[key]["pdm_score"].float(), atol=1e-4, rtol=1e-4):
                    raise RuntimeError(f"{key}: the extracted scoring branch does not reproduce DrivoR's own scores")
                for j, generator in enumerate(args.generators):
                    cross_score[rows, i, j] = drivor_score(model, outputs[generator]["proposals"].float(),
                                                           *context).cpu().numpy()
            done += len(rows)
            if done % (args.batch_size * 25) < len(rows) or done == n:
                rate = done / (time.time() - start)
                print(f"[shard {args.shard}] {done}/{n} scenes, {rate:.2f} scenes/s, "
                      f"ETA {(n - done) / rate / 60:.1f} min", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_dir / f"shard_{args.shard}.npz", tokens=np.array(tokens), sources=np.array(sources),
             generators=np.array(args.generators), scorers=np.array(args.scorers), proposals=proposals,
             own_score=own_score, cross_score=cross_score, human=human)


# ----------------------------------------------------------------------------- stage 2: official PDMS and summary

def _score_task(task):
    from navsim.evaluate.pairwise_pdm_score import pairwise_pdm_scores

    cache_path, trajectories = task
    with lzma.open(cache_path, "rb") as stream:
        metric_cache = pickle.load(stream)
    return pairwise_pdm_scores(metric_cache, trajectories, _SIMULATOR, _SCORER)


def merge_and_score(args):
    from hydra.utils import instantiate

    from navsim.agents.drivoR.proposal_analysis import compose_eval_config
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pairwise_pdm_score import PDM_FIELDS

    global _SIMULATOR, _SCORER
    shards = [np.load(path) for path in sorted(args.output_dir.glob("shard_*.npz"))]
    merged = {key: np.concatenate([shard[key] for shard in shards]) for key in
              ("tokens", "proposals", "own_score", "cross_score", "human")}
    merged.update({key: shards[0][key] for key in ("sources", "generators", "scorers")})
    cfg = compose_eval_config()
    _SIMULATOR, _SCORER = instantiate(cfg.simulator), instantiate(cfg.scorer)
    cache_paths = MetricCacheLoader(args.metric_cache).metric_cache_paths
    n, sets = merged["proposals"].shape[:2]
    # One task per scene: the 64 proposals of every set, then the human GT.
    tasks = ((cache_paths[t], np.concatenate([merged["proposals"][i].reshape(-1, 8, 3), merged["human"][i][None]]))
             for i, t in enumerate(merged["tokens"].tolist()))
    pdm = np.zeros((n, sets * 64 + 1, len(PDM_FIELDS)), dtype=np.float32)
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.score_workers, mp_context=multiprocessing.get_context("fork")) as pool:
        for i, rows in enumerate(pool.map(_score_task, tasks, chunksize=4)):
            pdm[i] = rows
            if (i + 1) % 500 == 0 or i + 1 == n:
                rate = (i + 1) / (time.time() - start)
                print(f"[scoring] {i + 1}/{n} scenes, {rate:.1f} scenes/s, ETA {(n - i - 1) / rate / 60:.1f} min",
                      flush=True)
    np.savez(args.output_dir / "pdm.npz", pdm=pdm, pdm_fields=np.array(PDM_FIELDS), **merged)


def summarize(args):
    data = np.load(args.output_dir / "pdm.npz")
    fields = data["pdm_fields"].tolist()
    sources, generators, scorers = (data[key].tolist() for key in ("sources", "generators", "scorers"))
    n = len(data["tokens"])
    pdm = data["pdm"][:, :-1].reshape(n, len(sources), 64, len(fields))
    score_col, columns = fields.index("score"), [fields.index(field) for _, field in METRICS]
    rows = np.arange(n)

    def pick(set_index, ranking):
        """[n, fields] of the proposal each scene's ranking puts first."""
        return pdm[rows, set_index, np.argmax(ranking, axis=1)]

    def entry(source, ranker, chosen, reference=None, changed=None):
        result = {"proposals": label(source), "ranked_by": ranker, "pdms": 100 * float(chosen[:, score_col].mean()),
                  **{name: 100 * float(chosen[:, col].mean()) for (name, _), col in zip(METRICS, columns)}}
        if reference is not None:  # paired difference to the own scorer on the same proposals
            diff = chosen[:, score_col] - reference[:, score_col]
            result["delta"] = 100 * float(diff.mean())
            result["delta_ci95"] = 196 * float(diff.std(ddof=1) / np.sqrt(n))
            result["changed_pick"] = float(changed)
        return result

    results = []
    for s, source in enumerate(scorers):
        index = sources.index(source)
        results.append(entry(source, "own scorer", pick(index, data["own_score"][:, index])))
        results[-1]["best_of_64"] = 100 * float(np.nanmax(pdm[:, index, :, score_col], axis=1).mean())
    for g, source in enumerate(generators):
        index = sources.index(source)
        own = pick(index, data["own_score"][:, index])
        results.append(entry(source, "own scorer", own))
        results[-1]["best_of_64"] = 100 * float(np.nanmax(pdm[:, index, :, score_col], axis=1).mean())
        for s, scorer in enumerate(scorers):
            ranking = data["cross_score"][:, s, g]
            changed = np.mean(np.argmax(ranking, axis=1) != np.argmax(data["own_score"][:, index], axis=1))
            results.append(entry(source, f"{label(scorer)} scorer", pick(index, ranking), own, changed))
    (args.output_dir / "summary.json").write_text(json.dumps({"scenes": n, "results": results}, indent=2),
                                                  encoding="utf-8")

    lines = ["# 用 DrivoR 的打分器给 Action-DiT 的 64 条候选排序", "",
             f"navtest 随机 {n} 个场景（`--max-scenes 0` 为全量）。PDMS 及子指标 ×100，均为被选中轨迹的官方分数。"
             "DrivoR 打分器连同它自己的图像编码器和自车状态编码一起使用；Action-DiT 每个场景用同一份初始噪声。"
             "差值为与该模型自带打分器在同一组候选上的配对差（±95% 置信区间）。", "",
             "| 候选来源 | 排序用的打分器 | PDMS | 差值 | 换选比例 | NC | DAC | TTC | C | EP | 64 条最优 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        delta = f"{r['delta']:+.2f} ± {r['delta_ci95']:.2f}" if "delta" in r else ""
        changed = f"{r['changed_pick']:.0%}" if "changed_pick" in r else ""
        best = f"{r['best_of_64']:.2f}" if "best_of_64" in r else ""
        lines.append(f"| {r['proposals']} | {r['ranked_by']} | {r['pdms']:.2f} | {delta} | {changed} | "
                     + " | ".join(f"{r[name]:.2f}" for name, _ in METRICS) + f" | {best} |")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    args = parse_args()
    if not args.report:
        compute_shard(args)
        return
    if not (args.output_dir / "pdm.npz").exists():
        merge_and_score(args)
    summarize(args)
    (args.output_dir / "run_config.json").write_text(json.dumps(
        {key: str(value) for key, value in vars(args).items()} | {"time": f"{datetime.now():%Y-%m-%d %H:%M}"},
        indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
