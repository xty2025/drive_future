"""Profile per-stage cost of the PASP-DrivoR navtest PDMS path (no ray worker).

Usage: python scripts/evaluation/profile_pasp_eval_stages.py [num_scenes]
"""
import sys
import time
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.evaluate.pdm_score import pdm_score

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "navsim/planning/script/config/pdm_scoring"
OVERRIDES = [
    "train_test_split=navtest",
    "agent.checkpoint_path="
    + str(REPO / "outputs/pasp_drivor_dpo_8gpu_50ep_fixed/lightning_logs/version_1/checkpoints/last.ckpt"),
    "agent.scheduler_args.num_epochs=1",
    "agent.batch_size=1",
    "metric_cache_path=/mnt/zhaozc_workspace/project/FastWAM/data/metric_cache_navtest",
    "navsim_log_path=/mnt/data/navsim/test_navsim_logs/test",
    "sensor_blobs_path=/mnt/data/navsim/test_sensor_blobs/test",
    "output_dir=/tmp/pasp_profile",
]


def main() -> None:
    num_scenes = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg: DictConfig = hydra.compose(config_name="default_run_pdm_score", overrides=OVERRIDES)

    agent = instantiate(cfg.agent)
    t0 = time.time()
    agent.initialize()
    print(f"[init] {time.time() - t0:.1f}s device={next(agent.parameters()).device}", flush=True)

    simulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.max_scenes = num_scenes
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    tokens = [t for t in scene_loader.tokens if t in set(metric_cache_loader.tokens)][:num_scenes]

    stages = {"input": 0.0, "trajectory": 0.0, "pdm": 0.0}
    for token in tokens:
        t0 = time.time()
        agent_input = scene_loader.get_agent_input_from_token(token)
        t1 = time.time()
        trajectory = agent.compute_trajectory(agent_input)
        t2 = time.time()
        import lzma
        import pickle

        with lzma.open(metric_cache_loader.metric_cache_paths[token], "rb") as f:
            metric_cache = pickle.load(f)
        pdm_score(
            metric_cache=metric_cache,
            model_trajectory=trajectory,
            future_sampling=simulator.proposal_sampling,
            simulator=simulator,
            scorer=scorer,
        )
        t3 = time.time()
        stages["input"] += t1 - t0
        stages["trajectory"] += t2 - t1
        stages["pdm"] += t3 - t2
        print(f"[scene {token[:8]}] input={t1 - t0:.2f}s traj={t2 - t1:.2f}s pdm={t3 - t2:.2f}s", flush=True)
    n = len(tokens)
    print(f"[mean over {n}] " + " ".join(f"{k}={v / n:.2f}s" for k, v in stages.items()), flush=True)


if __name__ == "__main__":
    main()
