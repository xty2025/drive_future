import os
import random
from typing import Tuple
from pathlib import Path
import logging
import pickle
from datetime import datetime

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import torch
import torch.distributed as dist
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"

def dist_ready():
    return dist.is_available() and dist.is_initialized()

def build_dataloader(dataset, params, *, shuffle: bool, seed: int) -> DataLoader:
    """Keep loader/worker RNG independent of model RNG and support zero workers."""
    options = dict(params)
    workers = int(options.get("num_workers", 0))
    if workers == 0:
        options.pop("prefetch_factor", None)
        options["persistent_workers"] = False
    else:
        options.setdefault("persistent_workers", True)
    options.setdefault("drop_last", bool(shuffle))  # Keep the final validation batch.
    options["shuffle"] = shuffle
    # All ranks share this seed; Lightning supplies rank-aware worker_init_fn
    # and a DistributedSampler using the unchanged PL_GLOBAL_SEED.
    options["generator"] = torch.Generator().manual_seed(int(seed))
    return DataLoader(dataset, **options)


def resolve_train_checkpoint(output_dir, explicit_path=None, auto_resume: bool = True):
    """Honor explicit resumes; automatic discovery is confined to this run."""
    if explicit_path is not None or not auto_resume:
        return explicit_path
    checkpoints = [path for path in Path(output_dir).rglob("*.ckpt") if path.is_file()]
    if not checkpoints:
        return None
    last_checkpoints = [path for path in checkpoints if path.name == "last.ckpt"]
    if last_checkpoints:
        checkpoints = last_checkpoints
    return str(max(checkpoints, key=lambda path: (path.stat().st_mtime_ns, str(path))))



def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    
    print("Train without caching....")
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    cfg.train_ckpt_path = resolve_train_checkpoint(
        cfg.output_dir, cfg.train_ckpt_path, bool(cfg.get("auto_resume", True)))
    if cfg.get("runtime", {}).get("require_resume", False) and not cfg.train_ckpt_path:
        raise ValueError("Resume required but no checkpoint exists in this run; specify train_ckpt_path.")
    logger.info("Training checkpoint: %s", cfg.train_ckpt_path)

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module_class = getattr(agent, "lightning_module_class", AgentLightningModule)
    lightning_module = lightning_module_class(
        agent=agent,
        log_local_steps=cfg.get("runtime", {}).get("log_local_steps", True),
        console_log_interval=cfg.get("runtime", {}).get("console_log_interval", 25),
        performance_diagnostics=cfg.get("runtime", {}).get("performance_diagnostics"),
        training_seed=cfg.seed if cfg.get("runtime", {}).get("rank_seeded_training", True) else None,
        validation_fp32=cfg.get("runtime", {}).get("validation_fp32", True),
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = build_dataloader(train_data, cfg.dataloader.params, shuffle=True, seed=cfg.seed)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = build_dataloader(val_data, cfg.dataloader.params, shuffle=False, seed=cfg.seed + 1)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())

    if cfg.validation_run:
        logger.info("Starting Validation")
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        dump_root = os.path.join(os.getenv('SUBSCORE_PATH'), "navsim1_pdm_scores", cfg.experiment_name)
        os.makedirs(dump_root, exist_ok=True)
        dump_path = os.path.join(dump_root, f"{timestamp}.pkl")
        trainer.validate(
            model=lightning_module,
            dataloaders=[val_dataloader],
            ckpt_path=cfg.train_ckpt_path,
            verbose=True
        )
        logger.info("Running predictions to collect trajectories")
        predictions = trainer.predict(
            AgentLightningModule(agent=agent, for_viz=True),
            val_dataloader,
            return_predictions=True
        )

        if dist_ready():
            dist.barrier()
        
        world_size = dist.get_world_size() if dist_ready() else 1
        all_predictions = [None for _ in range(world_size)]

        if dist_ready():
            dist.all_gather_object(all_predictions, predictions)
        else:
            all_predictions = [predictions]

        rank = dist.get_rank() if dist_ready() else 0
        if rank != 0:
            return None

        merged_predictions = {}
        for proc_prediction in all_predictions:
            for d in proc_prediction:
                merged_predictions.update(d)

        pickle.dump(predictions, open(dump_path, 'wb'))
    else:
        logger.info("Starting Training")
        trainer.fit(
            model=lightning_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=cfg.train_ckpt_path
        )


if __name__ == "__main__":
    main()
