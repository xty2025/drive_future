"""Train on the navtrain train/val union using the maintained training runtime."""

import json
import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, open_dict

from navsim.planning.script import run_training as training


logger = logging.getLogger(__name__)


def prepare_trainval_config(cfg: DictConfig):
    """Both raw-data and cache-only paths consume the same deduplicated union."""
    original_train = list(cfg.train_logs)
    validation = list(cfg.val_logs)
    combined = list(dict.fromkeys(original_train + validation))
    cfg.train_logs = combined
    with open_dict(cfg):
        if 'runtime' not in cfg:
            cfg.runtime = {}
        with open_dict(cfg.runtime):
            cfg.runtime.training_data_scope = 'trainval'
            cfg.runtime.scheduler_dataset_size_from_data = True
            cfg.runtime.validation_is_training_subset = True
    return dict(training_data_scope='trainval',
                original_train_log_count=len(set(original_train)),
                validation_log_count=len(set(validation)),
                combined_train_log_count=len(combined),
                validation_is_training_subset=True,
                effective_train_logs=combined, validation_logs=validation)


def build_datasets(cfg: DictConfig, agent):
    """Keep the original full-entry helper available to direct callers."""
    prepare_trainval_config(cfg)
    return training.build_datasets(cfg, agent)


def run(cfg: DictConfig):
    scope = prepare_trainval_config(cfg)
    logger.info('Full trainval: %s train logs + %s val logs -> %s unique training logs. '
                'Per-epoch val metrics monitor a subset included in training.',
                scope['original_train_log_count'], scope['validation_log_count'],
                scope['combined_train_log_count'])
    if int(os.environ.get('RANK', '0')) == 0:
        output = Path(cfg.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output/'trainval_full_scope.json').write_text(json.dumps(scope, indent=2)+'\n')
    # Reuse scoped checkpoint discovery, rank RNG, efficient loaders, FP32
    # validation, custom Lightning modules and all-epoch checkpoint callbacks.
    return training.main.__wrapped__(cfg)


@hydra.main(config_path='config/training', config_name='default_training_full', version_base=None)
def main(cfg: DictConfig):
    return run(cfg)


if __name__ == '__main__':
    main()
