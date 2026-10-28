"""Rank-specific model randomness, separate from Lightning's sampler seed."""

import random

import numpy as np
import torch


def seed_model_randomness(base_seed: int, global_rank: int) -> int:
    """Use FastWAM's base_seed + global_rank rule after distributed setup.

    Do not call seed_everything here: changing PL_GLOBAL_SEED would also change
    Lightning's DistributedSampler seed across ranks.
    """
    process_seed = int(base_seed) + int(global_rank)
    random.seed(process_seed)
    np.random.seed(process_seed % (2 ** 32))
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)
    return process_seed
