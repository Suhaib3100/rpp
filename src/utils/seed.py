"""Reproducibility helpers: global seeding and deterministic dataloader workers."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG this project draws from.

    With ``deterministic=True`` cuDNN autotuning is disabled so that repeated
    runs with the same seed produce bitwise-identical results. On CPU (this
    project's training device) the flags are harmless no-ops.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Give each dataloader worker a distinct, reproducible numpy/random seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
