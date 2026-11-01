"""Reproducibility: seed every RNG we use."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic_torch: bool = False) -> None:
    """Seed python, numpy and (if installed) torch.

    Note: PYTHONHASHSEED only affects hashing in *new* interpreter processes;
    it is set here for any subprocesses we spawn.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
