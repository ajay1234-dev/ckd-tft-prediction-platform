"""
src/utils/seeding.py
Fixed random seed management — must be called at the start of every script/notebook.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import os
import random

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and environment.
    Should be called at the top of every training/evaluation script.

    Parameters
    ----------
    seed : int
        Random seed value (default: 42 as defined in config.yaml).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Scikit-learn uses numpy's random state, covered above.
    # XGBoost / LightGBM accept seed via model params (handled in baselines).

    logger.info(f"Global random seed set to {seed}")
