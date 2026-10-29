"""
src/baselines/base.py
Abstract base class for all baseline models.
Phase 2 TFT comparison code iterates over BaselineModel instances generically.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaselineModel(ABC):
    """
    Common interface for all Phase 1 baseline models.

    Subclasses implement: fit(), predict(), predict_proba() (optional).
    save() / load() are provided here using joblib.
    """

    def __init__(self, model_name: str, random_seed: int = 42):
        self.model_name = model_name
        self.random_seed = random_seed
        self.model: Any = None
        self._feature_names: Optional[list] = None
        self._metadata: Dict = {}

    # ── Abstract interface ───────────────────────────────────────────────────

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineModel":
        """Fit model on training data. Must call assert_no_target_leakage first."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predictions."""
        ...

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Return class probabilities (classifiers only). Default: None."""
        return None

    # ── Save / Load ──────────────────────────────────────────────────────────

    def save(self, model_dir: str | Path) -> None:
        """
        Save model weights + metadata to model_dir/{model_name}/.
        """
        model_dir = Path(model_dir) / self.model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, model_dir / "model.joblib")

        self._metadata.update({
            "model_name": self.model_name,
            "random_seed": self.random_seed,
            "saved_at": datetime.now().isoformat(),
            "feature_names": self._feature_names,
        })
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

        logger.info(f"Model saved → {model_dir}")

    @classmethod
    def load(cls, model_dir: str | Path, model_name: str) -> "BaselineModel":
        """Load a saved model from disk."""
        path = Path(model_dir) / model_name
        model_path = path / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        instance = cls.__new__(cls)
        instance.model_name = model_name
        instance.model = joblib.load(model_path)

        meta_path = path / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                instance._metadata = json.load(f)
            instance.random_seed = instance._metadata.get("random_seed", 42)
            instance._feature_names = instance._metadata.get("feature_names", None)
        else:
            instance._metadata = {}
            instance.random_seed = 42
            instance._feature_names = None

        logger.info(f"Model loaded ← {path}")
        return instance

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _prepare_X(self, X: pd.DataFrame) -> np.ndarray:
        """Drop non-numeric and identifier columns before passing to sklearn."""
        drop_cols = ["patient_id", "visit_id", "visit_date", "ckd_stage"]
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")
        X = X.select_dtypes(include=[np.number])
        X = X.fillna(0)
        if self._feature_names is None:
            self._feature_names = X.columns.tolist()
        return X.values
