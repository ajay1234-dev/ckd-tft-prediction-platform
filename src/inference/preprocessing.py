"""
src/inference/preprocessing.py
Apply Phase 1 fitted preprocessing artifacts to inference-time patient data.

CRITICAL: Never calls fit() or fit_transform() — only transform().
All scalers/imputers/encoders were fitted exclusively on Phase 1 training data.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Default artifacts directory
_DEFAULT_ARTIFACTS_DIR = Path("models/preprocessing")

# Number of features the trained model accepts (from config)
_N_STATIC = 4
_N_TV = 29
_N_TEMPORAL = 2


class InferencePreprocessor:
    """
    Applies Phase 1's fitted preprocessing artifacts to new patient data.

    Loads imputer, scaler, and encoder from disk once and reuses them
    for every inference call. Never refits on inference data.
    """

    def __init__(self, artifacts_dir: Optional[str | Path] = None):
        if artifacts_dir is None:
            artifacts_dir = _DEFAULT_ARTIFACTS_DIR
        self.artifacts_dir = Path(artifacts_dir)
        self._loaded = False
        self._imputer = None
        self._scaler = None
        self._encoder = None
        self._feature_config: Optional[Dict] = None
        self._manifest: Optional[Dict] = None

    def load(self) -> "InferencePreprocessor":
        """Load all preprocessing artifacts from disk."""
        if self._loaded:
            return self

        required = ["imputer.joblib", "scaler.joblib", "encoders.joblib",
                    "feature_config.json", "preprocessing_manifest.json"]
        for fname in required:
            p = self.artifacts_dir / fname
            if not p.exists():
                raise FileNotFoundError(
                    f"Preprocessing artifact not found: {p}. "
                    "Ensure Phase 1 preprocessing has been run."
                )

        self._imputer = joblib.load(self.artifacts_dir / "imputer.joblib")
        self._scaler = joblib.load(self.artifacts_dir / "scaler.joblib")
        self._encoder = joblib.load(self.artifacts_dir / "encoders.joblib")

        with open(self.artifacts_dir / "feature_config.json") as f:
            self._feature_config = json.load(f)
        with open(self.artifacts_dir / "preprocessing_manifest.json") as f:
            self._manifest = json.load(f)

        self._loaded = True
        logger.info(f"Preprocessing artifacts loaded from {self.artifacts_dir}")
        return self

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_sequence_from_arrays(
        self,
        static_arr: np.ndarray,   # [N_static_stored] or [N_static_stored] (10 stored, use first 4)
        hist_arr: np.ndarray,      # [SeqLen, N_tv] (29)
        time_arr: np.ndarray,      # [SeqLen, N_temporal_stored] (3 stored, use first 2)
        mask_arr: np.ndarray,      # [SeqLen]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert pre-processed numpy arrays (as stored in the .npy files)
        to model-ready tensors, applying the correct feature slicing.

        Parameters
        ----------
        static_arr : [N_static_stored] — use first _N_STATIC columns
        hist_arr : [SeqLen, N_tv]
        time_arr : [SeqLen, N_temporal_stored] — use first _N_TEMPORAL columns
        mask_arr : [SeqLen]

        Returns
        -------
        (static_tensor [1, 4], hist_tensor [1, 8, 29],
         time_tensor [1, 8, 2], mask_tensor [1, 8])
        """
        s = torch.tensor(static_arr[:_N_STATIC], dtype=torch.float32).unsqueeze(0)
        h = torch.tensor(hist_arr, dtype=torch.float32).unsqueeze(0)
        t = torch.tensor(time_arr[:, :_N_TEMPORAL], dtype=torch.float32).unsqueeze(0)
        m = torch.tensor(mask_arr, dtype=torch.int8).unsqueeze(0)
        return s, h, t, m


# Module-level singleton
_preprocessor: Optional[InferencePreprocessor] = None


def get_preprocessor(artifacts_dir: Optional[str | Path] = None) -> InferencePreprocessor:
    """Get or create the module-level InferencePreprocessor singleton."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = InferencePreprocessor(artifacts_dir).load()
    return _preprocessor
