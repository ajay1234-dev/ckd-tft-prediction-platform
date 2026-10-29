"""
src/inference/predictor.py
Main predictor class — two modes:
  1. Prediction-only  (fast, no explanation capture overhead)
  2. Prediction + explanation (full explainability outputs)

Also provides patient lookup from the Phase 1 test/val/train sequence stores.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.inference.model_loader import ModelLoader, get_model_loader
from src.inference.preprocessing import InferencePreprocessor, get_preprocessor
from src.explainability.patient_explanation import assemble_patient_explanation
from src.features.ckd_stage import decode_stage, STAGE_ORDER
from src.utils.logging import get_logger

logger = get_logger(__name__)

_HORIZONS = ["3m", "6m", "12m"]
_DISCLAIMER = (
    "Research prototype using synthetically generated longitudinal CKD data. "
    "Not intended for clinical diagnosis or treatment."
)


class PatientStore:
    """
    Indexed lookup of synthetic patients from Phase 1 sequence arrays.
    Loads once at startup; O(1) lookup by patient index or string ID.
    """

    def __init__(self, sequences_dir: str | Path = "data/processed/sequences",
                 split: str = "test"):
        self.sequences_dir = Path(sequences_dir)
        self.split = split
        self._loaded = False
        self._patient_ids: Optional[np.ndarray] = None
        self._static: Optional[np.ndarray] = None
        self._hist: Optional[np.ndarray] = None
        self._time: Optional[np.ndarray] = None
        self._masks: Optional[np.ndarray] = None
        self._spec: Optional[Dict] = None
        self._id_to_idx: Dict[str, int] = {}

    def load(self) -> "PatientStore":
        if self._loaded:
            return self
        split_dir = self.sequences_dir / self.split
        logger.info(f"Loading patient store from {split_dir}...")

        self._patient_ids = np.load(split_dir / "patient_ids.npy", allow_pickle=True)
        self._static = np.load(split_dir / "static_features.npy")
        self._hist = np.load(split_dir / "historical_features.npy")
        self._time = np.load(split_dir / "time_features.npy")
        self._masks = np.load(split_dir / "masks.npy")
        with open(split_dir / "spec.json") as f:
            self._spec = json.load(f)

        # Build string-ID lookup
        for idx, pid in enumerate(self._patient_ids):
            self._id_to_idx[str(pid)] = idx

        self._loaded = True
        logger.info(f"Patient store loaded: {len(self._patient_ids)} patients")
        return self

    def __len__(self) -> int:
        return len(self._patient_ids) if self._patient_ids is not None else 0

    def get_patient(self, patient_id: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Look up a patient by string ID. Returns None if not found.

        Returns dict with keys: patient_id, static, hist, time, mask
        """
        if patient_id not in self._id_to_idx:
            return None
        idx = self._id_to_idx[patient_id]
        return {
            "patient_id": patient_id,
            "index": idx,
            "static": self._static[idx],
            "hist": self._hist[idx],
            "time": self._time[idx],
            "mask": self._masks[idx],
        }

    def list_patient_ids(self, limit: int = 100) -> List[str]:
        return [str(pid) for pid in self._patient_ids[:limit]]

    @property
    def feature_names(self) -> Optional[Dict[str, List[str]]]:
        return self._spec.get("feature_names") if self._spec else None


class CKDPredictor:
    """
    High-level predictor combining model + preprocessing + patient store.

    Supports two modes:
    - predict(patient_id)            → prediction-only dict
    - predict_with_explanation(...)  → full prediction + explanation dict
    """

    def __init__(
        self,
        model_loader: Optional[ModelLoader] = None,
        preprocessor: Optional[InferencePreprocessor] = None,
        patient_store: Optional[PatientStore] = None,
        device: str = "cpu",
    ):
        self._loader = model_loader or get_model_loader()
        self._preprocessor = preprocessor or get_preprocessor()
        self._patient_store = patient_store
        self._device = device
        self._model = None
        self._config = None

    def initialize(self) -> "CKDPredictor":
        """Load model + preprocessing (idempotent)."""
        self._model, self._config = self._loader.load(self._device)
        if not self._preprocessor.is_loaded:
            self._preprocessor.load()
        if self._patient_store is not None:
            self._patient_store.load()
        return self

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def predict(self, patient_id: str) -> Dict[str, Any]:
        """
        Prediction-only mode — runs forward pass, returns structured predictions.
        Does NOT capture explanation weights for maximum efficiency.
        """
        patient_data = self._get_patient_or_raise(patient_id)
        s, h, t, m = self._preprocessor.get_sequence_from_arrays(
            patient_data["static"], patient_data["hist"],
            patient_data["time"], patient_data["mask"],
        )
        s, h, t, m = s.to(self._device), h.to(self._device), t.to(self._device), m.to(self._device)

        with torch.no_grad():
            out = self._model(s, h, t, m)

        return self._format_prediction(patient_id, out)

    def predict_with_explanation(self, patient_id: str) -> Dict[str, Any]:
        """
        Full prediction + explanation mode — returns predictions AND
        TFT-native explainability (VSN weights + temporal attention).
        """
        patient_data = self._get_patient_or_raise(patient_id)
        s, h, t, m = self._preprocessor.get_sequence_from_arrays(
            patient_data["static"], patient_data["hist"],
            patient_data["time"], patient_data["mask"],
        )
        s, h, t, m = s.to(self._device), h.to(self._device), t.to(self._device), m.to(self._device)

        with torch.no_grad():
            out = self._model(s, h, t, m)

        # Extract visit metadata for attention display
        time_data = patient_data["time"]
        mask_data = patient_data["mask"]
        valid_indices = [i for i in range(len(mask_data)) if mask_data[i] > 0]
        visit_numbers = [int(time_data[i, 0]) for i in range(len(mask_data))]
        days_list = [float(time_data[i, 1]) for i in range(len(mask_data))]

        explanation = assemble_patient_explanation(
            patient_id=patient_id,
            raw_outputs=out,
            mask=m,
            visit_numbers=visit_numbers,
            days_from_baseline=days_list,
            checkpoint_id=self._loader.checkpoint_name,
        )
        return explanation

    def predict_from_tensors(
        self,
        patient_id: str,
        static: torch.Tensor,
        hist: torch.Tensor,
        time: torch.Tensor,
        mask: torch.Tensor,
        with_explanation: bool = False,
    ) -> Dict[str, Any]:
        """Raw tensor mode — for custom sequences not looked up by ID."""
        s = static.to(self._device)
        h = hist.to(self._device)
        t = time.to(self._device)
        m = mask.to(self._device)

        with torch.no_grad():
            out = self._model(s, h, t, m)

        if with_explanation:
            return assemble_patient_explanation(patient_id, out, mask=m,
                                                checkpoint_id=self._loader.checkpoint_name)
        return self._format_prediction(patient_id, out)

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _get_patient_or_raise(self, patient_id: str) -> Dict:
        if self._patient_store is None:
            raise RuntimeError("PatientStore not configured.")
        data = self._patient_store.get_patient(patient_id)
        if data is None:
            raise KeyError(f"Patient ID '{patient_id}' not found in {self._patient_store.split} split.")
        return data

    @staticmethod
    def _format_prediction(patient_id: str, out: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        stage_logits = out["stage_logits"][0]     # [3, 6]
        egfr_pred = out["egfr_pred"][0]           # [3]
        prog_logits = out["progression_logits"][0]  # [3]

        predictions: Dict[str, Any] = {"stage": {}, "egfr": {}, "progression": {}}

        for i, hz in enumerate(_HORIZONS):
            probs = torch.softmax(stage_logits[i], dim=-1).cpu().numpy()
            pred_idx = int(probs.argmax())
            try:
                predicted_stage = decode_stage(pred_idx + 1)
            except ValueError:
                predicted_stage = f"unknown_{pred_idx}"

            predictions["stage"][hz] = {
                "predicted_stage": predicted_stage,
                "probabilities": {stage: float(probs[j]) for j, stage in enumerate(STAGE_ORDER)},
            }
            predictions["egfr"][hz] = float(egfr_pred[i].item())

            logit_val = float(prog_logits[i].item())
            prog_prob = 1.0 / (1.0 + math.exp(-logit_val))
            predictions["progression"][hz] = {
                "probability": float(prog_prob),
                "predicted_class": int(prog_prob > 0.5),
            }

        return {
            "patient_id": patient_id,
            "prediction": predictions,
            "metadata": {
                "disclaimer": _DISCLAIMER,
                "data_type": "synthetic",
            },
        }
