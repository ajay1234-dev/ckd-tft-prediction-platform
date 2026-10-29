"""
src/explainability/variable_importance.py
Extract and interpret Variable Selection Network (VSN) weights from the TFT.

All values returned here come from an actual model forward pass — never fabricated.
These weights indicate which features the model assigned more importance to for
a given prediction. They do NOT establish causal relationships.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Canonical feature name ordering as stored in the sequence .npy arrays
# (first 4 of the 10 stored static features match n_static_features=4 in config)
STATIC_FEATURE_NAMES: List[str] = [
    "age",
    "height_cm",
    "family_history_ckd",
    "baseline_bmi",
]

# All 29 time-varying feature names (exact order in historical_features.npy)
TV_FEATURE_NAMES: List[str] = [
    "weight_kg",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "serum_creatinine_mg_dl",
    "egfr_ml_min_1_73m2",
    "bun_mg_dl",
    "uacr_mg_g",
    "fasting_glucose_mg_dl",
    "hba1c_pct",
    "hemoglobin_g_dl",
    "anemia",
    "sodium_meq_l",
    "potassium_meq_l",
    "calcium_mg_dl",
    "phosphate_mg_dl",
    "bicarbonate_meq_l",
    "uric_acid_mg_dl",
    "acei_arb",
    "sglt2_inhibitor",
    "diuretic",
    "antihypertensive",
    "statin",
    "epo",
    "iron_supplement",
    "metformin",
    "insulin",
    "on_dialysis",
    "kidney_transplant",
]

# The 2 temporal features used (first 2 of the 3 stored)
TEMPORAL_FEATURE_NAMES: List[str] = [
    "visit_number",
    "days_from_baseline",
]

# Combined temporal VSN feature names: tv (29) + temporal (2) = 31
TEMPORAL_VSN_FEATURE_NAMES: List[str] = TV_FEATURE_NAMES + TEMPORAL_FEATURE_NAMES


def extract_static_importance(
    static_vsn_weights: torch.Tensor,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Convert static VSN softmax weights to a named feature-importance dict.

    Parameters
    ----------
    static_vsn_weights : Tensor [Batch, N_static] or [N_static]
        Variable-selection softmax weights for the static covariate encoder.
        Values are non-negative and sum to 1.0.
    feature_names : list of str, optional
        Feature names in the same order as the weight vector.
        Defaults to STATIC_FEATURE_NAMES.

    Returns
    -------
    dict mapping feature_name → importance_score (float, [0, 1])

    Notes
    -----
    Importance values indicate the model's learned variable selection weight
    for this prediction. They do NOT establish causal relationships.
    """
    if feature_names is None:
        feature_names = STATIC_FEATURE_NAMES

    # Handle batch dimension — average if batch
    weights = static_vsn_weights.detach().cpu()
    if weights.dim() == 2:
        weights = weights.mean(dim=0)  # [N_static]

    weights_np = weights.numpy().tolist()

    if len(weights_np) != len(feature_names):
        logger.warning(
            f"VSN weight length {len(weights_np)} != feature_names length {len(feature_names)}. "
            "Using index-based names."
        )
        feature_names = [f"static_feature_{i}" for i in range(len(weights_np))]

    return {name: float(w) for name, w in zip(feature_names, weights_np)}


def extract_temporal_importance(
    temporal_vsn_weights: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Convert temporal VSN weights to a named feature-importance dict.

    The temporal VSN produces per-timestep selection weights [Batch, SeqLen, N_tv+N_t].
    We average over valid (non-padded) timesteps to get a single importance score
    per feature.

    Parameters
    ----------
    temporal_vsn_weights : Tensor [Batch, SeqLen, N_features] or [SeqLen, N_features]
        Per-timestep variable-selection weights.
    mask : Tensor [Batch, SeqLen] or [SeqLen], optional
        1 for valid timesteps, 0 for padding. If None, all steps treated as valid.
    feature_names : list of str, optional
        Feature names. Defaults to TEMPORAL_VSN_FEATURE_NAMES (31 names).

    Returns
    -------
    dict mapping feature_name → mean_importance_score (float, [0, 1])

    Notes
    -----
    Averaged over valid visits only. Padding positions are excluded.
    Importance values indicate model focus, not causal importance.
    """
    if feature_names is None:
        feature_names = TEMPORAL_VSN_FEATURE_NAMES

    weights = temporal_vsn_weights.detach().cpu()
    if weights.dim() == 3:
        weights = weights[0]  # Take first (only) batch → [SeqLen, N_features]

    if mask is not None:
        m = mask.detach().cpu()
        if m.dim() == 2:
            m = m[0]  # [SeqLen]
        valid_mask = (m > 0).unsqueeze(-1).float()  # [SeqLen, 1]
        valid_count = valid_mask.sum().clamp(min=1)
        mean_weights = (weights * valid_mask).sum(dim=0) / valid_count  # [N_features]
    else:
        mean_weights = weights.mean(dim=0)  # [N_features]

    weights_np = mean_weights.numpy().tolist()

    if len(weights_np) != len(feature_names):
        logger.warning(
            f"Temporal VSN weight length {len(weights_np)} != feature_names length {len(feature_names)}."
        )
        feature_names = [f"tv_feature_{i}" for i in range(len(weights_np))]

    return {name: float(w) for name, w in zip(feature_names, weights_np)}


def top_k_features(importance_dict: Dict[str, float], k: int = 10) -> Dict[str, float]:
    """Return top-k features by importance, sorted descending."""
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_items[:k])
