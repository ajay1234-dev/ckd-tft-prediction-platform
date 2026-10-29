"""
src/explainability/patient_explanation.py
Assemble a complete, patient-specific explanation object from TFT inference outputs.

All values come from an actual model forward pass on the patient's real sequence.
No values are fabricated, hard-coded, or sampled randomly.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

import torch

from src.explainability.variable_importance import (
    extract_static_importance,
    extract_temporal_importance,
    top_k_features,
    STATIC_FEATURE_NAMES,
    TEMPORAL_VSN_FEATURE_NAMES,
)
from src.explainability.temporal_attention import extract_attention_per_timestep
from src.features.ckd_stage import decode_stage, STAGE_ORDER
from src.utils.logging import get_logger

logger = get_logger(__name__)

_DISCLAIMER = (
    "Research prototype using synthetically generated longitudinal CKD data. "
    "Not intended for clinical diagnosis or treatment. "
    "Feature importance and attention values are model interpretability signals — "
    "they do not establish causal relationships."
)

_HORIZONS = ["3m", "6m", "12m"]


def _build_stage_output(stage_logits_1d: torch.Tensor) -> Dict[str, Any]:
    """Convert [6] stage logits to prediction dict with full probability distribution."""
    probs = torch.softmax(stage_logits_1d, dim=-1).cpu().numpy()
    pred_idx = int(probs.argmax())
    # Model class 0-5 maps to stage codes 1-6
    try:
        predicted_stage = decode_stage(pred_idx + 1)
    except ValueError:
        predicted_stage = f"unknown_{pred_idx}"

    prob_dict = {stage: float(probs[i]) for i, stage in enumerate(STAGE_ORDER)}
    return {
        "predicted_stage": predicted_stage,
        "probabilities": prob_dict,
    }


def _build_progression_output(prog_logit: float) -> Dict[str, Any]:
    """Convert scalar progression logit to dict."""
    import math
    prob = 1.0 / (1.0 + math.exp(-prog_logit))
    return {
        "probability": float(prob),
        "predicted_class": int(prob > 0.5),
    }


def assemble_patient_explanation(
    patient_id: str,
    raw_outputs: Dict[str, torch.Tensor],
    mask: Optional[torch.Tensor] = None,
    visit_numbers: Optional[List[int]] = None,
    days_from_baseline: Optional[List[float]] = None,
    checkpoint_id: str = "best_model",
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Assemble a full JSON-serializable explanation for one patient.

    Parameters
    ----------
    patient_id : str
    raw_outputs : dict
        Output from TemporalFusionTransformer.forward(), including:
        - stage_logits [1, 3, 6]
        - egfr_pred [1, 3]
        - progression_logits [1, 3]
        - attention_weights [1, SeqLen, SeqLen]
        - static_vsn_weights [1, N_static]
        - temporal_vsn_weights [1, SeqLen, N_tv+N_t]
    mask : Tensor [1, SeqLen], optional
    visit_numbers : list of int, optional
    days_from_baseline : list of float, optional
    checkpoint_id : str
    top_k : int
        Number of top features to include in explanation.

    Returns
    -------
    dict — fully JSON-serializable explanation object
    """
    stage_logits = raw_outputs["stage_logits"][0]       # [3, 6]
    egfr_pred = raw_outputs["egfr_pred"][0]             # [3]
    prog_logits = raw_outputs["progression_logits"][0]  # [3]
    attn_weights = raw_outputs["attention_weights"]     # [1, S, S]
    static_vsn = raw_outputs["static_vsn_weights"]      # [1, N_s]
    temporal_vsn = raw_outputs["temporal_vsn_weights"]  # [1, S, N_tv]

    # ── Predictions ──────────────────────────────────────────────────
    predictions: Dict[str, Any] = {
        "stage": {},
        "egfr": {},
        "progression": {},
    }
    for i, hz in enumerate(_HORIZONS):
        predictions["stage"][hz] = _build_stage_output(stage_logits[i])
        predictions["egfr"][hz] = float(egfr_pred[i].item())
        predictions["progression"][hz] = _build_progression_output(
            float(prog_logits[i].item())
        )

    # ── Feature importance ────────────────────────────────────────────
    static_importance = extract_static_importance(static_vsn, STATIC_FEATURE_NAMES)
    temporal_importance = extract_temporal_importance(temporal_vsn, mask, TEMPORAL_VSN_FEATURE_NAMES)

    feature_importance = {
        "static": top_k_features(static_importance, k=top_k),
        "time_varying": top_k_features(temporal_importance, k=top_k),
        "note": (
            "Importance scores are TFT variable-selection softmax weights averaged "
            "across valid visits. They indicate model focus, not causal importance."
        ),
    }

    # ── Temporal attention ─────────────────────────────────────────────
    temporal_attention = extract_attention_per_timestep(
        attn_weights,
        mask=mask,
        visit_numbers=visit_numbers,
        days_from_baseline=days_from_baseline,
    )

    # ── Metadata ──────────────────────────────────────────────────────
    metadata = {
        "patient_id": patient_id,
        "model_checkpoint": checkpoint_id,
        "explanation_method": "tft_native",
        "explanation_components": [
            "variable_selection_network_weights",
            "temporal_self_attention_weights",
        ],
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "disclaimer": _DISCLAIMER,
    }

    return {
        "patient_id": patient_id,
        "prediction": predictions,
        "feature_importance": feature_importance,
        "temporal_attention": temporal_attention,
        "metadata": metadata,
    }
