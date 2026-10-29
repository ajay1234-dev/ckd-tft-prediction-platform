"""
src/api/routes/model_info.py
GET /model/info — model metadata endpoint.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from fastapi import APIRouter, Depends
from src.api.dependencies import get_predictor
from src.inference.predictor import CKDPredictor

router = APIRouter(tags=["Model"])

_DISCLAIMER = (
    "Research prototype using synthetically generated longitudinal CKD data. "
    "Not intended for clinical diagnosis or treatment. "
    "This model has not been clinically validated."
)


@router.get(
    "/model/info",
    summary="Model metadata",
    description=(
        "Returns model architecture summary, supported tasks/horizons, "
        "and the synthetic-data disclaimer."
    ),
)
def model_info(predictor: CKDPredictor = Depends(get_predictor)):
    config = predictor._config or {}
    mc = config.get("model", {})
    tc = config.get("training", {})

    return {
        "model_type": "TemporalFusionTransformer",
        "checkpoint": predictor._loader.checkpoint_name,
        "data_type": "synthetic",
        "disclaimer": _DISCLAIMER,
        "architecture": {
            "hidden_size": mc.get("hidden_size"),
            "num_attention_heads": mc.get("num_attention_heads"),
            "n_static_features": mc.get("n_static_features"),
            "n_tv_features": mc.get("n_tv_features"),
            "n_temporal_features": mc.get("n_temporal_features"),
            "n_classes": mc.get("n_classes"),
        },
        "tasks": ["stage_classification", "egfr_regression", "progression_binary"],
        "horizons": ["3m", "6m", "12m"],
        "stage_labels": ["G1", "G2", "G3a", "G3b", "G4", "G5"],
        "explainability_methods": ["tft_variable_selection", "tft_temporal_attention"],
        "training": {
            "optimizer": tc.get("optimizer"),
            "learning_rate": tc.get("learning_rate"),
            "num_epochs": tc.get("num_epochs"),
            "random_seed": config.get("reproducibility", {}).get("random_seed"),
        },
        "limitations": [
            "Model trained on synthetic data only.",
            "Not clinically validated.",
            "Explainability outputs are model interpretability signals, not causal evidence.",
            "External validation required before any real-world clinical use.",
        ],
    }
