"""
src/schemas/explanation.py
Pydantic response models for /explain endpoint.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AttentionStep(BaseModel):
    """Attention weight for a single historical time step."""
    visit_index: int = Field(..., description="0-based position in the sequence")
    visit_number: int = Field(..., description="Visit number (1-based)")
    days_from_baseline: Optional[float] = Field(None, description="Days from first visit")
    attention_weight: float = Field(..., ge=0.0, description="Attention weight from last valid step to this step")
    is_padding: bool = Field(..., description="True if this step is padded (should not be interpreted)")
    disclaimer: str = Field(..., description="Interpretability disclaimer")


class FeatureImportance(BaseModel):
    """Variable selection importance scores for static and time-varying features."""
    static: Dict[str, float] = Field(
        ..., description="Static feature importance scores (VSN softmax weights)"
    )
    time_varying: Dict[str, float] = Field(
        ..., description="Time-varying feature importance scores (avg VSN weights over valid visits)"
    )
    note: str = Field(
        ...,
        description="Explains what these importance scores represent and their limitations"
    )


class ExplanationMeta(BaseModel):
    patient_id: str
    model_checkpoint: str
    explanation_method: str = Field("tft_native", description="tft_native = VSN weights + attention")
    explanation_components: List[str]
    generated_at: str
    disclaimer: str


class ExplanationResponse(BaseModel):
    """Full response schema for POST /explain."""
    patient_id: str
    prediction: Dict[str, Any] = Field(..., description="Same prediction output as /predict")
    feature_importance: FeatureImportance
    temporal_attention: List[AttentionStep]
    metadata: ExplanationMeta

    model_config = {"json_schema_extra": {
        "example": {
            "patient_id": "12345",
            "prediction": {"stage": {}, "egfr": {}, "progression": {}},
            "feature_importance": {
                "static": {"age": 0.31, "family_history_ckd": 0.28},
                "time_varying": {"egfr_ml_min_1_73m2": 0.19, "serum_creatinine_mg_dl": 0.14},
                "note": "Model interpretability signal, not causal importance."
            },
            "temporal_attention": [
                {"visit_index": 0, "visit_number": 1, "days_from_baseline": 0.0,
                 "attention_weight": 0.12, "is_padding": False,
                 "disclaimer": "Attention weights indicate model focus, not causal importance."}
            ],
            "metadata": {
                "patient_id": "12345",
                "model_checkpoint": "best_model.pt",
                "explanation_method": "tft_native",
                "explanation_components": ["variable_selection_network_weights", "temporal_self_attention_weights"],
                "generated_at": "2026-09-14T00:00:00Z",
                "disclaimer": "Research prototype. Not for clinical use."
            }
        }
    }}
