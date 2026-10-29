"""
src/schemas/prediction.py
Pydantic response models for /predict endpoint.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

from typing import Dict, Optional
from pydantic import BaseModel, Field


class HorizonStage(BaseModel):
    """CKD stage prediction for one horizon."""
    predicted_stage: str = Field(..., description="Predicted CKD stage label (G1, G2, G3a, G3b, G4, G5)")
    probabilities: Dict[str, float] = Field(
        ..., description="Full probability distribution over all 6 CKD stages"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "predicted_stage": "G3a",
            "probabilities": {"G1": 0.02, "G2": 0.08, "G3a": 0.61, "G3b": 0.19, "G4": 0.07, "G5": 0.03}
        }
    }}


class HorizonProgression(BaseModel):
    """Progression risk prediction for one horizon."""
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of CKD progression")
    predicted_class: int = Field(..., description="0=no progression, 1=progression (threshold 0.5)")


class StagePredictions(BaseModel):
    """Stage predictions across all 3 horizons."""
    h3m: HorizonStage = Field(..., alias="3m")
    h6m: HorizonStage = Field(..., alias="6m")
    h12m: HorizonStage = Field(..., alias="12m")

    model_config = {"populate_by_name": True}


class EGFRPredictions(BaseModel):
    """eGFR point-estimate predictions across all 3 horizons."""
    h3m: float = Field(..., alias="3m", description="Predicted eGFR at 3 months (standardised)")
    h6m: float = Field(..., alias="6m", description="Predicted eGFR at 6 months (standardised)")
    h12m: float = Field(..., alias="12m", description="Predicted eGFR at 12 months (standardised)")

    model_config = {"populate_by_name": True}


class ProgressionPredictions(BaseModel):
    """Progression predictions across all 3 horizons."""
    h3m: HorizonProgression = Field(..., alias="3m")
    h6m: HorizonProgression = Field(..., alias="6m")
    h12m: HorizonProgression = Field(..., alias="12m")

    model_config = {"populate_by_name": True}


class PredictionMeta(BaseModel):
    disclaimer: str
    data_type: str = Field("synthetic", description="Always 'synthetic' for this research prototype")


class AllPredictions(BaseModel):
    stage: Dict[str, HorizonStage]
    egfr: Dict[str, float]
    progression: Dict[str, HorizonProgression]


class PredictionResponse(BaseModel):
    """Full response schema for POST /predict."""
    patient_id: str
    prediction: AllPredictions
    metadata: PredictionMeta

    model_config = {"json_schema_extra": {
        "example": {
            "patient_id": "12345",
            "prediction": {
                "stage": {
                    "3m": {"predicted_stage": "G3a", "probabilities": {"G1": 0.02}},
                },
                "egfr": {"3m": -0.45, "6m": -0.52, "12m": -0.61},
                "progression": {
                    "3m": {"probability": 0.34, "predicted_class": 0},
                },
            },
            "metadata": {
                "disclaimer": "Research prototype. Not for clinical use.",
                "data_type": "synthetic"
            }
        }
    }}
