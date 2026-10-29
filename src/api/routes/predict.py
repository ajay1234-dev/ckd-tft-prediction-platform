"""
src/api/routes/predict.py
POST /predict — prediction-only endpoint.

Returns stage/eGFR/progression predictions for 3m/6m/12m horizons.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_predictor
from src.inference.predictor import CKDPredictor
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Prediction"])


class PredictRequest(BaseModel):
    patient_id: str = Field(..., description="Synthetic patient ID from the test cohort")

    model_config = {"json_schema_extra": {"example": {"patient_id": "12345"}}}


@router.post(
    "/predict",
    summary="Predict CKD progression",
    description=(
        "Run multi-task, multi-horizon prediction for a synthetic patient. "
        "Returns CKD stage, eGFR, and progression risk at 3, 6, and 12 months. "
        "This is a research prototype — predictions are not for clinical use."
    ),
)
def predict(
    request: PredictRequest,
    predictor: CKDPredictor = Depends(get_predictor),
):
    patient_id = request.patient_id.strip()
    if not patient_id:
        raise HTTPException(status_code=400, detail="patient_id must not be empty.")

    logger.info(f"Prediction request for patient '{patient_id}'")
    t0 = time.perf_counter()

    try:
        result = predictor.predict(patient_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error(f"Prediction failed for patient '{patient_id}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Inference failed. Check server logs for details.",
        )

    elapsed = time.perf_counter() - t0
    logger.info(f"Prediction completed for patient '{patient_id}' in {elapsed:.3f}s")
    return result
