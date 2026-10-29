"""
src/api/routes/explain.py
POST /explain — prediction + explanation endpoint.

Returns predictions AND TFT-native explainability:
- Variable Selection Network weights (static + time-varying)
- Temporal self-attention weights

All values come from actual model forward pass — never fabricated.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_predictor
from src.inference.predictor import CKDPredictor
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Explainability"])


class ExplainRequest(BaseModel):
    patient_id: str = Field(..., description="Synthetic patient ID from the test cohort")

    model_config = {"json_schema_extra": {"example": {"patient_id": "12345"}}}


@router.post(
    "/explain",
    summary="Predict + explain CKD progression",
    description=(
        "Run multi-task, multi-horizon prediction with full TFT-native explainability. "
        "Returns all /predict outputs PLUS: "
        "variable-selection importance scores (which features the model focused on) "
        "and temporal attention weights (which historical visits most influenced the prediction). "
        "IMPORTANT: Feature importance and attention are model interpretability signals — "
        "they do not establish causal relationships. "
        "This is a research prototype — outputs are not for clinical use."
    ),
)
def explain(
    request: ExplainRequest,
    predictor: CKDPredictor = Depends(get_predictor),
):
    patient_id = request.patient_id.strip()
    if not patient_id:
        raise HTTPException(status_code=400, detail="patient_id must not be empty.")

    logger.info(f"Explanation request for patient '{patient_id}'")
    t0 = time.perf_counter()

    try:
        result = predictor.predict_with_explanation(patient_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error(f"Explanation failed for patient '{patient_id}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Explanation generation failed. Check server logs for details.",
        )

    elapsed = time.perf_counter() - t0
    logger.info(f"Explanation completed for patient '{patient_id}' in {elapsed:.3f}s")
    return result
