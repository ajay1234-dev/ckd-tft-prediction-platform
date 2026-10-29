"""
src/api/routes/health.py
GET /health — liveness check endpoint.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from fastapi import APIRouter
from src.api.dependencies import get_predictor

router = APIRouter(tags=["Health"])

_DISCLAIMER = (
    "Research prototype using synthetically generated longitudinal CKD data. "
    "Not intended for clinical diagnosis or treatment."
)


@router.get(
    "/health",
    summary="Health check",
    description="Returns service liveness status and whether the model is loaded.",
)
def health_check():
    try:
        predictor = get_predictor()
        model_loaded = predictor._model is not None
    except Exception:
        model_loaded = False

    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "data_disclaimer": _DISCLAIMER,
    }
