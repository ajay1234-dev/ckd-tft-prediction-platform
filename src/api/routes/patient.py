"""
src/api/routes/patient.py
GET /patient/{patient_id} — retrieve synthetic patient history.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from src.api.dependencies import get_patient_store
from src.inference.predictor import PatientStore
from src.schemas.patient import PatientHistoryResponse, PatientVisit

router = APIRouter(tags=["Patient"])


@router.get(
    "/patient/{patient_id}",
    response_model=PatientHistoryResponse,
    summary="Get synthetic patient history",
    description=(
        "Returns the available longitudinal visit history for a synthetic patient. "
        "No real patient data is stored or returned."
    ),
)
def get_patient(
    patient_id: str,
    store: PatientStore = Depends(get_patient_store),
):
    if not patient_id or not patient_id.strip():
        raise HTTPException(status_code=400, detail="patient_id must not be empty.")

    data = store.get_patient(patient_id)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Patient '{patient_id}' not found in the {store.split} split.",
        )

    mask = data["mask"]
    time_data = data["time"]
    seq_len = len(mask)

    visits = []
    for i in range(seq_len):
        is_pad = bool(mask[i] == 0)
        vn = float(time_data[i, 0]) if not is_pad else None
        dfb = float(time_data[i, 1]) if not is_pad else None
        visits.append(PatientVisit(
            visit_index=i,
            visit_number=vn,
            days_from_baseline=dfb,
            is_padding=is_pad,
        ))

    n_visits = int(sum(1 for v in visits if not v.is_padding))

    return PatientHistoryResponse(
        patient_id=patient_id,
        split=store.split,
        n_visits=n_visits,
        sequence_length=seq_len,
        visits=visits,
        disclaimer="Synthetic patient data — not a real clinical record.",
    )


@router.get(
    "/patient/{patient_id}/prediction",
    summary="Predict for a patient (convenience endpoint)",
    description="Retrieves patient history and runs prediction in one call.",
)
def get_patient_prediction(
    patient_id: str,
    store: PatientStore = Depends(get_patient_store),
):
    """Convenience endpoint combining patient lookup and prediction."""
    from fastapi import Request
    from src.api.dependencies import get_predictor
    predictor = get_predictor()
    if store.get_patient(patient_id) is None:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")
    try:
        return predictor.predict(patient_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inference failed. Check server logs.")
