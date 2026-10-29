"""
src/schemas/patient.py
Pydantic response models for /patient/{patient_id} endpoint.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class PatientVisit(BaseModel):
    """One clinical visit in a patient's longitudinal history."""
    visit_index: int = Field(..., description="0-based index in the sequence")
    visit_number: Optional[float] = Field(None, description="Visit number from data")
    days_from_baseline: Optional[float] = Field(None, description="Days since first visit")
    is_padding: bool = Field(False, description="True if this slot is padding (no real visit)")


class PatientHistoryResponse(BaseModel):
    """Response schema for GET /patient/{patient_id}."""
    patient_id: str
    split: str = Field(..., description="Which data split this patient belongs to (test/val/train)")
    n_visits: int = Field(..., description="Number of real (non-padded) visits")
    sequence_length: int = Field(..., description="Total sequence length including padding")
    visits: List[PatientVisit]
    disclaimer: str = Field(
        "Synthetic patient data — not a real clinical record.",
        description="Reminder that this is synthetic data"
    )
