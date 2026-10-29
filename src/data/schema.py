"""
src/data/schema.py
Column schema definitions and validation for longitudinal and static-baseline datasets.
Raises SchemaValidationError with specific missing column details on failure.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class SchemaValidationError(Exception):
    """Raised when a dataset is missing required columns or has type mismatches."""
    pass


# ---------------------------------------------------------------------------
# Column schemas
# ---------------------------------------------------------------------------

# Required columns for longitudinal files (train / val / test)
LONGITUDINAL_REQUIRED_COLUMNS: List[str] = [
    # Identifiers
    "patient_id", "visit_id",
    # Temporal
    "visit_date", "visit_number", "days_from_baseline", "time_since_previous_visit_days",
    # Demographics
    "age", "sex", "height_cm", "weight_kg", "bmi", "smoking_status", "family_history_ckd",
    # Comorbidities
    "diabetes", "hypertension", "cardiovascular_disease", "heart_failure",
    "glomerulonephritis", "obesity",
    # Vitals
    "systolic_bp", "diastolic_bp",
    # Renal labs
    "serum_creatinine_mg_dl", "egfr_ml_min_1_73m2", "bun_mg_dl",
    "uacr_mg_g", "albuminuria_category",
    # Metabolic labs
    "fasting_glucose_mg_dl", "hba1c_pct", "hemoglobin_g_dl", "anemia",
    "sodium_meq_l", "potassium_meq_l", "calcium_mg_dl", "phosphate_mg_dl",
    "bicarbonate_meq_l", "uric_acid_mg_dl",
    # Medications / treatment
    "acei_arb", "sglt2_inhibitor", "diuretic", "antihypertensive", "statin",
    "epo", "iron_supplement", "metformin", "insulin",
    "on_dialysis", "dialysis_type", "kidney_transplant",
    # Current CKD stage
    "ckd_stage",
    # Future targets (present in file but NEVER used as input features)
    "stage_3m", "egfr_3m", "progression_3m",
    "stage_6m", "egfr_6m", "progression_6m",
    "stage_12m", "egfr_12m", "progression_12m",
    "stage_transition", "progression_event",
]

# Required columns for static baseline file
STATIC_BASELINE_REQUIRED_COLUMNS: List[str] = [
    "patient_id",
    "age", "sex", "height_cm", "weight_kg", "bmi", "smoking_status", "family_history_ckd",
    "diabetes", "hypertension", "cardiovascular_disease", "heart_failure",
    "glomerulonephritis", "obesity",
    "systolic_bp", "diastolic_bp",
    "serum_creatinine_mg_dl", "egfr_ml_min_1_73m2", "bun_mg_dl",
    "uacr_mg_g", "albuminuria_category",
    "fasting_glucose_mg_dl", "hba1c_pct", "hemoglobin_g_dl", "anemia",
    "sodium_meq_l", "potassium_meq_l", "calcium_mg_dl", "phosphate_mg_dl",
    "bicarbonate_meq_l", "uric_acid_mg_dl",
    "acei_arb", "sglt2_inhibitor", "diuretic", "antihypertensive", "statin",
    "epo", "iron_supplement", "metformin", "insulin",
    "on_dialysis", "dialysis_type", "kidney_transplant",
    "ckd_stage",
]

# Target columns that must NEVER appear in model feature matrices
TARGET_COLUMNS: List[str] = [
    "stage_3m", "stage_6m", "stage_12m",
    "egfr_3m", "egfr_6m", "egfr_12m",
    "progression_3m", "progression_6m", "progression_12m",
    "stage_transition", "progression_event",
]

# Expected dtypes for key columns (used for documentation; coercion happens in loader)
LONGITUDINAL_EXPECTED_DTYPES: Dict[str, str] = {
    "patient_id": "int64",
    "visit_number": "int64",
    "days_from_baseline": "int64",
    "time_since_previous_visit_days": "int64",
    "visit_date": "datetime64[ns]",
    "age": "float64",
    "height_cm": "float64",
    "weight_kg": "float64",
    "bmi": "float64",
    "systolic_bp": "float64",
    "diastolic_bp": "float64",
    "serum_creatinine_mg_dl": "float64",
    "egfr_ml_min_1_73m2": "float64",
    "egfr_3m": "float64",
    "egfr_6m": "float64",
    "egfr_12m": "float64",
    "progression_3m": "float64",
    "progression_6m": "float64",
    "progression_12m": "float64",
    "progression_event": "int64",
}


# ---------------------------------------------------------------------------
# Validation function
# ---------------------------------------------------------------------------

def validate_schema(
    df: pd.DataFrame,
    required_columns: List[str],
    dataset_name: str = "dataset",
) -> None:
    """
    Validate that a DataFrame contains all required columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    required_columns : list of str
        Columns that must be present.
    dataset_name : str
        Human-readable name used in error messages.

    Raises
    ------
    SchemaValidationError
        If any required columns are missing, with the exact list of missing columns.
    """
    present = set(df.columns)
    required = set(required_columns)
    missing = sorted(required - present)

    if missing:
        raise SchemaValidationError(
            f"Schema validation FAILED for '{dataset_name}'.\n"
            f"Missing {len(missing)} required column(s):\n"
            + "\n".join(f"  - {col}" for col in missing)
        )


def validate_longitudinal_schema(df: pd.DataFrame, dataset_name: str = "longitudinal") -> None:
    """Validate schema for train/val/test longitudinal files."""
    validate_schema(df, LONGITUDINAL_REQUIRED_COLUMNS, dataset_name)


def validate_static_baseline_schema(df: pd.DataFrame, dataset_name: str = "static_baseline") -> None:
    """Validate schema for static baseline file."""
    validate_schema(df, STATIC_BASELINE_REQUIRED_COLUMNS, dataset_name)
