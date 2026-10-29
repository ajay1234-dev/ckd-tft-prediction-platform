"""
src/features/static_feature_builder.py
Derives static (patient-level) baseline features.

Priority:
  1. Use the dedicated static baseline CSV (purpose-built as baseline snapshot).
  2. Fall back to first chronological visit in the longitudinal file if a column
     is absent from the static file.

"baseline_X" columns = comorbidity / BMI flags at the patient's FIRST visit.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

from typing import Optional
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Columns to derive from first visit / static file
_BASELINE_COMORBIDITY_MAP = {
    "baseline_diabetes":                "diabetes",
    "baseline_hypertension":            "hypertension",
    "baseline_cardiovascular_disease":  "cardiovascular_disease",
    "baseline_heart_failure":           "heart_failure",
    "baseline_glomerulonephritis":      "glomerulonephritis",
    "baseline_obesity":                 "obesity",
}


def build_baseline_static_features(
    longitudinal_df: pd.DataFrame,
    static_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build a patient-level static feature table.

    For each patient:
      - age, sex, height_cm, family_history_ckd, smoking_status → taken directly
        (these are recorded at baseline / relatively stable).
      - baseline_bmi → BMI at first chronological visit.
      - baseline_diabetes, baseline_hypertension, etc. → comorbidity flags at
        first chronological visit (or from static_df if provided).

    Parameters
    ----------
    longitudinal_df : pd.DataFrame
        Full longitudinal dataset sorted chronologically per patient.
    static_df : pd.DataFrame, optional
        Static baseline CSV. If provided, its values are preferred over
        first-visit derivation.

    Returns
    -------
    pd.DataFrame
        One row per patient_id with all static features.
    """
    logger.info("Building patient-level static feature table…")

    # Sort by patient + date to reliably get "first visit"
    df = longitudinal_df.copy()
    df = df.sort_values(["patient_id", "visit_date", "visit_number"])

    # First visit per patient
    first_visits = df.groupby("patient_id").first().reset_index()

    static_cols = ["patient_id", "age", "sex", "height_cm", "family_history_ckd", "smoking_status"]
    result = first_visits[static_cols].copy()

    # baseline_bmi from first visit
    result["baseline_bmi"] = first_visits["bmi"].values

    # Comorbidity flags: prefer static_df if available
    for baseline_col, source_col in _BASELINE_COMORBIDITY_MAP.items():
        if static_df is not None and source_col in static_df.columns:
            mapping = static_df.set_index("patient_id")[source_col]
            result[baseline_col] = result["patient_id"].map(mapping)
            n_missing = result[baseline_col].isna().sum()
            if n_missing > 0:
                # Fall back to first visit for patients not in static file
                fallback = first_visits.set_index("patient_id")[source_col]
                result[baseline_col] = result.apply(
                    lambda row: fallback.get(row["patient_id"], row[baseline_col])
                    if pd.isna(row[baseline_col]) else row[baseline_col],
                    axis=1,
                )
                logger.warning(
                    f"{baseline_col}: {n_missing} patients missing from static file, "
                    f"fell back to first-visit value."
                )
        else:
            # Derive entirely from first visit
            result[baseline_col] = first_visits[source_col].values

        # Ensure int64
        result[baseline_col] = result[baseline_col].fillna(0).astype("int64")

    logger.info(
        f"Static feature table built: {len(result):,} patients, "
        f"{len(result.columns)} columns"
    )
    return result.reset_index(drop=True)
