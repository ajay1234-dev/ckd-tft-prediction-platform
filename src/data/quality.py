"""
src/data/quality.py
DataQualityReport generation — computes per-dataset statistics and saves structured reports.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger
from src.utils.io import save_json, save_csv

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class DataQualityReport:
    """Structured quality metrics for a single dataset."""
    dataset_name: str
    n_patients: int
    n_rows: int
    n_columns: int
    overall_missing_pct: float
    duplicate_rows: int
    duplicate_patient_visit_pairs: int
    min_visits_per_patient: int
    max_visits_per_patient: int
    median_visits_per_patient: float
    visit_date_min: str
    visit_date_max: str
    # Per-column missing %
    missing_pct_per_column: Dict[str, float] = field(default_factory=dict)
    # CKD stage distribution
    ckd_stage_distribution: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Categorical unique value counts
    categorical_unique_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_row(self) -> dict:
        """Return a single-row summary suitable for a summary table."""
        return {
            "Dataset": self.dataset_name,
            "Patients": self.n_patients,
            "Visits": self.n_rows,
            "Features": self.n_columns,
            "Missing %": round(self.overall_missing_pct, 2),
            "Duplicate rows": self.duplicate_rows,
            "Min visits/patient": self.min_visits_per_patient,
            "Max visits/patient": self.max_visits_per_patient,
            "Median visits/patient": self.median_visits_per_patient,
        }


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def generate_quality_report(df: pd.DataFrame, dataset_name: str) -> DataQualityReport:
    """
    Compute a DataQualityReport for a given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    dataset_name : str

    Returns
    -------
    DataQualityReport
    """
    logger.info(f"Generating quality report for: {dataset_name}")

    # Basic counts
    n_patients = df["patient_id"].nunique() if "patient_id" in df.columns else 0
    n_rows = len(df)
    n_columns = len(df.columns)

    # Missing values
    missing_counts = df.isnull().sum()
    missing_pct_per_column = {
        col: round(100.0 * missing_counts[col] / n_rows, 4)
        for col in df.columns
    }
    overall_missing_pct = round(100.0 * df.isnull().sum().sum() / (n_rows * n_columns), 4)

    # Duplicates
    duplicate_rows = int(df.duplicated().sum())
    if "patient_id" in df.columns and "visit_id" in df.columns:
        duplicate_pv = int(df.duplicated(subset=["patient_id", "visit_id"]).sum())
    else:
        duplicate_pv = 0

    # Visits per patient
    if "patient_id" in df.columns:
        visits_per_patient = df.groupby("patient_id").size()
        min_v = int(visits_per_patient.min())
        max_v = int(visits_per_patient.max())
        median_v = float(visits_per_patient.median())
    else:
        min_v = max_v = 1
        median_v = 1.0

    # Date range
    if "visit_date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["visit_date"]):
        date_min = str(df["visit_date"].min().date()) if not df["visit_date"].isna().all() else "N/A"
        date_max = str(df["visit_date"].max().date()) if not df["visit_date"].isna().all() else "N/A"
    else:
        date_min = date_max = "N/A"

    # CKD stage distribution
    ckd_dist = {}
    if "ckd_stage" in df.columns:
        counts = df["ckd_stage"].value_counts().sort_index()
        for stage, count in counts.items():
            ckd_dist[str(stage)] = {
                "count": int(count),
                "pct": round(100.0 * count / n_rows, 2),
            }

    # Categorical unique value counts
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_unique = {col: int(df[col].nunique()) for col in cat_cols}

    report = DataQualityReport(
        dataset_name=dataset_name,
        n_patients=n_patients,
        n_rows=n_rows,
        n_columns=n_columns,
        overall_missing_pct=overall_missing_pct,
        duplicate_rows=duplicate_rows,
        duplicate_patient_visit_pairs=duplicate_pv,
        min_visits_per_patient=min_v,
        max_visits_per_patient=max_v,
        median_visits_per_patient=median_v,
        visit_date_min=date_min,
        visit_date_max=date_max,
        missing_pct_per_column=missing_pct_per_column,
        ckd_stage_distribution=ckd_dist,
        categorical_unique_counts=cat_unique,
    )

    logger.info(
        f"Quality report done — {dataset_name}: "
        f"{n_patients:,} patients, {n_rows:,} rows, "
        f"{overall_missing_pct:.2f}% missing overall"
    )
    return report


# ---------------------------------------------------------------------------
# Save reports
# ---------------------------------------------------------------------------

def save_quality_reports(
    reports: List[DataQualityReport],
    output_dir: str | Path,
) -> None:
    """
    Save a list of quality reports to:
      - reports/metrics/data_quality_report.csv  (summary table)
      - reports/metrics/data_quality_report.json (full detail)

    Parameters
    ----------
    reports : list of DataQualityReport
    output_dir : str or Path
        Should be reports/metrics/
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary CSV
    summary_rows = [r.summary_row() for r in reports]
    summary_df = pd.DataFrame(summary_rows)
    save_csv(summary_df, output_dir / "data_quality_report.csv")

    # Full JSON
    full = {r.dataset_name: r.to_dict() for r in reports}
    save_json(full, output_dir / "data_quality_report.json")

    logger.info(f"Quality reports saved to {output_dir}")
