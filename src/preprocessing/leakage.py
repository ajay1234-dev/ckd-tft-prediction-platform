"""
src/preprocessing/leakage.py
Patient-level leakage detection and chronological order verification.

- check_patient_leakage: pairwise intersection of train/val/test patient IDs.
- verify_static_baseline_patients: static patients must be subset of train.
- verify_chronological_order: visit_date order must match visit_number order.

Raises PatientLeakageError (non-zero exit) if any leakage is found.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import sys
from typing import Optional, Set

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class PatientLeakageError(Exception):
    """
    Raised when the same patient_id appears in more than one data split.
    This is a hard failure — the pipeline must not continue.
    """
    pass


# ---------------------------------------------------------------------------
# Patient leakage check
# ---------------------------------------------------------------------------

def check_patient_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str = "patient_id",
    max_display_ids: int = 20,
) -> None:
    """
    Verify that train / val / test datasets have fully disjoint patient sets.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
    id_col : str
    max_display_ids : int
        Maximum overlapping IDs to include in error message.

    Raises
    ------
    PatientLeakageError
        If any pairwise intersection is non-empty.
    """
    train_ids: Set = set(train_df[id_col].unique())
    val_ids:   Set = set(val_df[id_col].unique())
    test_ids:  Set = set(test_df[id_col].unique())

    logger.info(
        f"Patient counts — train: {len(train_ids):,} | "
        f"val: {len(val_ids):,} | test: {len(test_ids):,}"
    )

    errors = []

    train_val = train_ids & val_ids
    if train_val:
        sample = sorted(train_val)[:max_display_ids]
        errors.append(
            f"train ∩ val: {len(train_val):,} overlapping patients. "
            f"Sample IDs: {sample}"
            + (" (truncated)" if len(train_val) > max_display_ids else "")
        )

    train_test = train_ids & test_ids
    if train_test:
        sample = sorted(train_test)[:max_display_ids]
        errors.append(
            f"train ∩ test: {len(train_test):,} overlapping patients. "
            f"Sample IDs: {sample}"
            + (" (truncated)" if len(train_test) > max_display_ids else "")
        )

    val_test = val_ids & test_ids
    if val_test:
        sample = sorted(val_test)[:max_display_ids]
        errors.append(
            f"val ∩ test: {len(val_test):,} overlapping patients. "
            f"Sample IDs: {sample}"
            + (" (truncated)" if len(val_test) > max_display_ids else "")
        )

    if errors:
        msg = (
            "PatientLeakageError: Patient IDs overlap across data splits!\n"
            + "\n".join(f"  ✗ {e}" for e in errors)
            + "\n\nData splitting must be performed at the PATIENT level with no overlap."
        )
        raise PatientLeakageError(msg)

    logger.info(
        "✓ No patient leakage detected across train / validation / test. "
        f"({len(train_ids):,} | {len(val_ids):,} | {len(test_ids):,} patients)"
    )


# ---------------------------------------------------------------------------
# Static baseline consistency check
# ---------------------------------------------------------------------------

def verify_static_baseline_patients(
    static_df: pd.DataFrame,
    train_df: pd.DataFrame,
    id_col: str = "patient_id",
) -> None:
    """
    Verify the static baseline patients are a subset of (or equal to) training patients.
    Reports mismatches but does NOT raise — static file may be a subset.

    Parameters
    ----------
    static_df : pd.DataFrame
    train_df : pd.DataFrame
    """
    static_ids = set(static_df[id_col].unique())
    train_ids  = set(train_df[id_col].unique())

    extra = static_ids - train_ids
    if extra:
        logger.warning(
            f"Static baseline has {len(extra):,} patient(s) NOT in training set. "
            f"Sample extra IDs: {sorted(extra)[:10]}"
        )
    else:
        logger.info(
            f"✓ All {len(static_ids):,} static baseline patients are present in training set."
        )

    missing_from_static = train_ids - static_ids
    if missing_from_static:
        logger.info(
            f"Info: {len(missing_from_static):,} training patients have no static baseline row "
            f"(will use first-visit derivation for baseline features)."
        )


# ---------------------------------------------------------------------------
# Chronological order verification
# ---------------------------------------------------------------------------

def verify_chronological_order(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    date_col: str = "visit_date",
    number_col: str = "visit_number",
    days_col: str = "days_from_baseline",
) -> pd.DataFrame:
    """
    For every patient, verify that sorting by visit_date gives the same
    order as visit_number ascending.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame of patients with ordering violations (empty if none).
    """
    logger.info("Verifying per-patient chronological ordering…")
    violations = []

    for pid, group in df.groupby(patient_col):
        by_date   = group.sort_values(date_col)[number_col].values
        by_number = group.sort_values(number_col)[number_col].values
        if not (by_date == by_number).all():
            violations.append({
                "patient_id": pid,
                "n_visits": len(group),
                "visit_numbers_by_date": list(by_date),
                "visit_numbers_sorted": list(by_number),
            })

    if violations:
        logger.warning(
            f"Chronological order violations found in {len(violations):,} patients. "
            f"These will be re-sorted by visit_date during preprocessing."
        )
    else:
        logger.info("✓ All patients have consistent chronological ordering.")

    return pd.DataFrame(violations)


# ---------------------------------------------------------------------------
# Combined runner (used at start of every pipeline)
# ---------------------------------------------------------------------------

def run_all_leakage_checks(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    static_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Run all leakage and ordering checks. Call this at the start of
    the preprocessing pipeline before any fitting.

    Raises PatientLeakageError on leakage (exits pipeline loudly).
    """
    check_patient_leakage(train_df, val_df, test_df)

    if static_df is not None:
        verify_static_baseline_patients(static_df, train_df)

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        violations = verify_chronological_order(split_df)
        if not violations.empty:
            logger.warning(
                f"{name}: {len(violations)} patients with ordering violations — "
                "will be corrected during preprocessing."
            )
