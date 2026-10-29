"""
src/data/loader.py
Dataset loading functions with explicit dtype handling and date parsing.
Validates schema immediately after loading.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.schema import (
    validate_longitudinal_schema,
    validate_static_baseline_schema,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Dtype maps (prevent silent coercion of IDs and flags)
# ---------------------------------------------------------------------------
_LONGITUDINAL_DTYPES = {
    "patient_id": "int64",
    "visit_id": "str",
    "visit_number": "int64",
    "days_from_baseline": "int64",
    "time_since_previous_visit_days": "int64",
    "family_history_ckd": "int64",
    "diabetes": "int64",
    "hypertension": "int64",
    "cardiovascular_disease": "int64",
    "heart_failure": "int64",
    "glomerulonephritis": "int64",
    "obesity": "int64",
    "anemia": "int64",
    "acei_arb": "int64",
    "sglt2_inhibitor": "int64",
    "diuretic": "int64",
    "antihypertensive": "int64",
    "statin": "int64",
    "epo": "int64",
    "iron_supplement": "int64",
    "metformin": "int64",
    "insulin": "int64",
    "on_dialysis": "int64",
    "kidney_transplant": "int64",
    "progression_event": "int64",
}

_STATIC_DTYPES = {
    "patient_id": "int64",
    "family_history_ckd": "int64",
    "diabetes": "int64",
    "hypertension": "int64",
    "cardiovascular_disease": "int64",
    "heart_failure": "int64",
    "glomerulonephritis": "int64",
    "obesity": "int64",
    "anemia": "int64",
    "acei_arb": "int64",
    "sglt2_inhibitor": "int64",
    "diuretic": "int64",
    "antihypertensive": "int64",
    "statin": "int64",
    "epo": "int64",
    "iron_supplement": "int64",
    "metformin": "int64",
    "insulin": "int64",
    "on_dialysis": "int64",
    "kidney_transplant": "int64",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_longitudinal(
    path: str | Path,
    dataset_name: str = "longitudinal",
    validate: bool = True,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a longitudinal CKD dataset (train / val / test) from CSV.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    dataset_name : str
        Human-readable label for logging and error messages.
    validate : bool
        If True, validate schema after loading (recommended).
    nrows : int, optional
        Load only the first N rows (useful for quick tests).

    Returns
    -------
    pd.DataFrame
        Loaded and date-parsed DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Please ensure the CSV is placed in data/raw/ as described in README.md.\n"
            "NOTE: This is a research prototype using synthetic data only."
        )

    logger.info(f"Loading {dataset_name} from: {path}")
    df = pd.read_csv(
        path,
        dtype=_LONGITUDINAL_DTYPES,
        low_memory=False,
        nrows=nrows,
    )

    # Parse visit_date
    df = _parse_dates(df, ["visit_date"])

    if validate:
        validate_longitudinal_schema(df, dataset_name)

    logger.info(
        f"Loaded {dataset_name}: {df['patient_id'].nunique():,} patients, "
        f"{len(df):,} rows, {len(df.columns)} columns"
    )
    return df


def load_static_baseline(
    path: str | Path,
    dataset_name: str = "static_baseline",
    validate: bool = True,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load the static (patient-level) baseline dataset.

    Parameters
    ----------
    path : str or Path
    dataset_name : str
    validate : bool
    nrows : int, optional

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Static baseline file not found: {path}")

    logger.info(f"Loading {dataset_name} from: {path}")
    df = pd.read_csv(
        path,
        dtype=_STATIC_DTYPES,
        low_memory=False,
        nrows=nrows,
    )

    if validate:
        validate_static_baseline_schema(df, dataset_name)

    logger.info(
        f"Loaded {dataset_name}: {df['patient_id'].nunique():,} patients, "
        f"{len(df):,} rows, {len(df.columns)} columns"
    )
    return df


def load_all_datasets(config) -> dict:
    """
    Load all 4 datasets using paths from CKDConfig.

    Parameters
    ----------
    config : CKDConfig

    Returns
    -------
    dict with keys: 'train', 'val', 'test', 'static'
    """
    root = Path(config.data.train_file).parent.parent  # project root / data
    return {
        "train": load_longitudinal(config.data.train_file, dataset_name="train"),
        "val":   load_longitudinal(config.data.val_file,   dataset_name="val"),
        "test":  load_longitudinal(config.data.test_file,  dataset_name="test"),
        "static": load_static_baseline(config.data.static_file, dataset_name="static_baseline"),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dates(df: pd.DataFrame, date_columns: list) -> pd.DataFrame:
    """Parse specified columns as datetime."""
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            n_failed = df[col].isna().sum()
            if n_failed > 0:
                logger.warning(
                    f"Date parsing: {n_failed} values in '{col}' could not be parsed → NaT"
                )
    return df
