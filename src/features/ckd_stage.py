"""
src/features/ckd_stage.py
Bidirectional CKD stage encoding utilities.

Clinical stage labels G1–G5 (with G3a/G3b split) are the canonical form.
Numeric codes are for ML model input ONLY and must always be reversible.
This is the single source of truth — no other module may hardcode this mapping.

Stage mapping:
    G1  → 1  (eGFR ≥ 90, kidney damage markers present)
    G2  → 2  (eGFR 60–89)
    G3a → 3  (eGFR 45–59)
    G3b → 4  (eGFR 30–44)
    G4  → 5  (eGFR 15–29)
    G5  → 6  (eGFR < 15, kidney failure)

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical mapping — single source of truth
# ---------------------------------------------------------------------------

CKD_STAGE_MAP: Dict[str, int] = {
    "G1":  1,
    "G2":  2,
    "G3a": 3,
    "G3b": 4,
    "G4":  5,
    "G5":  6,
}

_REVERSE_MAP: Dict[int, str] = {v: k for k, v in CKD_STAGE_MAP.items()}

VALID_STAGE_LABELS: List[str] = list(CKD_STAGE_MAP.keys())
VALID_STAGE_CODES: List[int] = list(_REVERSE_MAP.keys())

# Ordered list for display / ordinal purposes
STAGE_ORDER: List[str] = ["G1", "G2", "G3a", "G3b", "G4", "G5"]


# ---------------------------------------------------------------------------
# Core encode / decode
# ---------------------------------------------------------------------------

def encode_stage(label: str) -> int:
    """
    Convert a clinical CKD stage label to its numeric code.

    Parameters
    ----------
    label : str
        One of: G1, G2, G3a, G3b, G4, G5

    Returns
    -------
    int
        Numeric code (1–6)

    Raises
    ------
    ValueError
        If label is not a recognized CKD stage string.
        No silent NaN or default fallback — unrecognized labels always raise.
    """
    if not isinstance(label, str):
        raise ValueError(
            f"encode_stage expects a string, got {type(label).__name__}: {label!r}"
        )
    label = label.strip()
    if label not in CKD_STAGE_MAP:
        raise ValueError(
            f"Unrecognized CKD stage label: {label!r}\n"
            f"Valid labels: {VALID_STAGE_LABELS}\n"
            "NOTE: Do NOT use numeric codes as input — use clinical string labels."
        )
    return CKD_STAGE_MAP[label]


def decode_stage(code: int) -> str:
    """
    Convert a numeric CKD stage code back to its clinical label.

    Parameters
    ----------
    code : int
        One of: 1, 2, 3, 4, 5, 6

    Returns
    -------
    str
        Clinical label (G1, G2, G3a, G3b, G4, G5)

    Raises
    ------
    ValueError
        If code is not a recognized stage code.
    """
    if not isinstance(code, (int, float)):
        raise ValueError(
            f"decode_stage expects an int or float, got {type(code).__name__}: {code!r}"
        )
    code_int = int(code)
    if code_int not in _REVERSE_MAP:
        raise ValueError(
            f"Unrecognized CKD stage code: {code_int}\n"
            f"Valid codes: {VALID_STAGE_CODES}\n"
            f"Valid labels: {VALID_STAGE_LABELS}"
        )
    return _REVERSE_MAP[code_int]


# ---------------------------------------------------------------------------
# Vectorized helpers for DataFrames
# ---------------------------------------------------------------------------

def encode_stage_series(series: pd.Series, allow_null: bool = False) -> pd.Series:
    """
    Apply encode_stage across a pandas Series.

    Parameters
    ----------
    series : pd.Series
        Series of clinical stage labels.
    allow_null : bool
        If True, NaN values are preserved as NaN (useful for target columns
        that may have missing 12m predictions). Default is False.

    Returns
    -------
    pd.Series of int (or float where NaN preserved)
    """
    if allow_null:
        return series.apply(lambda x: encode_stage(x) if pd.notna(x) else float("nan"))
    return series.apply(encode_stage)


def decode_stage_series(series: pd.Series, allow_null: bool = False) -> pd.Series:
    """
    Apply decode_stage across a pandas Series of numeric codes.
    """
    if allow_null:
        return series.apply(lambda x: decode_stage(x) if pd.notna(x) else None)
    return series.apply(decode_stage)


def is_valid_stage_label(label: str) -> bool:
    """Return True if label is a recognized CKD stage string."""
    return isinstance(label, str) and label.strip() in CKD_STAGE_MAP


def stage_severity(label: str) -> int:
    """Return ordinal severity (1=least, 6=most) — same as encode_stage."""
    return encode_stage(label)
