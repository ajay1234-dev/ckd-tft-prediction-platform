"""
src/features/sequence_builder.py
Longitudinal patient sequence structures for Phase 2 TFT input.

Builds per-patient records:
  - static_features:    1D array  [num_static_features]
  - historical_features: 2D array [seq_len, num_time_varying_features]
  - time_features:       2D array [seq_len, num_temporal_features]
  - future_targets:      dict with keys stage/egfr/progression → [3] arrays
  - mask:                1D bool  [seq_len] — True = real visit, False = padded

Shapes documented in data/processed/sequences/SEQUENCE_SPEC.md

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features.feature_groups import FeatureGroups
from src.features.ckd_stage import encode_stage, is_valid_stage_label
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Temporal features used in time_features array
_TEMPORAL_COLS = ["visit_number", "days_from_baseline", "time_since_previous_visit_days"]

# Static features derived columns (those starting with "baseline_" + core static)
_CORE_STATIC_COLS = ["age", "sex", "height_cm", "family_history_ckd", "smoking_status"]


def _encode_stage_safe(label) -> float:
    """Encode stage to float, returning NaN for missing/invalid values."""
    if pd.isna(label):
        return float("nan")
    try:
        return float(encode_stage(str(label)))
    except ValueError:
        return float("nan")


def build_patient_sequences(
    df: pd.DataFrame,
    feature_groups: FeatureGroups,
    static_feature_df: Optional[pd.DataFrame] = None,
    max_seq_len: Optional[int] = None,
    pad_value: float = 0.0,
    mask_value: int = -1,
) -> Dict[str, np.ndarray]:
    """
    Build sequence arrays for all patients in df.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed longitudinal dataframe (post pipeline.transform).
    feature_groups : FeatureGroups
    static_feature_df : pd.DataFrame, optional
        Patient-level static features. If None, derived from df.
    max_seq_len : int, optional
        Pad/truncate to this length. If None, infer from data.
    pad_value : float
        Value used for padded positions.
    mask_value : int
        Value in mask for padded positions (real=1, padded=0).

    Returns
    -------
    dict with keys:
        patient_ids, static_features, historical_features,
        time_features, future_targets, masks
    """
    logger.info("Building patient sequence arrays…")

    # Sort chronologically
    df = df.sort_values(["patient_id", "visit_date", "visit_number"]).reset_index(drop=True)

    patient_ids = df["patient_id"].unique()
    N = len(patient_ids)

    # Infer max_seq_len
    if max_seq_len is None:
        visits_per_patient = df.groupby("patient_id").size()
        max_seq_len = int(visits_per_patient.max())
        logger.info(f"Inferred max_seq_len = {max_seq_len} from data")

    # Determine feature columns
    tv_cols = _get_available_cols(df, feature_groups.time_varying_features)
    t_cols  = _get_available_cols(df, _TEMPORAL_COLS)

    # Numeric only for arrays (drop string columns that haven't been encoded)
    df_numeric = df.select_dtypes(include=[np.number])
    tv_cols  = [c for c in tv_cols if c in df_numeric.columns]
    t_cols   = [c for c in t_cols  if c in df_numeric.columns]

    F_tv = len(tv_cols)
    F_t  = len(t_cols)

    # Static features
    if static_feature_df is not None:
        static_pid_map = static_feature_df.set_index("patient_id")
        static_num_cols = static_feature_df.select_dtypes(include=[np.number]).columns.tolist()
        static_num_cols = [c for c in static_num_cols if c != "patient_id"]
        F_s = len(static_num_cols)
    else:
        # Derive from first visit (numeric cols only)
        first_visits = df.groupby("patient_id").first().reset_index()
        static_pid_map = first_visits.set_index("patient_id")
        static_num_cols = [c for c in feature_groups.static_features if c in df_numeric.columns]
        F_s = len(static_num_cols)

    logger.info(
        f"Sequence spec: N={N}, seq_len={max_seq_len}, "
        f"F_static={F_s}, F_tv={F_tv}, F_time={F_t}"
    )

    # Allocate arrays
    static_arr   = np.full((N, F_s), pad_value, dtype=np.float32)
    hist_arr     = np.full((N, max_seq_len, F_tv), pad_value, dtype=np.float32)
    time_arr     = np.full((N, max_seq_len, F_t),  pad_value, dtype=np.float32)
    masks        = np.zeros((N, max_seq_len), dtype=np.int8)

    # Targets: stage (encoded), egfr, progression — each shape (N, 3)
    tgt_stage    = np.full((N, 3), np.nan, dtype=np.float32)
    tgt_egfr     = np.full((N, 3), np.nan, dtype=np.float32)
    tgt_prog     = np.full((N, 3), np.nan, dtype=np.float32)

    target_stage_cols = ["stage_3m", "stage_6m", "stage_12m"]
    target_egfr_cols  = ["egfr_3m",  "egfr_6m",  "egfr_12m"]
    target_prog_cols  = ["progression_3m", "progression_6m", "progression_12m"]

    groups = df.groupby("patient_id")

    for i, pid in enumerate(patient_ids):
        group = groups.get_group(pid)

        # Static features
        if pid in static_pid_map.index:
            row = static_pid_map.loc[pid]
            for j, col in enumerate(static_num_cols):
                val = row.get(col, pad_value)
                static_arr[i, j] = float(val) if pd.notna(val) else pad_value

        # Time-varying + temporal histories
        seq = group.head(max_seq_len)
        T = len(seq)
        masks[i, :T] = 1  # real visits

        for j, col in enumerate(tv_cols):
            vals = seq[col].values if col in seq.columns else np.full(T, pad_value)
            vals = np.where(np.isfinite(vals.astype(float)), vals.astype(float), pad_value)
            hist_arr[i, :T, j] = vals

        for j, col in enumerate(t_cols):
            vals = seq[col].values if col in seq.columns else np.full(T, pad_value)
            vals = np.where(np.isfinite(vals.astype(float)), vals.astype(float), pad_value)
            time_arr[i, :T, j] = vals

        # Targets — use the LAST visit's targets (most recent horizon)
        last_row = group.iloc[-1]

        for k, col in enumerate(target_stage_cols):
            if col in last_row.index:
                tgt_stage[i, k] = _encode_stage_safe(last_row[col])

        for k, col in enumerate(target_egfr_cols):
            if col in last_row.index and pd.notna(last_row[col]):
                tgt_egfr[i, k] = float(last_row[col])

        for k, col in enumerate(target_prog_cols):
            if col in last_row.index and pd.notna(last_row[col]):
                tgt_prog[i, k] = float(last_row[col])

    result = {
        "patient_ids": patient_ids,
        "static_features": static_arr,
        "historical_features": hist_arr,
        "time_features": time_arr,
        "masks": masks,
        "future_targets": {
            "stage": tgt_stage,
            "egfr": tgt_egfr,
            "progression": tgt_prog,
        },
        "feature_names": {
            "static": static_num_cols,
            "time_varying": tv_cols,
            "temporal": t_cols,
        },
        "seq_len": max_seq_len,
        "n_patients": N,
    }

    logger.info(
        f"Sequences built: {N:,} patients | "
        f"static {static_arr.shape} | hist {hist_arr.shape} | time {time_arr.shape}"
    )
    return result


def save_sequences(
    sequences: Dict,
    output_dir: str | Path,
    split_name: str = "train",
) -> None:
    """Save sequence arrays to data/processed/sequences/{split_name}/."""
    output_dir = Path(output_dir) / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "patient_ids.npy",          sequences["patient_ids"])
    np.save(output_dir / "static_features.npy",      sequences["static_features"])
    np.save(output_dir / "historical_features.npy",  sequences["historical_features"])
    np.save(output_dir / "time_features.npy",        sequences["time_features"])
    np.save(output_dir / "masks.npy",                sequences["masks"])
    np.save(output_dir / "targets_stage.npy",        sequences["future_targets"]["stage"])
    np.save(output_dir / "targets_egfr.npy",         sequences["future_targets"]["egfr"])
    np.save(output_dir / "targets_progression.npy",  sequences["future_targets"]["progression"])

    # Save feature names & spec as JSON
    spec = {
        "split": split_name,
        "n_patients":  sequences["n_patients"],
        "seq_len":     sequences["seq_len"],
        "shapes": {
            "static_features":       list(sequences["static_features"].shape),
            "historical_features":   list(sequences["historical_features"].shape),
            "time_features":         list(sequences["time_features"].shape),
            "masks":                 list(sequences["masks"].shape),
            "future_targets_stage":  list(sequences["future_targets"]["stage"].shape),
            "future_targets_egfr":   list(sequences["future_targets"]["egfr"].shape),
            "future_targets_prog":   list(sequences["future_targets"]["progression"].shape),
        },
        "feature_names": sequences["feature_names"],
    }
    with open(output_dir / "spec.json", "w") as f:
        json.dump(spec, f, indent=2, default=str)

    logger.info(f"Sequences saved to {output_dir}")


def load_sequences(sequences_dir: str | Path, split_name: str) -> Dict:
    """Load previously saved sequence arrays."""
    d = Path(sequences_dir) / split_name
    return {
        "patient_ids":       np.load(d / "patient_ids.npy", allow_pickle=True),
        "static_features":   np.load(d / "static_features.npy"),
        "historical_features": np.load(d / "historical_features.npy"),
        "time_features":     np.load(d / "time_features.npy"),
        "masks":             np.load(d / "masks.npy"),
        "future_targets": {
            "stage":       np.load(d / "targets_stage.npy"),
            "egfr":        np.load(d / "targets_egfr.npy"),
            "progression": np.load(d / "targets_progression.npy"),
        },
    }


def _get_available_cols(df: pd.DataFrame, wanted: List[str]) -> List[str]:
    """Return intersection of wanted columns and df columns, preserving order."""
    present = set(df.columns)
    return [c for c in wanted if c in present]
