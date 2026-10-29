"""
src/explainability/global_explanations.py
Aggregate explainability signals across many patients (e.g., test cohort).

All values derived from actual model inference over real (synthetic) test data.
No values are fabricated.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from src.explainability.variable_importance import (
    extract_static_importance,
    extract_temporal_importance,
    STATIC_FEATURE_NAMES,
    TEMPORAL_VSN_FEATURE_NAMES,
)
from src.explainability.temporal_attention import (
    extract_attention_per_timestep,
    summarize_attention_across_patients,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def generate_global_feature_importance(
    model: torch.nn.Module,
    sequences_dir: str | Path,
    split: str = "test",
    max_patients: Optional[int] = None,
    n_static_features: int = 4,
    n_temporal_features: int = 2,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Run the model over many patients and aggregate VSN weights.

    Parameters
    ----------
    model : TemporalFusionTransformer (loaded, eval mode)
    sequences_dir : path to data/processed/sequences/
    split : 'train', 'val', or 'test'
    max_patients : limit for speed (None = all)
    n_static_features : how many static cols the model uses
    n_temporal_features : how many temporal cols the model uses

    Returns
    -------
    pd.DataFrame with columns: feature, group, mean_importance, std_importance
    """
    from src.features.sequence_builder import load_sequences

    logger.info(f"Generating global feature importance on {split} split...")
    seqs = load_sequences(sequences_dir, split)

    static_arr = seqs["static_features"][:, :n_static_features]
    hist_arr = seqs["historical_features"]
    time_arr = seqs["time_features"][:, :, :n_temporal_features]
    masks_arr = seqs["masks"]

    n = len(static_arr) if max_patients is None else min(max_patients, len(static_arr))
    logger.info(f"Running inference on {n} patients for global importance...")

    all_static_weights: List[Dict[str, float]] = []
    all_temporal_weights: List[Dict[str, float]] = []

    model.eval()
    with torch.no_grad():
        for i in range(n):
            s = torch.tensor(static_arr[i:i+1], dtype=torch.float32).to(device)
            h = torch.tensor(hist_arr[i:i+1], dtype=torch.float32).to(device)
            t = torch.tensor(time_arr[i:i+1], dtype=torch.float32).to(device)
            m = torch.tensor(masks_arr[i:i+1], dtype=torch.int8).to(device)

            out = model(s, h, t, m)

            sw = extract_static_importance(out["static_vsn_weights"], STATIC_FEATURE_NAMES)
            tw = extract_temporal_importance(out["temporal_vsn_weights"], m, TEMPORAL_VSN_FEATURE_NAMES)
            all_static_weights.append(sw)
            all_temporal_weights.append(tw)

    # Aggregate
    static_df = pd.DataFrame(all_static_weights)
    temporal_df = pd.DataFrame(all_temporal_weights)

    rows = []
    for col in static_df.columns:
        rows.append({
            "feature": col,
            "group": "static",
            "mean_importance": float(static_df[col].mean()),
            "std_importance": float(static_df[col].std()),
        })
    for col in temporal_df.columns:
        rows.append({
            "feature": col,
            "group": "time_varying",
            "mean_importance": float(temporal_df[col].mean()),
            "std_importance": float(temporal_df[col].std()),
        })

    result = pd.DataFrame(rows).sort_values("mean_importance", ascending=False)
    logger.info(f"Global feature importance computed over {n} patients.")
    return result


def generate_temporal_attention_summary(
    model: torch.nn.Module,
    sequences_dir: str | Path,
    split: str = "test",
    max_patients: Optional[int] = None,
    n_static_features: int = 4,
    n_temporal_features: int = 2,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Run the model over many patients and aggregate temporal attention by visit position.

    Returns
    -------
    pd.DataFrame with columns: visit_index, mean_attention_weight, std_attention_weight
    """
    from src.features.sequence_builder import load_sequences

    logger.info(f"Generating temporal attention summary on {split} split...")
    seqs = load_sequences(sequences_dir, split)

    static_arr = seqs["static_features"][:, :n_static_features]
    hist_arr = seqs["historical_features"]
    time_arr = seqs["time_features"][:, :, :n_temporal_features]
    masks_arr = seqs["masks"]

    n = len(static_arr) if max_patients is None else min(max_patients, len(static_arr))
    logger.info(f"Running inference on {n} patients for attention summary...")

    all_patient_attentions: List[List[Dict]] = []

    model.eval()
    with torch.no_grad():
        for i in range(n):
            s = torch.tensor(static_arr[i:i+1], dtype=torch.float32).to(device)
            h = torch.tensor(hist_arr[i:i+1], dtype=torch.float32).to(device)
            t = torch.tensor(time_arr[i:i+1], dtype=torch.float32).to(device)
            m = torch.tensor(masks_arr[i:i+1], dtype=torch.int8).to(device)

            out = model(s, h, t, m)
            attn_steps = extract_attention_per_timestep(out["attention_weights"], mask=m)
            all_patient_attentions.append(attn_steps)

    # Aggregate across patients by visit_index
    from collections import defaultdict
    acc: Dict[int, List[float]] = defaultdict(list)
    for patient_attn in all_patient_attentions:
        for step in patient_attn:
            if not step["is_padding"]:
                acc[step["visit_index"]].append(step["attention_weight"])

    rows = []
    for idx in sorted(acc.keys()):
        vals = acc[idx]
        rows.append({
            "visit_index": idx,
            "visit_number": idx + 1,
            "mean_attention_weight": float(np.mean(vals)),
            "std_attention_weight": float(np.std(vals)),
            "n_patients": len(vals),
        })

    result = pd.DataFrame(rows)
    logger.info(f"Temporal attention summary computed over {n} patients.")
    return result
