"""
src/evaluation/reports.py
Save metrics, confusion matrices (image + array), and consolidated summary CSV.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logging import get_logger
from src.utils.io import save_json, save_csv

logger = get_logger(__name__)


def save_confusion_matrix(
    cm: List[List[int]],
    labels: List[str],
    metrics_dir: str | Path,
    figures_dir: str | Path,
    model_name: str,
    target: str,
    horizon: str,
) -> None:
    """
    Save confusion matrix as:
      - CSV array:  reports/metrics/cm_{model}_{target}_{horizon}.csv
      - PNG figure: reports/figures/cm_{model}_{target}_{horizon}.png
    """
    cm_arr = np.array(cm)
    metrics_dir = Path(metrics_dir)
    figures_dir = Path(figures_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    stem = f"cm_{model_name}_{target}_{horizon}"

    # CSV
    cm_df = pd.DataFrame(cm_arr, index=labels, columns=labels)
    cm_df.to_csv(metrics_dir / f"{stem}.csv")

    # PNG
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels))))
    sns.heatmap(
        cm_arr, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}\n{target} ({horizon})")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{stem}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Confusion matrix saved: {stem}")


def save_metrics_report(
    metrics: Dict,
    metrics_dir: str | Path,
    model_name: str,
    task_type: str,
    target: str,
    horizon: str,
) -> None:
    """Save a metrics dict to JSON."""
    path = Path(metrics_dir) / f"{model_name}_{task_type}_{target}_{horizon}.json"
    metrics["model_name"] = model_name
    metrics["task_type"]  = task_type
    metrics["target"]     = target
    metrics["horizon"]    = horizon
    save_json(metrics, path)


def generate_baseline_summary(all_results: List[Dict], metrics_dir: str | Path) -> pd.DataFrame:
    """
    Consolidate all baseline results into a single summary CSV.
    This is the primary Phase 2 comparison file.

    Parameters
    ----------
    all_results : list of dict
        Each dict contains model_name, task_type, target, horizon, and metric values.
    metrics_dir : str or Path

    Returns
    -------
    pd.DataFrame
    """
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(all_results)

    # Move key columns to front
    priority_cols = ["model_name", "task_type", "target", "horizon", "n_samples"]
    existing = [c for c in priority_cols if c in summary_df.columns]
    rest = [c for c in summary_df.columns if c not in existing]
    summary_df = summary_df[existing + rest]

    path = metrics_dir / "baseline_results_summary.csv"
    summary_df.to_csv(path, index=False)
    logger.info(f"Baseline summary saved → {path} ({len(summary_df)} rows)")
    return summary_df
