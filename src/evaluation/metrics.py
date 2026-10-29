"""
src/evaluation/metrics.py
Metric computation functions for classification, binary, and regression tasks.
All metrics computed from REAL model predictions on held-out data — never fabricated.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    labels: Optional[List] = None,
) -> Dict:
    """
    Compute multiclass classification metrics.

    Returns
    -------
    dict with keys: accuracy, macro_precision, macro_recall, macro_f1, weighted_f1, confusion_matrix
    """
    metrics = {
        "accuracy":        round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall":    round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_f1":        round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1":     round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n_samples":       int(len(y_true)),
    }
    return metrics


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compute binary classification metrics.

    Returns
    -------
    dict with keys: roc_auc, pr_auc, precision, recall, f1, specificity, sensitivity
    """
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "precision":   round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":      round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":          round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "sensitivity": round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0, 4),
        "specificity": round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0, 4),
        "n_samples":   int(len(y_true)),
        "n_positive":  int(y_true.sum()),
    }

    if y_proba is not None:
        proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, proba_pos)), 4)
            metrics["pr_auc"]  = round(float(average_precision_score(y_true, proba_pos)), 4)
        except ValueError as e:
            logger.warning(f"Could not compute AUC: {e}")
            metrics["roc_auc"] = None
            metrics["pr_auc"]  = None
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"]  = None

    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict:
    """
    Compute regression metrics.

    Returns
    -------
    dict with keys: mae, rmse, r2
    """
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    # Remove NaN pairs
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    return {
        "mae":      round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse":     round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2":       round(float(r2_score(y_true, y_pred)), 4),
        "n_samples": int(len(y_true)),
    }
