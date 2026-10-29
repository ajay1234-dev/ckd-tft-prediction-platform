"""
scripts/evaluate_baselines.py
CLI: Load saved baseline models, evaluate on val+test, save all metrics and figures.

Usage:
    python scripts/evaluate_baselines.py
    python scripts/evaluate_baselines.py --config configs/config.yaml --split test

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.seeding import set_global_seed
from src.utils.logging import get_logger
from src.data.loader import load_all_datasets
from src.features.feature_groups import load_feature_groups
from src.preprocessing.pipeline import CKDPreprocessingPipeline
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_binary_metrics,
    compute_regression_metrics,
)
from src.evaluation.reports import (
    save_confusion_matrix,
    save_metrics_report,
    generate_baseline_summary,
)
from src.features.ckd_stage import encode_stage_series, STAGE_ORDER

logger = get_logger("evaluate_baselines")


def main(config_path=None, split="test"):
    logger.info("=" * 70)
    logger.info(f"CKD Phase 1 — Baseline Evaluation on '{split}' split")
    logger.info("Research prototype | Synthetic data | Not for clinical use")
    logger.info("=" * 70)

    cfg = load_config(config_path)
    set_global_seed(cfg.random_seed)
    fg = load_feature_groups(cfg.data.feature_groups_file)

    # Load and transform evaluation split
    datasets = load_all_datasets(cfg)
    pipeline = CKDPreprocessingPipeline.load(cfg.output.preprocessing_dir, fg)

    eval_df = datasets[split] if split in datasets else datasets["test"]
    X_eval, y_eval = pipeline.transform(eval_df, split_name=split)

    baselines_dir = Path(cfg.output.baselines_dir)
    all_results = []

    if not baselines_dir.exists() or not any(baselines_dir.iterdir()):
        logger.error(f"No models found in {baselines_dir}. Run train_baselines.py first.")
        sys.exit(1)

    # Discover saved models
    for model_dir in sorted(baselines_dir.iterdir()):
        model_path = model_dir / "model.joblib"
        meta_path  = model_dir / "metadata.json"
        if not model_path.exists():
            continue

        model = joblib.load(model_path)
        with open(meta_path) as f:
            meta = json.load(f)

        model_name = model_dir.name  # e.g. "LogisticRegression_classification_stage_3m"
        parts = model_name.split("_")
        # Reconstruct task and target from directory name
        if "classification_stage" in model_name:
            task_type = "classification_stage"
            target = [p for p in cfg.targets.classification_stage if p in model_name]
            target = target[0] if target else "stage_?m"
        elif "binary_progression" in model_name:
            task_type = "binary_progression"
            target = [p for p in cfg.targets.binary_progression if p in model_name]
            target = target[0] if target else "progression_?m"
        elif "regression_egfr" in model_name:
            task_type = "regression_egfr"
            target = [p for p in cfg.targets.regression_egfr if p in model_name]
            target = target[0] if target else "egfr_?m"
        else:
            logger.warning(f"Unknown task type for model: {model_name} — skipping")
            continue

        horizon = target.split("_")[-1] if "_" in target else "?"

        # Prepare target
        if target not in y_eval.columns:
            logger.warning(f"Target {target} not in eval set — skipping {model_name}")
            continue

        if task_type == "classification_stage":
            y_true_raw = y_eval[target].dropna()
            y_true = encode_stage_series(y_true_raw, allow_null=True).dropna().astype(int)
        elif task_type == "binary_progression":
            y_true = pd.to_numeric(y_eval[target], errors="coerce").dropna().astype(int)
        else:
            y_true = pd.to_numeric(y_eval[target], errors="coerce").dropna()

        valid_idx = y_true.index
        X_sub = X_eval.loc[valid_idx] if hasattr(X_eval, "loc") else X_eval

        # Drop non-numeric / ID cols for prediction
        drop_cols = ["patient_id", "visit_id", "visit_date", "ckd_stage"]
        X_sub_arr = X_sub.drop(columns=[c for c in drop_cols if c in X_sub.columns], errors="ignore")
        X_sub_arr = X_sub_arr.select_dtypes(include=[np.number]).fillna(0).values

        try:
            y_pred = model.predict(X_sub_arr)
        except Exception as e:
            logger.error(f"Prediction failed for {model_name}: {e}")
            continue

        # Compute metrics
        if task_type == "classification_stage":
            present_labels = sorted(y_true.unique())
            label_names = [f"G{l}" if l <= 2 else ("G3a" if l == 3 else ("G3b" if l == 4 else ("G4" if l == 5 else "G5"))) for l in present_labels]
            metrics = compute_classification_metrics(y_true.values, y_pred, labels=present_labels)
            save_confusion_matrix(
                metrics["confusion_matrix"], label_names,
                cfg.output.metrics_dir, cfg.output.figures_dir,
                model_name, target, horizon
            )

        elif task_type == "binary_progression":
            try:
                y_proba = model.predict_proba(X_sub_arr)
            except Exception:
                y_proba = None
            metrics = compute_binary_metrics(y_true.values, y_pred, y_proba)
            save_confusion_matrix(
                compute_binary_metrics(y_true.values, y_pred)["confusion_matrix"]
                if "confusion_matrix" in compute_binary_metrics(y_true.values, y_pred)
                else [[0,0],[0,0]],
                ["No Progression", "Progression"],
                cfg.output.metrics_dir, cfg.output.figures_dir,
                model_name, target, horizon
            )
        else:
            metrics = compute_regression_metrics(y_true.values, y_pred)

        save_metrics_report(
            metrics.copy(), cfg.output.metrics_dir,
            model_name, task_type, target, horizon
        )

        row = {"model_name": model_name, "task_type": task_type,
               "target": target, "horizon": horizon, "split": split}
        row.update({k: v for k, v in metrics.items() if k != "confusion_matrix"})
        all_results.append(row)
        logger.info(f"  ✓ {model_name} | {target} evaluated")

    if all_results:
        summary = generate_baseline_summary(all_results, cfg.output.metrics_dir)
        logger.info(f"\n{'='*70}")
        logger.info("BASELINE EVALUATION SUMMARY")
        logger.info(f"{'='*70}")
        print(summary[["model_name", "task_type", "target", "horizon"] +
                       [c for c in summary.columns if c not in
                        ["model_name","task_type","target","horizon","split","confusion_matrix","n_samples","n_positive"]]
                      ].to_string(index=False))
    else:
        logger.warning("No results generated. Check that models were trained first.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()
    main(args.config, args.split)
