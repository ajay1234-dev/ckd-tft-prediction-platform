"""
scripts/train_baselines.py
CLI: Preprocess data + train all baseline models for all targets and horizons.
Saves models and preprocessing artifacts.

Usage:
    python scripts/train_baselines.py
    python scripts/train_baselines.py --config configs/config.yaml --skip-sequences

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from src.utils.config import load_config
from src.utils.seeding import set_global_seed
from src.utils.logging import get_logger
from src.data.loader import load_all_datasets
from src.features.feature_groups import load_feature_groups
from src.features.static_feature_builder import build_baseline_static_features
from src.features.sequence_builder import build_patient_sequences, save_sequences
from src.preprocessing.pipeline import CKDPreprocessingPipeline
from src.preprocessing.leakage import run_all_leakage_checks, PatientLeakageError
from src.baselines.classifiers import get_stage_classifiers
from src.baselines.binary_classifiers import get_binary_classifiers
from src.baselines.regressors import get_regressors
from src.features.ckd_stage import encode_stage_series

logger = get_logger("train_baselines")


def main(config_path=None, skip_sequences=False):
    logger.info("=" * 70)
    logger.info("CKD Phase 1 — Baseline Model Training")
    logger.info("Research prototype | Synthetic data | Not for clinical use")
    logger.info("=" * 70)

    cfg = load_config(config_path)
    set_global_seed(cfg.random_seed)
    fg = load_feature_groups(cfg.data.feature_groups_file)

    # ── 1. Load data ──────────────────────────────────────────────────────
    logger.info("Loading datasets…")
    datasets = load_all_datasets(cfg)
    train_df, val_df, test_df, static_df = (
        datasets["train"], datasets["val"], datasets["test"], datasets["static"]
    )

    # ── 2. Leakage check (FAIL FAST) ──────────────────────────────────────
    try:
        run_all_leakage_checks(train_df, val_df, test_df, static_df)
    except PatientLeakageError as e:
        logger.error(str(e))
        sys.exit(1)

    # ── 3. Preprocessing ──────────────────────────────────────────────────
    logger.info("Fitting preprocessing pipeline on training data…")
    pipeline = CKDPreprocessingPipeline(
        fg, unknown_category_token=cfg.preprocessing.unknown_category_token,
        random_seed=cfg.random_seed
    )
    X_train, y_train = pipeline.fit_transform(train_df)
    X_val,   y_val   = pipeline.transform(val_df, "val")
    X_test,  y_test  = pipeline.transform(test_df, "test")

    pipeline.save(cfg.output.preprocessing_dir)

    # Save feature config snapshot
    import json
    feat_cfg = {
        "static_features": fg.static_features,
        "time_varying_features": fg.time_varying_features,
        "temporal_features": fg.temporal_features,
        "all_targets": fg.all_targets,
        "binary_columns": fg.binary_columns,
        "numerical_columns": fg.numerical_columns,
    }
    Path(cfg.output.preprocessing_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.output.preprocessing_dir) / "feature_config.json", "w") as f:
        json.dump(feat_cfg, f, indent=2)

    # ── 4. Build sequences (for Phase 2) ──────────────────────────────────
    if not skip_sequences:
        logger.info("Building patient sequences for Phase 2…")
        for split_name, (df_proc, df_raw) in {
            "train": (X_train, train_df),
            "val":   (X_val,   val_df),
            "test":  (X_test,  test_df),
        }.items():
            # Attach target columns back for sequence builder
            target_cols = [c for c in fg.all_targets if c in df_raw.columns]
            df_with_targets = df_proc.copy()
            for tc in target_cols:
                if tc not in df_with_targets.columns:
                    df_with_targets[tc] = df_raw[tc].values

            static_feat_df = build_baseline_static_features(df_raw, static_df if split_name == "train" else None)
            seqs = build_patient_sequences(
                df_with_targets, fg,
                static_feature_df=static_feat_df,
                max_seq_len=cfg.sequence.max_seq_len,
                pad_value=cfg.sequence.pad_value,
            )
            save_sequences(seqs, cfg.data.sequences_dir, split_name=split_name)

        # Write SEQUENCE_SPEC.md
        _write_sequence_spec(cfg.data.sequences_dir)

    # ── 5. Train all baseline models ──────────────────────────────────────
    target_groups = {
        "classification_stage": cfg.targets.classification_stage,
        "binary_progression":   cfg.targets.binary_progression,
        "regression_egfr":      cfg.targets.regression_egfr,
    }

    for group_name, targets in target_groups.items():
        for target_col in targets:
            horizon = target_col.split("_")[-1]  # "3m", "6m", "12m"
            logger.info(f"\n── Training {group_name} | target={target_col} ──")

            if group_name == "classification_stage":
                models = get_stage_classifiers(cfg, fg)
                y_tr = _prepare_stage_target(y_train, target_col)
                y_v  = _prepare_stage_target(y_val,   target_col)
            elif group_name == "binary_progression":
                models = get_binary_classifiers(cfg, fg)
                y_tr = _prepare_binary_target(y_train, target_col)
                y_v  = _prepare_binary_target(y_val,   target_col)
            else:
                models = get_regressors(cfg, fg)
                y_tr = _prepare_regression_target(y_train, target_col)
                y_v  = _prepare_regression_target(y_val,   target_col)

            if y_tr is None:
                logger.warning(f"Target {target_col} not found in training data — skipping.")
                continue

            mask_tr = ~y_tr.isna()
            mask_v  = ~y_v.isna() if y_v is not None else pd.Series(False)

            for model_name, model in models:
                try:
                    model.fit(X_train[mask_tr], y_tr[mask_tr], feature_groups=fg)
                    safe_name = f"{model_name}_{group_name}_{target_col}"
                    model.model_name = safe_name
                    model.save(cfg.output.baselines_dir)
                except Exception as e:
                    logger.error(f"Failed to fit {model_name} for {target_col}: {e}")

    logger.info("\n✓ Training complete. All models saved to: " + cfg.output.baselines_dir)


# ── Helpers ────────────────────────────────────────────────────────────────

def _prepare_stage_target(y_df, col):
    if col not in y_df.columns: return None
    s = y_df[col].dropna()
    return encode_stage_series(y_df[col], allow_null=True)

def _prepare_binary_target(y_df, col):
    if col not in y_df.columns: return None
    return pd.to_numeric(y_df[col], errors="coerce")

def _prepare_regression_target(y_df, col):
    if col not in y_df.columns: return None
    return pd.to_numeric(y_df[col], errors="coerce")

def _write_sequence_spec(sequences_dir):
    spec_path = Path(sequences_dir) / "SEQUENCE_SPEC.md"
    content = """# Sequence Tensor Specification — Phase 2 Interface

> Research Prototype | Synthetic Data | Not for Clinical Use

## File Locations

Each split has its own subdirectory under `data/processed/sequences/{train,val,test}/`.

## Array Shapes

| Array | Shape | dtype | Description |
|---|---|---|---|
| `patient_ids.npy` | `(N,)` | int64 | Patient IDs in order |
| `static_features.npy` | `(N, S)` | float32 | Static per-patient features |
| `historical_features.npy` | `(N, T, F_tv)` | float32 | Time-varying features per visit |
| `time_features.npy` | `(N, T, F_t)` | float32 | Temporal meta-features per visit |
| `masks.npy` | `(N, T)` | int8 | 1=real visit, 0=padded |
| `targets_stage.npy` | `(N, 3)` | float32 | Stage codes [3m, 6m, 12m] |
| `targets_egfr.npy` | `(N, 3)` | float32 | eGFR values [3m, 6m, 12m] |
| `targets_progression.npy` | `(N, 3)` | float32 | Binary progression [3m, 6m, 12m] |

Where:
- `N` = number of patients
- `T` = max_seq_len (see spec.json in each split directory)
- `S` = number of static features
- `F_tv` = number of time-varying features
- `F_t` = number of temporal meta-features (visit_number, days_from_baseline, etc.)

## Loading

```python
from src.features.sequence_builder import load_sequences
seqs = load_sequences("data/processed/sequences", split_name="train")
```

## Phase 2 Usage

The TFT model should consume:
- `static_features` -> static encoder
- `historical_features` + `time_features` + `masks` -> temporal encoder
- `targets_*` -> multi-task prediction heads

Feature names for each axis are stored in `spec.json` alongside the arrays.
"""
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    with open(spec_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"SEQUENCE_SPEC.md written to {spec_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-sequences", action="store_true",
                        help="Skip sequence building (faster for model-only runs)")
    args = parser.parse_args()
    main(args.config, args.skip_sequences)
