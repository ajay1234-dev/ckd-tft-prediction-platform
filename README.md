# CKD Progression Prediction — Phase 1: Data & ML Foundation

> **Research Prototype** — An Explainable Temporal Fusion Transformer-Based Multi-Task Deep Learning Framework with Temporal Attention, Dynamic Feature Selection, and Multi-Horizon Forecasting for Predicting Chronic Kidney Disease Stage Transitions, Progression Risk, and Longitudinal Renal Function Trajectories

---

## ⚠️ Synthetic Data Disclaimer

> **This is a research prototype using synthetically generated longitudinal CKD data. It is not intended for clinical diagnosis or treatment.**
>
> **Predictions are experimental and should not be used for medical decision-making.**
>
> All datasets are entirely synthetic. No real patient information is present anywhere in this codebase, configurations, or generated reports.

---

## Project Overview

This repository implements **Phase 1 of 4** of the CKD TFT research system:

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | Data ingestion, preprocessing, EDA, baseline ML models | ✅ **This repo** |
| **Phase 2** | Temporal Fusion Transformer + multi-task multi-horizon prediction | ✅ **Complete** |
| Phase 3 | Explainability (SHAP/attention), prediction API, model serving | ✅ **Complete** |
| Phase 4 | Frontend dashboard, full integration, deployment | ✅ **Complete** |

---

## Datasets

Four CSV files reside in `data/raw/`:

| Dataset | File | Patients | Visits |
|---|---|---|---|
| Training | `synthetic_ckd_longitudinal_50000_patients_400000_visits.csv` | 50,000 | 400,000 |
| Validation | `synthetic_ckd_validation_10000_patients_80000_visits.csv` | 10,000 | 80,000 |
| Test | `synthetic_ckd_test_10000_patients_80000_visits.csv` | 10,000 | 80,000 |
| Static baseline | `synthetic_ckd_baseline_static_50000_patients.csv` | 50,000 | 1/patient |

**Column groups (60 columns per longitudinal file):**
- **Identifiers:** `patient_id`, `visit_id`
- **Temporal:** `visit_date`, `visit_number`, `days_from_baseline`, `time_since_previous_visit_days`
- **Demographics:** `age`, `sex`, `height_cm`, `weight_kg`, `bmi`, `smoking_status`, `family_history_ckd`
- **Comorbidities:** `diabetes`, `hypertension`, `cardiovascular_disease`, `heart_failure`, `glomerulonephritis`, `obesity`
- **Vitals:** `systolic_bp`, `diastolic_bp`
- **Renal labs:** `serum_creatinine_mg_dl`, `egfr_ml_min_1_73m2`, `bun_mg_dl`, `uacr_mg_g`, `albuminuria_category`
- **Metabolic labs:** `fasting_glucose_mg_dl`, `hba1c_pct`, `hemoglobin_g_dl`, `anemia`, + electrolytes
- **Medications:** 12 medication/treatment flag columns
- **CKD stage:** `ckd_stage` (G1/G2/G3a/G3b/G4/G5)
- **Targets (never input features):** `stage_3m/6m/12m`, `egfr_3m/6m/12m`, `progression_3m/6m/12m`, `stage_transition`, `progression_event`

---

## Installation

```bash
# 1. Clone / navigate to project root
cd "e:/college works/mohan sir/CKD"

# 2. Create and activate virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

### Step 1 — Data Ingestion & Quality Report
```bash
python scripts/run_ingestion.py
```
Outputs: `reports/metrics/data_quality_report.{csv,json}`

### Step 2 — Patient Leakage Check (runs automatically in training, also standalone)
```bash
python scripts/check_leakage.py
```
Exits with code 0 if clean, code 1 if leakage found.

### Step 3 — Exploratory Data Analysis
```bash
python scripts/run_eda.py
# Optional: control trajectory sample size
python scripts/run_eda.py --sample-patients 100
```
Outputs: `reports/figures/` (17 figures) and `reports/metrics/eda_summary.json`

### Step 4 — Train Baseline Models
```bash
python scripts/train_baselines.py
# Skip sequence building for faster runs:
python scripts/train_baselines.py --skip-sequences
```
Outputs:
- `models/preprocessing/` — fitted imputer, scaler, encoders, manifest
- `models/baselines/` — one directory per model × target × horizon
- `data/processed/sequences/` — train/val/test sequence arrays

### Step 5 — Evaluate Baselines
```bash
python scripts/evaluate_baselines.py --split test
python scripts/evaluate_baselines.py --split val
```
Outputs: `reports/metrics/` and `reports/figures/`

### Step 6 — Start the Inference API (Phase 3)
```bash
uvicorn src.api.main:app --reload --port 8000
```
API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

### Step 7 — Launch the Dashboard (Phase 4)
```bash
streamlit run dashboard.py
```
Dashboard URL: [http://localhost:8501](http://localhost:8501)

### Run All Tests
```bash
pytest tests/ -v --tb=short
# With coverage:
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Project Structure

```
CKD/
├── configs/
│   ├── config.yaml                  # Central config (paths, seed, targets)
│   └── feature_groups.yaml          # Single source of truth for feature definitions
├── data/
│   ├── raw/                         # Original CSVs (never modified)
│   └── processed/
│       └── sequences/               # Phase 2 sequence arrays (train/val/test)
├── models/
│   ├── baselines/                   # Saved baseline model weights
│   └── preprocessing/               # Fitted imputer, scaler, encoders, manifest
├── notebooks/
│   └── 01_eda.ipynb                 # Thin EDA notebook calling src/data/eda.py
├── reports/
│   ├── figures/                     # EDA + confusion matrix PNGs
│   └── metrics/                     # JSON/CSV metric reports
├── scripts/
│   ├── run_ingestion.py
│   ├── check_leakage.py
│   ├── run_eda.py
│   ├── train_baselines.py
│   └── evaluate_baselines.py
├── src/
│   ├── data/
│   │   ├── loader.py                # Dataset loading with explicit dtypes
│   │   ├── schema.py                # Column schema + SchemaValidationError
│   │   ├── quality.py               # DataQualityReport
│   │   └── eda.py                   # 17-figure EDA module
│   ├── features/
│   │   ├── ckd_stage.py             # G1-G5 codec (single source of truth)
│   │   ├── feature_groups.py        # FeatureGroups + assert_no_target_leakage
│   │   ├── static_feature_builder.py
│   │   └── sequence_builder.py      # Phase 2 sequence arrays
│   ├── preprocessing/
│   │   ├── pipeline.py              # CKDPreprocessingPipeline (fit-on-train only)
│   │   └── leakage.py               # PatientLeakageError + all checks
│   ├── baselines/
│   │   ├── base.py                  # BaselineModel abstract class
│   │   ├── classifiers.py           # Stage multiclass (LR/RF/XGB/LGBM)
│   │   ├── binary_classifiers.py    # Progression binary (LR/RF/XGB/LGBM)
│   │   └── regressors.py            # eGFR regression (Ridge/RF/XGB/LGBM)
│   ├── evaluation/
│   │   ├── metrics.py               # Metric computation functions
│   │   └── reports.py               # Save metrics, CMs, summary CSV
│   └── utils/
│       ├── config.py                # Pydantic config loader
│       ├── logging.py               # Structured logging
│       ├── seeding.py               # set_global_seed()
│       └── io.py                    # save/load JSON, CSV, joblib
├── tests/
│   ├── test_ckd_stage.py
│   ├── test_leakage.py
│   ├── test_leakage_guard.py
│   ├── test_schema.py
│   ├── test_preprocessing.py
│   └── test_sequences.py
├── app/                             # Empty placeholder for Phase 3/4
├── requirements.txt
└── README.md
```

---

## Expected Outputs

| Artifact | Location |
|---|---|
| Data quality reports | `reports/metrics/data_quality_report.{csv,json}` |
| EDA figures (17) | `reports/figures/01_*.png … 17_*.png` |
| EDA numeric summary | `reports/metrics/eda_summary.json` |
| Fitted imputer | `models/preprocessing/imputer.joblib` |
| Fitted scaler | `models/preprocessing/scaler.joblib` |
| Fitted encoders | `models/preprocessing/encoders.joblib` |
| Preprocessing manifest | `models/preprocessing/preprocessing_manifest.json` |
| Feature config snapshot | `models/preprocessing/feature_config.json` |
| Baseline models | `models/baselines/{ModelName}_{task}_{target}/` |
| Per-model metrics | `reports/metrics/{ModelName}_{task}_{target}_{horizon}.json` |
| Confusion matrices (arrays) | `reports/metrics/cm_*.csv` |
| Confusion matrices (figures) | `reports/figures/cm_*.png` |
| Consolidated summary | `reports/metrics/baseline_results_summary.csv` |
| Sequence arrays | `data/processed/sequences/{train,val,test}/` |
| Sequence spec | `data/processed/sequences/SEQUENCE_SPEC.md` |

---

## Interface for Phase 2

Phase 2 (TFT implementation) should use the following modules and artifacts:

### Loading Sequence Data
```python
from src.features.sequence_builder import load_sequences

train_seqs = load_sequences("data/processed/sequences", split_name="train")
# Keys: patient_ids, static_features, historical_features, time_features,
#       masks, future_targets (stage, egfr, progression)
```

**Tensor shapes** (documented in `data/processed/sequences/SEQUENCE_SPEC.md`):
- `static_features`: `(N, S)` — static per-patient features
- `historical_features`: `(N, T, F_tv)` — time-varying features per visit
- `time_features`: `(N, T, F_t)` — temporal meta-features per visit
- `masks`: `(N, T)` — 1=real visit, 0=padded
- `future_targets.stage`: `(N, 3)` — [3m, 6m, 12m] encoded stage
- `future_targets.egfr`: `(N, 3)` — [3m, 6m, 12m] eGFR values
- `future_targets.progression`: `(N, 3)` — [3m, 6m, 12m] binary flags

### Loading Preprocessing Pipeline (reuse for Phase 2 inference)
```python
from src.preprocessing.pipeline import CKDPreprocessingPipeline
from src.features.feature_groups import load_feature_groups

fg = load_feature_groups("configs/feature_groups.yaml")
pipeline = CKDPreprocessingPipeline.load("models/preprocessing/", fg)
X_new, y_new = pipeline.transform(new_df)
```

### Loading Baseline Results (for TFT benchmarking)
```python
import pandas as pd
summary = pd.read_csv("reports/metrics/baseline_results_summary.csv")
# Columns: model_name, task_type, target, horizon, + metric columns
```

### Feature Config
```python
import json
with open("models/preprocessing/feature_config.json") as f:
    feat_cfg = json.load(f)
# Keys: static_features, time_varying_features, temporal_features, all_targets, ...
```

### CKD Stage Codec
```python
from src.features.ckd_stage import encode_stage, decode_stage
code = encode_stage("G3a")   # → 3
label = decode_stage(3)      # → "G3a"
```

### Leakage Guard (call before every TFT fit)
```python
from src.features.feature_groups import assert_no_target_leakage, load_feature_groups
fg = load_feature_groups()
assert_no_target_leakage(feature_list, fg, context="TFT training")
```

---

## Hard Constraints (Enforced in Code)

1. All train/val/test splits are at **patient level** — verified by `check_patient_leakage()`.
2. **No future target columns** (`*_3m`, `*_6m`, `*_12m`, `stage_transition`, `progression_event`) ever appear in model input matrices — enforced by `assert_no_target_leakage()`.
3. All imputers/scalers/encoders are **fit on training data only** — never refit on val/test.
4. All metrics are computed from **real model predictions** on held-out data.
5. **Fixed random seed 42** everywhere.
6. CKD stages G1–G5 (with G3a/G3b) are the **canonical clinical labels** — numeric codes are only used internally by models and are always reversible via `decode_stage()`.
