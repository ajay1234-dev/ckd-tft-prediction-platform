"""
tests/test_preprocessing.py
Tests for the preprocessing pipeline — imputation, encoding, determinism.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import pytest
import numpy as np
import pandas as pd
from src.features.feature_groups import load_feature_groups
from src.preprocessing.pipeline import CKDPreprocessingPipeline, _canonicalize_binary


@pytest.fixture
def fg():
    return load_feature_groups()


def make_small_train(n=50, seed=42):
    """Synthetic minimal training DataFrame for pipeline tests."""
    rng = np.random.default_rng(seed)
    n_stages = 6
    stage_labels = ["G1", "G2", "G3a", "G3b", "G4", "G5"]
    n_per = n // n_stages

    rows = []
    pid = 1
    for stage in stage_labels:
        for _ in range(n_per):
            for v in range(1, 3):
                rows.append({
                    "patient_id": pid,
                    "visit_id": f"P{pid:04d}_V{v:02d}",
                    "visit_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=(v-1)*90),
                    "visit_number": v,
                    "days_from_baseline": (v-1)*90,
                    "time_since_previous_visit_days": 90 if v > 1 else 0,
                    "age": float(rng.integers(30, 80)),
                    "sex": rng.choice(["Male", "Female"]),
                    "height_cm": float(rng.uniform(150, 190)),
                    "weight_kg": float(rng.uniform(50, 120)),
                    "bmi": float(rng.uniform(18, 40)),
                    "smoking_status": rng.choice(["Never", "Former", "Current"]),
                    "family_history_ckd": int(rng.integers(0, 2)),
                    "diabetes": int(rng.integers(0, 2)),
                    "hypertension": int(rng.integers(0, 2)),
                    "cardiovascular_disease": 0,
                    "heart_failure": 0,
                    "glomerulonephritis": 0,
                    "obesity": 0,
                    "systolic_bp": float(rng.uniform(100, 180)),
                    "diastolic_bp": float(rng.uniform(60, 110)),
                    "serum_creatinine_mg_dl": float(rng.uniform(0.5, 10)),
                    "egfr_ml_min_1_73m2": float(rng.uniform(5, 100)),
                    "bun_mg_dl": float(rng.uniform(5, 100)),
                    "uacr_mg_g": float(rng.uniform(0, 3000)),
                    "albuminuria_category": rng.choice(["A1", "A2", "A3"]),
                    "fasting_glucose_mg_dl": float(rng.uniform(70, 300)),
                    "hba1c_pct": float(rng.uniform(4, 14)),
                    "hemoglobin_g_dl": float(rng.uniform(7, 17)),
                    "anemia": int(rng.integers(0, 2)),
                    "sodium_meq_l": float(rng.uniform(130, 150)),
                    "potassium_meq_l": float(rng.uniform(3, 6)),
                    "calcium_mg_dl": float(rng.uniform(7, 12)),
                    "phosphate_mg_dl": float(rng.uniform(2, 8)),
                    "bicarbonate_meq_l": float(rng.uniform(15, 30)),
                    "uric_acid_mg_dl": float(rng.uniform(3, 12)),
                    "acei_arb": int(rng.integers(0, 2)),
                    "sglt2_inhibitor": 0,
                    "diuretic": 0,
                    "antihypertensive": int(rng.integers(0, 2)),
                    "statin": 0,
                    "epo": 0,
                    "iron_supplement": 0,
                    "metformin": 0,
                    "insulin": 0,
                    "on_dialysis": 0,
                    "dialysis_type": float("nan"),
                    "kidney_transplant": 0,
                    "ckd_stage": stage,
                    # Targets
                    "stage_3m": stage,
                    "egfr_3m": float(rng.uniform(5, 100)),
                    "progression_3m": float(rng.integers(0, 2)),
                    "stage_6m": stage,
                    "egfr_6m": float(rng.uniform(5, 100)),
                    "progression_6m": float(rng.integers(0, 2)),
                    "stage_12m": stage,
                    "egfr_12m": float(rng.uniform(5, 100)),
                    "progression_12m": float(rng.integers(0, 2)),
                    "stage_transition": "Stable",
                    "progression_event": int(rng.integers(0, 2)),
                })
            pid += 1
    return pd.DataFrame(rows)


class TestBinaryCanonicalization:
    def test_int_values(self):
        s = pd.Series([0, 1, 0, 1])
        result = _canonicalize_binary(s)
        assert list(result) == [0, 1, 0, 1]

    def test_boolean_values(self):
        s = pd.Series([True, False, True])
        result = _canonicalize_binary(s)
        assert list(result) == [1, 0, 1]

    def test_float_values(self):
        s = pd.Series([1.0, 0.0, 1.0])
        result = _canonicalize_binary(s)
        assert list(result) == [1, 0, 1]

    def test_string_yn(self):
        s = pd.Series(["Y", "N", "y", "n"])
        result = _canonicalize_binary(s)
        assert list(result) == [1, 0, 1, 0]


class TestPreprocessingPipeline:
    @pytest.fixture
    def pipeline_and_data(self, fg):
        train_df = make_small_train(n=60, seed=42)
        val_df   = make_small_train(n=30, seed=99)
        # Assign different patient IDs to val
        val_df["patient_id"] = val_df["patient_id"] + 1000
        pipeline = CKDPreprocessingPipeline(fg, random_seed=42)
        X_train, y_train = pipeline.fit_transform(train_df)
        X_val,   y_val   = pipeline.transform(val_df, "val")
        return pipeline, X_train, y_train, X_val, y_val, train_df

    def test_no_target_in_X_train(self, pipeline_and_data, fg):
        _, X_train, _, _, _, _ = pipeline_and_data
        target_set = set(fg.all_targets)
        leaked = [c for c in X_train.columns if c in target_set]
        assert len(leaked) == 0, f"Target columns leaked into X_train: {leaked}"

    def test_targets_in_y_train(self, pipeline_and_data, fg):
        _, _, y_train, _, _, _ = pipeline_and_data
        assert "stage_3m" in y_train.columns

    def test_train_has_no_nans_in_numerical(self, pipeline_and_data, fg):
        _, X_train, _, _, _, _ = pipeline_and_data
        num_cols = [c for c in fg.numerical_columns if c in X_train.columns]
        assert X_train[num_cols].isnull().sum().sum() == 0, "NaNs remain in numerical columns after imputation"

    def test_val_has_no_nans_in_numerical(self, pipeline_and_data, fg):
        _, _, _, X_val, _, _ = pipeline_and_data
        num_cols = [c for c in fg.numerical_columns if c in X_val.columns]
        assert X_val[num_cols].isnull().sum().sum() == 0

    def test_determinism(self, fg):
        """Same input processed twice with same seed yields identical output."""
        df = make_small_train(n=30, seed=42)
        p1 = CKDPreprocessingPipeline(fg, random_seed=42)
        X1, _ = p1.fit_transform(df.copy())

        p2 = CKDPreprocessingPipeline(fg, random_seed=42)
        X2, _ = p2.fit_transform(df.copy())

        num_cols = [c for c in fg.numerical_columns if c in X1.columns and c in X2.columns]
        pd.testing.assert_frame_equal(X1[num_cols].reset_index(drop=True),
                                      X2[num_cols].reset_index(drop=True))
