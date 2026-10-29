"""
tests/test_schema.py
Tests for schema validation and data loading.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import pytest
import pandas as pd
import numpy as np
from src.data.schema import (
    validate_longitudinal_schema,
    validate_static_baseline_schema,
    SchemaValidationError,
    LONGITUDINAL_REQUIRED_COLUMNS,
    STATIC_BASELINE_REQUIRED_COLUMNS,
    TARGET_COLUMNS,
)


def make_minimal_longitudinal() -> pd.DataFrame:
    """Create a DataFrame with all required longitudinal columns."""
    data = {col: [None] for col in LONGITUDINAL_REQUIRED_COLUMNS}
    return pd.DataFrame(data)


def make_minimal_static() -> pd.DataFrame:
    data = {col: [None] for col in STATIC_BASELINE_REQUIRED_COLUMNS}
    return pd.DataFrame(data)


class TestLongitudinalSchema:
    def test_valid_df_passes(self):
        df = make_minimal_longitudinal()
        validate_longitudinal_schema(df)  # Should not raise

    def test_missing_one_column_raises(self):
        df = make_minimal_longitudinal().drop(columns=["patient_id"])
        with pytest.raises(SchemaValidationError, match="patient_id"):
            validate_longitudinal_schema(df)

    def test_missing_multiple_columns_raises(self):
        df = make_minimal_longitudinal().drop(columns=["patient_id", "egfr_ml_min_1_73m2", "ckd_stage"])
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_longitudinal_schema(df)
        msg = str(exc_info.value)
        assert "patient_id" in msg
        assert "egfr_ml_min_1_73m2" in msg
        assert "ckd_stage" in msg

    def test_extra_columns_ok(self):
        """Extra columns beyond the required set should not cause errors."""
        df = make_minimal_longitudinal()
        df["extra_column"] = 42
        validate_longitudinal_schema(df)  # Should not raise


class TestStaticBaselineSchema:
    def test_valid_static_df_passes(self):
        df = make_minimal_static()
        validate_static_baseline_schema(df)

    def test_missing_column_raises(self):
        df = make_minimal_static().drop(columns=["patient_id"])
        with pytest.raises(SchemaValidationError, match="patient_id"):
            validate_static_baseline_schema(df)


class TestTargetColumns:
    def test_target_columns_defined(self):
        """Verify the target column list is non-empty and contains expected entries."""
        assert len(TARGET_COLUMNS) == 11
        assert "stage_3m" in TARGET_COLUMNS
        assert "egfr_12m" in TARGET_COLUMNS
        assert "progression_event" in TARGET_COLUMNS

    def test_target_columns_not_in_feature_space(self):
        """Target columns must not appear as both required features and targets."""
        required_features = set(LONGITUDINAL_REQUIRED_COLUMNS) - set(TARGET_COLUMNS)
        for tc in TARGET_COLUMNS:
            assert tc not in required_features, f"{tc} appeared in feature space!"
