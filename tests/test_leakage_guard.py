"""
tests/test_leakage_guard.py
Tests for assert_no_target_leakage — verifies that target columns
never appear in model feature matrices.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import pytest
from src.features.feature_groups import load_feature_groups, assert_no_target_leakage


@pytest.fixture
def fg():
    return load_feature_groups()


class TestLeakageGuard:
    def test_clean_feature_list_passes(self, fg):
        """Non-target features should pass the guard without error."""
        clean_features = ["age", "systolic_bp", "egfr_ml_min_1_73m2", "bmi"]
        assert_no_target_leakage(clean_features, fg, context="test_clean")

    def test_stage_target_raises(self, fg):
        """stage_3m in feature list must raise ValueError."""
        with pytest.raises(ValueError, match="TARGET LEAKAGE DETECTED"):
            assert_no_target_leakage(
                ["age", "stage_3m", "bmi"],
                fg, context="test_stage_leak"
            )

    def test_egfr_target_raises(self, fg):
        with pytest.raises(ValueError, match="egfr_12m"):
            assert_no_target_leakage(["egfr_ml_min_1_73m2", "egfr_12m"], fg)

    def test_progression_target_raises(self, fg):
        with pytest.raises(ValueError, match="progression_6m"):
            assert_no_target_leakage(["age", "progression_6m"], fg)

    def test_stage_transition_raises(self, fg):
        with pytest.raises(ValueError, match="stage_transition"):
            assert_no_target_leakage(["ckd_stage", "stage_transition"], fg)

    def test_progression_event_raises(self, fg):
        with pytest.raises(ValueError, match="progression_event"):
            assert_no_target_leakage(["age", "progression_event"], fg)

    def test_all_targets_raise(self, fg):
        """Every single target column must trigger the guard."""
        for target_col in fg.all_targets:
            with pytest.raises(ValueError, match="TARGET LEAKAGE DETECTED"):
                assert_no_target_leakage([target_col], fg, context=f"single_{target_col}")

    def test_empty_feature_list_passes(self, fg):
        """Empty feature list has nothing to leak."""
        assert_no_target_leakage([], fg, context="empty")

    def test_no_target_in_model_input_matrix(self, fg):
        """
        Hard guarantee: assembling the standard input feature set
        must pass the leakage guard.
        """
        # Standard input features for a baseline model
        input_features = (
            fg.static_features
            + fg.time_varying_features
            + fg.temporal_features
        )
        # Remove any target columns that happen to be in the lists (none should be)
        target_set = set(fg.all_targets)
        leaked = [f for f in input_features if f in target_set]
        assert len(leaked) == 0, (
            f"Target columns found in standard input feature set: {leaked}"
        )
