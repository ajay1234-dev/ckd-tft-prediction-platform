"""
tests/test_sequences.py
Tests for longitudinal sequence builder — array shapes match SEQUENCE_SPEC.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import pytest
import numpy as np
import pandas as pd
from src.features.feature_groups import load_feature_groups
from src.features.sequence_builder import build_patient_sequences
from tests.test_preprocessing import make_small_train


@pytest.fixture
def fg():
    return load_feature_groups()


@pytest.fixture
def small_df():
    df = make_small_train(n=30, seed=42)
    # Simulate preprocessed numeric types
    return df


class TestSequenceShapes:
    def test_basic_shapes(self, fg, small_df):
        seqs = build_patient_sequences(small_df, fg)
        N = seqs["n_patients"]
        T = seqs["seq_len"]
        assert seqs["static_features"].ndim == 2
        assert seqs["historical_features"].ndim == 3
        assert seqs["time_features"].ndim == 3
        assert seqs["masks"].ndim == 2

        assert seqs["static_features"].shape[0] == N
        assert seqs["historical_features"].shape[0] == N
        assert seqs["historical_features"].shape[1] == T
        assert seqs["time_features"].shape[0] == N
        assert seqs["time_features"].shape[1] == T
        assert seqs["masks"].shape == (N, T)

    def test_target_shapes(self, fg, small_df):
        seqs = build_patient_sequences(small_df, fg)
        N = seqs["n_patients"]
        for key in ["stage", "egfr", "progression"]:
            arr = seqs["future_targets"][key]
            assert arr.shape == (N, 3), (
                f"future_targets['{key}'] shape {arr.shape} != ({N}, 3)"
            )

    def test_mask_values(self, fg, small_df):
        """Real visits must be marked 1; padded positions 0."""
        seqs = build_patient_sequences(small_df, fg)
        masks = seqs["masks"]
        assert set(np.unique(masks)).issubset({0, 1})

    def test_n_patients_correct(self, fg, small_df):
        n_expected = small_df["patient_id"].nunique()
        seqs = build_patient_sequences(small_df, fg)
        assert seqs["n_patients"] == n_expected

    def test_feature_names_present(self, fg, small_df):
        seqs = build_patient_sequences(small_df, fg)
        assert "time_varying" in seqs["feature_names"]
        assert "temporal" in seqs["feature_names"]

    def test_fixed_seq_len_respected(self, fg, small_df):
        seqs = build_patient_sequences(small_df, fg, max_seq_len=4)
        assert seqs["seq_len"] == 4
        assert seqs["historical_features"].shape[1] == 4
