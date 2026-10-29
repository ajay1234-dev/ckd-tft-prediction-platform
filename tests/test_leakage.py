"""
tests/test_leakage.py
Unit tests for patient leakage detection and chronological ordering.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import pytest
import pandas as pd
import numpy as np
import logging.handlers
from src.preprocessing.leakage import (
    check_patient_leakage,
    PatientLeakageError,
    verify_chronological_order,
    verify_static_baseline_patients,
)


def make_df(patient_ids, visits_per_patient=2, start_date="2020-01-01"):
    """Create a minimal longitudinal dataframe for testing."""
    rows = []
    for pid in patient_ids:
        for v in range(1, visits_per_patient + 1):
            rows.append({
                "patient_id": pid,
                "visit_id": f"P{pid:04d}_V{v:02d}",
                "visit_number": v,
                "visit_date": pd.Timestamp(start_date) + pd.Timedelta(days=(v - 1) * 90),
                "days_from_baseline": (v - 1) * 90,
            })
    return pd.DataFrame(rows)


class TestPatientLeakage:
    def test_clean_splits_pass(self):
        """Disjoint patient IDs should not raise."""
        train = make_df([1, 2, 3])
        val   = make_df([4, 5])
        test  = make_df([6, 7])
        # Should not raise
        check_patient_leakage(train, val, test)

    def test_train_val_overlap_raises(self):
        """Patient in both train and val should trigger PatientLeakageError."""
        train = make_df([1, 2, 3])
        val   = make_df([3, 4, 5])  # patient 3 overlaps!
        test  = make_df([6, 7])
        with pytest.raises(PatientLeakageError, match="train ∩ val"):
            check_patient_leakage(train, val, test)

    def test_train_test_overlap_raises(self):
        train = make_df([1, 2])
        val   = make_df([3, 4])
        test  = make_df([2, 5])    # patient 2 overlaps!
        with pytest.raises(PatientLeakageError, match="train ∩ test"):
            check_patient_leakage(train, val, test)

    def test_val_test_overlap_raises(self):
        train = make_df([1, 2])
        val   = make_df([3, 4])
        test  = make_df([4, 5])    # patient 4 overlaps!
        with pytest.raises(PatientLeakageError, match="val ∩ test"):
            check_patient_leakage(train, val, test)

    def test_patient_never_in_multiple_splits(self):
        """Hard guarantee: after passing check, no patient is in >1 split."""
        train_ids = {1, 2, 3, 4, 5}
        val_ids   = {6, 7, 8}
        test_ids  = {9, 10}
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0


class TestChronologicalOrder:
    def test_correctly_ordered_returns_empty(self):
        df = make_df([1, 2], visits_per_patient=4)
        violations = verify_chronological_order(df)
        assert violations.empty

    def test_shuffled_visits_detected(self):
        """A patient with dates out of sync with visit_number should be flagged."""
        rows = [
            {"patient_id": 1, "visit_id": "P1_V1", "visit_number": 1,
             "visit_date": pd.Timestamp("2020-04-01"), "days_from_baseline": 90},
            {"patient_id": 1, "visit_id": "P1_V2", "visit_number": 2,
             "visit_date": pd.Timestamp("2020-01-01"), "days_from_baseline": 0},   # WRONG ORDER
        ]
        df = pd.DataFrame(rows)
        violations = verify_chronological_order(df)
        assert len(violations) == 1
        assert violations.iloc[0]["patient_id"] == 1


class TestStaticBaselineConsistency:
    def test_all_static_in_train(self):
        """Should log info without errors when static ⊆ train."""
        train = make_df([1, 2, 3, 4, 5])
        static = pd.DataFrame({"patient_id": [1, 2, 3]})
        # Should not raise
        verify_static_baseline_patients(static, train)

    def test_extra_in_static_warns(self):
        """Extra patients in static not in train should produce a warning message."""
        import logging as _logging

        train = make_df([1, 2])
        static = pd.DataFrame({"patient_id": [1, 2, 99]})

        # Attach a MemoryHandler to capture log records from the leakage logger
        captured_records = []
        handler = _logging.handlers.MemoryHandler(capacity=100, flushLevel=_logging.ERROR)
        handler.buffer = captured_records  # direct reference to mutable list

        class RecordCollector(_logging.Handler):
            def emit(self, record):
                captured_records.append(record)

        collector = RecordCollector()
        leakage_logger = _logging.getLogger("src.preprocessing.leakage")
        leakage_logger.addHandler(collector)

        try:
            verify_static_baseline_patients(static, train)
        finally:
            leakage_logger.removeHandler(collector)

        messages = [r.getMessage() for r in captured_records]
        assert any("99" in msg for msg in messages), (
            f"Expected warning mentioning '99' in: {messages}"
        )
