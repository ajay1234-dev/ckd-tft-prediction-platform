"""
tests/test_ckd_stage.py
Unit tests for CKD stage encoding/decoding.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import pytest
from src.features.ckd_stage import (
    encode_stage, decode_stage, encode_stage_series, decode_stage_series,
    is_valid_stage_label, CKD_STAGE_MAP, STAGE_ORDER
)
import pandas as pd


class TestEncodeStage:
    def test_all_valid_labels(self):
        expected = {"G1": 1, "G2": 2, "G3a": 3, "G3b": 4, "G4": 5, "G5": 6}
        for label, code in expected.items():
            assert encode_stage(label) == code

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError, match="Unrecognized CKD stage label"):
            encode_stage("G6")

    def test_numeric_input_raises(self):
        with pytest.raises(ValueError):
            encode_stage(1)

    def test_none_raises(self):
        with pytest.raises(ValueError):
            encode_stage(None)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            encode_stage("")

    def test_lowercase_raises(self):
        with pytest.raises(ValueError):
            encode_stage("g1")  # Must be exact case


class TestDecodeStage:
    def test_all_valid_codes(self):
        expected = {1: "G1", 2: "G2", 3: "G3a", 4: "G3b", 5: "G4", 6: "G5"}
        for code, label in expected.items():
            assert decode_stage(code) == label

    def test_invalid_code_raises(self):
        with pytest.raises(ValueError, match="Unrecognized CKD stage code"):
            decode_stage(7)

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            decode_stage(0)


class TestRoundTrip:
    def test_encode_decode_all_stages(self):
        for label in STAGE_ORDER:
            assert decode_stage(encode_stage(label)) == label

    def test_decode_encode_all_codes(self):
        for code in range(1, 7):
            assert encode_stage(decode_stage(code)) == code


class TestSeriesEncoding:
    def test_encode_series(self):
        s = pd.Series(["G1", "G2", "G3a", "G3b", "G4", "G5"])
        encoded = encode_stage_series(s)
        assert list(encoded) == [1, 2, 3, 4, 5, 6]

    def test_encode_series_with_null(self):
        s = pd.Series(["G1", None, "G3a"])
        encoded = encode_stage_series(s, allow_null=True)
        assert encoded[0] == 1
        assert pd.isna(encoded[1])
        assert encoded[2] == 3


class TestValidLabel:
    def test_valid_labels_return_true(self):
        for label in STAGE_ORDER:
            assert is_valid_stage_label(label) is True

    def test_invalid_label_returns_false(self):
        assert is_valid_stage_label("G6") is False
        assert is_valid_stage_label("") is False
        assert is_valid_stage_label(None) is False
