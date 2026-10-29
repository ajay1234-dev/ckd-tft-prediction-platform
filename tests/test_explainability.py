"""
tests/test_explainability.py
Phase 3 tests — explainability outputs.

Tests that explanation values come from real model outputs,
use real feature names, and are correctly structured.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

# ── Fixtures (reuse from test_inference_engine via shared setup) ──────────────

@pytest.fixture(scope="module")
def predictor_and_store():
    from src.inference.model_loader import ModelLoader
    from src.inference.predictor import CKDPredictor, PatientStore
    from src.inference.preprocessing import InferencePreprocessor

    loader = ModelLoader()
    preprocessor = InferencePreprocessor().load()
    store = PatientStore("data/processed/sequences", split="test").load()
    pred = CKDPredictor(
        model_loader=loader,
        preprocessor=preprocessor,
        patient_store=store,
        device="cpu",
    ).initialize()
    return pred, store


@pytest.fixture(scope="module")
def explanation(predictor_and_store):
    predictor, store = predictor_and_store
    pid = store.list_patient_ids(1)[0]
    return predictor.predict_with_explanation(pid)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestExplanationStructure:
    def test_explanation_has_required_keys(self, explanation):
        """Test 16: Explanation output structure matches documented schema."""
        required = ["patient_id", "prediction", "feature_importance",
                    "temporal_attention", "metadata"]
        for key in required:
            assert key in explanation, f"Missing key: {key}"

    def test_explanation_is_json_serializable(self, explanation):
        """Test 16b: Explanation is fully JSON-serializable (no raw tensors/numpy)."""
        serialized = json.dumps(explanation)
        assert len(serialized) > 100  # non-trivial output

    def test_feature_importance_uses_real_names(self, explanation):
        """Test 18: Variable importance uses real feature names, not placeholders."""
        from src.explainability.variable_importance import STATIC_FEATURE_NAMES, TV_FEATURE_NAMES
        fi = explanation["feature_importance"]

        # Static names should match real feature list
        for name in fi["static"].keys():
            assert name in STATIC_FEATURE_NAMES, f"Unknown static feature: {name}"

        # Time-varying names should match real feature list
        for name in fi["time_varying"].keys():
            assert name in TV_FEATURE_NAMES + ["visit_number", "days_from_baseline"], \
                f"Unknown TV feature: {name}"

    def test_feature_importance_values_in_range(self, explanation):
        """Feature importance values should be in [0, 1] (softmax outputs)."""
        fi = explanation["feature_importance"]
        for group in ["static", "time_varying"]:
            for name, val in fi[group].items():
                assert 0.0 <= val <= 1.0, f"{group}/{name} out of range: {val}"

    def test_static_importance_sums_to_one(self, explanation):
        """Static VSN weights should sum to ~1 (softmax)."""
        fi = explanation["feature_importance"]
        total = sum(fi["static"].values())
        # Only top-k may be returned; if all returned, should sum to 1
        # With top-k=10 and 4 static features, all are returned
        assert abs(total - 1.0) < 0.01, f"Static importance sums to {total}, not 1"


class TestTemporalAttention:
    def test_attention_has_one_entry_per_timestep(self, explanation):
        """Test 17: Temporal attention has one entry per time step."""
        attn = explanation["temporal_attention"]
        assert len(attn) == 8, f"Expected 8 attention steps, got {len(attn)}"

    def test_attention_weights_non_negative(self, explanation):
        """All attention weights should be non-negative."""
        for step in explanation["temporal_attention"]:
            assert step["attention_weight"] >= 0.0, \
                f"Negative attention weight at visit {step['visit_index']}"

    def test_attention_has_padding_flags(self, explanation):
        """Test 17b: Attention entries have is_padding flag."""
        for step in explanation["temporal_attention"]:
            assert "is_padding" in step

    def test_padding_steps_labeled(self, explanation):
        """Padding steps should be distinguishable from valid steps."""
        attn = explanation["temporal_attention"]
        # At least one step should be valid (not padding)
        valid_steps = [s for s in attn if not s["is_padding"]]
        assert len(valid_steps) > 0, "No valid (non-padded) steps in attention output"

    def test_attention_has_disclaimer(self, explanation):
        """Test: Each attention step includes the interpretability disclaimer."""
        for step in explanation["temporal_attention"]:
            assert "disclaimer" in step
            assert len(step["disclaimer"]) > 10


class TestExplanationMetadata:
    def test_metadata_has_required_fields(self, explanation):
        meta = explanation["metadata"]
        assert "model_checkpoint" in meta
        assert "explanation_method" in meta
        assert "generated_at" in meta
        assert "disclaimer" in meta

    def test_explanation_method_is_tft_native(self, explanation):
        """Explanation method should be tft_native (not SHAP or random)."""
        assert explanation["metadata"]["explanation_method"] == "tft_native"

    def test_disclaimer_present(self, explanation):
        """Disclaimer must be present and non-empty."""
        disc = explanation["metadata"]["disclaimer"]
        assert isinstance(disc, str) and len(disc) > 20


class TestPredictionInExplanation:
    def test_explanation_contains_all_prediction_horizons(self, explanation):
        """Test: /explain includes all 3 prediction horizons."""
        pred = explanation["prediction"]
        for task in ["stage", "egfr", "progression"]:
            for hz in ["3m", "6m", "12m"]:
                assert hz in pred[task], f"Missing {hz} in explanation/{task}"

    def test_stage_probs_sum_to_one_in_explanation(self, explanation):
        """Stage probabilities in explanation should sum to ~1."""
        for hz in ["3m", "6m", "12m"]:
            probs = explanation["prediction"]["stage"][hz]["probabilities"]
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-4
