"""
tests/test_inference_engine.py
Phase 3 tests — inference engine (model loading, prediction, schema).

All tests use the actual trained Phase 2 checkpoint and real synthetic test data.
No mocking of model outputs.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model_and_config():
    """Load the trained TFT model once for the whole module."""
    from src.inference.model_loader import ModelLoader
    loader = ModelLoader()
    model, config = loader.load(device="cpu")
    return model, config


@pytest.fixture(scope="module")
def patient_store():
    """Load the test patient store once."""
    from src.inference.predictor import PatientStore
    store = PatientStore("data/processed/sequences", split="test").load()
    return store


@pytest.fixture(scope="module")
def preprocessor():
    from src.inference.preprocessing import InferencePreprocessor
    return InferencePreprocessor().load()


@pytest.fixture(scope="module")
def predictor(model_and_config, patient_store, preprocessor):
    """Build a full predictor."""
    from src.inference.model_loader import ModelLoader
    from src.inference.predictor import CKDPredictor
    loader = ModelLoader()
    pred = CKDPredictor(
        model_loader=loader,
        preprocessor=preprocessor,
        patient_store=patient_store,
        device="cpu",
    ).initialize()
    return pred


@pytest.fixture(scope="module")
def first_patient_id(patient_store):
    return patient_store.list_patient_ids(1)[0]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestModelLoading:
    def test_checkpoint_loads(self, model_and_config):
        """Test 1: Checkpoint and config load correctly."""
        model, config = model_and_config
        assert model is not None
        assert config is not None
        assert "model" in config

    def test_model_in_eval_mode(self, model_and_config):
        """Loaded model should be in eval mode."""
        model, _ = model_and_config
        assert not model.training

    def test_missing_checkpoint_fails_fast(self, tmp_path):
        """Test: Missing checkpoint raises FileNotFoundError, not a confusing error."""
        from src.inference.model_loader import ModelLoader
        # Reset singleton to test fresh loading from a bad path
        ModelLoader.reset()
        loader = ModelLoader(tmp_path)
        with pytest.raises(FileNotFoundError, match="config"):
            loader.load()
        # Restore: reset again so other tests reload cleanly
        ModelLoader.reset()

    def test_cpu_inference_works(self, model_and_config):
        """Test 20: CPU inference works with no GPU present."""
        model, config = model_and_config
        mc = config["model"]
        s = torch.zeros(1, mc["n_static_features"])
        h = torch.zeros(1, 8, mc["n_tv_features"])
        t = torch.zeros(1, 8, mc["n_temporal_features"])
        m = torch.ones(1, 8, dtype=torch.int8)
        with torch.no_grad():
            out = model(s, h, t, m)
        assert "stage_logits" in out


class TestPredictionOutput:
    def test_prediction_runs_end_to_end(self, predictor, first_patient_id):
        """Test 2: Inference engine runs end-to-end on a fixture patient."""
        result = predictor.predict(first_patient_id)
        assert result is not None
        assert "prediction" in result

    def test_prediction_has_correct_structure(self, predictor, first_patient_id):
        """Test 3: Prediction output structure matches schema."""
        result = predictor.predict(first_patient_id)
        pred = result["prediction"]
        assert "stage" in pred
        assert "egfr" in pred
        assert "progression" in pred

    def test_all_three_horizons_present(self, predictor, first_patient_id):
        """Test 7: All three horizons (3m/6m/12m) present and distinct in outputs."""
        result = predictor.predict(first_patient_id)
        pred = result["prediction"]
        for task in ["stage", "egfr", "progression"]:
            assert "3m" in pred[task], f"Missing 3m in {task}"
            assert "6m" in pred[task], f"Missing 6m in {task}"
            assert "12m" in pred[task], f"Missing 12m in {task}"

    def test_stage_probabilities_valid(self, predictor, first_patient_id):
        """Test 4: Stage probabilities are valid [0,1] and sum ≈ 1 for all horizons."""
        result = predictor.predict(first_patient_id)
        for hz in ["3m", "6m", "12m"]:
            probs = result["prediction"]["stage"][hz]["probabilities"]
            for stage, p in probs.items():
                assert 0.0 <= p <= 1.0, f"Stage prob out of range: {stage}={p}"
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-4, f"Probs don't sum to 1: {total}"

    def test_egfr_outputs_are_finite(self, predictor, first_patient_id):
        """Test 5: eGFR outputs are finite floats for all 3 horizons."""
        result = predictor.predict(first_patient_id)
        for hz in ["3m", "6m", "12m"]:
            val = result["prediction"]["egfr"][hz]
            assert isinstance(val, float), f"eGFR is not a float: {type(val)}"
            assert math.isfinite(val), f"eGFR is not finite: {val}"

    def test_progression_probabilities_valid(self, predictor, first_patient_id):
        """Test 6: Progression probabilities are valid [0,1] for all horizons."""
        result = predictor.predict(first_patient_id)
        for hz in ["3m", "6m", "12m"]:
            prob = result["prediction"]["progression"][hz]["probability"]
            assert 0.0 <= prob <= 1.0, f"Progression prob out of range at {hz}: {prob}"

    def test_inference_determinism(self, predictor, first_patient_id):
        """Test 21: Same model + same input → same output (CPU is deterministic)."""
        r1 = predictor.predict(first_patient_id)
        r2 = predictor.predict(first_patient_id)
        for hz in ["3m", "6m", "12m"]:
            assert r1["prediction"]["egfr"][hz] == r2["prediction"]["egfr"][hz], \
                "eGFR outputs differ between identical runs"
            assert r1["prediction"]["stage"][hz]["predicted_stage"] == \
                   r2["prediction"]["stage"][hz]["predicted_stage"]


class TestPatientLookup:
    def test_patient_lookup_returns_correct_data(self, patient_store, first_patient_id):
        """Test 8: Patient lookup returns the correct history for a known patient ID."""
        data = patient_store.get_patient(first_patient_id)
        assert data is not None
        assert data["patient_id"] == first_patient_id
        assert "static" in data
        assert "hist" in data
        assert "mask" in data

    def test_nonexistent_patient_returns_none(self, patient_store):
        """Test 9: Invalid patient ID returns None (not a crash)."""
        result = patient_store.get_patient("NONEXISTENT_PATIENT_XYZ")
        assert result is None

    def test_predictor_raises_key_error_for_missing_patient(self, predictor):
        """Missing patient in predictor raises KeyError (caught by API as 404)."""
        with pytest.raises(KeyError):
            predictor.predict("NONEXISTENT_PATIENT_XYZ")
