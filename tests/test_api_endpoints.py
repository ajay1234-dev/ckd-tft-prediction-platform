"""
tests/test_api_endpoints.py
Phase 3 tests — FastAPI endpoint integration tests.

Uses FastAPI's TestClient (httpx) for synchronous API testing.
Model is loaded once at module scope via the lifespan mechanism.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# ── Setup ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Create a TestClient with the FastAPI app (triggers lifespan startup)."""
    from src.api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def valid_patient_id():
    """Get a valid patient ID from the test split for endpoint tests."""
    from src.inference.predictor import PatientStore
    store = PatientStore("data/processed/sequences", split="test").load()
    return store.list_patient_ids(1)[0]


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        """Test 10: GET /health returns success."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_ok(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_shows_model_loaded(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["model_loaded"] is True

    def test_health_has_disclaimer(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "data_disclaimer" in data


# ── Model Info ────────────────────────────────────────────────────────────────

class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client):
        """Test 11: GET /model/info returns expected metadata fields."""
        resp = client.get("/model/info")
        assert resp.status_code == 200

    def test_model_info_has_required_fields(self, client):
        data = client.get("/model/info").json()
        required = ["model_type", "checkpoint", "data_type", "disclaimer",
                    "tasks", "horizons", "stage_labels", "limitations"]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_model_info_data_type_is_synthetic(self, client):
        """Test 11b: data_type must be 'synthetic' — explicit disclaimer."""
        data = client.get("/model/info").json()
        assert data["data_type"] == "synthetic"

    def test_model_info_horizons_correct(self, client):
        data = client.get("/model/info").json()
        assert set(data["horizons"]) == {"3m", "6m", "12m"}

    def test_model_info_has_disclaimer_text(self, client):
        data = client.get("/model/info").json()
        disc = data["disclaimer"]
        assert isinstance(disc, str) and len(disc) > 20


# ── Patient ───────────────────────────────────────────────────────────────────

class TestPatientEndpoint:
    def test_valid_patient_returns_200(self, client, valid_patient_id):
        """Test: GET /patient/{id} returns patient history."""
        resp = client.get(f"/patient/{valid_patient_id}")
        assert resp.status_code == 200

    def test_patient_response_has_visits(self, client, valid_patient_id):
        data = client.get(f"/patient/{valid_patient_id}").json()
        assert "visits" in data
        assert len(data["visits"]) > 0

    def test_nonexistent_patient_returns_404(self, client):
        """Test 9b: Invalid/nonexistent patient returns 404."""
        resp = client.get("/patient/NONEXISTENT_XYZ_999")
        assert resp.status_code == 404

    def test_patient_response_has_disclaimer(self, client, valid_patient_id):
        data = client.get(f"/patient/{valid_patient_id}").json()
        assert "disclaimer" in data


# ── Predict ───────────────────────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_predict_returns_200(self, client, valid_patient_id):
        """Test 12: POST /predict returns valid response."""
        resp = client.post("/predict", json={"patient_id": valid_patient_id})
        assert resp.status_code == 200

    def test_predict_response_has_all_tasks(self, client, valid_patient_id):
        data = client.post("/predict", json={"patient_id": valid_patient_id}).json()
        pred = data["prediction"]
        assert "stage" in pred and "egfr" in pred and "progression" in pred

    def test_predict_stage_has_probabilities(self, client, valid_patient_id):
        data = client.post("/predict", json={"patient_id": valid_patient_id}).json()
        for hz in ["3m", "6m", "12m"]:
            assert "probabilities" in data["prediction"]["stage"][hz]

    def test_predict_missing_patient_returns_404(self, client):
        resp = client.post("/predict", json={"patient_id": "NONEXISTENT_XYZ_999"})
        assert resp.status_code == 404

    def test_predict_empty_patient_id_returns_400(self, client):
        """Test 14: Malformed input is rejected with 400."""
        resp = client.post("/predict", json={"patient_id": ""})
        assert resp.status_code == 400

    def test_predict_missing_patient_id_field_returns_422(self, client):
        """Test 15: Missing required fields are rejected."""
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_predict_response_schema_conformant(self, client, valid_patient_id):
        """Test 12b: Schema conformance — all expected keys present."""
        data = client.post("/predict", json={"patient_id": valid_patient_id}).json()
        assert "patient_id" in data
        assert "prediction" in data
        assert "metadata" in data


# ── Explain ───────────────────────────────────────────────────────────────────

class TestExplainEndpoint:
    def test_explain_returns_200(self, client, valid_patient_id):
        """Test 13: POST /explain returns valid response."""
        resp = client.post("/explain", json={"patient_id": valid_patient_id})
        assert resp.status_code == 200

    def test_explain_includes_feature_importance(self, client, valid_patient_id):
        data = client.post("/explain", json={"patient_id": valid_patient_id}).json()
        assert "feature_importance" in data
        fi = data["feature_importance"]
        assert "static" in fi
        assert "time_varying" in fi

    def test_explain_includes_temporal_attention(self, client, valid_patient_id):
        data = client.post("/explain", json={"patient_id": valid_patient_id}).json()
        assert "temporal_attention" in data
        assert len(data["temporal_attention"]) > 0

    def test_explain_missing_patient_returns_404(self, client):
        resp = client.post("/explain", json={"patient_id": "NONEXISTENT_XYZ_999"})
        assert resp.status_code == 404

    def test_explain_empty_patient_id_returns_400(self, client):
        resp = client.post("/explain", json={"patient_id": ""})
        assert resp.status_code == 400

    def test_explain_missing_field_returns_422(self, client):
        resp = client.post("/explain", json={})
        assert resp.status_code == 422

    def test_explain_response_has_disclaimer(self, client, valid_patient_id):
        data = client.post("/explain", json={"patient_id": valid_patient_id}).json()
        disc = data["metadata"]["disclaimer"]
        assert isinstance(disc, str) and len(disc) > 20

    def test_explain_response_is_json_serializable(self, client, valid_patient_id):
        """Test 16c: Full explain response is JSON-serializable."""
        import json
        resp = client.post("/explain", json={"patient_id": valid_patient_id})
        assert resp.status_code == 200
        # If this doesn't raise, the response is valid JSON
        data = resp.json()
        reserialized = json.dumps(data)
        assert len(reserialized) > 200
