"""
tests/test_tft_inference.py
Unit tests for TFT inference function.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import pytest
import torch
import json
from pathlib import Path
import tempfile

from src.models.tft.model import TemporalFusionTransformer
from src.training.inference import load_tft_model, predict_patient

@pytest.fixture
def mock_model_dir(tmp_path):
    """Creates a temporary directory with a dummy model and config."""
    config = {
        'model': {
            'n_static_features': 10,
            'n_tv_features': 29,
            'n_temporal_features': 3,
            'hidden_size': 16,
            'num_attention_heads': 2,
            'dropout': 0.1,
            'n_classes': 6,
        }
    }
    
    with open(tmp_path / "config.json", "w") as f:
        json.dump(config, f)
        
    model = TemporalFusionTransformer(**config['model'])
    torch.save(model.state_dict(), tmp_path / "best_model.pt")
    
    return tmp_path

def test_inference_signature_and_output(mock_model_dir):
    """Test that predict_patient returns the correctly structured dictionary."""
    # Reset cache in case of contamination
    import src.training.inference
    src.training.inference._CACHED_MODEL = None
    
    static = torch.randn(10)
    tv = torch.randn(8, 29)
    temp = torch.randn(8, 3)
    
    result = predict_patient(
        patient_id="P123",
        static_features=static,
        historical_features=tv,
        time_features=temp,
        model_dir=mock_model_dir
    )
    
    assert result["patient_id"] == "P123"
    
    for hz in ["3m", "6m", "12m"]:
        assert hz in result["stage_predictions"]
        assert "predicted_stage" in result["stage_predictions"][hz]
        assert "probabilities" in result["stage_predictions"][hz]
        for stage in ["G1", "G2", "G3a", "G3b", "G4", "G5"]:
            assert stage in result["stage_predictions"][hz]["probabilities"]
            
        assert hz in result["egfr_predictions"]
        assert isinstance(result["egfr_predictions"][hz], float)
        
        assert hz in result["progression_predictions"]
        assert "probability" in result["progression_predictions"][hz]
        assert "predicted_class" in result["progression_predictions"][hz]

def test_inference_determinism(mock_model_dir):
    """Test that calling inference twice with the same input yields identical results."""
    import src.training.inference
    src.training.inference._CACHED_MODEL = None
    
    static = torch.randn(10)
    tv = torch.randn(8, 29)
    temp = torch.randn(8, 3)
    
    res1 = predict_patient("P1", static, tv, temp, model_dir=mock_model_dir)
    res2 = predict_patient("P1", static, tv, temp, model_dir=mock_model_dir)
    
    assert res1 == res2
