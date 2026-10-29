"""
src/training/inference.py
Reusable single-patient inference function for the TFT.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from src.models.tft.model import TemporalFusionTransformer
from src.features.ckd_stage import decode_stage

# Cache for the loaded model to avoid reloading on every call
_CACHED_MODEL: Optional[TemporalFusionTransformer] = None
_CACHED_CONFIG: Optional[Dict[str, Any]] = None

def load_tft_model(model_dir: str | Path = "models/tft", device: str = "cpu") -> Tuple[TemporalFusionTransformer, Dict[str, Any]]:
    """Loads the TFT model and its config from disk."""
    global _CACHED_MODEL, _CACHED_CONFIG
    
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL, _CACHED_CONFIG
        
    model_dir = Path(model_dir)
    with open(model_dir / "config.json", "r") as f:
        config = json.load(f)
        
    model = TemporalFusionTransformer(
        n_static_features=config['model']['n_static_features'],
        n_tv_features=config['model']['n_tv_features'],
        n_temporal_features=config['model']['n_temporal_features'],
        hidden_size=config['model']['hidden_size'],
        num_attention_heads=config['model']['num_attention_heads'],
        dropout=config['model']['dropout'],
        n_classes=config['model']['n_classes'],
    ).to(device)
    
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    model.eval()
    
    _CACHED_MODEL = model
    _CACHED_CONFIG = config
    
    return model, config


def predict_patient(
    patient_id: str,
    static_features: torch.Tensor,
    historical_features: torch.Tensor,
    time_features: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    model_dir: str | Path = "models/tft",
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run inference for a single patient's sequence.
    
    Parameters
    ----------
    patient_id : str
    static_features : Tensor [1, F_s]
    historical_features : Tensor [1, SeqLen, F_tv]
    time_features : Tensor [1, SeqLen, F_t]
    mask : Tensor [1, SeqLen], optional
    
    Returns
    -------
    Dictionary structured as required by Phase 3 backend.
    """
    model, config = load_tft_model(model_dir, device)
    
    # Ensure inputs are on correct device and batch size 1
    if static_features.dim() == 1:
        static_features = static_features.unsqueeze(0)
    if historical_features.dim() == 2:
        historical_features = historical_features.unsqueeze(0)
    if time_features.dim() == 2:
        time_features = time_features.unsqueeze(0)
    if mask is not None and mask.dim() == 1:
        mask = mask.unsqueeze(0)
        
    static_features = static_features.to(device)
    historical_features = historical_features.to(device)
    time_features = time_features.to(device)
    if mask is not None:
        mask = mask.to(device)
        
    with torch.no_grad():
        preds = model(static_features, historical_features, time_features, mask)
        
    s_logits = preds['stage_logits'][0] # [3, 6]
    e_pred = preds['egfr_pred'][0] # [3]
    p_logits = preds['progression_logits'][0] # [3]
    
    # Process stage
    s_probs = torch.softmax(s_logits, dim=-1).cpu().numpy()
    s_preds = torch.argmax(s_logits, dim=-1).cpu().numpy()
    
    # Process progression
    p_probs = torch.sigmoid(p_logits).cpu().numpy()
    p_preds = (p_probs > 0.5).astype(int)
    
    e_pred = e_pred.cpu().numpy()
    
    horizons = ["3m", "6m", "12m"]
    
    result = {
        "patient_id": patient_id,
        "stage_predictions": {},
        "egfr_predictions": {},
        "progression_predictions": {}
    }
    
    for i, hz in enumerate(horizons):
        # Stage
        pred_label = decode_stage(int(s_preds[i]) + 1)
        probs_dict = {f"G{c+1}": float(s_probs[i, c]) for c in range(6)}
        # Fix G3a/G3b keys
        probs_dict["G3a"] = probs_dict.pop("G3")
        probs_dict["G3b"] = probs_dict.pop("G4")
        probs_dict["G4"] = probs_dict.pop("G5")
        probs_dict["G5"] = probs_dict.pop("G6")
        
        result["stage_predictions"][hz] = {
            "predicted_stage": pred_label,
            "probabilities": probs_dict
        }
        
        # eGFR
        result["egfr_predictions"][hz] = float(e_pred[i])
        
        # Progression
        result["progression_predictions"][hz] = {
            "probability": float(p_probs[i]),
            "predicted_class": int(p_preds[i])
        }
        
    return result
