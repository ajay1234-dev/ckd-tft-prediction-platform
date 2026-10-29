"""
src/training/evaluate.py
Test set evaluation for the TFT model.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error, r2_score
)

from src.models.tft.model import TemporalFusionTransformer
from src.training.dataset import get_dataloader
from src.features.ckd_stage import decode_stage
from src.utils.logging import get_logger

logger = get_logger(__name__)

def evaluate_tft(config: Dict[str, Any]) -> None:
    """Run evaluation on the test set."""
    device = torch.device("cuda" if torch.cuda.is_available() and config['training']['device'] != "cpu" else "cpu")
    logger.info(f"Evaluating on device: {device}")
    
    # 1. DataLoader
    test_loader = get_dataloader(
        config['data']['sequence_dir'], 
        "test", 
        config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    # 2. Model (load best)
    model = TemporalFusionTransformer(
        n_static_features=config['model']['n_static_features'],
        n_tv_features=config['model']['n_tv_features'],
        n_temporal_features=config['model']['n_temporal_features'],
        hidden_size=config['model']['hidden_size'],
        num_attention_heads=config['model']['num_attention_heads'],
        dropout=config['model']['dropout'],
        n_classes=config['model']['n_classes'],
    ).to(device)
    
    model_path = Path("models/tft/best_model.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {model_path}")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 3. Collect Predictions
    all_preds = {'stage_logits': [], 'egfr_pred': [], 'progression_logits': []}
    all_targets = {'stage': [], 'egfr': [], 'progression': []}
    patient_ids = []
    
    with torch.no_grad():
        for static, hist, temp, mask, targets in test_loader:
            static = static.to(device)
            hist = hist.to(device)
            temp = temp.to(device)
            mask = mask.to(device)
            
            preds = model(static, hist, temp, mask)
            
            all_preds['stage_logits'].append(preds['stage_logits'].cpu().numpy())
            all_preds['egfr_pred'].append(preds['egfr_pred'].cpu().numpy())
            all_preds['progression_logits'].append(preds['progression_logits'].cpu().numpy())
            
            all_targets['stage'].append(targets['stage'].numpy())
            all_targets['egfr'].append(targets['egfr'].numpy())
            all_targets['progression'].append(targets['progression'].numpy())
            
    # Concat
    for k in all_preds: all_preds[k] = np.concatenate(all_preds[k], axis=0)
    for k in all_targets: all_targets[k] = np.concatenate(all_targets[k], axis=0)
    
    # Also load the patient IDs from the dataset for predictions CSV
    patient_ids = test_loader.dataset.patient_ids
    
    # Setup output dirs
    metrics_dir = Path("reports/phase2/metrics")
    preds_dir = Path("reports/phase2/predictions")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Compute Metrics per horizon
    horizons = ["3m", "6m", "12m"]
    
    # 4a. Stage Metrics
    stage_metrics = []
    s_logits = all_preds['stage_logits']
    s_tgt = all_targets['stage']
    s_probs = torch.softmax(torch.tensor(s_logits), dim=-1).numpy()
    s_preds = np.argmax(s_logits, axis=-1)
    
    for i, hz in enumerate(horizons):
        valid = (s_tgt[:, i] >= 0) & (s_tgt[:, i] < 6)
        y_true = s_tgt[valid, i]
        y_pred = s_preds[valid, i]
        if len(y_true) > 0:
            stage_metrics.append({
                "horizon": hz,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
                "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
                "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
                "f1_weighted": f1_score(y_true, y_pred, average='weighted', zero_division=0),
            })
            
            # Save confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(6))
            pd.DataFrame(cm).to_csv(metrics_dir / f"tft_cm_stage_{hz}.csv", index=False)
            
    pd.DataFrame(stage_metrics).to_csv(metrics_dir / "tft_stage_metrics.csv", index=False)
    
    # 4b. Progression Metrics
    prog_metrics = []
    p_logits = all_preds['progression_logits']
    p_tgt = all_targets['progression']
    def expit(x): return 1/(1+np.exp(-np.clip(x, -500, 500)))
    p_probs = expit(p_logits)
    p_preds = (p_probs > 0.5).astype(int)
    
    for i, hz in enumerate(horizons):
        valid = ~np.isnan(p_tgt[:, i])
        y_true = p_tgt[valid, i]
        y_prob = p_probs[valid, i]
        y_pred = p_preds[valid, i]
        
        if len(y_true) > 0:
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, y_prob)
                pr_auc = average_precision_score(y_true, y_prob)
            else:
                auc, pr_auc = 0.5, 0.5
                
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                
            prog_metrics.append({
                "horizon": hz,
                "roc_auc": auc,
                "pr_auc": pr_auc,
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall_sensitivity": sensitivity,
                "specificity": specificity,
            })
            
    pd.DataFrame(prog_metrics).to_csv(metrics_dir / "tft_progression_metrics.csv", index=False)
    
    # 4c. eGFR Metrics
    egfr_metrics = []
    e_pred = all_preds['egfr_pred']
    e_tgt = all_targets['egfr']
    
    for i, hz in enumerate(horizons):
        valid = ~np.isnan(e_tgt[:, i])
        y_true = e_tgt[valid, i]
        y_p = e_pred[valid, i]
        
        if len(y_true) > 0:
            egfr_metrics.append({
                "horizon": hz,
                "mae": mean_absolute_error(y_true, y_p),
                "rmse": np.sqrt(mean_squared_error(y_true, y_p)),
                "r2": r2_score(y_true, y_p),
            })
            
    pd.DataFrame(egfr_metrics).to_csv(metrics_dir / "tft_egfr_metrics.csv", index=False)
    
    # 5. Save Predictions File
    # We will build a unified DataFrame
    pred_data = {"patient_id": patient_ids}
    
    for i, hz in enumerate(horizons):
        # Stage
        pred_data[f"stage_{hz}_pred"] = [decode_stage(int(c) + 1) for c in s_preds[:, i]]
        for c in range(6):
            pred_data[f"stage_{hz}_prob_G{c+1}"] = s_probs[:, i, c]
            
        # eGFR
        pred_data[f"egfr_{hz}_pred"] = e_pred[:, i]
        
        # Progression
        pred_data[f"progression_{hz}_prob"] = p_probs[:, i]
        pred_data[f"progression_{hz}_pred"] = p_preds[:, i]
        
    df_preds = pd.DataFrame(pred_data)
    df_preds.to_csv(preds_dir / "tft_predictions.csv", index=False)
    
    logger.info(f"Evaluation complete. Metrics saved to {metrics_dir}, Predictions to {preds_dir}")
