"""
src/training/train.py
Training loop for the Temporal Fusion Transformer.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import json
import time

from sklearn.metrics import f1_score, roc_auc_score, mean_absolute_error

from src.models.tft.model import TemporalFusionTransformer
from src.models.tft.losses import JointMultiTaskLoss
from src.training.dataset import get_dataloader
from src.utils.logging import get_logger

logger = get_logger(__name__)

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_validation_metrics(all_preds: Dict[str, np.ndarray], all_targets: Dict[str, np.ndarray]) -> Tuple[float, float, float, float]:
    """
    Computes validation metrics for early stopping.
    Returns:
        stage_macro_f1, prog_auc, egfr_mae, combined_score
    """
    # 1. Stage Macro F1
    s_logits = all_preds['stage_logits'] # [N, 3, 6]
    s_tgt = all_targets['stage'] # [N, 3]
    
    valid_s_mask = (s_tgt >= 0) & (s_tgt < 6)
    if valid_s_mask.any():
        s_preds_flat = np.argmax(s_logits[valid_s_mask], axis=-1)
        s_tgt_flat = s_tgt[valid_s_mask]
        stage_macro_f1 = f1_score(s_tgt_flat, s_preds_flat, average='macro', zero_division=0)
    else:
        stage_macro_f1 = 0.0
        
    # 2. Progression AUC
    p_logits = all_preds['progression_logits'] # [N, 3]
    p_tgt = all_targets['progression'] # [N, 3]
    
    valid_p_mask = ~np.isnan(p_tgt)
    if valid_p_mask.any():
        # sigmoid for probability
        def expit(x): return 1/(1+np.exp(-np.clip(x, -500, 500)))
        p_probs_flat = expit(p_logits[valid_p_mask])
        p_tgt_flat = p_tgt[valid_p_mask]
        
        # Check if both classes are present
        if len(np.unique(p_tgt_flat)) > 1:
            prog_auc = roc_auc_score(p_tgt_flat, p_probs_flat)
        else:
            prog_auc = 0.5
    else:
        prog_auc = 0.5
        
    # 3. eGFR MAE
    e_pred = all_preds['egfr_pred'] # [N, 3]
    e_tgt = all_targets['egfr'] # [N, 3]
    
    valid_e_mask = ~np.isnan(e_tgt)
    if valid_e_mask.any():
        egfr_mae = mean_absolute_error(e_tgt[valid_e_mask], e_pred[valid_e_mask])
    else:
        egfr_mae = float('inf')
        
    # Combined score (higher is better)
    # 1/MAE to make it higher-is-better. If MAE=0, limit to 100
    e_score = 1.0 / egfr_mae if egfr_mae > 0.01 else 100.0
    combined_score = stage_macro_f1 + prog_auc + e_score
    
    return stage_macro_f1, prog_auc, egfr_mae, combined_score

def train_tft(config: Dict[str, Any]) -> None:
    """Run full training pipeline."""
    set_seed(config['reproducibility']['random_seed'])
    
    device = torch.device("cuda" if torch.cuda.is_available() and config['training']['device'] != "cpu" else "cpu")
    logger.info(f"Using device: {device}")
    
    # 1. DataLoaders
    train_loader = get_dataloader(
        config['data']['sequence_dir'], 
        "train", 
        config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    val_loader = get_dataloader(
        config['data']['sequence_dir'], 
        "val", 
        config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    # 2. Model
    model = TemporalFusionTransformer(
        n_static_features=config['model']['n_static_features'],
        n_tv_features=config['model']['n_tv_features'],
        n_temporal_features=config['model']['n_temporal_features'],
        hidden_size=config['model']['hidden_size'],
        num_attention_heads=config['model']['num_attention_heads'],
        dropout=config['model']['dropout'],
        n_classes=config['model']['n_classes'],
    ).to(device)
    
    # 3. Loss & Optimizer
    criterion = JointMultiTaskLoss(
        stage_weight=config['loss']['stage_weight'],
        egfr_weight=config['loss']['egfr_weight'],
        progression_weight=config['loss']['progression_weight'],
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Setup tracking
    epochs = config['training']['num_epochs']
    patience = config['training']['early_stopping_patience']
    clip_val = config['training'].get('gradient_clip_val', 1.0)
    
    best_val_score = -float('inf')
    epochs_without_improvement = 0
    
    history = []
    
    # Output dirs
    out_dir = Path("models/tft")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path("reports/phase2/training")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # --- TRAIN ---
        model.train()
        train_losses = {"total_loss": 0.0, "stage_loss": 0.0, "egfr_loss": 0.0, "progression_loss": 0.0}
        
        for batch_idx, (static, hist, temp, mask, targets) in enumerate(train_loader):
            static = static.to(device)
            hist = hist.to(device)
            temp = temp.to(device)
            mask = mask.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            optimizer.zero_grad()
            
            preds = model(static, hist, temp, mask)
            loss, metrics = criterion(preds, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            optimizer.step()
            
            for k, v in metrics.items():
                train_losses[k] += v
                
        # average train losses
        num_train_batches = len(train_loader)
        train_losses = {k: v / num_train_batches for k, v in train_losses.items()}
        
        # --- VAL ---
        model.eval()
        val_losses = {"total_loss": 0.0, "stage_loss": 0.0, "egfr_loss": 0.0, "progression_loss": 0.0}
        
        all_preds = {'stage_logits': [], 'egfr_pred': [], 'progression_logits': []}
        all_targets = {'stage': [], 'egfr': [], 'progression': []}
        
        with torch.no_grad():
            for static, hist, temp, mask, targets in val_loader:
                static = static.to(device)
                hist = hist.to(device)
                temp = temp.to(device)
                mask = mask.to(device)
                targets_d = {k: v.to(device) for k, v in targets.items()}
                
                preds = model(static, hist, temp, mask)
                loss, metrics = criterion(preds, targets_d)
                
                for k, v in metrics.items():
                    val_losses[k] += v
                    
                all_preds['stage_logits'].append(preds['stage_logits'].cpu().numpy())
                all_preds['egfr_pred'].append(preds['egfr_pred'].cpu().numpy())
                all_preds['progression_logits'].append(preds['progression_logits'].cpu().numpy())
                
                all_targets['stage'].append(targets['stage'].numpy())
                all_targets['egfr'].append(targets['egfr'].numpy())
                all_targets['progression'].append(targets['progression'].numpy())
                
        # average val losses
        num_val_batches = len(val_loader)
        val_losses = {k: v / num_val_batches for k, v in val_losses.items()}
        
        # compute validation metrics
        for k in all_preds: all_preds[k] = np.concatenate(all_preds[k], axis=0)
        for k in all_targets: all_targets[k] = np.concatenate(all_targets[k], axis=0)
        
        s_f1, p_auc, e_mae, val_score = compute_validation_metrics(all_preds, all_targets)
        
        elapsed = time.time() - start_time
        
        logger.info(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_losses['total_loss']:.4f} | "
            f"Val Loss: {val_losses['total_loss']:.4f} | "
            f"Val F1: {s_f1:.4f} | Val AUC: {p_auc:.4f} | Val MAE: {e_mae:.4f} | "
            f"Score: {val_score:.4f} | Time: {elapsed:.1f}s"
        )
        
        scheduler.step(val_score)
        
        # Record history
        record = {
            "epoch": epoch,
            "val_score": val_score,
            "val_stage_f1": s_f1,
            "val_prog_auc": p_auc,
            "val_egfr_mae": e_mae,
            "time_sec": elapsed,
        }
        for k, v in train_losses.items(): record[f"train_{k}"] = v
        for k, v in val_losses.items(): record[f"val_{k}"] = v
        history.append(record)
        
        # Checkpointing
        if val_score > best_val_score:
            best_val_score = val_score
            epochs_without_improvement = 0
            
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            with open(out_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"  → Best model saved (score: {val_score:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement.")
                break
                
    # Save final model
    torch.save(model.state_dict(), out_dir / "final_model.pt")
    
    # Save history
    df_history = pd.DataFrame(history)
    df_history.to_csv(report_dir / "tft_training_history.csv", index=False)
    logger.info("Training complete.")
