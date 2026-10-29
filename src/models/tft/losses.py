"""
src/models/tft/losses.py
Joint Multi-Task Loss for the TFT model.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple

class JointMultiTaskLoss(nn.Module):
    """
    Computes a weighted sum of losses across the three prediction tasks:
      1. Stage classification (CrossEntropy)
      2. eGFR regression (SmoothL1)
      3. Progression risk (BCEWithLogits)
      
    All losses are computed per horizon and then averaged over the 3 horizons.
    """
    def __init__(
        self,
        stage_weight: float = 1.0,
        egfr_weight: float = 0.01,
        progression_weight: float = 1.0,
    ):
        super().__init__()
        self.stage_weight = stage_weight
        self.egfr_weight = egfr_weight
        self.progression_weight = progression_weight
        
        # Loss functions
        # For stage, reduction=mean means mean over batch and horizons
        self.stage_loss_fn = nn.CrossEntropyLoss()
        
        # For eGFR, Huber loss is robust to outliers
        self.egfr_loss_fn = nn.SmoothL1Loss()
        
        # For progression, BCE with logits handles the binary classification
        self.progression_loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        predictions: dict containing:
            'stage_logits': [Batch, 3, 6]
            'egfr_pred': [Batch, 3]
            'progression_logits': [Batch, 3]
        targets: dict containing:
            'stage': [Batch, 3] (values 0-5)
            'egfr': [Batch, 3] (continuous)
            'progression': [Batch, 3] (values 0, 1)
            
        Returns
        -------
        total_loss : scalar Tensor
        metrics : dict with individual loss floats for logging
        """
        # Ensure target shapes and types
        stage_tgt = targets['stage'].long() # CrossEntropy requires long [Batch, 3]
        egfr_tgt = targets['egfr'].float()  # [Batch, 3]
        prog_tgt = targets['progression'].float() # [Batch, 3]
        
        # Predictions
        stage_logits = predictions['stage_logits'] # [Batch, 3, 6]
        egfr_pred = predictions['egfr_pred'] # [Batch, 3]
        prog_logits = predictions['progression_logits'] # [Batch, 3]
        
        # 1. Stage Loss
        # CrossEntropy expects logits of shape [Batch, C, d1, d2...] or [Batch*d1, C]
        # We reshape to [Batch * 3, 6] and target to [Batch * 3]
        s_logits_flat = stage_logits.view(-1, 6)
        s_tgt_flat = stage_tgt.view(-1)
        
        # Mask out NaN targets (if any)
        valid_stage_mask = (s_tgt_flat >= 0) & (s_tgt_flat < 6)
        if valid_stage_mask.any():
            stage_loss = self.stage_loss_fn(s_logits_flat[valid_stage_mask], s_tgt_flat[valid_stage_mask])
        else:
            stage_loss = 0.0 * stage_logits.sum()
            
        # 2. eGFR Loss
        valid_egfr_mask = ~torch.isnan(egfr_tgt)
        if valid_egfr_mask.any():
            egfr_loss = self.egfr_loss_fn(egfr_pred[valid_egfr_mask], egfr_tgt[valid_egfr_mask])
        else:
            egfr_loss = 0.0 * egfr_pred.sum()
            
        # 3. Progression Loss
        valid_prog_mask = ~torch.isnan(prog_tgt)
        if valid_prog_mask.any():
            prog_loss = self.progression_loss_fn(prog_logits[valid_prog_mask], prog_tgt[valid_prog_mask])
        else:
            prog_loss = 0.0 * prog_logits.sum()
            
        # Weighted Total
        total_loss = (
            self.stage_weight * stage_loss +
            self.egfr_weight * egfr_loss +
            self.progression_weight * prog_loss
        )
        
        metrics = {
            "total_loss": total_loss.item(),
            "stage_loss": stage_loss.item(),
            "egfr_loss": egfr_loss.item(),
            "progression_loss": prog_loss.item(),
        }
        
        return total_loss, metrics
