"""
src/models/tft/heads.py
Multi-task, multi-horizon output heads for the TFT model.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import torch
import torch.nn as nn

class StageHead(nn.Module):
    """
    Predicts CKD Stage (6 classes) for 3 horizons.
    Input: Shared representation [Batch, HiddenSize]
    Output: Logits [Batch, 3, 6]
    """
    def __init__(self, hidden_size: int, num_classes: int = 6):
        super().__init__()
        self.num_classes = num_classes
        # Linear layer mapping hidden state to 3 horizons x 6 classes
        self.proj = nn.Linear(hidden_size, 3 * num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, HiddenSize]
        out = self.proj(x)
        # Reshape to [Batch, 3_horizons, num_classes]
        return out.view(x.size(0), 3, self.num_classes)


class EGFRHead(nn.Module):
    """
    Predicts continuous eGFR for 3 horizons.
    Input: Shared representation [Batch, HiddenSize]
    Output: Values [Batch, 3]
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns [Batch, 3]
        return self.proj(x)


class ProgressionHead(nn.Module):
    """
    Predicts binary progression risk (logits) for 3 horizons.
    Input: Shared representation [Batch, HiddenSize]
    Output: Logits [Batch, 3]
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns [Batch, 3] (these are logits, Sigmoid applied at inference/loss)
        return self.proj(x)


class MultiTaskHeads(nn.Module):
    """
    Wraps all three task heads.
    Takes the final shared temporal representation and produces all outputs.
    """
    def __init__(self, hidden_size: int, num_classes: int = 6):
        super().__init__()
        self.stage_head = StageHead(hidden_size, num_classes)
        self.egfr_head = EGFRHead(hidden_size)
        self.progression_head = ProgressionHead(hidden_size)
        
    def forward(self, shared_repr: torch.Tensor):
        """
        Parameters
        ----------
        shared_repr : Tensor [Batch, HiddenSize]
            The final output representation from the TFT encoder.
            
        Returns
        -------
        Dictionary containing:
            'stage_logits': [Batch, 3, 6]
            'egfr_pred': [Batch, 3]
            'progression_logits': [Batch, 3]
        """
        return {
            "stage_logits": self.stage_head(shared_repr),
            "egfr_pred": self.egfr_head(shared_repr),
            "progression_logits": self.progression_head(shared_repr),
        }
