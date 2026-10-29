"""
src/models/tft/variable_selection.py
Variable Selection Networks for TFT.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .gating import GatedResidualNetwork

class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN).
    Computes variable importances and produces a weighted sum of inputs.
    Weights are retained in self.selection_weights for Phase 3 explainability.
    """
    def __init__(
        self,
        num_inputs: int,
        hidden_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        
        # Flatten inputs for the variable selection weighting network
        self.flattened_grn = GatedResidualNetwork(
            input_size=num_inputs * hidden_size,
            hidden_size=hidden_size,
            output_size=num_inputs,
            dropout=dropout,
            context_size=context_size,
        )
        
        # Per-variable processing networks
        self.single_variable_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
                context_size=context_size,
            )
            for _ in range(num_inputs)
        ])
        
        self.softmax = nn.Softmax(dim=-1)
        self.selection_weights = None  # Store weights for Phase 3

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input of shape [Batch, SeqLen, NumInputs, HiddenSize] (or [Batch, NumInputs, HiddenSize] for static)
        context : Tensor, optional
            Context vector of shape [Batch, ContextSize]
        
        Returns
        -------
        Tensor of shape [Batch, SeqLen, HiddenSize] (or [Batch, HiddenSize])
        """
        # Flatten all variables: [..., NumInputs * HiddenSize]
        flattened_x = x.flatten(start_dim=-2)
        
        # Compute selection weights
        # sparse_weights: [..., NumInputs]
        sparse_weights = self.flattened_grn(flattened_x, context)
        sparse_weights = self.softmax(sparse_weights)
        
        # Store for explainability
        self.selection_weights = sparse_weights.detach()
        
        # Expand weights to match hidden size: [..., NumInputs, 1]
        sparse_weights_expanded = sparse_weights.unsqueeze(-1)
        
        # Process each variable individually
        processed_inputs = []
        for i in range(self.num_inputs):
            var_x = x[..., i, :]  # [..., HiddenSize]
            var_out = self.single_variable_grns[i](var_x, context)
            processed_inputs.append(var_out)
            
        # Stack processed inputs: [..., NumInputs, HiddenSize]
        processed_inputs = torch.stack(processed_inputs, dim=-2)
        
        # Weighted sum across variables
        # [..., HiddenSize]
        output = torch.sum(processed_inputs * sparse_weights_expanded, dim=-2)
        
        return output
