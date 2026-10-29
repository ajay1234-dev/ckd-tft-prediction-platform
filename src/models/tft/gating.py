"""
src/models/tft/gating.py
Gated Residual Networks and GLU components for TFT.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import torch
import torch.nn as nn
from typing import Optional

class GLU(nn.Module):
    """
    Gated Linear Unit.
    Computes: GLU(x) = x_1 ⊙ sigmoid(x_2)
    where x is split in half along the last dimension.
    """
    def __init__(self, input_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, input_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN).
    Applies non-linear processing with a residual connection and gating mechanism.
    Supports optional context vector conditioning.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size

        # Skip connection projection if dimensions differ
        if self.input_size != self.output_size:
            self.skip_layer = nn.Linear(self.input_size, self.output_size)
        else:
            self.skip_layer = nn.Identity()

        # Context conditioning
        if self.context_size is not None:
            self.context_layer = nn.Linear(self.context_size, hidden_size, bias=False)

        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gate = GLU(hidden_size)
        
        # Projection from hidden to output
        if hidden_size != self.output_size:
            self.out_proj = nn.Linear(hidden_size, self.output_size)
        else:
            self.out_proj = nn.Identity()

        self.layer_norm = nn.LayerNorm(self.output_size)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input of shape [..., input_size]
        context : Tensor, optional
            Context vector of shape [..., context_size].
            Can broadcast over sequence dimension if context is static.
        """
        residual = self.skip_layer(x)

        hidden = self.fc1(x)
        if context is not None and self.context_size is not None:
            # Context may be [Batch, ContextSize], need to broadcast to [Batch, SeqLen, ContextSize] if x is 3D
            context_proj = self.context_layer(context)
            if x.dim() == 3 and context_proj.dim() == 2:
                context_proj = context_proj.unsqueeze(1)
            hidden = hidden + context_proj

        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        
        gated = self.gate(hidden)
        gated = self.out_proj(gated)

        return self.layer_norm(residual + gated)
