"""
src/models/tft/attention.py
Interpretable Multi-Head Self-Attention for TFT.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention layer.
    As proposed in the TFT paper, values are shared across heads and the attention
    weights are averaged, enabling interpretability of the temporal attention.
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
            
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        # Q, K have distinct weights per head.
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        
        # Value is shared across all heads (mapped to hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        q : Tensor [Batch, SeqLen_q, HiddenSize]
        k : Tensor [Batch, SeqLen_k, HiddenSize]
        v : Tensor [Batch, SeqLen_k, HiddenSize]
        mask : Tensor [Batch, SeqLen_k] or [Batch, SeqLen_q, SeqLen_k] (1 for real, 0 for pad)
        
        Returns
        -------
        output: [Batch, SeqLen_q, HiddenSize]
        attention_weights: [Batch, SeqLen_q, SeqLen_k] (averaged across heads)
        """
        batch_size = q.size(0)
        seq_len_q = q.size(1)
        seq_len_k = k.size(1)

        # [Batch, SeqLen, NumHeads, HeadSize] -> [Batch, NumHeads, SeqLen, HeadSize]
        queries = self.q_proj(q).view(batch_size, seq_len_q, self.num_heads, self.head_size).transpose(1, 2)
        keys = self.k_proj(k).view(batch_size, seq_len_k, self.num_heads, self.head_size).transpose(1, 2)
        
        # V is shared across heads, so we don't project into heads yet, 
        # but for the math to work we can just expand it across heads.
        # Actually, original TFT averages the attention weights across heads before multiplying by V!
        
        # Attention scores: [Batch, NumHeads, SeqLen_q, SeqLen_k]
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_size)
        
        if mask is not None:
            # Format mask for addition
            if mask.dim() == 2:
                # [Batch, SeqLen_k] -> [Batch, 1, 1, SeqLen_k]
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # [Batch, SeqLen_q, SeqLen_k] -> [Batch, 1, SeqLen_q, SeqLen_k]
                mask = mask.unsqueeze(1)
                
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Attention weights: [Batch, NumHeads, SeqLen_q, SeqLen_k]
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Mean across heads for interpretable attention: [Batch, SeqLen_q, SeqLen_k]
        mean_attn_weights = attn_weights.mean(dim=1)
        
        # Multiply averaged weights by values: [Batch, SeqLen_q, SeqLen_k] x [Batch, SeqLen_k, HiddenSize]
        values = self.v_proj(v) # [Batch, SeqLen_k, HiddenSize]
        
        # Result: [Batch, SeqLen_q, HiddenSize]
        out = torch.matmul(mean_attn_weights, values)
        out = self.out_proj(out)
        
        return out, mean_attn_weights
