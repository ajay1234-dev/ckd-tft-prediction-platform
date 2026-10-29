"""
src/models/tft/model.py
Main Temporal Fusion Transformer (TFT) Architecture.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import torch
import torch.nn as nn
from typing import Optional, Dict

from .embeddings import TFTEmbeddingModule
from .variable_selection import VariableSelectionNetwork
from .gating import GatedResidualNetwork
from .attention import InterpretableMultiHeadAttention
from .heads import MultiTaskHeads

class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for multi-task, multi-horizon prediction.
    Architecture based on Lim et al. (2021).
    """
    def __init__(
        self,
        n_static_features: int,
        n_tv_features: int,
        n_temporal_features: int,
        hidden_size: int = 64,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        n_classes: int = 6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 1. Input Projections
        self.embeddings = TFTEmbeddingModule(
            n_static_features, n_tv_features, n_temporal_features, hidden_size
        )
        
        # 2. Static Covariate Encoders (using GRNs)
        # Static variable selection
        self.static_vsn = VariableSelectionNetwork(
            num_inputs=n_static_features,
            hidden_size=hidden_size,
            dropout=dropout,
            context_size=None,
        )
        
        # Context vectors derived from static representation
        self.static_context_grn_cs = GatedResidualNetwork(
            input_size=hidden_size, hidden_size=hidden_size, output_size=hidden_size, dropout=dropout
        )
        self.static_context_grn_ce = GatedResidualNetwork(
            input_size=hidden_size, hidden_size=hidden_size, output_size=hidden_size, dropout=dropout
        )
        self.static_context_grn_ch = GatedResidualNetwork(
            input_size=hidden_size, hidden_size=hidden_size, output_size=hidden_size, dropout=dropout
        )
        self.static_context_grn_cc = GatedResidualNetwork(
            input_size=hidden_size, hidden_size=hidden_size, output_size=hidden_size, dropout=dropout
        )
        
        # 3. Temporal Variable Selection
        self.temporal_vsn = VariableSelectionNetwork(
            num_inputs=n_tv_features + n_temporal_features,
            hidden_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size, # Context is cs (from static)
        )
        
        # 4. Locality Enhancement (LSTM)
        # We use LSTM to process sequence locally before attention
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0.0 else 0.0
        )
        
        # Gate & Add after LSTM (Post-LSTM GRN/Gate)
        self.post_lstm_gate = nn.Linear(hidden_size, hidden_size) # simplified GLU for skip connection
        self.post_lstm_norm = nn.LayerNorm(hidden_size)
        
        # 5. Static Enrichment
        self.static_enrichment_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size, # Context is ce
        )
        
        # 6. Temporal Self-Attention
        self.attention = InterpretableMultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        self.post_attn_gate = nn.Linear(hidden_size, hidden_size)
        self.post_attn_norm = nn.LayerNorm(hidden_size)
        
        # 7. Position-wise Feed-Forward
        self.pos_ff_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
            context_size=None
        )
        self.post_ff_gate = nn.Linear(hidden_size, hidden_size)
        self.post_ff_norm = nn.LayerNorm(hidden_size)
        
        # 8. Multi-Task Multi-Horizon Heads
        self.heads = MultiTaskHeads(hidden_size=hidden_size, num_classes=n_classes)

    def forward(
        self,
        static_features: torch.Tensor,
        historical_features: torch.Tensor,
        time_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        static_features: [Batch, F_s]
        historical_features: [Batch, SeqLen, F_tv]
        time_features: [Batch, SeqLen, F_t]
        mask: [Batch, SeqLen] (1 for real, 0 for pad)
        
        Returns
        -------
        Dictionary with task outputs
        """
        # 1. Input Projections
        # static_emb: [Batch, F_s, Hidden]
        # temp_emb: [Batch, SeqLen, F_tv+F_t, Hidden]
        static_emb, temp_emb = self.embeddings(static_features, historical_features, time_features)
        
        # 2. Static Covariate Encoders
        # static_repr: [Batch, Hidden]
        static_repr = self.static_vsn(static_emb)
        
        # Context vectors: [Batch, Hidden]
        cs = self.static_context_grn_cs(static_repr)
        ce = self.static_context_grn_ce(static_repr)
        ch = self.static_context_grn_ch(static_repr)
        cc = self.static_context_grn_cc(static_repr)
        
        # 3. Temporal Variable Selection
        # temp_repr: [Batch, SeqLen, Hidden]
        temp_repr = self.temporal_vsn(temp_emb, context=cs)
        
        # 4. Locality Enhancement (LSTM)
        # Initialize LSTM hidden/cell states using static contexts
        batch_size = static_features.size(0)
        h0 = ch.unsqueeze(0).expand(2, batch_size, -1).contiguous()
        c0 = cc.unsqueeze(0).expand(2, batch_size, -1).contiguous()
        
        lstm_out, _ = self.lstm(temp_repr, (h0, c0))
        
        # Residual connection over LSTM
        # Using a simple linear gate instead of full GLU here for standard TFT
        gated_lstm = torch.sigmoid(self.post_lstm_gate(lstm_out)) * lstm_out
        lstm_out = self.post_lstm_norm(temp_repr + gated_lstm)
        
        # 5. Static Enrichment
        # Enrich temporal repr with static context ce
        enriched = self.static_enrichment_grn(lstm_out, context=ce)
        
        # 6. Temporal Self-Attention
        # Self attention (queries, keys, values are all 'enriched')
        attn_out, attn_weights = self.attention(q=enriched, k=enriched, v=enriched, mask=mask)
        
        # Gate and add (residual over attention)
        gated_attn = torch.sigmoid(self.post_attn_gate(attn_out)) * attn_out
        attn_out = self.post_attn_norm(enriched + gated_attn)
        
        # 7. Position-wise Feed-Forward
        ff_out = self.pos_ff_grn(attn_out)
        gated_ff = torch.sigmoid(self.post_ff_gate(ff_out)) * ff_out
        final_temporal_repr = self.post_ff_norm(attn_out + gated_ff)
        
        # Extract the representation of the LAST timestep (most recent visit)
        # Assuming padding is at the end, if mask is provided we should pick the last valid step.
        # But Phase 1 sequence_builder says: "sort chronologically... take head(max_seq_len)... 
        # Targets — use the LAST visit's targets". And mask=1 means real, mask=0 means pad.
        # For our data (no padding, all length 8), it's always the last index.
        # For safety, let's extract the last valid timestep for each sequence.
        if mask is not None:
            device = final_temporal_repr.device
            lengths = mask.sum(dim=1) # [Batch]
            last_idx = (lengths - 1).clamp(min=0).long()
            batch_indices = torch.arange(batch_size, device=device)
            shared_repr = final_temporal_repr[batch_indices, last_idx, :] # [Batch, Hidden]
        else:
            shared_repr = final_temporal_repr[:, -1, :] # [Batch, Hidden]
            
        # 8. Multi-Task Multi-Horizon Heads
        outputs = self.heads(shared_repr)
        
        # Attach attention weights (and VSN weights) for Phase 3 interpretability
        outputs["attention_weights"] = attn_weights
        outputs["static_vsn_weights"] = self.static_vsn.selection_weights
        outputs["temporal_vsn_weights"] = self.temporal_vsn.selection_weights
        
        return outputs
