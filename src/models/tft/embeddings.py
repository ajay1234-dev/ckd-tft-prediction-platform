"""
src/models/tft/embeddings.py
Input projections for TFT static and time-varying features.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import torch
import torch.nn as nn

class ContinuousFeatureEmbedding(nn.Module):
    """Projects continuous features into a hidden embedding space."""
    def __init__(self, num_features: int, hidden_size: int):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        # One linear layer per feature
        self.projections = nn.ModuleList([
            nn.Linear(1, hidden_size) for _ in range(num_features)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input shape [..., num_features]
            
        Returns
        -------
        Tensor of shape [..., num_features, hidden_size]
        """
        # We project each feature i individually:
        # x[..., i:i+1] -> [..., 1] -> Linear -> [..., hidden_size]
        embeddings = []
        for i in range(self.num_features):
            feature_slice = x[..., i : i + 1]
            emb = self.projections[i](feature_slice)
            embeddings.append(emb)
        return torch.stack(embeddings, dim=-2)


class TFTEmbeddingModule(nn.Module):
    """
    Main embedding module handling all inputs.
    Projects static and temporal/time-varying inputs to the uniform hidden size.
    """
    def __init__(self, n_static_features: int, n_tv_features: int, n_temporal_features: int, hidden_size: int):
        super().__init__()
        self.static_embedding = ContinuousFeatureEmbedding(n_static_features, hidden_size)
        
        # We treat time-varying (history) and temporal (meta) as just a unified set of continuous inputs per step
        self.tv_embedding = ContinuousFeatureEmbedding(n_tv_features + n_temporal_features, hidden_size)
        
    def forward(self, static: torch.Tensor, tv: torch.Tensor, temp: torch.Tensor):
        """
        Parameters
        ----------
        static : Tensor [Batch, S]
        tv : Tensor [Batch, SeqLen, F_tv]
        temp : Tensor [Batch, SeqLen, F_t]
        
        Returns
        -------
        static_emb : Tensor [Batch, S, HiddenSize]
        temporal_emb : Tensor [Batch, SeqLen, F_tv + F_t, HiddenSize]
        """
        static_emb = self.static_embedding(static)
        
        # Combine historical features and temporal meta-features
        combined_temporal = torch.cat([tv, temp], dim=-1)
        temporal_emb = self.tv_embedding(combined_temporal)
        
        return static_emb, temporal_emb
