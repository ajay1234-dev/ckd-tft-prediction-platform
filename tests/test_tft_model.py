"""
tests/test_tft_model.py
Unit tests for the TFT model architecture.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import pytest
import torch
from src.models.tft.model import TemporalFusionTransformer
from src.models.tft.attention import InterpretableMultiHeadAttention
from src.models.tft.variable_selection import VariableSelectionNetwork
from src.models.tft.losses import JointMultiTaskLoss

@pytest.fixture
def tft_model():
    return TemporalFusionTransformer(
        n_static_features=10,
        n_tv_features=20,
        n_temporal_features=3,
        hidden_size=16,
        num_attention_heads=2,
        dropout=0.1,
        n_classes=6,
    )

@pytest.fixture
def dummy_inputs():
    batch_size = 4
    seq_len = 8
    static = torch.randn(batch_size, 10)
    tv = torch.randn(batch_size, seq_len, 20)
    temp = torch.randn(batch_size, seq_len, 3)
    mask = torch.ones(batch_size, seq_len)
    return static, tv, temp, mask

def test_model_initialization(tft_model):
    """Test that all submodules are properly instantiated."""
    assert isinstance(tft_model.static_vsn, VariableSelectionNetwork)
    assert isinstance(tft_model.attention, InterpretableMultiHeadAttention)

def test_forward_pass_shapes(tft_model, dummy_inputs):
    """Test forward pass output shapes."""
    static, tv, temp, mask = dummy_inputs
    preds = tft_model(static, tv, temp, mask)
    
    assert 'stage_logits' in preds
    assert 'egfr_pred' in preds
    assert 'progression_logits' in preds
    assert 'attention_weights' in preds
    
    # Check shapes
    batch_size = static.size(0)
    assert preds['stage_logits'].shape == (batch_size, 3, 6)
    assert preds['egfr_pred'].shape == (batch_size, 3)
    assert preds['progression_logits'].shape == (batch_size, 3)
    
    assert preds['attention_weights'].shape == (batch_size, 8, 8)
    assert preds['static_vsn_weights'].shape == (batch_size, 10)
    assert preds['temporal_vsn_weights'].shape == (batch_size, 8, 23)

def test_loss_computation(dummy_inputs):
    """Test JointMultiTaskLoss."""
    static, tv, temp, mask = dummy_inputs
    preds = {
        'stage_logits': torch.randn(4, 3, 6),
        'egfr_pred': torch.randn(4, 3),
        'progression_logits': torch.randn(4, 3),
    }
    
    targets = {
        'stage': torch.randint(0, 6, (4, 3)),
        'egfr': torch.randn(4, 3),
        'progression': torch.randint(0, 2, (4, 3)).float(),
    }
    
    criterion = JointMultiTaskLoss()
    loss, metrics = criterion(preds, targets)
    
    assert loss.dim() == 0  # scalar
    assert 'total_loss' in metrics
    assert 'stage_loss' in metrics
    assert 'egfr_loss' in metrics
    assert 'progression_loss' in metrics
    assert not torch.isnan(loss)

def test_masking(tft_model, dummy_inputs):
    """Test that padded steps do not affect the output."""
    static, tv, temp, mask = dummy_inputs
    
    # Sequence 1: full length
    # Sequence 2: padded length (last 3 steps are padding)
    mask[1, 5:] = 0 
    
    preds1 = tft_model(static, tv, temp, torch.ones_like(mask))
    preds2 = tft_model(static, tv, temp, mask)
    
    # We only care that the attention weights for padded steps are exactly zero
    attn_weights = preds2['attention_weights']
    
    # For batch 1 (padded), the attention to padded keys (indices 5,6,7) should be zero
    padded_keys_attn = attn_weights[1, :, 5:]
    assert torch.allclose(padded_keys_attn, torch.zeros_like(padded_keys_attn))
