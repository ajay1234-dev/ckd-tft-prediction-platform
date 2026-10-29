"""
tests/test_tft_training.py
Unit tests for the TFT training pipeline.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import pytest
import torch
import numpy as np
from pathlib import Path

from src.models.tft.model import TemporalFusionTransformer
from src.models.tft.losses import JointMultiTaskLoss
from src.training.dataset import CKDSequenceDataset

# Test on a small mock subset if possible, but testing dataset loading might require the real files
# Let's mock a dataloader yield

@pytest.fixture
def mock_batch():
    batch_size = 2
    seq_len = 8
    
    static = torch.randn(batch_size, 10)
    tv = torch.randn(batch_size, seq_len, 29)
    temp = torch.randn(batch_size, seq_len, 3)
    mask = torch.ones(batch_size, seq_len)
    
    targets = {
        'stage': torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long),
        'egfr': torch.randn(batch_size, 3),
        'progression': torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.float32)
    }
    
    return static, tv, temp, mask, targets

def test_gradient_flow(mock_batch):
    """Test that gradients flow into all heads and the shared encoder."""
    static, tv, temp, mask, targets = mock_batch
    
    model = TemporalFusionTransformer(
        n_static_features=10,
        n_tv_features=29,
        n_temporal_features=3,
        hidden_size=16,
        num_attention_heads=2,
        dropout=0.1,
        n_classes=6,
    )
    
    criterion = JointMultiTaskLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    optimizer.zero_grad()
    preds = model(static, tv, temp, mask)
    loss, _ = criterion(preds, targets)
    loss.backward()
    
    # Check that heads received gradients
    assert model.heads.stage_head.proj.weight.grad is not None
    assert model.heads.egfr_head.proj.weight.grad is not None
    assert model.heads.progression_head.proj.weight.grad is not None
    
    # Check that shared encoder received gradients from all losses
    assert model.lstm.weight_ih_l0.grad is not None
    assert model.static_vsn.single_variable_grns[0].fc1.weight.grad is not None
    assert model.attention.q_proj.weight.grad is not None

def test_cpu_execution(mock_batch):
    """Test that the model runs correctly on CPU."""
    static, tv, temp, mask, targets = mock_batch
    
    model = TemporalFusionTransformer(
        n_static_features=10,
        n_tv_features=29,
        n_temporal_features=3,
        hidden_size=16,
    )
    
    # Default device is CPU for tensors
    assert static.device.type == 'cpu'
    
    preds = model(static, tv, temp, mask)
    assert preds['stage_logits'].device.type == 'cpu'

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_execution(mock_batch):
    """Test that the model runs correctly on GPU."""
    static, tv, temp, mask, targets = mock_batch
    device = torch.device('cuda')
    
    model = TemporalFusionTransformer(
        n_static_features=10,
        n_tv_features=29,
        n_temporal_features=3,
        hidden_size=16,
    ).to(device)
    
    static = static.to(device)
    tv = tv.to(device)
    temp = temp.to(device)
    mask = mask.to(device)
    
    preds = model(static, tv, temp, mask)
    assert preds['stage_logits'].device.type == 'cuda'
