"""
src/explainability/temporal_attention.py
Extract and interpret TFT temporal self-attention weights.

Attention weights indicate which historical time steps the model focused on
when making its prediction. This is a model interpretability signal —
it does NOT establish causal relationships between time steps and outcomes.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)

ATTENTION_DISCLAIMER = (
    "Attention weights indicate model focus on historical time steps. "
    "This is a model interpretability signal, not causal evidence."
)


def extract_attention_per_timestep(
    attention_weights: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    visit_numbers: Optional[List[int]] = None,
    days_from_baseline: Optional[List[float]] = None,
) -> List[Dict]:
    """
    Convert raw TFT attention weights into a structured, human-readable list.

    The TFT attention module returns mean-head attention of shape [B, SeqLen, SeqLen].
    For interpretability, we extract the attention FROM the last valid timestep
    (the prediction position) TO every other time step. Padded positions are
    flagged and should not be interpreted as meaningful.

    Parameters
    ----------
    attention_weights : Tensor [Batch, SeqLen_q, SeqLen_k] or [SeqLen_q, SeqLen_k]
        Averaged attention weights from InterpretableMultiHeadAttention.
    mask : Tensor [Batch, SeqLen] or [SeqLen], optional
        1 = real visit, 0 = padding. If None, all steps treated as valid.
    visit_numbers : list of int, optional
        Visit numbers for each time step (for display). Defaults to 1-indexed.
    days_from_baseline : list of float, optional
        Days from baseline for each time step. Defaults to None.

    Returns
    -------
    list of dict, one per time step, each containing:
        visit_index      : int   (0-based position in sequence)
        visit_number     : int   (1-based if no metadata provided)
        days_from_baseline : float or None
        attention_weight : float (from the final valid step to this step)
        is_padding       : bool  (True = padded, should not be interpreted)
        disclaimer       : str
    """
    attn = attention_weights.detach().cpu()

    # Remove batch dim if present
    if attn.dim() == 3:
        attn = attn[0]  # [SeqLen_q, SeqLen_k]

    seq_len = attn.shape[-1]

    # Determine padding mask
    if mask is not None:
        m = mask.detach().cpu()
        if m.dim() == 2:
            m = m[0]
        is_padding = [(m[i].item() == 0) for i in range(seq_len)]
    else:
        is_padding = [False] * seq_len

    # Find last valid timestep (this is the prediction anchor)
    last_valid_idx = seq_len - 1
    for i in range(seq_len - 1, -1, -1):
        if not is_padding[i]:
            last_valid_idx = i
            break

    # Extract attention FROM last valid step TO all steps
    attn_row = attn[last_valid_idx].numpy()  # [SeqLen_k]

    results = []
    for i in range(seq_len):
        entry = {
            "visit_index": i,
            "visit_number": (visit_numbers[i] if visit_numbers else i + 1),
            "days_from_baseline": (
                float(days_from_baseline[i]) if days_from_baseline else None
            ),
            "attention_weight": float(attn_row[i]),
            "is_padding": bool(is_padding[i]),
            "disclaimer": ATTENTION_DISCLAIMER,
        }
        results.append(entry)

    return results


def summarize_attention_across_patients(
    all_attention: List[List[Dict]],
) -> Dict[str, float]:
    """
    Average attention weights across many patients by visit_index.

    Parameters
    ----------
    all_attention : list of (list of dicts from extract_attention_per_timestep)

    Returns
    -------
    dict: visit_index_str → mean_attention_weight (valid steps only)
    """
    from collections import defaultdict

    acc: Dict[int, List[float]] = defaultdict(list)
    for patient_attn in all_attention:
        for step in patient_attn:
            if not step["is_padding"]:
                acc[step["visit_index"]].append(step["attention_weight"])

    return {
        f"visit_{idx}": float(np.mean(weights))
        for idx, weights in sorted(acc.items())
    }
