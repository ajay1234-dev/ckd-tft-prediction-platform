"""
src/explainability/__init__.py
Phase 3 Explainability Package.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from .variable_importance import extract_static_importance, extract_temporal_importance
from .temporal_attention import extract_attention_per_timestep
from .patient_explanation import assemble_patient_explanation

__all__ = [
    "extract_static_importance",
    "extract_temporal_importance",
    "extract_attention_per_timestep",
    "assemble_patient_explanation",
]
