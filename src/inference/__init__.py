"""
src/inference/__init__.py
Phase 3 Inference Engine Package.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from .model_loader import ModelLoader, get_model_loader
from .predictor import CKDPredictor

__all__ = ["ModelLoader", "get_model_loader", "CKDPredictor"]
