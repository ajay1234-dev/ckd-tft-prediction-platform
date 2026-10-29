"""
src/api/dependencies.py
FastAPI shared dependencies via dependency injection.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from src.inference.model_loader import ModelLoader
from src.inference.predictor import CKDPredictor, PatientStore
from src.inference.preprocessing import InferencePreprocessor
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Singletons — initialized once at application startup
_predictor: Optional[CKDPredictor] = None
_patient_store: Optional[PatientStore] = None


def get_predictor() -> CKDPredictor:
    """FastAPI dependency — returns the initialized CKDPredictor singleton."""
    if _predictor is None:
        raise RuntimeError("CKDPredictor not initialized. Call initialize_dependencies() at startup.")
    return _predictor


def get_patient_store() -> PatientStore:
    """FastAPI dependency — returns the initialized PatientStore singleton."""
    if _patient_store is None:
        raise RuntimeError("PatientStore not initialized. Call initialize_dependencies() at startup.")
    return _patient_store


def initialize_dependencies(
    model_dir: Optional[str] = None,
    sequences_dir: Optional[str] = None,
    split: str = "test",
    device: str = "cpu",
) -> None:
    """
    Initialize all shared singletons. Called once at application startup.
    Fails fast if any required artifact is missing.
    """
    global _predictor, _patient_store

    model_dir = model_dir or os.environ.get("CKD_MODEL_DIR", "models/tft")
    sequences_dir = sequences_dir or os.environ.get("CKD_SEQUENCES_DIR", "data/processed/sequences")

    logger.info("Initializing Phase 3 dependencies...")

    loader = ModelLoader(model_dir)
    preprocessor = InferencePreprocessor().load()
    store = PatientStore(sequences_dir, split=split).load()

    predictor = CKDPredictor(
        model_loader=loader,
        preprocessor=preprocessor,
        patient_store=store,
        device=device,
    ).initialize()

    _patient_store = store
    _predictor = predictor
    logger.info("All Phase 3 dependencies initialized successfully.")
