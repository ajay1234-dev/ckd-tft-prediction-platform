"""
src/inference/model_loader.py
Singleton TFT model loader for Phase 3 inference.

Loads the Phase 2 checkpoint + config exactly ONCE at application startup.
Subsequent calls return the cached instance immediately.
Fails fast with a clear error if the checkpoint is missing.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch

from src.models.tft.model import TemporalFusionTransformer
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Default model directory — overridable via env var CKD_MODEL_DIR
_DEFAULT_MODEL_DIR = Path("models/tft")


class ModelLoader:
    """
    Thread-safe singleton that loads and caches the TFT model and its config.

    Usage
    -----
    loader = ModelLoader()
    model, config = loader.load()
    """

    _instance_lock = threading.Lock()
    _model: Optional[TemporalFusionTransformer] = None
    _config: Optional[Dict[str, Any]] = None
    _checkpoint_path: Optional[Path] = None

    def __init__(self, model_dir: Optional[str | Path] = None):
        if model_dir is None:
            model_dir = os.environ.get("CKD_MODEL_DIR", str(_DEFAULT_MODEL_DIR))
        self.model_dir = Path(model_dir)

    def load(self, device: Optional[str] = None) -> Tuple[TemporalFusionTransformer, Dict[str, Any]]:
        """
        Load and cache the TFT model. Thread-safe.

        Parameters
        ----------
        device : str, optional
            'cpu', 'cuda', or 'auto'. Defaults to 'auto' (uses GPU if available).

        Returns
        -------
        (TemporalFusionTransformer, config_dict)

        Raises
        ------
        FileNotFoundError
            If checkpoint or config files are missing — fails immediately with
            a clear, actionable error message.
        RuntimeError
            If the checkpoint cannot be loaded (corrupted, wrong format, etc.).
        """
        with ModelLoader._instance_lock:
            if ModelLoader._model is not None:
                return ModelLoader._model, ModelLoader._config

            device = self._resolve_device(device)
            logger.info(f"Loading TFT model from {self.model_dir} on {device}...")

            config_path = self.model_dir / "config.json"
            checkpoint_path = self.model_dir / "best_model.pt"

            # Fail fast — do not silently start a broken service
            if not config_path.exists():
                raise FileNotFoundError(
                    f"TFT config not found at {config_path}. "
                    "Ensure Phase 2 training has completed and 'models/tft/config.json' exists. "
                    "You can override the model directory with the CKD_MODEL_DIR env var."
                )
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"TFT checkpoint not found at {checkpoint_path}. "
                    "Ensure Phase 2 training has completed and 'models/tft/best_model.pt' exists. "
                    "You can override the model directory with the CKD_MODEL_DIR env var."
                )

            with open(config_path, "r") as f:
                config = json.load(f)

            mc = config["model"]
            model = TemporalFusionTransformer(
                n_static_features=mc["n_static_features"],
                n_tv_features=mc["n_tv_features"],
                n_temporal_features=mc["n_temporal_features"],
                hidden_size=mc["hidden_size"],
                num_attention_heads=mc["num_attention_heads"],
                dropout=mc["dropout"],
                n_classes=mc["n_classes"],
            )

            try:
                ckpt = torch.load(checkpoint_path, map_location=device)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"])
                    epoch_str = ckpt.get("epoch", "?")
                    score_str = f", val_score={ckpt['val_score']:.4f}" if "val_score" in ckpt else ""
                    logger.info(f"Loaded checkpoint (epoch={epoch_str}{score_str})")
                else:
                    model.load_state_dict(ckpt)
                    logger.info("Loaded checkpoint (raw state dict)")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load checkpoint from {checkpoint_path}: {exc}"
                ) from exc

            model = model.to(device)
            model.eval()

            ModelLoader._model = model
            ModelLoader._config = config
            ModelLoader._checkpoint_path = checkpoint_path

            n_params = sum(p.numel() for p in model.parameters())
            logger.info(
                f"TFT model loaded successfully. Parameters: {n_params:,}. "
                f"hidden_size={mc['hidden_size']}, heads={mc['num_attention_heads']}, "
                f"device={device}"
            )
            return model, config

    @property
    def is_loaded(self) -> bool:
        return ModelLoader._model is not None

    @property
    def checkpoint_name(self) -> str:
        if ModelLoader._checkpoint_path:
            return ModelLoader._checkpoint_path.name
        return "not_loaded"

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        if device is None or device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return device

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton cache (useful for testing)."""
        with cls._instance_lock:
            cls._model = None
            cls._config = None
            cls._checkpoint_path = None


# Module-level singleton — used by the API via dependency injection
_loader: Optional[ModelLoader] = None


def get_model_loader(model_dir: Optional[str | Path] = None) -> ModelLoader:
    """Get or create the module-level ModelLoader singleton."""
    global _loader
    if _loader is None:
        _loader = ModelLoader(model_dir)
    return _loader
