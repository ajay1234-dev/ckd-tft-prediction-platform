"""
src/utils/io.py
I/O helper functions for saving/loading JSON, CSV, and joblib artifacts.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def save_json(data: Dict[str, Any], path: str | Path, indent: int = 2) -> None:
    """Serialize dict to JSON file, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    logger.info(f"JSON saved -> {path}")


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load JSON from file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    """Save DataFrame to CSV, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    logger.info(f"CSV saved -> {path} ({len(df)} rows)")


def save_joblib(obj: Any, path: str | Path) -> None:
    """Serialize an object (model, scaler, etc.) to joblib file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info(f"Joblib artifact saved -> {path}")


def load_joblib(path: str | Path) -> Any:
    """Load a joblib-serialized object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Joblib artifact not found: {path}")
    obj = joblib.load(path)
    logger.info(f"Joblib artifact loaded <- {path}")
    return obj
