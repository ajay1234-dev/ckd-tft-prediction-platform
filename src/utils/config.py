"""
src/utils/config.py
Pydantic-based configuration loader.
All scripts must import configuration from this module — no script re-parses YAML independently.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class DataConfig(BaseModel):
    raw_dir: str
    processed_dir: str
    sequences_dir: str
    train_file: str
    val_file: str
    test_file: str
    static_file: str
    feature_groups_file: str


class SequenceConfig(BaseModel):
    max_seq_len: Optional[int] = None
    pad_value: float = 0.0
    mask_value: int = -1


class PreprocessingConfig(BaseModel):
    numerical_imputation_strategy: str = "median"
    scaling: str = "standard"
    unknown_category_token: str = "__unknown__"
    chronological_sort_key: str = "visit_date"


class TargetsConfig(BaseModel):
    classification_stage: List[str]
    regression_egfr: List[str]
    binary_progression: List[str]
    other: List[str]


class BaselineConfig(BaseModel):
    n_estimators: Optional[int] = None
    max_depth: Optional[int] = None
    min_samples_split: Optional[int] = None
    n_jobs: Optional[int] = None
    max_iter: Optional[int] = None
    solver: Optional[str] = None
    multi_class: Optional[str] = None
    C: Optional[float] = None
    alpha: Optional[float] = None
    learning_rate: Optional[float] = None

    class Config:
        extra = "allow"


class BaselinesConfig(BaseModel):
    logistic_regression: BaselineConfig
    random_forest: BaselineConfig
    ridge_regression: BaselineConfig
    xgboost: BaselineConfig
    lightgbm: BaselineConfig


class OutputConfig(BaseModel):
    models_dir: str
    baselines_dir: str
    preprocessing_dir: str
    reports_dir: str
    figures_dir: str
    metrics_dir: str


# ---------------------------------------------------------------------------
# Root config model
# ---------------------------------------------------------------------------

class CKDConfig(BaseModel):
    project_name: str
    random_seed: int
    phase: int
    data: DataConfig
    sequence: SequenceConfig
    preprocessing: PreprocessingConfig
    targets: TargetsConfig
    all_targets: List[str]
    ckd_stage_map: Dict[str, int]
    baselines: BaselinesConfig
    output: OutputConfig

    @validator("random_seed")
    def seed_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("random_seed must be non-negative")
        return v

    def all_target_columns(self) -> List[str]:
        """Return the flat union of all target column names."""
        return self.all_targets

    def resolve_path(self, rel_path: str, root: Optional[Path] = None) -> Path:
        """Resolve a relative path against project root."""
        if root is None:
            root = _find_project_root()
        return (root / rel_path).resolve()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_CONFIG_CACHE: Optional[CKDConfig] = None
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "config.yaml"


def _find_project_root() -> Path:
    """Walk upward from this file to find the project root (contains configs/)."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "configs").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not locate project root (no 'configs/' directory found).")


def load_config(config_path: Optional[str | Path] = None, force_reload: bool = False) -> CKDConfig:
    """
    Load and cache the central config.yaml.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to config.yaml. Defaults to configs/config.yaml relative to project root.
    force_reload : bool
        If True, bypass cache and reload from disk.

    Returns
    -------
    CKDConfig
        Validated configuration object.
    """
    global _CONFIG_CACHE

    if _CONFIG_CACHE is not None and not force_reload:
        return _CONFIG_CACHE

    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Expected at configs/config.yaml relative to project root."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    _CONFIG_CACHE = CKDConfig(**raw)
    return _CONFIG_CACHE


def get_project_root() -> Path:
    """Return the project root directory."""
    return _find_project_root()
