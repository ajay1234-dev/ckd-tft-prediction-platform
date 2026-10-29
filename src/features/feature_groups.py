"""
src/features/feature_groups.py
Load feature group definitions from feature_groups.yaml into typed Python structures.
Provides the leakage guard used by every module that assembles model input matrices.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class FeatureGroups:
    """Typed feature group definitions loaded from feature_groups.yaml."""

    identifiers: List[str]
    temporal_features: List[str]
    static_features: List[str]
    time_varying_features: List[str]
    raw_longitudinal_columns: List[str]

    # Encoding metadata
    categorical_encoding: Dict[str, List[str]]    # {"onehot": [...], "ordinal": [...]}
    binary_columns: List[str]
    numerical_columns: List[str]

    # Targets (grouped)
    targets_classification_stage: List[str]
    targets_regression_egfr: List[str]
    targets_binary_progression: List[str]
    targets_other: List[str]

    @property
    def all_targets(self) -> List[str]:
        """Flat union of all target column names."""
        return (
            self.targets_classification_stage
            + self.targets_regression_egfr
            + self.targets_binary_progression
            + self.targets_other
        )

    @property
    def all_input_features(self) -> List[str]:
        """All columns that may legitimately appear as model inputs (no targets, no identifiers)."""
        return (
            self.static_features
            + self.time_varying_features
            + self.temporal_features
        )

    @property
    def target_set(self) -> Set[str]:
        return set(self.all_targets)

    @property
    def onehot_columns(self) -> List[str]:
        return self.categorical_encoding.get("onehot", [])

    @property
    def ordinal_columns(self) -> List[str]:
        return self.categorical_encoding.get("ordinal", [])


# ---------------------------------------------------------------------------
# Leakage guard — must be called before every fit/predict
# ---------------------------------------------------------------------------

def assert_no_target_leakage(
    feature_list: List[str],
    feature_groups: FeatureGroups,
    context: str = "model input",
) -> None:
    """
    Assert that no target column appears in a proposed feature list.
    Raises a hard error — this must never silently pass when targets are present.

    Parameters
    ----------
    feature_list : list of str
        Columns that are about to be used as model features.
    feature_groups : FeatureGroups
        Feature group config providing the authoritative target set.
    context : str
        Description of where this check is being called (for error messages).

    Raises
    ------
    ValueError
        If any target column is found in feature_list.
        The error message lists every offending column.
    """
    targets = feature_groups.target_set
    leaked = sorted(set(feature_list) & targets)
    if leaked:
        raise ValueError(
            f"TARGET LEAKAGE DETECTED in '{context}'!\n"
            f"The following target column(s) appear in the feature list:\n"
            + "\n".join(f"  ✗ {col}" for col in leaked)
            + "\n\nRemove these columns from model inputs before training."
        )
    logger.debug(f"Leakage guard passed for '{context}' ({len(feature_list)} features checked)")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_FG_CACHE: Optional[FeatureGroups] = None


def load_feature_groups(
    yaml_path: Optional[str | Path] = None,
    force_reload: bool = False,
) -> FeatureGroups:
    """
    Load feature group definitions from YAML.

    Parameters
    ----------
    yaml_path : str or Path, optional
        Path to feature_groups.yaml. Defaults to configs/feature_groups.yaml
        relative to project root.
    force_reload : bool
        Bypass cache and reload from disk.

    Returns
    -------
    FeatureGroups
    """
    global _FG_CACHE

    if _FG_CACHE is not None and not force_reload:
        return _FG_CACHE

    if yaml_path is None:
        # Find project root: walk up from this file
        current = Path(__file__).resolve().parent
        for _ in range(10):
            candidate = current / "configs" / "feature_groups.yaml"
            if candidate.exists():
                yaml_path = candidate
                break
            current = current.parent
        else:
            raise FileNotFoundError(
                "Could not find configs/feature_groups.yaml. "
                "Run from project root or pass yaml_path explicitly."
            )

    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"feature_groups.yaml not found at: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    fg = FeatureGroups(
        identifiers=raw.get("identifiers", []),
        temporal_features=raw.get("temporal_features", []),
        static_features=raw.get("static_features", []),
        time_varying_features=raw.get("time_varying_features", []),
        raw_longitudinal_columns=raw.get("raw_longitudinal_columns", []),
        categorical_encoding=raw.get("categorical_encoding", {}),
        binary_columns=raw.get("binary_columns", []),
        numerical_columns=raw.get("numerical_columns", []),
        targets_classification_stage=raw["targets"]["classification_stage"],
        targets_regression_egfr=raw["targets"]["regression_egfr"],
        targets_binary_progression=raw["targets"]["binary_progression"],
        targets_other=raw["targets"]["other"],
    )

    logger.info(
        f"Feature groups loaded: "
        f"{len(fg.static_features)} static, "
        f"{len(fg.time_varying_features)} time-varying, "
        f"{len(fg.temporal_features)} temporal, "
        f"{len(fg.all_targets)} targets"
    )

    _FG_CACHE = fg
    return fg
