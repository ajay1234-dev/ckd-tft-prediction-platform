"""
src/baselines/classifiers.py
CKD stage multiclass classification baselines (G1/G2/G3a/G3b/G4/G5).
Predicts stage_3m, stage_6m, stage_12m separately.

Models:
  - Logistic Regression (multinomial)
  - Random Forest Classifier
  - XGBoost Classifier (if available)
  - LightGBM Classifier (if available)

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.baselines.base import BaselineModel
from src.features.feature_groups import FeatureGroups, assert_no_target_leakage
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Optional XGBoost / LightGBM
try:
    from xgboost import XGBClassifier as _XGBClf
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available — XGBoostClassifier will raise if instantiated.")

try:
    from lightgbm import LGBMClassifier as _LGBMClf
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available — LightGBMClassifier will raise if instantiated.")


# ---------------------------------------------------------------------------
# Logistic Regression (multiclass)
# ---------------------------------------------------------------------------

class LogisticRegressionClassifier(BaselineModel):
    def __init__(self, random_seed: int = 42, max_iter: int = 1000, C: float = 1.0):
        super().__init__("LogisticRegression_classifier", random_seed)
        self.model = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=max_iter,
            C=C,
            random_state=random_seed,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, feature_groups: Optional[FeatureGroups] = None) -> "LogisticRegressionClassifier":
        if feature_groups:
            assert_no_target_leakage(X.columns.tolist(), feature_groups, context=self.model_name)
        Xv = self._prepare_X(X)
        yv = y.values
        logger.info(f"Fitting {self.model_name} on {Xv.shape[0]:,} samples…")
        self.model.fit(Xv, yv)
        self._metadata = {"params": self.model.get_params(), "classes": list(self.model.classes_)}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._prepare_X(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._prepare_X(X))


# ---------------------------------------------------------------------------
# Random Forest (multiclass)
# ---------------------------------------------------------------------------

class RandomForestStageClassifier(BaselineModel):
    def __init__(self, random_seed: int = 42, n_estimators: int = 100, max_depth=None):
        super().__init__("RandomForest_classifier", random_seed)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_seed,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, feature_groups: Optional[FeatureGroups] = None) -> "RandomForestStageClassifier":
        if feature_groups:
            assert_no_target_leakage(X.columns.tolist(), feature_groups, context=self.model_name)
        Xv = self._prepare_X(X)
        logger.info(f"Fitting {self.model_name} on {Xv.shape[0]:,} samples…")
        self.model.fit(Xv, y.values)
        self._metadata = {"params": self.model.get_params()}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._prepare_X(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._prepare_X(X))


# ---------------------------------------------------------------------------
# XGBoost (multiclass)
# ---------------------------------------------------------------------------

class XGBoostClassifier(BaselineModel):
    def __init__(self, random_seed: int = 42, n_estimators: int = 100,
                 max_depth: int = 6, learning_rate: float = 0.1):
        if not _XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed. Install with: pip install xgboost\n"
                "Or proceed without it — Logistic Regression and Random Forest are available."
            )
        super().__init__("XGBoost_classifier", random_seed)
        self.model = _XGBClf(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_seed,
            n_jobs=-1,
            eval_metric="mlogloss",
            use_label_encoder=False,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, feature_groups: Optional[FeatureGroups] = None) -> "XGBoostClassifier":
        if feature_groups:
            assert_no_target_leakage(X.columns.tolist(), feature_groups, context=self.model_name)
        Xv = self._prepare_X(X)
        logger.info(f"Fitting {self.model_name}…")
        # XGBoost requires 0-indexed labels for multiclass. Stages are 1-6.
        y_encoded = y.values - 1
        self.model.fit(Xv, y_encoded)
        self._metadata = {"params": self.model.get_params()}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._prepare_X(X)) + 1

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._prepare_X(X))


# ---------------------------------------------------------------------------
# LightGBM (multiclass)
# ---------------------------------------------------------------------------

class LightGBMClassifier(BaselineModel):
    def __init__(self, random_seed: int = 42, n_estimators: int = 100,
                 max_depth: int = -1, learning_rate: float = 0.1):
        if not _LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM is not installed. Install with: pip install lightgbm\n"
                "Or proceed without it — sklearn models are available."
            )
        super().__init__("LightGBM_classifier", random_seed)
        self.model = _LGBMClf(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_seed,
            n_jobs=-1,
            verbosity=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, feature_groups: Optional[FeatureGroups] = None) -> "LightGBMClassifier":
        if feature_groups:
            assert_no_target_leakage(X.columns.tolist(), feature_groups, context=self.model_name)
        Xv = self._prepare_X(X)
        logger.info(f"Fitting {self.model_name}…")
        # LightGBM requires 0-indexed labels for multiclass. Stages are 1-6.
        y_encoded = y.values - 1
        self.model.fit(Xv, y_encoded)
        self._metadata = {"params": self.model.get_params()}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._prepare_X(X)) + 1

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._prepare_X(X))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_stage_classifiers(config, feature_groups: FeatureGroups) -> list:
    """Return list of (name, model) tuples for all available stage classifiers."""
    seed = config.random_seed
    lr_params = config.baselines.logistic_regression
    rf_params = config.baselines.random_forest
    xgb_params = config.baselines.xgboost

    models = [
        ("LogisticRegression", LogisticRegressionClassifier(
            random_seed=seed,
            max_iter=lr_params.max_iter or 1000,
            C=lr_params.C or 1.0,
        )),
        ("RandomForest", RandomForestStageClassifier(
            random_seed=seed,
            n_estimators=rf_params.n_estimators or 100,
        )),
    ]

    if _XGBOOST_AVAILABLE:
        models.append(("XGBoost", XGBoostClassifier(
            random_seed=seed,
            n_estimators=xgb_params.n_estimators or 100,
            max_depth=xgb_params.max_depth or 6,
            learning_rate=xgb_params.learning_rate or 0.1,
        )))

    if _LIGHTGBM_AVAILABLE:
        lgbm_params = config.baselines.lightgbm
        models.append(("LightGBM", LightGBMClassifier(
            random_seed=seed,
            n_estimators=lgbm_params.n_estimators or 100,
            learning_rate=lgbm_params.learning_rate or 0.1,
        )))

    return models
