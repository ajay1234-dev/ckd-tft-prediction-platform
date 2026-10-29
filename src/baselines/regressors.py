"""
src/baselines/regressors.py
eGFR regression baselines.
Predicts egfr_3m, egfr_6m, egfr_12m separately.

Models:
  - Ridge Regression
  - Random Forest Regressor
  - XGBoost Regressor (if available)
  - LightGBM Regressor (if available)

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from src.baselines.base import BaselineModel
from src.features.feature_groups import FeatureGroups, assert_no_target_leakage
from src.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from xgboost import XGBRegressor as _XGBReg
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor as _LGBMReg
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False


class RidgeRegressorModel(BaselineModel):
    def __init__(self, random_seed=42, alpha=1.0):
        super().__init__("Ridge_regressor", random_seed)
        self.model = Ridge(alpha=alpha, random_state=random_seed)

    def fit(self, X, y, feature_groups=None):
        if feature_groups:
            assert_no_target_leakage(X.columns.tolist(), feature_groups, self.model_name)
        Xv = self._prepare_X(X)
        self.model.fit(Xv, y.values.ravel())
        self._metadata = {"params": self.model.get_params()}
        logger.info(f"Fitted {self.model_name} on {Xv.shape[0]:,} samples.")
        return self

    def predict(self, X): return self.model.predict(self._prepare_X(X))


class RandomForestRegressorModel(BaselineModel):
    def __init__(self, random_seed=42, n_estimators=100):
        super().__init__("RandomForest_regressor", random_seed)
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_seed, n_jobs=-1
        )

    def fit(self, X, y, feature_groups=None):
        if feature_groups:
            assert_no_target_leakage(X.columns.tolist(), feature_groups, self.model_name)
        Xv = self._prepare_X(X)
        self.model.fit(Xv, y.values.ravel())
        self._metadata = {"params": self.model.get_params()}
        logger.info(f"Fitted {self.model_name} on {Xv.shape[0]:,} samples.")
        return self

    def predict(self, X): return self.model.predict(self._prepare_X(X))


class XGBoostRegressorModel(BaselineModel):
    def __init__(self, random_seed=42, n_estimators=100, max_depth=6, learning_rate=0.1):
        if not _XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed.")
        super().__init__("XGBoost_regressor", random_seed)
        self.model = _XGBReg(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=random_seed,
            n_jobs=-1,
        )

    def fit(self, X, y, feature_groups=None):
        if feature_groups:
            assert_no_target_leakage(X.columns.tolist(), feature_groups, self.model_name)
        self.model.fit(self._prepare_X(X), y.values.ravel())
        self._metadata = {"params": self.model.get_params()}
        return self

    def predict(self, X): return self.model.predict(self._prepare_X(X))


class LightGBMRegressorModel(BaselineModel):
    def __init__(self, random_seed=42, n_estimators=100, learning_rate=0.1):
        if not _LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed.")
        super().__init__("LightGBM_regressor", random_seed)
        self.model = _LGBMReg(
            n_estimators=n_estimators, learning_rate=learning_rate,
            random_state=random_seed, n_jobs=-1, verbosity=-1,
        )

    def fit(self, X, y, feature_groups=None):
        if feature_groups:
            assert_no_target_leakage(X.columns.tolist(), feature_groups, self.model_name)
        self.model.fit(self._prepare_X(X), y.values.ravel())
        self._metadata = {"params": self.model.get_params()}
        return self

    def predict(self, X): return self.model.predict(self._prepare_X(X))


def get_regressors(config, feature_groups: FeatureGroups) -> list:
    seed = config.random_seed
    rf_p  = config.baselines.random_forest
    xgb_p = config.baselines.xgboost

    models = [
        ("Ridge",        RidgeRegressorModel(seed, config.baselines.ridge_regression.alpha or 1.0)),
        ("RandomForest", RandomForestRegressorModel(seed, rf_p.n_estimators or 100)),
    ]
    if _XGBOOST_AVAILABLE:
        models.append(("XGBoost", XGBoostRegressorModel(seed, xgb_p.n_estimators or 100)))
    if _LIGHTGBM_AVAILABLE:
        lgbm_p = config.baselines.lightgbm
        models.append(("LightGBM", LightGBMRegressorModel(seed, lgbm_p.n_estimators or 100)))
    return models
