"""
src/baselines/binary_classifiers.py
Binary progression risk classification baselines.
Predicts progression_3m, progression_6m, progression_12m separately.

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

try:
    from xgboost import XGBClassifier as _XGBClf
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier as _LGBMClf
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False


class LogisticRegressionBinary(BaselineModel):
    def __init__(self, random_seed=42, max_iter=1000, C=1.0):
        super().__init__("LogisticRegression_binary", random_seed)
        self.model = LogisticRegression(
            solver="lbfgs", max_iter=max_iter, C=C, random_state=random_seed
        )

    def fit(self, X, y, feature_groups=None):
        if feature_groups:
            assert_no_target_leakage(X.columns.tolist(), feature_groups, self.model_name)
        Xv = self._prepare_X(X)
        self.model.fit(Xv, y.values.ravel())
        self._metadata = {"params": self.model.get_params()}
        return self

    def predict(self, X): return self.model.predict(self._prepare_X(X))
    def predict_proba(self, X): return self.model.predict_proba(self._prepare_X(X))


class RandomForestBinary(BaselineModel):
    def __init__(self, random_seed=42, n_estimators=100):
        super().__init__("RandomForest_binary", random_seed)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_seed, n_jobs=-1
        )

    def fit(self, X, y, feature_groups=None):
        if feature_groups:
            assert_no_target_leakage(X.columns.tolist(), feature_groups, self.model_name)
        Xv = self._prepare_X(X)
        self.model.fit(Xv, y.values.ravel())
        self._metadata = {"params": self.model.get_params()}
        return self

    def predict(self, X): return self.model.predict(self._prepare_X(X))
    def predict_proba(self, X): return self.model.predict_proba(self._prepare_X(X))


class XGBoostBinary(BaselineModel):
    def __init__(self, random_seed=42, n_estimators=100, max_depth=6, learning_rate=0.1):
        if not _XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed.")
        super().__init__("XGBoost_binary", random_seed)
        self.model = _XGBClf(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=random_seed,
            n_jobs=-1, eval_metric="logloss", use_label_encoder=False,
        )

    def fit(self, X, y, feature_groups=None):
        if feature_groups:
            assert_no_target_leakage(X.columns.tolist(), feature_groups, self.model_name)
        self.model.fit(self._prepare_X(X), y.values.ravel())
        self._metadata = {"params": self.model.get_params()}
        return self

    def predict(self, X): return self.model.predict(self._prepare_X(X))
    def predict_proba(self, X): return self.model.predict_proba(self._prepare_X(X))


class LightGBMBinary(BaselineModel):
    def __init__(self, random_seed=42, n_estimators=100, learning_rate=0.1):
        if not _LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed.")
        super().__init__("LightGBM_binary", random_seed)
        self.model = _LGBMClf(
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
    def predict_proba(self, X): return self.model.predict_proba(self._prepare_X(X))


def get_binary_classifiers(config, feature_groups: FeatureGroups) -> list:
    seed = config.random_seed
    lr_p  = config.baselines.logistic_regression
    rf_p  = config.baselines.random_forest
    xgb_p = config.baselines.xgboost

    models = [
        ("LogisticRegression", LogisticRegressionBinary(seed, lr_p.max_iter or 1000, lr_p.C or 1.0)),
        ("RandomForest",       RandomForestBinary(seed, rf_p.n_estimators or 100)),
    ]
    if _XGBOOST_AVAILABLE:
        models.append(("XGBoost", XGBoostBinary(seed, xgb_p.n_estimators or 100)))
    if _LIGHTGBM_AVAILABLE:
        lgbm_p = config.baselines.lightgbm
        models.append(("LightGBM", LightGBMBinary(seed, lgbm_p.n_estimators or 100)))
    return models
