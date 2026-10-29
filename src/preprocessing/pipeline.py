"""
src/preprocessing/pipeline.py
Full preprocessing pipeline: imputation → encoding → scaling.

Rules:
  - ALL fitters are trained on TRAINING data only; applied to val/test.
  - Target columns are NEVER imputed or scaled as features.
  - Fitted objects are persisted to models/preprocessing/.
  - A preprocessing_manifest.json records exactly which columns went through
    which transform.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from src.features.ckd_stage import CKD_STAGE_MAP, STAGE_ORDER, encode_stage_series
from src.features.feature_groups import FeatureGroups, assert_no_target_leakage
from src.utils.logging import get_logger
from src.utils.io import save_json

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Binary canonicalization helper
# ---------------------------------------------------------------------------

def _canonicalize_binary(series: pd.Series) -> pd.Series:
    """Coerce any boolean / Y-N / 1.0-0.0 encoding into strict int 0/1."""
    mapping = {
        True: 1, False: 0,
        "true": 1, "false": 0,
        "yes": 1, "no": 0,
        "y": 1, "n": 0,
        1: 1, 0: 0,
        1.0: 1, 0.0: 0,
    }
    def _convert(v):
        if pd.isna(v):
            return 0   # default missing binary to 0 (absent)
        key = v.lower().strip() if isinstance(v, str) else v
        if key in mapping:
            return mapping[key]
        try:
            return int(float(v))
        except Exception:
            return 0
    return series.apply(_convert).astype("int64")


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class CKDPreprocessingPipeline:
    """
    End-to-end preprocessing pipeline for the CKD longitudinal dataset.

    Lifecycle:
        pipeline = CKDPreprocessingPipeline(feature_groups, config)
        X_train = pipeline.fit_transform(train_df)
        X_val   = pipeline.transform(val_df)
        X_test  = pipeline.transform(test_df)
        pipeline.save(output_dir)

    The pipeline never touches target columns — they are separated before
    any transformation and returned separately.
    """

    def __init__(
        self,
        feature_groups: FeatureGroups,
        unknown_category_token: str = "__unknown__",
        random_seed: int = 42,
    ):
        self.fg = feature_groups
        self.unknown_token = unknown_category_token
        self.seed = random_seed

        # Fitted objects (populated during fit)
        self._num_imputer: Optional[SimpleImputer] = None
        self._scaler: Optional[StandardScaler] = None
        self._onehot_enc: Optional[OneHotEncoder] = None
        self._onehot_cols: List[str] = []
        self._onehot_feature_names: List[str] = []
        self._fitted: bool = False

        # Manifest
        self._manifest: Dict = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def fit_transform(self, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit all transformers on training data and transform it.

        Returns (X_train, y_train_df) where y_train_df contains all target columns.
        """
        logger.info("Fitting preprocessing pipeline on training data…")
        X, y = self._split_xy(train_df)
        X = self._sort_chronologically(X)
        X = self._canonicalize_binaries(X)
        X = self._fit_and_apply_imputer(X, fit=True)
        X = self._encode_ckd_stage(X)
        X = self._fit_and_apply_onehot(X, fit=True)
        X = self._fit_and_apply_scaler(X, fit=True)
        self._fitted = True
        self._build_manifest(X)
        logger.info(f"Fit complete — training set shape: {X.shape}")
        return X, y

    def transform(self, df: pd.DataFrame, split_name: str = "split") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform val/test data using ONLY fitted transformers (never refit).

        Returns (X, y_df).
        """
        if not self._fitted:
            raise RuntimeError("Pipeline has not been fitted. Call fit_transform() first.")
        logger.info(f"Transforming {split_name}…")
        X, y = self._split_xy(df)
        X = self._sort_chronologically(X)
        X = self._canonicalize_binaries(X)
        X = self._apply_imputer(X)
        X = self._encode_ckd_stage(X)
        X = self._apply_onehot(X)
        X = self._apply_scaler(X)
        logger.info(f"Transform complete — {split_name} shape: {X.shape}")
        return X, y

    def save(self, output_dir: str | Path) -> None:
        """Persist all fitted objects and manifest to output_dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self._num_imputer, output_dir / "imputer.joblib")
        joblib.dump(self._scaler, output_dir / "scaler.joblib")
        joblib.dump(self._onehot_enc, output_dir / "encoders.joblib")

        manifest_path = output_dir / "preprocessing_manifest.json"
        save_json(self._manifest, manifest_path)

        logger.info(f"Preprocessing artifacts saved to {output_dir}")

    @classmethod
    def load(cls, output_dir: str | Path, feature_groups: FeatureGroups) -> "CKDPreprocessingPipeline":
        """Load a previously fitted pipeline from disk."""
        output_dir = Path(output_dir)
        pipeline = cls(feature_groups)
        pipeline._num_imputer = joblib.load(output_dir / "imputer.joblib")
        pipeline._scaler = joblib.load(output_dir / "scaler.joblib")
        pipeline._onehot_enc = joblib.load(output_dir / "encoders.joblib")

        manifest_path = output_dir / "preprocessing_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                pipeline._manifest = json.load(f)
            pipeline._onehot_cols = pipeline._manifest.get("onehot_input_columns", [])
            pipeline._onehot_feature_names = pipeline._manifest.get("onehot_output_features", [])

        pipeline._fitted = True
        logger.info(f"Preprocessing pipeline loaded from {output_dir}")
        return pipeline

    # -------------------------------------------------------------------------
    # Internal steps
    # -------------------------------------------------------------------------

    def _split_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Separate feature columns from target columns."""
        target_cols = [c for c in self.fg.all_targets if c in df.columns]
        id_cols = [c for c in self.fg.identifiers if c in df.columns]
        # Keep IDs in X for merging / sequence building
        feature_cols = [c for c in df.columns if c not in target_cols]
        assert_no_target_leakage(feature_cols, self.fg, context="pipeline._split_xy")
        return df[feature_cols].copy(), df[target_cols].copy()

    def _sort_chronologically(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure chronological ordering per patient."""
        if "patient_id" in df.columns and "visit_date" in df.columns:
            df = df.sort_values(["patient_id", "visit_date", "visit_number"]).reset_index(drop=True)
        return df

    def _canonicalize_binaries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert all binary flag columns to strict 0/1 int."""
        binary_cols = [c for c in self.fg.binary_columns if c in df.columns]
        for col in binary_cols:
            df[col] = _canonicalize_binary(df[col])
        return df

    def _get_numerical_cols(self, df: pd.DataFrame) -> List[str]:
        """Columns to impute + scale: numerical_columns present in df, excluding identifiers/targets."""
        exclude = set(self.fg.all_targets) | set(self.fg.identifiers) | {"visit_date"}
        num_cols = [
            c for c in self.fg.numerical_columns
            if c in df.columns and c not in exclude
        ]
        return num_cols

    def _fit_and_apply_imputer(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        num_cols = self._get_numerical_cols(df)
        if fit:
            self._num_imputer = SimpleImputer(strategy="median")
            df[num_cols] = self._num_imputer.fit_transform(df[num_cols])
            logger.info(f"Imputer fitted on {len(num_cols)} numerical columns (strategy=median)")
        return df

    def _apply_imputer(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = [c for c in self._get_numerical_cols(df) if c in df.columns]
        if self._num_imputer and num_cols:
            df[num_cols] = self._num_imputer.transform(df[num_cols])
        return df

    def _encode_ckd_stage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode ckd_stage column using the CKD stage codec."""
        if "ckd_stage" in df.columns:
            df["ckd_stage_encoded"] = encode_stage_series(df["ckd_stage"], allow_null=True)
            # Keep original string for reports; model uses _encoded
            df["ckd_stage_encoded"] = df["ckd_stage_encoded"].fillna(-1).astype("int64")
        return df

    def _fit_and_apply_onehot(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        onehot_cols = [c for c in self.fg.onehot_columns if c in df.columns]
        self._onehot_cols = onehot_cols
        if not onehot_cols:
            return df
        if fit:
            self._onehot_enc = OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                sparse_output=False,
                dtype=np.float32,
            )
            encoded = self._onehot_enc.fit_transform(df[onehot_cols].fillna(self.unknown_token).astype(str))
            self._onehot_feature_names = list(self._onehot_enc.get_feature_names_out(onehot_cols))
            logger.info(
                f"OneHotEncoder fitted: {len(onehot_cols)} columns → "
                f"{len(self._onehot_feature_names)} features"
            )
        else:
            encoded = self._onehot_enc.transform(df[onehot_cols].fillna(self.unknown_token).astype(str))

        encoded_df = pd.DataFrame(encoded, columns=self._onehot_feature_names, index=df.index)
        df = pd.concat([df.drop(columns=onehot_cols), encoded_df], axis=1)
        return df

    def _apply_onehot(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._fit_and_apply_onehot(df, fit=False)

    def _fit_and_apply_scaler(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        num_cols = self._get_numerical_cols(df)
        num_cols = [c for c in num_cols if c in df.columns]
        if not num_cols:
            return df
        if fit:
            self._scaler = StandardScaler()
            df[num_cols] = self._scaler.fit_transform(df[num_cols])
            logger.info(f"StandardScaler fitted on {len(num_cols)} columns")
        else:
            df[num_cols] = self._scaler.transform(df[num_cols])
        return df

    def _apply_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._fit_and_apply_scaler(df, fit=False)

    def _build_manifest(self, X: pd.DataFrame) -> None:
        num_cols = self._get_numerical_cols(X)
        self._manifest = {
            "created_at": datetime.now().isoformat(),
            "imputer_strategy": "median",
            "scaler": "StandardScaler",
            "imputed_columns": num_cols,
            "scaled_columns": num_cols,
            "onehot_input_columns": self._onehot_cols,
            "onehot_output_features": self._onehot_feature_names,
            "ckd_stage_encoding": CKD_STAGE_MAP,
            "binary_canonicalized_columns": self.fg.binary_columns,
            "output_shape": list(X.shape),
        }
