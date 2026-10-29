"""
scripts/generate_phase3_artifacts.py
Generate Phase 3 research artifacts from the trained checkpoint.

Produces:
  - reports/phase3/global_importance/global_feature_importance.csv
  - reports/phase3/attention/temporal_attention_summary.csv
  - reports/phase3/patient_examples/patient_explanation_example.json

All values come from actual model inference. Nothing is fabricated.

Usage:
    python -m scripts.generate_phase3_artifacts

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.inference.model_loader import ModelLoader
from src.inference.predictor import CKDPredictor, PatientStore
from src.inference.preprocessing import InferencePreprocessor
from src.explainability.global_explanations import (
    generate_global_feature_importance,
    generate_temporal_attention_summary,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports" / "phase3"
MAX_PATIENTS_FOR_GLOBAL = 500  # Cap for speed — increase for production


def main():
    logger.info("=== Phase 3 Artifact Generation ===")

    # ── Directories ────────────────────────────────────────────────────
    (REPORTS_DIR / "global_importance").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "attention").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "patient_examples").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "explainability").mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────
    loader = ModelLoader()
    model, config = loader.load(device="cpu")
    mc = config["model"]

    preprocessor = InferencePreprocessor().load()
    store = PatientStore("data/processed/sequences", split="test").load()
    predictor = CKDPredictor(
        model_loader=loader,
        preprocessor=preprocessor,
        patient_store=store,
        device="cpu",
    ).initialize()

    # ── 1. Patient explanation example ────────────────────────────────
    logger.info("Generating patient explanation example...")
    first_pid = store.list_patient_ids(1)[0]
    example_explanation = predictor.predict_with_explanation(first_pid)

    example_path = REPORTS_DIR / "patient_examples" / "patient_explanation_example.json"
    with open(example_path, "w", encoding="utf-8") as f:
        json.dump(example_explanation, f, indent=2, default=str)
    logger.info(f"Patient explanation saved → {example_path}")

    # ── 2. Global feature importance ──────────────────────────────────
    logger.info(f"Generating global feature importance (max {MAX_PATIENTS_FOR_GLOBAL} patients)...")
    global_importance_df = generate_global_feature_importance(
        model=model,
        sequences_dir="data/processed/sequences",
        split="test",
        max_patients=MAX_PATIENTS_FOR_GLOBAL,
        n_static_features=mc["n_static_features"],
        n_temporal_features=mc["n_temporal_features"],
        device="cpu",
    )
    importance_path = REPORTS_DIR / "global_importance" / "global_feature_importance.csv"
    global_importance_df.to_csv(importance_path, index=False)
    logger.info(f"Global feature importance saved → {importance_path}")
    print("\nTop 10 features by mean importance:")
    print(global_importance_df.head(10).to_string(index=False))

    # ── 3. Temporal attention summary ─────────────────────────────────
    logger.info(f"Generating temporal attention summary (max {MAX_PATIENTS_FOR_GLOBAL} patients)...")
    attn_summary_df = generate_temporal_attention_summary(
        model=model,
        sequences_dir="data/processed/sequences",
        split="test",
        max_patients=MAX_PATIENTS_FOR_GLOBAL,
        n_static_features=mc["n_static_features"],
        n_temporal_features=mc["n_temporal_features"],
        device="cpu",
    )
    attn_path = REPORTS_DIR / "attention" / "temporal_attention_summary.csv"
    attn_summary_df.to_csv(attn_path, index=False)
    logger.info(f"Temporal attention summary saved → {attn_path}")
    print("\nTemporal attention by visit position:")
    print(attn_summary_df.to_string(index=False))

    # ── 4. Summary ────────────────────────────────────────────────────
    print("\n=== Phase 3 Artifacts Generated ===")
    print(f"  Patient example: {example_path}")
    print(f"  Global feature importance: {importance_path}")
    print(f"  Temporal attention summary: {attn_path}")
    print("\nDISCLAIMER: All values derived from actual model inference on synthetic data.")
    print("No values are fabricated. For research purposes only.")


if __name__ == "__main__":
    main()
