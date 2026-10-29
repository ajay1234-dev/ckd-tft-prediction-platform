"""
scripts/check_leakage.py
CLI: Verify patient-level disjointness across train/val/test. Exits non-zero if leakage found.

Usage:
    python scripts/check_leakage.py
    python scripts/check_leakage.py --config configs/config.yaml

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.logging import get_logger
from src.data.loader import load_all_datasets
from src.preprocessing.leakage import (
    run_all_leakage_checks,
    PatientLeakageError,
    verify_chronological_order,
)

logger = get_logger("check_leakage")


def main(config_path=None):
    logger.info("=" * 60)
    logger.info("CKD Phase 1 — Patient Leakage & Ordering Check")
    logger.info("=" * 60)

    cfg = load_config(config_path)
    datasets = load_all_datasets(cfg)

    try:
        run_all_leakage_checks(
            train_df=datasets["train"],
            val_df=datasets["val"],
            test_df=datasets["test"],
            static_df=datasets["static"],
        )
        logger.info("✓ All leakage checks passed. Pipeline may proceed.")
        sys.exit(0)

    except PatientLeakageError as e:
        logger.error(f"\n{'='*60}\nPATIENT LEAKAGE ERROR\n{'='*60}\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
