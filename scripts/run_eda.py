"""
scripts/run_eda.py
CLI: Run full EDA — generates all figures and eda_summary.json.

Usage:
    python scripts/run_eda.py
    python scripts/run_eda.py --config configs/config.yaml --sample-patients 100

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.seeding import set_global_seed
from src.utils.logging import get_logger
from src.data.loader import load_longitudinal
from src.data.eda import run_full_eda

logger = get_logger("run_eda")


def main(config_path=None, sample_patients=50):
    logger.info("=" * 60)
    logger.info("CKD Phase 1 — Exploratory Data Analysis")
    logger.info("NOTE: Synthetic data — not real patient records.")
    logger.info("=" * 60)

    cfg = load_config(config_path)
    set_global_seed(cfg.random_seed)

    logger.info("Loading training dataset for EDA…")
    train_df = load_longitudinal(cfg.data.train_file, dataset_name="train_eda")

    summary = run_full_eda(
        train_df=train_df,
        figures_dir=cfg.output.figures_dir,
        metrics_dir=cfg.output.metrics_dir,
        sample_patients=sample_patients,
    )

    logger.info(f"\nEDA complete. {len(summary)} figure sets generated.")
    logger.info(f"Figures → {cfg.output.figures_dir}/")
    logger.info(f"Summary → {cfg.output.metrics_dir}/eda_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CKD EDA")
    parser.add_argument("--config", default=None)
    parser.add_argument("--sample-patients", type=int, default=50,
                        help="Number of random patients for trajectory plot")
    args = parser.parse_args()
    main(args.config, args.sample_patients)
