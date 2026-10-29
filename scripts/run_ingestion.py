"""
scripts/run_ingestion.py
CLI: Load all 4 datasets, validate schemas, generate & save data quality reports.

Usage:
    python scripts/run_ingestion.py
    python scripts/run_ingestion.py --config configs/config.yaml

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.seeding import set_global_seed
from src.utils.logging import get_logger
from src.data.loader import load_all_datasets
from src.data.quality import generate_quality_report, save_quality_reports

logger = get_logger("run_ingestion")


def main(config_path: str = None):
    logger.info("=" * 60)
    logger.info("CKD Phase 1 — Data Ingestion & Quality Report")
    logger.info("NOTE: Research prototype using SYNTHETIC data only.")
    logger.info("=" * 60)

    cfg = load_config(config_path)
    set_global_seed(cfg.random_seed)

    # Load all datasets
    datasets = load_all_datasets(cfg)

    # Generate quality reports
    reports = []
    for name, df in datasets.items():
        report = generate_quality_report(df, dataset_name=name)
        reports.append(report)

    # Save reports
    save_quality_reports(reports, output_dir=cfg.output.metrics_dir)

    # Print summary table
    print("\n" + "=" * 80)
    print("DATA QUALITY SUMMARY")
    print("=" * 80)
    import pandas as pd
    rows = [r.summary_row() for r in reports]
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    print(f"\nFull reports saved to: {cfg.output.metrics_dir}/")
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CKD data ingestion & quality report")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
