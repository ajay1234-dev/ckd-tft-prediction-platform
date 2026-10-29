"""
scripts/evaluate_tft.py
CLI entry point to evaluate the Phase 2 TFT model on the test set.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import yaml
import argparse
from pathlib import Path
from src.training.evaluate import evaluate_tft

def main():
    parser = argparse.ArgumentParser(description="Evaluate the TFT model")
    parser.add_argument("--config", type=str, default="configs/tft_config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    print("Starting TFT Evaluation on Test Set...")
    evaluate_tft(config)
    print("Evaluation finished.")

if __name__ == "__main__":
    main()
