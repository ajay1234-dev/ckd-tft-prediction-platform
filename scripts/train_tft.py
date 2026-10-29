"""
scripts/train_tft.py
CLI entry point to train the Phase 2 TFT model.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import yaml
import argparse
from pathlib import Path
from src.training.train import train_tft

def main():
    parser = argparse.ArgumentParser(description="Train the TFT model")
    parser.add_argument("--config", type=str, default="configs/tft_config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    print("Starting TFT Training...")
    train_tft(config)
    print("Training finished.")

if __name__ == "__main__":
    main()
