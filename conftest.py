"""
conftest.py — pytest root configuration.
Adds project root to sys.path so all src.* imports work in tests.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
