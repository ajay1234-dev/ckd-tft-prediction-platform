#!/usr/bin/env bash
# scripts/run_tests.sh
# Run the full CKD Phase 1 test suite.
# 
# Usage:
#   bash scripts/run_tests.sh
#   bash scripts/run_tests.sh --cov   (with coverage)
#
# Research Prototype | Synthetic Data | Not for Clinical Use

set -e
cd "$(dirname "$0")/.."

echo "========================================"
echo "CKD Phase 1 — Test Suite"
echo "Research prototype | Synthetic data"
echo "========================================"

if [[ "$1" == "--cov" ]]; then
    pytest tests/ -v --tb=short --cov=src --cov-report=term-missing
else
    pytest tests/ -v --tb=short
fi
