"""
src/utils/logging.py
Structured logging setup with consistent format across all CKD pipeline scripts.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str, level: int = logging.INFO, log_file: Optional[Path] = None) -> logging.Logger:
    """
    Create or retrieve a logger with consistent formatting.

    Parameters
    ----------
    name : str
        Logger name (typically __name__ from calling module).
    level : int
        Logging level (default: INFO).
    log_file : Path, optional
        If provided, also write logs to this file.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # Console handler — use UTF-8 encoding to avoid Windows cp1252 issues
    import sys, io as _io
    stdout_utf8 = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8") if hasattr(sys.stdout, "buffer") else sys.stdout
    ch = logging.StreamHandler(stdout_utf8)
    ch.setLevel(level)
    ch.setFormatter(_FORMATTER)
    logger.addHandler(ch)

    # Optional file handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(_FORMATTER)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
