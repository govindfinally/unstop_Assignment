"""Structured application logging for console and file outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def setup_logging(log_dir: Path, log_file_name: str = "integrity_engine.log", level: str = "INFO") -> None:
    """Configure root logger with both console and file handlers."""
    global _CONFIGURED
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_file_name

    root_logger = logging.getLogger("integrity_engine")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if _CONFIGURED:
        return

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    root_logger.propagate = False
    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child logger under the integrity_engine namespace."""
    base = "integrity_engine"
    if not name:
        return logging.getLogger(base)
    return logging.getLogger(f"{base}.{name}")
