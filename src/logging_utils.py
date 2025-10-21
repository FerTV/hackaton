"""
Logging utilities for the YOLO dual-head fine-tuning pipeline.
"""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s | %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"


def configure_logging(verbose: bool = True) -> None:
    """
    Initialise the root logger with a concise formatter.

    Parameters
    ----------
    verbose:
        When True (the default), the global logging level is set to INFO.
        Otherwise, only warnings and errors are emitted.
    """

    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)


def get_logger(name: str = "yolo_def") -> logging.Logger:
    """
    Return a named logger used across the project.

    Parameters
    ----------
    name:
        The logger name. Defaults to ``"yolo_def"``.
    """

    return logging.getLogger(name)
