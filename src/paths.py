"""
Path utilities and configuration structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PathConfig:
    """
    Collection of filesystem locations used throughout the pipeline.

    Attributes
    ----------
    root:
        Project root directory.
    dataset_raw:
        Path to the raw dataset (must contain ``images/`` and ``labels/``).
    dataset_processed:
        Destination for the remapped dataset structure.
    dataset_yaml:
        Location of the generated ``data.yaml`` file.
    config_path:
        Path to the dual-head YOLO configuration file.
    output_dir:
        Directory where artefacts (runs, weights, etc.) are saved.
    """

    root: Path
    dataset_raw: Path
    dataset_processed: Path
    dataset_yaml: Path
    config_path: Path
    output_dir: Path


def resolve_path(root: Path, value: Path | str) -> Path:
    """
    Resolve ``value`` relative to ``root`` unless it is already absolute.

    Parameters
    ----------
    root:
        Base directory used for relative resolution.
    value:
        Absolute or relative path.
    """

    path = Path(value)
    return path if path.is_absolute() else (root / path)
