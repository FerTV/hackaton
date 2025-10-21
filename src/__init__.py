"""
Convenience exports for the YOLO dual-head fine-tuning utilities.
"""

from .dataset import COCO_NAMES
from .logging_utils import configure_logging, get_logger
from .patches import ensure_ultralytics_multihead_support
from .paths import PathConfig, resolve_path
from .config import ensure_dual_head_config, update_model_cfg
from .trainer import train_and_merge

__all__ = [
    "COCO_NAMES",
    "configure_logging",
    "ensure_dual_head_config",
    "ensure_ultralytics_multihead_support",
    "get_logger",
    "PathConfig",
    "resolve_path",
    "train_and_merge",
    "update_model_cfg",
]
