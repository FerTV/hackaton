#!/usr/bin/env python3
"""
Entry point for the YOLO11 dual-head fine-tuning workflow.

The script orchestrates the following stages:
    1. Runtime patching of Ultralytics so the ConcatHead module and multiple Detect
       heads are supported without cloning the repository.
    2. Dataset preparation (class ID remapping, symlinking images, and data.yaml generation).
    3. Dual-head configuration generation/update.
    4. Fine-tuning of the additional detection head and export of the merged weights.
"""

from __future__ import annotations

import argparse

from pathlib import Path

from logging_utils import configure_logging, get_logger
from config import update_model_cfg, ensure_dual_head_config
from patches import ensure_ultralytics_multihead_support
from paths import resolve_path, PathConfig
from trainer import train_and_merge
from dataset import COCO_NAMES, prepare_dataset, write_dataset_yaml
logger = get_logger()


def parse_args() -> argparse.Namespace:
    """CLI argument parsing."""

    parser = argparse.ArgumentParser(description="YOLO11 dual-head fine-tuning pipeline (no repo clone required).")
    default_root = Path(__file__).resolve().parent.parent

    parser.add_argument("--root", type=Path, default=default_root, help="Project root.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/datasets/drones"),
        help="Path to the raw dataset (expected structure: images/<split>, labels/<split>).",
    )
    parser.add_argument(
        "--processed-dataset-dir",
        type=Path,
        default=Path("data/processed/drones"),
        help="Where to materialise the remapped dataset used for training.",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=Path("data/dataset.yaml"),
        help="Output path for the generated YOLO data.yaml file.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("config/yolo11n-2xhead.yaml"),
        help="Location of the dual-head YOLO11 config file (created automatically if missing).",
    )
    parser.add_argument("--dataset-name", default="drones", help="Name used for logging/tracking runs.")
    parser.add_argument(
        "--class-names",
        nargs="+",
        help="Optional override for class names (one per class after remapping).",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=320, help="Training image size.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size used by Ultralytics during training.",
    )
    parser.add_argument("--freeze", type=int, default=23, help="Number of layers to freeze during training.")
    parser.add_argument(
        "--base-model",
        default="yolo11n.pt",
        help="Pretrained checkpoint to fine-tune and load into the dual-head architecture.",
    )
    parser.add_argument(
        "--base-class-count",
        type=int,
        default=len(COCO_NAMES),
        help="Number of classes in the pretrained model (default: 80 for COCO).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to store training outputs and exported models.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce logging verbosity.")

    return parser.parse_args()


def build_paths(args: argparse.Namespace) -> PathConfig:
    """Resolve user-supplied paths relative to the project root."""

    root = args.root.resolve()
    return PathConfig(
        root=root,
        dataset_raw=resolve_path(root, args.dataset_dir),
        dataset_processed=resolve_path(root, args.processed_dataset_dir),
        dataset_yaml=resolve_path(root, args.data_yaml),
        config_path=resolve_path(root, args.config_path),
        output_dir=resolve_path(root, args.output_dir),
    )


def main() -> None:
    args = parse_args()
    configure_logging(verbose=not args.quiet)

    paths = build_paths(args)
    base_class_count = args.base_class_count
    if base_class_count <= 0:
        raise ValueError("--base-class-count must be a positive integer.")

    logger.info("Project root: %s", paths.root)
    logger.info("Raw dataset dir: %s", paths.dataset_raw)
    logger.info("Processed dataset dir: %s", paths.dataset_processed)
    logger.info("Data YAML path: %s", paths.dataset_yaml)
    logger.info("Output dir: %s", paths.output_dir)
    logger.info("Dual-head config path: %s", paths.config_path)
    logger.info("Base model class count: %s", base_class_count)

    ensure_ultralytics_multihead_support()
    ensure_dual_head_config(paths.config_path)

    class_names, class_mapping, splits = prepare_dataset(
        paths.dataset_raw, paths.dataset_processed, args.dataset_name, args.class_names
    )
    logger.info("Detected class ids: %s", class_mapping)
    logger.info("Using class names: %s", class_names)

    write_dataset_yaml(paths.dataset_yaml, paths.dataset_processed, class_names, splits)

    added_classes = len(class_names)
    update_model_cfg(paths.config_path, added_classes, base_class_count=base_class_count)

    summary = train_and_merge(
        config_path=paths.config_path,
        data_yaml=paths.dataset_yaml,
        new_class_names=class_names,
        freeze_layers=args.freeze,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        output_dir=paths.output_dir,
        dataset_name=args.dataset_name,
        base_model=args.base_model,
    )

    logger.info("")
    logger.info("Pipeline completed. Key artifacts:")
    for label, path in summary.items():
        logger.info("  • %s: %s", label, path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
