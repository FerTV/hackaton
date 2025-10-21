"""
Training and model-merging utilities for the YOLO dual-head workflow.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Dict, Sequence

from dataset import COCO_NAMES  # reuse the constant from dataset module


logger = logging.getLogger(__name__)


def build_full_class_names(new_class_names: Sequence[str]) -> list[str]:
    """Concatenate the default COCO names with the new dataset-specific names."""

    return list(COCO_NAMES) + list(new_class_names)


def train_and_merge(
    config_path: Path,
    data_yaml: Path,
    added_classes: int,
    new_class_names: Sequence[str],
    freeze_layers: int,
    epochs: int,
    imgsz: int,
    output_dir: Path,
    dataset_name: str,
    base_model: str,
) -> Dict[str, Path | None]:
    """Train the custom head, merge it into the dual-head model, and export artefacts."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics package is not importable. Install it before running this script.") from exc

    import torch

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(base_model)
    original_state = copy.deepcopy(model.state_dict())
    head_prefix = f"model.model.{freeze_layers}"

    def put_in_eval_mode(trainer, n_layers: int = freeze_layers) -> None:
        model_ref = getattr(trainer, "model", None)
        if model_ref is None or not hasattr(model_ref, "named_modules"):
            return
        for name, module in model_ref.named_modules():
            if not name.endswith("bn"):
                continue
            parts = [part for part in name.split(".") if part.isdigit()]
            if not parts:
                continue
            layer_idx = int(parts[0])
            if layer_idx < n_layers and hasattr(module, "track_running_stats"):
                module.eval()
                module.track_running_stats = False

    model.add_callback("on_train_epoch_start", put_in_eval_mode)
    model.add_callback("on_pretrain_routine_start", put_in_eval_mode)

    project_dir = output_dir / "runs"
    project_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training...")
    results = model.train(
        data=str(data_yaml),
        freeze=freeze_layers,
        epochs=epochs,
        imgsz=imgsz,
        project=str(project_dir),
        name=dataset_name,
        exist_ok=True,
    )

    trainer = getattr(model, "trainer", None)
    if trainer is not None and getattr(trainer, "save_dir", None):
        save_dir = Path(trainer.save_dir)
    elif hasattr(results, "save_dir"):
        save_dir = Path(results.save_dir)
    else:
        save_dir = project_dir / dataset_name

    best_weights = save_dir / "weights" / "best.pt"

    updated_state = model.state_dict()
    for key, tensor in original_state.items():
        if key not in updated_state:
            continue
        if tensor.shape != updated_state[key].shape:
            continue
        if not torch.equal(tensor, updated_state[key]) and "bn" in key and head_prefix not in key:
            updated_state[key] = tensor

    head_weights: Dict[str, torch.Tensor] = {}
    for key, tensor in updated_state.items():
        if key.startswith(head_prefix):
            renamed = key.replace(f".{freeze_layers}", f".{freeze_layers + 1}", 1)
            head_weights[renamed] = tensor.clone()

    head_weights_path = output_dir / f"{Path(base_model).stem}_{dataset_name}_head.pth"
    torch.save(head_weights, head_weights_path)
    logger.info("Saved remapped head weights to %s", head_weights_path)

    merged_model = YOLO(str(config_path), task="detect").load(base_model)

    state_dict = torch.load(head_weights_path, map_location="cpu")
    missing, unexpected = merged_model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys when loading new head weights: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading new head weights: %s", unexpected)

    merged_model.model.names = {idx: name for idx, name in enumerate(build_full_class_names(new_class_names))}
    merged_model.ckpt = {"model": merged_model.model}

    merged_weights_path = output_dir / f"{Path(base_model).stem}_{dataset_name}_merged.pt"
    merged_model.save(str(merged_weights_path))
    logger.info("Merged model saved to %s", merged_weights_path)

    return {
        "best_weights": best_weights,
        "head_weights": head_weights_path,
        "merged_weights": merged_weights_path,
    }
