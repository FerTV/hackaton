#!/usr/bin/env python3
"""
YOLO11 dual-head fine-tuning pipeline without cloning the Ultralytics repository.

This script mirrors the behaviour of the reference Colab notebook by:
  • Preparing the counterfeit_nike dataset (remapping class ids, writing data.yaml).
  • Fine-tuning a pretrained YOLO11 checkpoint while freezing the backbone.
  • Extracting the newly trained detection head and merging it into a dual-head model
    that retains the original COCO predictions.
  • Saving the resulting weights and (optionally) exporting ONNX if the dependencies exist.

All required Ultralytics tweaks (ConcatHead module, multi-head parsing, stride handling)
are injected at runtime, so the upstream package remains untouched and no git clone is needed.

Run from the project root:
    python src/train.py
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


YOLO11_COCO_HEAD_LINE = "  - [[16, 19, 22], 1, Detect, [80]] # Base COCO head"
YOLO11_NEW_HEAD_LINE = "  - [[16, 19, 22], 1, Detect, [2]] # Added classes head"
YOLO11_CONCAT_LINE = "  - [[23, 24], 1, ConcatHead, [80, 2]] # Merge detection heads"

# COCO class names used by the base YOLO models.
COCO_NAMES: Tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def ensure_ultralytics_multihead_support() -> None:
    """Inject ConcatHead support and multi-detect handling into the Ultralytics runtime."""
    import contextlib
    import inspect

    import torch
    import torch.nn as nn
    from ultralytics.nn import modules as modules_init
    from ultralytics.nn.modules import conv as conv_module
    from ultralytics.nn import tasks as tasks_module
    from ultralytics.utils import LOGGER, colorstr
    from ultralytics.utils.ops import make_divisible

    if getattr(tasks_module, "_yolo_def_patched", False):
        return

    ConcatHead = getattr(conv_module, "ConcatHead", None)

    if ConcatHead is None:

        class ConcatHead(nn.Module):
            """Concatenate predictions from two Detect heads."""

            def __init__(self, nc1=80, nc2=1, ch=()):
                super().__init__()
                self.nc1 = nc1
                self.nc2 = nc2

            def forward(self, x):
                """Merge bounding boxes and class scores from two detection heads."""
                if isinstance(x[0], tuple):
                    preds1, preds2 = x[0][0], x[1][0]
                elif isinstance(x[0], list):
                    return [torch.cat((x0, x1), dim=1) for x0, x1 in zip(x[0], x[1])]
                else:
                    preds1, preds2 = x[0], x[1]

                concatenated = torch.cat((preds1[:, :4, :], preds2[:, :4, :]), dim=2)

                def extend(preds, pad_left: bool) -> torch.Tensor:
                    shape = list(preds.shape)
                    shape[-1] = preds1.shape[-1] + preds2.shape[-1]
                    extended = torch.zeros(shape, device=preds.device, dtype=preds.dtype)
                    if pad_left:
                        extended[..., preds.shape[-1] :] = preds
                    else:
                        extended[..., : preds.shape[-1]] = preds
                    return extended

                preds1_extended = extend(preds1, pad_left=False)
                preds2_extended = extend(preds2, pad_left=True)
                concatenated = torch.cat((concatenated, preds1_extended[:, 4:, :]), dim=1)
                concatenated = torch.cat((concatenated, preds2_extended[:, 4:, :]), dim=1)

                if isinstance(x[0], tuple):
                    return concatenated, x[0][1]
                return concatenated

        conv_module.ConcatHead = ConcatHead
        if hasattr(conv_module, "__all__") and "ConcatHead" not in conv_module.__all__:
            conv_module.__all__ = tuple(list(conv_module.__all__) + ["ConcatHead"])
    else:
        ConcatHead = conv_module.ConcatHead

    modules_init.ConcatHead = ConcatHead
    if hasattr(modules_init, "__all__") and "ConcatHead" not in modules_init.__all__:
        modules_init.__all__ = tuple(list(modules_init.__all__) + ["ConcatHead"])
    tasks_module.ConcatHead = ConcatHead

    BaseModel = tasks_module.BaseModel
    if not getattr(BaseModel._apply, "_yolo_def_patched", False):
        original_apply = BaseModel._apply

        def _apply_multi(self, fn):
            self = original_apply(self, fn)
            model_ref = getattr(self, "model", [])
            for module in model_ref:
                if isinstance(module, tasks_module.Detect):
                    module.stride = fn(module.stride)
                    module.anchors = fn(module.anchors)
                    module.strides = fn(module.strides)
            return self

        _apply_multi._yolo_def_patched = True  # type: ignore[attr-defined]
        BaseModel._apply = _apply_multi  # type: ignore[assignment]

    DetectionModel = tasks_module.DetectionModel
    detection_src = inspect.getsource(DetectionModel.__init__)
    if "detect_modules =" not in detection_src:
        original_init = DetectionModel.__init__

        def _init_multi(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True):
            original_init(self, cfg, ch, nc, verbose)
            detect_modules = [module for module in self.model if isinstance(module, tasks_module.Detect)]
            if not detect_modules:
                return

            s = 256
            primary = detect_modules[-1]
            for module in detect_modules:
                module.inplace = self.inplace

            def _forward(x):
                if self.end2end:
                    return self.forward(x)["one2many"]
                return self.forward(x)[0] if isinstance(
                    primary, (tasks_module.Segment, tasks_module.YOLOESegment, tasks_module.Pose, tasks_module.OBB)
                ) else self.forward(x)

            self.model.eval()
            for module in detect_modules:
                module.training = True
            stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])
            self.stride = stride
            for module in detect_modules:
                module.stride = stride
                module.bias_init()
            self.model.train()

        DetectionModel.__init__ = _init_multi  # type: ignore[assignment]

    parse_src = inspect.getsource(tasks_module.parse_model)
    if "ConcatHead" not in parse_src:
        original_parse = tasks_module.parse_model

        def parse_model_patched(d, ch, verbose=True):
            import ast

            legacy = True
            max_channels = float("inf")
            nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
            depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
            scale = d.get("scale")
            if scales:
                if not scale:
                    scale = tuple(scales.keys())[0]
                    LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
                depth, width, max_channels = scales[scale]

            if act:
                tasks_module.Conv.default_act = eval(act)
                if verbose:
                    LOGGER.info(f"{colorstr('activation:')} {act}")

            if verbose:
                LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

            ch = [ch]
            layers, save, c2 = [], [], ch[-1]
            base_modules = frozenset(
                {
                    tasks_module.Classify,
                    tasks_module.Conv,
                    tasks_module.ConvTranspose,
                    tasks_module.GhostConv,
                    tasks_module.Bottleneck,
                    tasks_module.GhostBottleneck,
                    tasks_module.SPP,
                    tasks_module.SPPF,
                    tasks_module.C2fPSA,
                    tasks_module.C2PSA,
                    tasks_module.DWConv,
                    tasks_module.Focus,
                    tasks_module.BottleneckCSP,
                    tasks_module.C1,
                    tasks_module.C2,
                    tasks_module.C2f,
                    tasks_module.C3k2,
                    tasks_module.RepNCSPELAN4,
                    tasks_module.ELAN1,
                    tasks_module.ADown,
                    tasks_module.AConv,
                    tasks_module.SPPELAN,
                    tasks_module.C2fAttn,
                    tasks_module.C3,
                    tasks_module.C3TR,
                    tasks_module.C3Ghost,
                    torch.nn.ConvTranspose2d,
                    tasks_module.DWConvTranspose2d,
                    tasks_module.C3x,
                    tasks_module.RepC3,
                    tasks_module.PSA,
                    tasks_module.SCDown,
                    tasks_module.C2fCIB,
                    tasks_module.A2C2f,
                }
            )
            repeat_modules = frozenset(
                {
                    tasks_module.BottleneckCSP,
                    tasks_module.C1,
                    tasks_module.C2,
                    tasks_module.C2f,
                    tasks_module.C3k2,
                    tasks_module.C2fAttn,
                    tasks_module.C3,
                    tasks_module.C3TR,
                    tasks_module.C3Ghost,
                    tasks_module.C3x,
                    tasks_module.RepC3,
                    tasks_module.C2fPSA,
                    tasks_module.C2fCIB,
                    tasks_module.C2PSA,
                    tasks_module.A2C2f,
                }
            )
            for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
                m = (
                    getattr(torch.nn, m[3:])
                    if "nn." in m
                    else getattr(__import__("torchvision").ops, m[16:])
                    if "torchvision.ops." in m
                    else tasks_module.__dict__[m]
                )
                for j, a in enumerate(args):
                    if isinstance(a, str):
                        with contextlib.suppress(ValueError):
                            args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
                n = n_ = max(round(n * depth), 1) if n > 1 else n
                if m in base_modules:
                    c1, c2 = ch[f], args[0]
                    if c2 != nc:
                        c2 = make_divisible(min(c2, max_channels) * width, 8)
                    if m is tasks_module.C2fAttn:
                        args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                        args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])

                    args = [c1, c2, *args[1:]]
                    if m in repeat_modules:
                        args.insert(2, n)
                        n = 1
                    if m is tasks_module.C3k2:
                        legacy = False
                        if scale in "mlx":
                            args[3] = True
                    if m is tasks_module.A2C2f:
                        legacy = False
                        if scale in "lx":
                            args.extend((True, 1.2))
                    if m is tasks_module.C2fCIB:
                        legacy = False
                elif m is tasks_module.AIFI:
                    args = [ch[f], *args]
                elif m in frozenset({tasks_module.HGStem, tasks_module.HGBlock}):
                    c1, cm, c2 = ch[f], args[0], args[1]
                    args = [c1, cm, c2, *args[2:]]
                    if m is tasks_module.HGBlock:
                        args.insert(4, n)
                        n = 1
                elif m is tasks_module.ResNetLayer:
                    c2 = args[1] if args[3] else args[1] * 4
                elif m is torch.nn.BatchNorm2d:
                    args = [ch[f]]
                elif m is tasks_module.Concat:
                    c2 = sum(ch[x] for x in f)
                elif m in frozenset(
                    {
                        tasks_module.ConcatHead,
                        tasks_module.Detect,
                        tasks_module.WorldDetect,
                        tasks_module.YOLOEDetect,
                        tasks_module.Segment,
                        tasks_module.YOLOESegment,
                        tasks_module.Pose,
                        tasks_module.OBB,
                        tasks_module.ImagePoolingAttn,
                        tasks_module.v10Detect,
                    }
                ):
                    args.append([ch[x] for x in f])
                    if m is tasks_module.Segment or m is tasks_module.YOLOESegment:
                        args[2] = make_divisible(min(args[2], max_channels) * width, 8)
                    if m in {
                        tasks_module.Detect,
                        tasks_module.YOLOEDetect,
                        tasks_module.Segment,
                        tasks_module.YOLOESegment,
                        tasks_module.Pose,
                        tasks_module.OBB,
                    }:
                        m.legacy = legacy
                elif m is tasks_module.RTDETRDecoder:
                    args.insert(1, [ch[x] for x in f])
                elif m is tasks_module.CBLinear:
                    c2 = args[0]
                    c1 = ch[f]
                    args = [c1, c2, *args[1:]]
                elif m is tasks_module.CBFuse:
                    c2 = ch[f[-1]]
                elif m in frozenset({tasks_module.TorchVision, tasks_module.Index}):
                    c2 = args[0]
                    c1 = ch[f]
                    args = [*args[1:]]
                else:
                    c2 = ch[f]

                module = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
                t = str(m)[8:-2].replace("__main__.", "")
                module.np = sum(x.numel() for x in module.parameters())
                module.i, module.f, module.type = i, f, t
                if verbose:
                    LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{module.np:10.0f}  {t:<45}{str(args):<30}")
                save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
                layers.append(module)
                if i == 0:
                    ch = []
                ch.append(c2)
            return torch.nn.Sequential(*layers), sorted(save)

        tasks_module.parse_model = parse_model_patched  # type: ignore[assignment]

    tasks_module._yolo_def_patched = True


def run_command(command: Sequence[str], cwd: Path | None = None) -> None:
    """Run a shell command and raise if it fails."""
    display_cmd = " ".join(str(part) for part in command)
    print(f"[cmd] {display_cmd} (cwd={cwd or Path.cwd()})")
    subprocess.run(command, cwd=cwd, check=True)


def ensure_multihead_config(cfg_path: Path) -> None:
    """Create the dual-head YOLO11 config if it does not already exist."""
    if cfg_path.exists():
        return

    import ultralytics

    base_cfg = Path(ultralytics.__file__).resolve().parent / "cfg" / "models" / "11" / "yolo11.yaml"
    if not base_cfg.exists():
        raise FileNotFoundError(f"Base YOLO11 config not found: {base_cfg}")

    base_text = base_cfg.read_text()
    modified = base_text.replace("nc: 80 # number of classes", "nc: 82 # number of classes", 1)
    modified = modified.replace(
        "  - [[16, 19, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)",
        "\n".join((YOLO11_COCO_HEAD_LINE, YOLO11_NEW_HEAD_LINE, YOLO11_CONCAT_LINE)),
        1,
    )
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(modified)
    print(f"Created dual-head config at {cfg_path}")


def update_model_cfg(cfg_path: Path, added_classes: int) -> Path:
    """Update the dual-head config with the requested number of new classes."""
    content = cfg_path.read_text()
    total_classes = len(COCO_NAMES) + added_classes
    content, replaced_nc = re.subn(
        r"nc:\s+\d+\s+# number of classes",
        f"nc: {total_classes} # number of classes",
        content,
        count=1,
    )
    if replaced_nc == 0:
        raise ValueError(f"Failed to update class count in {cfg_path}")

    detect_pattern = r"  - \[\[16, 19, 22\], 1, Detect, \[\d+\]\] # Added classes head"
    content, replaced_detect = re.subn(
        detect_pattern,
        f"  - [[16, 19, 22], 1, Detect, [{added_classes}]] # Added classes head",
        content,
        count=1,
    )
    if replaced_detect == 0:
        raise ValueError(f"Failed to update new head definition in {cfg_path}")

    concat_pattern = r"  - \[\[23, 24\], 1, ConcatHead, \[80, \d+\]\] # Merge detection heads"
    content, replaced_concat = re.subn(
        concat_pattern,
        f"  - [[23, 24], 1, ConcatHead, [80, {added_classes}]] # Merge detection heads",
        content,
        count=1,
    )
    if replaced_concat == 0:
        raise ValueError(f"Failed to update ConcatHead definition in {cfg_path}")

    cfg_path.write_text(content)
    print(f"Configured {cfg_path} for {added_classes} new classes ({total_classes} total).")
    return cfg_path


def symlink_or_copy(src: Path, dst: Path) -> None:
    """Create a symlink where possible, otherwise copy the file."""
    if dst.exists():
        return
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def collect_class_ids(label_dir: Path) -> List[int]:
    """Collect sorted class IDs from a YOLO label directory."""
    class_ids = set()
    for label_file in label_dir.glob("*.txt"):
        with label_file.open() as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                class_ids.add(int(stripped.split()[0]))
    return sorted(class_ids)


def prepare_dataset(
    src_dir: Path,
    processed_dir: Path,
    dataset_name: str,
    class_names_override: Sequence[str] | None = None,
) -> Tuple[List[str], Dict[int, int], List[str]]:
    """Remap class IDs to a dense range starting at zero and mirror the dataset."""
    if not src_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {src_dir}")

    image_root = src_dir / "images"
    label_root = src_dir / "labels"
    if not image_root.exists() or not label_root.exists():
        raise FileNotFoundError(f"Expected 'images' and 'labels' directories in {src_dir}")

    splits = [p.name for p in image_root.iterdir() if p.is_dir()]
    if not splits:
        raise ValueError(f"No dataset splits found under {image_root}")

    all_class_ids: set[int] = set()
    for split in splits:
        split_dir = label_root / split
        if split_dir.exists():
            all_class_ids.update(collect_class_ids(split_dir))

    if not all_class_ids:
        raise ValueError(f"No labels found in {label_root}")

    sorted_ids = sorted(all_class_ids)
    class_mapping = {original: idx for idx, original in enumerate(sorted_ids)}

    if class_names_override:
        if len(class_names_override) != len(sorted_ids):
            raise ValueError(
                f"Provided {len(class_names_override)} class names but dataset exposes {len(sorted_ids)} unique IDs."
            )
        class_names = list(class_names_override)
    else:
        slug = dataset_name.replace(" ", "_").replace("-", "_")
        class_names = [slug] if len(sorted_ids) == 1 else [f"{slug}_{idx}" for idx in range(len(sorted_ids))]

    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_images_root = processed_dir / "images"
    processed_labels_root = processed_dir / "labels"

    for split in splits:
        src_images_split = image_root / split
        src_labels_split = label_root / split
        dst_images_split = processed_images_root / split
        dst_labels_split = processed_labels_root / split

        dst_images_split.mkdir(parents=True, exist_ok=True)
        dst_labels_split.mkdir(parents=True, exist_ok=True)

        label_lookup = {p.name: p for p in src_labels_split.glob("*.txt")} if src_labels_split.exists() else {}

        for image_path in sorted(src_images_split.glob("*")):
            if not image_path.is_file():
                continue

            dst_image_path = dst_images_split / image_path.name
            symlink_or_copy(image_path.resolve(), dst_image_path)

            label_filename = image_path.with_suffix(".txt").name
            src_label = label_lookup.get(label_filename)
            dst_label = dst_labels_split / label_filename

            if src_label is None:
                dst_label.write_text("")
                continue

            remapped_lines: List[str] = []
            with src_label.open() as label_handle:
                for line in label_handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    parts = stripped.split()
                    original_cls = int(parts[0])
                    if original_cls not in class_mapping:
                        raise ValueError(f"Unexpected class ID {original_cls} in {src_label}")
                    parts[0] = str(class_mapping[original_cls])
                    remapped_lines.append(" ".join(parts))

            dst_label.write_text("\n".join(remapped_lines) + ("\n" if remapped_lines else ""))

    processed_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = processed_dir / "class_mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "source_dataset": str(src_dir.resolve()),
                "class_id_map": {str(k): v for k, v in class_mapping.items()},
                "class_names": class_names,
            },
            indent=2,
        )
    )

    return class_names, class_mapping, splits


def select_split(splits: Sequence[str], target: str) -> str | None:
    """Pick a dataset split matching the target alias."""
    if target in splits:
        return target
    aliases = {"val": ["val", "valid", "validation"], "test": ["test", "testing"]}
    for alias in aliases.get(target, []):
        if alias in splits:
            return alias
    return None


def write_dataset_yaml(
    yaml_path: Path,
    processed_dir: Path,
    class_names: Sequence[str],
    splits: Sequence[str],
) -> Path:
    """Generate a YOLO data.yaml file for the processed dataset."""
    processed_dir = processed_dir.resolve()
    train_split = select_split(splits, "train")
    val_split = select_split(splits, "val")
    test_split = select_split(splits, "test")

    if train_split is None:
        raise ValueError("Dataset must include a 'train' split")
    if val_split is None:
        raise ValueError("Dataset must include a validation split ('val' or 'valid')")

    lines = [
        "# Auto-generated by src/train.py",
        f"path: {processed_dir}",
        f"train: {processed_dir / 'images' / train_split}",
        f"val: {processed_dir / 'images' / val_split}",
    ]
    if test_split is not None:
        lines.append(f"test: {processed_dir / 'images' / test_split}")
    lines.append(f"nc: {len(class_names)}")
    lines.append("names:")
    for name in class_names:
        lines.append(f"  - {name}")

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(lines) + "\n")
    return yaml_path


def build_full_class_names(new_class_names: Sequence[str]) -> List[str]:
    """Concatenate COCO + new dataset names for the merged model."""
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
    """Train, extract the new head, merge into the dual-head architecture, and export artifacts."""
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

    print("Starting training...")
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
    print(f"Saved remapped head weights to {head_weights_path}")

    merged_model = YOLO(str(config_path), task="detect").load(base_model)

    state_dict = torch.load(head_weights_path, map_location="cpu")
    missing, unexpected = merged_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys when loading new head weights: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys when loading new head weights: {unexpected}")

    merged_model.model.names = {idx: name for idx, name in enumerate(build_full_class_names(new_class_names))}
    merged_model.ckpt = {"model": merged_model.model}

    merged_weights_path = output_dir / f"{Path(base_model).stem}_{dataset_name}_merged.pt"
    merged_model.save(str(merged_weights_path))
    print(f"Merged model saved to {merged_weights_path}")

    # onnx_path: Path | None = None
    # try:
    #     export_path = merged_model.export(
    #         format="onnx",
    #         imgsz=imgsz,
    #         project=str(output_dir),
    #         name=f"{dataset_name}_onnx",
    #         exist_ok=True,
    #     )
    #     onnx_path = Path(export_path)
    #     print(f"Exported ONNX model to {onnx_path}")
    # except ModuleNotFoundError as exc:
    #     print(f"Skipping ONNX export: {exc}")

    # return {
    #     "best_weights": best_weights,
    #     "head_weights": head_weights_path,
    #     "merged_weights": merged_weights_path,
    #     "onnx_path": onnx_path,
    # }


def resolve_path(root: Path, value: Path | str) -> Path:
    """Resolve a (possibly relative) path against the project root."""
    path = Path(value)
    return path if path.is_absolute() else (root / path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Recreate the YOLO multi-head workflow without cloning the repo.")
    default_root = Path(__file__).resolve().parent.parent

    parser.add_argument("--root", type=Path, default=default_root, help="Project root (defaults to repository root).")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/datasets/counterfeit-nike"),
        help="Path to the raw dataset (images/<split>, labels/<split>).",
    )
    parser.add_argument(
        "--processed-dataset-dir",
        type=Path,
        default=Path("data/processed/counterfeit-nike"),
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
    parser.add_argument(
        "--dataset-name",
        default="counterfeit_nike",
        help="Name used for logging/tracking runs.",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        help="Optional override for class names (one per class after remapping).",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training and export image size.")
    parser.add_argument("--freeze", type=int, default=23, help="Number of layers to freeze during training.")
    parser.add_argument(
        "--base-model",
        default="yolo11n.pt",
        help="Pretrained checkpoint to fine-tune and load into the dual-head architecture.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to store training outputs and exported models.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    dataset_dir = resolve_path(root, args.dataset_dir)
    processed_dir = resolve_path(root, args.processed_dataset_dir)
    data_yaml_path = resolve_path(root, args.data_yaml)
    output_dir = resolve_path(root, args.output_dir)
    config_path = resolve_path(root, args.config_path)

    print(f"Project root: {root}")
    print(f"Raw dataset dir: {dataset_dir}")
    print(f"Processed dataset dir: {processed_dir}")
    print(f"Data YAML path: {data_yaml_path}")
    print(f"Output dir: {output_dir}")
    print(f"Dual-head config path: {config_path}")

    ensure_ultralytics_multihead_support()
    ensure_multihead_config(config_path)

    class_names, class_mapping, splits = prepare_dataset(
        dataset_dir, processed_dir, args.dataset_name, args.class_names
    )
    print(f"Detected class ids: {class_mapping}")
    print(f"Using class names: {class_names}")

    write_dataset_yaml(data_yaml_path, processed_dir, class_names, splits)

    added_classes = len(class_names)
    update_model_cfg(config_path, added_classes)

    summary = train_and_merge(
        config_path=config_path,
        data_yaml=data_yaml_path,
        added_classes=added_classes,
        new_class_names=class_names,
        freeze_layers=args.freeze,
        epochs=args.epochs,
        imgsz=args.imgsz,
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        base_model=args.base_model,
    )

    print("\nPipeline completed. Key artifacts:")
    for label, path in summary.items():
        print(f"  - {label}: {path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
