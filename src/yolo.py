import copy
import traceback
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from ultralytics import YOLO
from pathlib import Path
import yaml
import torch
import logging

COCO_NAMES: tuple[str, ...] = (
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

class YOLO11n():
    def __init__(
        self,
        base_model_path: str | Path = "yolo11n.pt",
        config_path: str | Path | None = None,
        freeze_layers: int = 23,
        new_class_names: Sequence[str] | None = None,
    ):
        self._base_model_path = Path(base_model_path)
        self._config_path = Path(config_path) if config_path else None
        self._freeze_layers = freeze_layers
        self._new_class_names = list(new_class_names) if new_class_names else []
        self._head_module_index: int | None = None
        self._model = self._load_model()
        self._head_module_index = self._detect_last_head_index()

    def _load_model(self):
        try:
            if self._config_path and self._config_path.exists():
                return YOLO(str(self._config_path), task="detect").load(str(self._base_model_path))
        except Exception:
            logging.error(traceback.format_exc())
        return YOLO(str(self._base_model_path))

    def _detect_last_head_index(self) -> int | None:
        modules = getattr(self._model.model, "model", None)
        if modules is None:
            return None
        detect_indices = [
            idx for idx, module in enumerate(modules)
            if module.__class__.__name__ == "Detect"
        ]
        return detect_indices[-1] if detect_indices else None

    def load_state_dict(self, params, new_class_names: Sequence[str] | None = None):
        if new_class_names is not None:
            self._new_class_names = list(new_class_names)
        try:
            current_state = self._model.model.state_dict()
            param_keys = set(params.keys())
            current_keys = set(current_state.keys())

            unknown_keys = param_keys - current_keys

            if unknown_keys:
                self._update_head(params, self._freeze_layers)
            else:
                current_state.update(params)
                self._model.model.load_state_dict(current_state, strict=False)
        except Exception as e:
            logging.error(traceback.format_exc())

    def get_model_parameters(self):
        head_index = self._head_module_index
        if head_index is None:
            head_index = self._detect_last_head_index()
            self._head_module_index = head_index
        if head_index is None:
            return {}

        state_dict = self._model.model.state_dict()
        prefix = f"model.{head_index}"
        head_items: Dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            if not name.startswith(prefix):
                continue
            adjusted_name = name.replace("model.", "model.model.", 1)
            head_items[adjusted_name] = tensor.clone()
        return head_items
    
    # def get_model_weight(self):
    #     #TODO ver cuantos samples hay en el dataset
    #     pass
    
    def _update_head(self, head_state, freeze_layers):
        """
        Actualiza el modelo local añadiendo una nueva cabeza.
        head_state: dict con los pesos de la cabeza (enviados desde otro nodo)
        freeze_layers: número de capas congeladas (para localizar la cabeza)
        """

        try:
            base_model_path = str(self._base_model_path)
            if self._config_path and self._config_path.exists():
                merged_model = YOLO(str(self._config_path), task="detect").load(base_model_path)
            else:
                merged_model = YOLO(base_model_path)

            merged_model.model.load_state_dict(self._model.model.state_dict(), strict=False)

            remapped_head = {
                key.replace("model.model.", "model.", 1): value
                for key, value in head_state.items()
            }

            missing, unexpected = merged_model.model.load_state_dict(remapped_head, strict=False)
            if missing:
                logging.debug("Missing keys during head merge: {0}".format(missing))
            if unexpected:
                logging.debug("Unexpected keys during head merge: {0}".format(unexpected))

            if hasattr(merged_model.model, "freeze"):
                merged_model.model.freeze(freeze_layers)
            else:
                for idx, module in enumerate(getattr(merged_model.model, "model", [])[:freeze_layers]):
                    for param in module.parameters():
                        param.requires_grad = False

            custom_names = self._new_class_names if self._new_class_names else ["custom"]
            merged_model.model.names = {
                idx: name for idx, name in enumerate(self.build_full_class_names(custom_names))
            }
            merged_model.ckpt = {"model": merged_model.model}

            self._model = merged_model
            self._head_module_index = self._detect_last_head_index()

        except Exception as e:
            logging.error(traceback.format_exc())
                
    # def train(self):
    #     self._train_and_merge()    
        
    def build_full_class_names(self, new_class_names: Sequence[str]) -> list[str]:
        """Concatenate the default COCO names with the new dataset-specific names."""

        return list(COCO_NAMES) + list(new_class_names) 
