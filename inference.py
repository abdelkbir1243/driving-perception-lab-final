"""Prétraitement, inférence et post-traitement des détections."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from config import CLASS_NAMES, INPUT_HEIGHT, INPUT_WIDTH


@dataclass(frozen=True)
class LetterboxMetadata:
    scale: float
    pad_left: int
    pad_top: int
    resized_width: int
    resized_height: int
    original_width: int
    original_height: int


def letterbox_image(
    image: Image.Image,
    target_width: int = INPUT_WIDTH,
    target_height: int = INPUT_HEIGHT,
) -> tuple[Image.Image, LetterboxMetadata]:
    image = image.convert("RGB")
    original_width, original_height = image.size
    if original_width <= 0 or original_height <= 0:
        raise ValueError("L'image possède une résolution invalide.")

    scale = min(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))
    resized = image.resize(
        (resized_width, resized_height), Image.Resampling.BILINEAR
    )

    pad_left = (target_width - resized_width) // 2
    pad_top = (target_height - resized_height) // 2
    canvas = Image.new("RGB", (target_width, target_height), color=(0, 0, 0))
    canvas.paste(resized, (pad_left, pad_top))
    metadata = LetterboxMetadata(
        scale=float(scale),
        pad_left=pad_left,
        pad_top=pad_top,
        resized_width=resized_width,
        resized_height=resized_height,
        original_width=original_width,
        original_height=original_height,
    )
    return canvas, metadata


def _boxes_to_original(
    boxes: np.ndarray, metadata: LetterboxMetadata
) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32)
    converted = boxes.astype(np.float32, copy=True)
    converted[:, [0, 2]] = (
        converted[:, [0, 2]] - metadata.pad_left
    ) / metadata.scale
    converted[:, [1, 3]] = (
        converted[:, [1, 3]] - metadata.pad_top
    ) / metadata.scale
    converted[:, [0, 2]] = np.clip(
        converted[:, [0, 2]], 0, metadata.original_width - 1
    )
    converted[:, [1, 3]] = np.clip(
        converted[:, [1, 3]], 0, metadata.original_height - 1
    )
    return converted


def run_inference(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
) -> dict[str, Any]:
    """Exécute une seule inférence et conserve toutes les sorties >= 0.05."""

    original = image.convert("RGB")
    prepared, metadata = letterbox_image(original)
    tensor = pil_to_tensor(prepared).float().div(255.0).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        prediction = model([tensor])[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    boxes = prediction["boxes"].detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy().astype(int)
    scores = prediction["scores"].detach().cpu().numpy()
    boxes = _boxes_to_original(boxes, metadata)

    valid = (
        (scores >= 0.05)
        & np.isfinite(boxes).all(axis=1)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
    )
    detections = []
    image_area = float(original.width * original.height)
    for box, label, score in zip(boxes[valid], labels[valid], scores[valid]):
        if label not in CLASS_NAMES or label == 0:
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        area_ratio = ((x2 - x1) * (y2 - y1)) / image_area
        center_x = ((x1 + x2) / 2.0) / original.width
        zone = "Gauche" if center_x < 0.34 else "Droite" if center_x > 0.66 else "Centre"
        visual_scale = (
            "Proche" if area_ratio >= 0.08 else "Intermédiaire" if area_ratio >= 0.02 else "Lointain / petit"
        )
        detections.append(
            {
                "label_id": int(label),
                "class_name": CLASS_NAMES[int(label)],
                "confidence": float(score),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": x2 - x1,
                "height": y2 - y1,
                "area_ratio": float(area_ratio),
                "zone": zone,
                "visual_scale": visual_scale,
            }
        )

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return {
        "detections": detections,
        "inference_seconds": elapsed,
        "fps": 1.0 / elapsed if elapsed > 0 else 0.0,
        "device": "CUDA" if device.type == "cuda" else "CPU",
        "original_width": original.width,
        "original_height": original.height,
    }


def filter_detections(
    detections: list[dict[str, Any]],
    confidence_threshold: float,
    selected_classes: list[str],
) -> list[dict[str, Any]]:
    selected = set(selected_classes)
    return [
        detection
        for detection in detections
        if detection["confidence"] >= confidence_threshold
        and detection["class_name"] in selected
    ]
