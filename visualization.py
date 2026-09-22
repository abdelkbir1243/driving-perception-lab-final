"""Rendu HUD des détections et préparation des exports."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from config import CLASS_COLORS


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _draw_corner_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int, int],
    width: int,
) -> None:
    x1, y1, x2, y2 = box
    corner = max(10, min(34, int(min(x2 - x1, y2 - y1) * 0.24)))
    segments = [
        ((x1, y1 + corner), (x1, y1), (x1 + corner, y1)),
        ((x2 - corner, y1), (x2, y1), (x2, y1 + corner)),
        ((x1, y2 - corner), (x1, y2), (x1 + corner, y2)),
        ((x2 - corner, y2), (x2, y2), (x2, y2 - corner)),
    ]
    for segment in segments:
        draw.line(segment, fill=color, width=width, joint="curve")


def draw_detections(
    image: Image.Image, detections: list[dict[str, Any]]
) -> Image.Image:
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    image_width, image_height = canvas.size
    thickness = max(2, round(min(image_width, image_height) / 260))
    label_font = _font(max(12, round(min(image_width, image_height) / 42)), bold=True)
    micro_font = _font(max(10, round(min(image_width, image_height) / 55)))

    # Repères discrets de la caméra, inspirés d'un HUD de perception.
    draw.line(
        (image_width // 2, int(image_height * 0.46), image_width // 2, int(image_height * 0.54)),
        fill=(255, 255, 255, 75),
        width=1,
    )
    draw.line(
        (int(image_width * 0.47), image_height // 2, int(image_width * 0.53), image_height // 2),
        fill=(255, 255, 255, 75),
        width=1,
    )

    for index, detection in enumerate(detections, start=1):
        class_name = detection["class_name"]
        rgb = CLASS_COLORS.get(class_name, (255, 255, 255))
        color = (*rgb, 255)
        x1 = max(0, int(round(detection["x1"])))
        y1 = max(0, int(round(detection["y1"])))
        x2 = min(image_width - 1, int(round(detection["x2"])))
        y2 = min(image_height - 1, int(round(detection["y2"])))
        _draw_corner_box(draw, (x1, y1, x2, y2), color, thickness)

        label = f"{index:02d}  {class_name.upper()}  {detection['confidence'] * 100:.0f}%"
        text_box = draw.textbbox((0, 0), label, font=label_font)
        label_width = text_box[2] - text_box[0] + 16
        label_height = text_box[3] - text_box[1] + 10
        label_top = max(0, y1 - label_height - 3)
        label_right = min(image_width - 1, x1 + label_width)
        draw.rounded_rectangle(
            (x1, label_top, label_right, label_top + label_height),
            radius=5,
            fill=(7, 16, 29, 225),
            outline=color,
            width=1,
        )
        draw.text((x1 + 8, label_top + 4), label, fill=(245, 249, 255, 255), font=label_font)

    status = f"PERCEPTION ACTIVE  ·  {len(detections):02d} OBJECTS"
    status_box = draw.textbbox((0, 0), status, font=micro_font)
    status_width = status_box[2] - status_box[0] + 20
    draw.rounded_rectangle((12, 12, 12 + status_width, 40), radius=7, fill=(5, 15, 28, 210))
    draw.ellipse((21, 21, 29, 29), fill=(47, 224, 164, 255))
    draw.text((36, 18), status, fill=(221, 233, 245, 255), font=micro_font)
    return Image.alpha_composite(canvas, overlay).convert("RGB")


def detections_dataframe(
    detections: list[dict[str, Any]], small_ratio: float
) -> pd.DataFrame:
    rows = []
    for index, detection in enumerate(detections, start=1):
        rows.append(
            {
                "ID": f"OBJ-{index:02d}",
                "Classe": detection["class_name"],
                "Confiance": detection["confidence"],
                "Zone": detection.get("zone", "—"),
                "Échelle visuelle": detection.get("visual_scale", "—"),
                "X1": round(detection["x1"], 1),
                "Y1": round(detection["y1"], 1),
                "X2": round(detection["x2"], 1),
                "Y2": round(detection["y2"], 1),
                "Surface image (%)": round(detection["area_ratio"] * 100, 3),
                "Petit objet": detection["area_ratio"] < small_ratio,
            }
        )
    return pd.DataFrame(rows)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
