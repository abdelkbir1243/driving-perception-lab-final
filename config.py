"""Configuration centralisée du démonstrateur Streamlit."""

from pathlib import Path


APP_TITLE = "Driving Perception Lab"
APP_SUBTITLE = "Driving-Aware I-JEPA · Visual Perception Console"
APP_DESCRIPTION = (
    "Analyse interactive de scènes routières par détection d'objets 2D."
)
MODEL_VERSION = "E2 · Best checkpoint · Epoch 35"

PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINTS = {
    "Driving-Aware I-JEPA": PROJECT_DIR / "models" / "ijepa" / "best_detector.pt",
    "ImageNet baseline": PROJECT_DIR / "models" / "imagenet" / "best_detector.pt",
}
DEFAULT_CHECKPOINT = CHECKPOINTS["Driving-Aware I-JEPA"]

INPUT_WIDTH = 1024
INPUT_HEIGHT = 320
PATCH_SIZE = 16
EMBED_DIM = 384

CLASS_NAMES = {
    0: "__background__",
    1: "Pedestrian",
    2: "Cyclist",
    3: "Car",
    4: "Van",
}

DISPLAY_CLASSES = ["Pedestrian", "Cyclist", "Car", "Van"]

# Couleurs RGB utilisées sur l'image annotée.
CLASS_COLORS = {
    "Pedestrian": (255, 92, 122),
    "Cyclist": (255, 190, 74),
    "Car": (47, 224, 164),
    "Van": (72, 166, 255),
}

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

DEFAULT_CONFIDENCE = 0.50
IOU_DISPLAY_VALUE = 0.50
DEFAULT_SMALL_OBJECT_RATIO = 0.01

# Performances mesurées sur les 1 497 images de validation KITTI,
# avec le meilleur checkpoint de l'époque 35.
VALIDATION_METRICS = {
    "Pedestrian": {
        "AP@50": 0.726110,
        "Recall E2": 0.799107,
        "Precision@0.50": 0.718468,
        "Recall@0.50": 0.712054,
        "F1@0.50": 0.715247,
    },
    "Cyclist": {
        "AP@50": 0.835158,
        "Recall E2": 0.869281,
        "Precision@0.50": 0.800000,
        "Recall@0.50": 0.797386,
        "F1@0.50": 0.798691,
    },
    "Car": {
        "AP@50": 0.932775,
        "Recall E2": 0.944014,
        "Precision@0.50": 0.887526,
        "Recall@0.50": 0.916901,
        "F1@0.50": 0.901974,
    },
    "Van": {
        "AP@50": 0.910020,
        "Recall E2": 0.936057,
        "Precision@0.50": 0.808989,
        "Recall@0.50": 0.895204,
        "F1@0.50": 0.849916,
    },
}

COMPARISON_METRICS = {
    "ImageNet baseline": {
        "Best epoch": 25,
        "Pedestrian AP@50": 0.794920,
        "Pedestrian Recall": 0.879464,
    },
    "Driving-Aware I-JEPA": {
        "Best epoch": 35,
        "Pedestrian AP@50": 0.726110,
        "Pedestrian Recall": 0.799107,
    },
}
