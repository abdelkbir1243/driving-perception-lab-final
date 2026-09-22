"""Validation rapide du dépôt avant envoi sur GitHub."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch

from checkpoint_files import checkpoint_parts, inspect_checkpoint, resolve_checkpoint
from config import CHECKPOINTS
from model import load_model


def main() -> None:
    root = Path(__file__).resolve().parent
    required = [
        root / name
        for name in (
            "app.py",
            "model.py",
            "inference.py",
            "checkpoint_files.py",
            "style.css",
            "requirements.txt",
        )
    ]
    missing = [str(path.name) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Fichiers manquants : {', '.join(missing)}")
    print(f"Python : {sys.version.split()[0]}")
    print(f"PyTorch : {torch.__version__}")
    if sys.version_info[:2] != (3, 12):
        print(
            "NOTE : cet environnement sert à préparer les modèles. "
            "Dans Streamlit Cloud, sélectionnez toujours Python 3.12."
        )
    for name, checkpoint_source in CHECKPOINTS.items():
        report = inspect_checkpoint(checkpoint_source)
        if not report.ready:
            raise RuntimeError(f"{name} — {report.message}")
        checkpoint = resolve_checkpoint(checkpoint_source)
        size_mb = checkpoint.stat().st_size / (1024 * 1024)
        parts = checkpoint_parts(checkpoint_source)
        print(f"\n{name}")
        print(f"Morceaux GitHub : {len(parts)}")
        print(f"Checkpoint : {size_mb:.1f} MiB")
        print("Intégrité du manifeste : OK")
        model = load_model(checkpoint, torch.device("cpu"))
        print(f"Architecture chargée : {type(model.backbone).__name__}")
        del model
        gc.collect()
    print("VALIDATION RÉUSSIE — dépôt prêt pour Streamlit.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"VALIDATION ÉCHOUÉE — {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
