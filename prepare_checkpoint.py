"""Crée un checkpoint de déploiement compact à partir du checkpoint Colab."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch


CHUNK_SIZE = 20 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "detector_state_dict", "state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                return candidate
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Aucun state_dict de détecteur trouvé dans le checkpoint.")


def identify_architecture(state: Mapping[str, torch.Tensor]) -> str:
    keys = tuple(state.keys())
    if "backbone.vit.cls_token" in keys:
        return "ImageNet ViT-S/16 timm + SFP P2-P6 + Faster R-CNN"
    if any(key.startswith("backbone.p2_up_1.") for key in keys):
        return "Driving-Aware I-JEPA ViT-S/16 + SFP P2-P6 + Faster R-CNN"
    if any(key.startswith("backbone.body.vit.") for key in keys):
        return "Driving-Aware I-JEPA ViT-S/16 + legacy FPN + Faster R-CNN"
    raise ValueError("Architecture non reconnue : checkpoint non compatible avec cette application.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="best_detector.pt produit par Colab")
    parser.add_argument(
        "destination",
        type=Path,
        nargs="?",
        default=Path("models/best_detector.pt"),
    )
    args = parser.parse_args()

    if not args.source.is_file():
        raise FileNotFoundError(args.source)

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    state = extract_state_dict(checkpoint)
    architecture = identify_architecture(state)
    compact = {
        key: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
        for key, value in state.items()
    }
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    for old_part in args.destination.parent.glob(f"{args.destination.name}.part*"):
        old_part.unlink()
    metadata = {}
    if isinstance(checkpoint, Mapping):
        for key in (
            "backbone_name",
            "epoch",
            "best_ap50_pedestrian",
            "classes",
            "resolution",
        ):
            if key in checkpoint:
                metadata[key] = checkpoint[key]
    torch.save(
        {
            "model_state_dict": compact,
            "metadata": metadata,
            "format": "streamlit-fp16-storage-v2",
        },
        args.destination,
    )

    total_bytes = args.destination.stat().st_size
    size_mb = total_bytes / (1024 * 1024)
    print(f"Checkpoint créé : {args.destination}")
    print(f"Taille : {size_mb:.1f} MiB")

    checksum = sha256_file(args.destination)
    part_records = []
    if args.destination.stat().st_size > CHUNK_SIZE:
        with args.destination.open("rb") as source:
            index = 1
            while chunk := source.read(CHUNK_SIZE):
                part = args.destination.with_name(f"{args.destination.name}.part{index:03d}")
                part.write_bytes(chunk)
                part_records.append({"name": part.name, "bytes": part.stat().st_size})
                print(f"Morceau : {part.name} ({part.stat().st_size / 1024**2:.1f} MiB)")
                index += 1
        args.destination.unlink()
        print("OK : morceaux compatibles avec l'upload web GitHub, sans Git LFS.")
    else:
        part_records.append({"name": args.destination.name, "bytes": args.destination.stat().st_size})
        print("OK : fichier compatible avec l'upload web GitHub.")

    manifest = {
        "format": "driving-perception-checkpoint-v2",
        "architecture": architecture,
        "storage_dtype": "float16",
        "runtime_dtype": "float32",
        "sha256": checksum,
        "total_bytes": total_bytes,
        "parts": part_records,
    }
    manifest_path = args.destination.parent / "checkpoint_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest : {manifest_path}")


if __name__ == "__main__":
    main()
