"""Découverte, validation et reconstitution du checkpoint de déploiement."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path


MANIFEST_NAME = "checkpoint_manifest.json"


@dataclass(frozen=True)
class CheckpointReport:
    ready: bool
    mode: str
    message: str
    files: tuple[str, ...]
    total_bytes: int = 0


def checkpoint_parts(path: Path) -> list[Path]:
    return sorted(path.parent.glob(f"{path.name}.part[0-9][0-9][0-9]"))


def _load_manifest(path: Path) -> dict | None:
    manifest_path = path.parent / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _manifest_parts(path: Path, manifest: dict) -> tuple[list[Path], str | None]:
    """Retourne les morceaux dans l'ordre déclaré et une erreur éventuelle."""
    records = manifest.get("parts")
    if not isinstance(records, list) or not records:
        return [], "Manifeste invalide : liste des morceaux absente."

    ordered: list[Path] = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            return [], "Manifeste invalide : description d'un morceau incorrecte."
        part = path.parent / record["name"]
        if not part.is_file():
            return [], f"Checkpoint incomplet : {part.name} est absent."
        expected_bytes = record.get("bytes")
        if not isinstance(expected_bytes, int) or part.stat().st_size != expected_bytes:
            return [], f"Checkpoint corrompu : taille incorrecte pour {part.name}."
        ordered.append(part)

    detected = {part.name for part in checkpoint_parts(path)}
    expected = {part.name for part in ordered}
    unexpected = sorted(detected - expected)
    if unexpected:
        return [], "Morceaux inattendus : " + ", ".join(unexpected)
    return ordered, None


def inspect_checkpoint(path: Path) -> CheckpointReport:
    if path.is_file():
        size = path.stat().st_size
        if size < 1_000_000:
            return CheckpointReport(False, "invalid", "Checkpoint anormalement petit.", (path.name,), size)
        with path.open("rb") as handle:
            if handle.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
                return CheckpointReport(False, "lfs", "GitHub contient un pointeur LFS, pas le modèle réel.", (path.name,), size)
        return CheckpointReport(True, "single", "Checkpoint complet disponible.", (path.name,), size)

    manifest = _load_manifest(path)
    parts = checkpoint_parts(path)
    if not parts:
        return CheckpointReport(
            False,
            "missing",
            "Poids du modèle absents du dossier models/.",
            (),
        )

    if not manifest:
        return CheckpointReport(
            False,
            "manifest_missing",
            "Manifeste checkpoint_manifest.json absent ou invalide.",
            tuple(part.name for part in parts),
            sum(part.stat().st_size for part in parts),
        )

    parts, manifest_error = _manifest_parts(path, manifest)
    if manifest_error:
        detected_parts = checkpoint_parts(path)
        return CheckpointReport(
            False,
            "incomplete",
            manifest_error,
            tuple(part.name for part in detected_parts),
            sum(part.stat().st_size for part in detected_parts),
        )

    expected_total = manifest.get("total_bytes")
    total = sum(part.stat().st_size for part in parts)
    if not isinstance(expected_total, int) or total != expected_total:
        return CheckpointReport(
            False,
            "size_mismatch",
            "Checkpoint corrompu : taille totale différente du manifeste.",
            tuple(part.name for part in parts),
            total,
        )

    return CheckpointReport(
        True,
        "parts",
        f"Checkpoint prêt · {len(parts)} morceaux vérifiés.",
        tuple(part.name for part in parts),
        total,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(path: Path) -> Path:
    """Retourne le checkpoint réel, en fusionnant ses morceaux dans /tmp."""
    report = inspect_checkpoint(path)
    if not report.ready:
        raise FileNotFoundError(report.message)
    if report.mode == "single":
        return path

    manifest = _load_manifest(path)
    if manifest is None:
        raise RuntimeError("Manifeste checkpoint_manifest.json absent ou invalide.")
    parts, manifest_error = _manifest_parts(path, manifest)
    if manifest_error:
        raise RuntimeError(manifest_error)
    signature = "|".join(
        f"{part.name}:{part.stat().st_size}:{part.stat().st_mtime_ns}"
        for part in parts
    )
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
    merged = Path(tempfile.gettempdir()) / f"driving_perception_{digest}.pt"
    expected_size = sum(part.stat().st_size for part in parts)

    if not (merged.is_file() and merged.stat().st_size == expected_size):
        temporary = merged.with_suffix(".tmp")
        with temporary.open("wb") as output:
            for part in parts:
                with part.open("rb") as source:
                    while chunk := source.read(8 * 1024 * 1024):
                        output.write(chunk)
        temporary.replace(merged)

    if manifest:
        expected_hash = manifest.get("sha256")
        if expected_hash and _sha256(merged) != expected_hash:
            merged.unlink(missing_ok=True)
            raise RuntimeError("Échec de l'intégrité SHA-256 du checkpoint reconstitué.")
    return merged
