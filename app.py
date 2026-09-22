"""Driving Perception Lab — comparaison ImageNet et Driving-Aware I-JEPA."""

from __future__ import annotations

import hashlib
import json
import gc
from io import BytesIO

import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from checkpoint_files import inspect_checkpoint, resolve_checkpoint
from config import (
    APP_TITLE,
    CHECKPOINTS,
    COMPARISON_METRICS,
    DEFAULT_CONFIDENCE,
    DEFAULT_SMALL_OBJECT_RATIO,
    DISPLAY_CLASSES,
    VALIDATION_METRICS,
)
from inference import filter_detections, run_inference
from model import load_model
from ui import hero, load_css, metric_card, section_label, system_card
from visualization import detections_dataframe, draw_detections, image_to_png_bytes


IJEPA = "Driving-Aware I-JEPA"
IMAGENET = "ImageNet baseline"
COMPARE = "Comparaison côte à côte"


st.set_page_config(page_title=APP_TITLE, page_icon="◉", layout="wide")
load_css()


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner=False)
def load_model_cached(path: str, device_name: str, signature: int):
    del signature
    return load_model(path, torch.device(device_name))


def count_class(detections: list[dict], name: str) -> int:
    return sum(item["class_name"] == name for item in detections)


def load_uploaded_image(uploaded_file) -> tuple[Image.Image | None, str | None, str | None]:
    if uploaded_file is None:
        return None, None, None
    try:
        payload = uploaded_file.getvalue()
        if len(payload) > 20 * 1024 * 1024:
            return None, None, "L'image dépasse 20 MiB."
        candidate = Image.open(BytesIO(payload))
        candidate.verify()
        image = Image.open(BytesIO(payload)).convert("RGB")
        image.load()
        return image, hashlib.sha256(payload).hexdigest(), None
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return None, None, f"Image invalide ou illisible : {exc}"


def requested_models(mode: str) -> list[str]:
    return [IJEPA, IMAGENET] if mode == COMPARE else [mode]


def render_diagnostic(name: str, report) -> None:
    st.markdown(f"**{name}**")
    system_card(report.message, report.ready)
    with st.expander(f"Diagnostic · {name}", expanded=False):
        st.write(f"Mode : `{report.mode}`")
        if report.total_bytes:
            st.write(f"Taille : `{report.total_bytes / 1024**2:.1f} MiB`")
        if report.files:
            st.code("\n".join(report.files))
        else:
            st.caption("Exécutez le notebook Colab pour ajouter ce checkpoint.")


def filtered_result(result: dict, threshold: float, classes: list[str]) -> list[dict]:
    return filter_detections(result["detections"], threshold, classes)


def render_scene_metrics(detections: list[dict], result: dict, small_ratio: float) -> None:
    pedestrians = count_class(detections, "Pedestrian")
    cyclists = count_class(detections, "Cyclist")
    vehicles = count_class(detections, "Car") + count_class(detections, "Van")
    small_objects = sum(item["area_ratio"] < small_ratio for item in detections)
    cards = [
        ("Objets", str(len(detections)), "après filtrage"),
        ("Usagers vulnérables", str(pedestrians + cyclists), f"{pedestrians} piéton · {cyclists} cycliste"),
        ("Véhicules", str(vehicles), "cars + vans"),
        ("Petits objets", str(small_objects), "surface apparente"),
        ("Latence", f"{result['inference_seconds'] * 1000:.0f} ms", f"{result['fps']:.2f} FPS · {result['device']}"),
    ]
    for column, card in zip(st.columns(5, gap="small"), cards):
        with column:
            metric_card(*card)


def render_registry(name: str, image: Image.Image, detections: list[dict], small_ratio: float) -> None:
    dataframe = detections_dataframe(detections, small_ratio)
    if dataframe.empty:
        st.info("Aucun objet ne dépasse les filtres sélectionnés.")
    else:
        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Confiance": st.column_config.ProgressColumn(
                    "Confiance", min_value=0.0, max_value=1.0, format="%.3f"
                )
            },
        )
    slug = "ijepa" if name == IJEPA else "imagenet"
    annotated = draw_detections(image, detections)
    columns = st.columns(3)
    columns[0].download_button(
        "Image annotée",
        image_to_png_bytes(annotated),
        file_name=f"{slug}_perception.png",
        mime="image/png",
        key=f"png_{slug}",
        use_container_width=True,
    )
    columns[1].download_button(
        "Registre CSV",
        dataframe.to_csv(index=False).encode("utf-8"),
        file_name=f"{slug}_detections.csv",
        mime="text/csv",
        key=f"csv_{slug}",
        disabled=dataframe.empty,
        use_container_width=True,
    )
    columns[2].download_button(
        "Détections JSON",
        json.dumps(detections, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name=f"{slug}_detections.json",
        mime="application/json",
        key=f"json_{slug}",
        use_container_width=True,
    )


device = choose_device()
reports = {name: inspect_checkpoint(path) for name, path in CHECKPOINTS.items()}

with st.sidebar:
    st.markdown("### ◉ Perception Console")
    st.caption("Deux représentations, un protocole identique")
    st.divider()
    mode = st.radio(
        "Mode d'évaluation",
        [IJEPA, IMAGENET, COMPARE],
        index=0,
        help="Le mode comparaison exécute les deux modèles sur la même image.",
    )
    st.divider()
    confidence_threshold = st.slider(
        "Seuil de confiance", 0.10, 0.95, DEFAULT_CONFIDENCE, 0.05
    )
    selected_classes = st.multiselect(
        "Classes visibles", DISPLAY_CLASSES, default=DISPLAY_CLASSES
    )
    small_ratio_percent = st.slider(
        "Seuil petit objet (% image)",
        0.1,
        5.0,
        DEFAULT_SMALL_OBJECT_RATIO * 100,
        0.1,
    )
    small_ratio = small_ratio_percent / 100.0
    st.divider()
    st.markdown("#### État des modèles")
    for name in (IJEPA, IMAGENET):
        icon = "✅" if reports[name].ready else "❌"
        st.caption(f"{icon} {name}")
    st.caption(f"Device · {'CUDA' if device.type == 'cuda' else 'CPU'}")
    st.divider()
    st.caption("Démonstrateur académique — aucune décision de conduite réelle.")

required = requested_models(mode)
mode_ready = all(reports[name].ready for name in required)
hero(mode_ready)

section_label("01 · Protocole d'essai")
upload_column, status_column = st.columns([1.05, 0.95], gap="large")
with upload_column:
    uploaded_file = st.file_uploader(
        "Déposer une image de scène routière",
        type=None,
        help="JPG, JPEG, PNG, JFIF ou WEBP · 20 MiB maximum",
    )
    st.caption(f"Mode actif : **{mode}** · même image, même résolution et mêmes filtres.")
with status_column:
    diagnostic_columns = st.columns(2)
    for column, name in zip(diagnostic_columns, (IJEPA, IMAGENET)):
        with column:
            render_diagnostic(name, reports[name])

image, image_digest, image_error = load_uploaded_image(uploaded_file)
if image_error:
    st.error(image_error)
if image_digest and st.session_state.get("image_digest") != image_digest:
    st.session_state["image_digest"] = image_digest
    st.session_state.pop("inference_results", None)

analyze_clicked = st.button(
    "Exécuter le protocole de perception",
    type="primary",
    use_container_width=True,
    disabled=image is None or not mode_ready,
)

if analyze_clicked and image is not None:
    try:
        outputs = dict(st.session_state.get("inference_results", {}))
        progress = st.progress(0, text="Préparation des modèles…")
        if mode == COMPARE:
            # La comparaison charge un seul modèle à la fois. Cela évite de
            # conserver deux détecteurs lourds dans la RAM de Streamlit Cloud.
            load_model_cached.clear()
            gc.collect()
        for index, name in enumerate(required, start=1):
            progress.progress((index - 1) / len(required), text=f"Analyse · {name}")
            checkpoint = resolve_checkpoint(CHECKPOINTS[name])
            if mode == COMPARE:
                model = load_model(checkpoint, device)
            else:
                model = load_model_cached(
                    str(checkpoint), device.type, checkpoint.stat().st_mtime_ns
                )
            outputs[name] = run_inference(model, image, device)
            if mode == COMPARE:
                del model
                gc.collect()
        progress.progress(1.0, text="Analyse terminée")
        progress.empty()
        st.session_state["inference_results"] = outputs
        st.toast("Protocole terminé", icon="✅")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            st.error("Mémoire insuffisante. Testez les modèles séparément sur cette instance.")
        else:
            st.error(f"Échec de l'inférence : {exc}")
    except Exception as exc:
        st.error(f"Impossible de lancer l'analyse : {exc}")

results = st.session_state.get("inference_results", {})
has_results = image is not None and all(name in results for name in required)

if image is None:
    section_label("02 · Vue capteur")
    st.info("Importez une scène routière pour commencer le test.")
elif not has_results:
    section_label("02 · Scène chargée")
    st.image(image, caption=f"Image source · {image.width} × {image.height}", use_container_width=True)
    if not mode_ready:
        missing = [name for name in required if not reports[name].ready]
        st.warning("Checkpoint requis : " + ", ".join(missing))
else:
    filtered = {
        name: filtered_result(results[name], confidence_threshold, selected_classes)
        for name in required
    }
    annotated = {name: draw_detections(image, filtered[name]) for name in required}
    section_label("02 · Résultat du test")

    if mode == COMPARE:
        left, right = st.columns(2, gap="medium")
        with left:
            st.markdown(f"#### {IJEPA}")
            st.image(annotated[IJEPA], use_container_width=True)
        with right:
            st.markdown(f"#### {IMAGENET}")
            st.image(annotated[IMAGENET], use_container_width=True)

        section_label("03 · Comparaison sur la scène")
        comparison_rows = []
        for name in (IJEPA, IMAGENET):
            detections = filtered[name]
            comparison_rows.append(
                {
                    "Modèle": name,
                    "Objets": len(detections),
                    "Piétons": count_class(detections, "Pedestrian"),
                    "Cyclistes": count_class(detections, "Cyclist"),
                    "Cars": count_class(detections, "Car"),
                    "Vans": count_class(detections, "Van"),
                    "Latence (ms)": round(results[name]["inference_seconds"] * 1000),
                }
            )
        st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, use_container_width=True)
        st.caption(
            "Une différence de nombre de boxes sur une image ne désigne pas automatiquement un meilleur modèle : "
            "la qualité doit être jugée avec les annotations de vérité terrain et les métriques du jeu de validation."
        )
    else:
        st.markdown(f"#### {mode}")
        view = st.radio(
            "Affichage", ["Perception IA", "Image originale"], horizontal=True, label_visibility="collapsed"
        )
        st.image(annotated[mode] if view == "Perception IA" else image, use_container_width=True)
        section_label("03 · Télémétrie de la scène")
        render_scene_metrics(filtered[mode], results[mode], small_ratio)

    registry_tab, validation_tab, protocol_tab = st.tabs(
        ["Registre des objets", "Résultats de validation", "Protocole scientifique"]
    )
    with registry_tab:
        selected_registry = (
            st.radio("Modèle affiché", [IJEPA, IMAGENET], horizontal=True)
            if mode == COMPARE
            else mode
        )
        render_registry(selected_registry, image, filtered[selected_registry], small_ratio)

    with validation_tab:
        comparison_table = pd.DataFrame.from_dict(COMPARISON_METRICS, orient="index")
        comparison_table.index.name = "Pré-entraînement"
        st.dataframe(comparison_table.style.format(precision=4), use_container_width=True)
        st.info(
            "Sur KITTI, la baseline ImageNet obtient le meilleur AP@50 Pedestrian. "
            "Driving-Aware I-JEPA reste la contribution SSL étudiée ; l'objectif de l'application "
            "est d'observer et comparer les comportements, pas de déclarer une supériorité artificielle."
        )
        with st.expander("Métriques détaillées du modèle final I-JEPA"):
            detailed = pd.DataFrame.from_dict(VALIDATION_METRICS, orient="index")
            detailed.index.name = "Classe"
            st.dataframe(detailed.style.format("{:.4f}"), use_container_width=True)

    with protocol_tab:
        st.markdown(
            """
            #### Comparaison contrôlée
            - même image et même letterbox 1024 × 320 ;
            - même Faster R-CNN, mêmes anchors et même SFP P2–P6 ;
            - mêmes classes et même seuil de confiance ;
            - seule l'initialisation du ViT-S/16 diffère : ImageNet supervisé ou Driving-Aware I-JEPA.

            Les sorties affichées sont des observations qualitatives. Une comparaison quantitative
            exige les annotations KITTI et le calcul AP/Recall sur tout le split de validation.
            """
        )

st.divider()
st.caption(
    "PFE · LAMNAOUAR Abdelkabir · ImageNet baseline vs Driving-Aware I-JEPA · "
    "Analyse expérimentale uniquement."
)
