"""Petits composants visuels HTML sans dépendance externe."""

from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st


def load_css() -> None:
    css_path = Path(__file__).resolve().parent / "style.css"
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def hero(model_ready: bool) -> None:
    model_class = "" if model_ready else " warning"
    model_text = "Modèle prêt" if model_ready else "Modèle à installer"
    st.markdown(
        f"""
        <section class="app-hero">
          <div class="eyebrow">Autonomous driving · perception stack</div>
          <h1>Driving Perception Lab</h1>
          <p>Analysez une scène routière avec le détecteur final Driving-Aware I-JEPA,
          visualisez les objets perçus et inspectez les indicateurs de la scène.</p>
          <div class="status-row">
            <span class="status-chip{model_class}"><span class="dot"></span>{model_text}</span>
            <span class="status-chip">ViT-S/16</span>
            <span class="status-chip">Faster R-CNN</span>
            <span class="status-chip">KITTI · 1024 × 320</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{escape(text)}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, detail: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="label">{escape(label)}</div>
          <div class="value">{escape(value)}</div>
          <div class="detail">{escape(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def system_card(message: str, ready: bool) -> None:
    state = "ok" if ready else "error"
    title = "Système opérationnel" if ready else "Checkpoint requis"
    st.markdown(
        f'<div class="system-card {state}"><strong>{title}</strong><br>{escape(message)}</div>',
        unsafe_allow_html=True,
    )
