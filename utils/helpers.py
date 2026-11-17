"""Shared UI, formatting, and feature-naming helpers for PredictIQ."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import streamlit as st

from utils.config import RISK_THRESHOLDS, STYLES_PATH


def load_css(styles_path: Path = STYLES_PATH) -> None:
    """Inject custom enterprise CSS into the Streamlit application."""
    if styles_path.exists():
        css = styles_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str) -> None:
    """Render a consistent SAS-Viya-inspired page header."""
    safe_title = html.escape(title)
    safe_subtitle = html.escape(subtitle)
    st.markdown(
        f"""
        <div class="page-header">
            <h1>{safe_title}</h1>
            <p>{safe_subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, subtext: str = "") -> None:
    """Render a structured KPI card with optional subtext."""
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    safe_sub = html.escape(str(subtext))
    sub_html = f'<div class="kpi-sub">{safe_sub}</div>' if subtext else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{safe_label}</div>
            <div class="kpi-value">{safe_value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def classify_risk(
    probability: float,
    low_max: float = RISK_THRESHOLDS["low_max"],
    high_min: float = RISK_THRESHOLDS["high_min"],
) -> str:
    """Classify a churn probability into Low Risk, Medium Risk, or High Risk."""
    prob = float(probability)
    if prob < low_max:
        return "Low Risk"
    if prob < high_min:
        return "Medium Risk"
    return "High Risk"


def risk_badge_html(risk_category: str) -> str:
    """Return styled HTML badge for a risk category."""
    css_class = {
        "Low Risk": "badge-low",
        "Medium Risk": "badge-medium",
        "High Risk": "badge-high",
    }.get(risk_category, "badge-medium")
    return f'<span class="badge {css_class}">{html.escape(risk_category)}</span>'


def format_currency(amount: float) -> str:
    """Format a numeric value as USD currency."""
    return f"${amount:,.2f}"


def format_percent(ratio: float, decimals: int = 1) -> str:
    """Format a 0-1 ratio as a percentage string."""
    return f"{ratio * 100:.{decimals}f}%"


def clean_feature_name(raw_name: str) -> str:
    """Convert ColumnTransformer output feature names into readable labels."""
    name = str(raw_name)
    if name.startswith("num__"):
        return name[len("num__") :]
    if name.startswith("cat__"):
        remainder = name[len("cat__") :]
        if "_" in remainder:
            col, val = remainder.split("_", 1)
            return f"{col}: {val}"
        return remainder
    return name


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float safely, returning default on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
