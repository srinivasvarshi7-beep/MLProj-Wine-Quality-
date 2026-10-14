"""Helper functions for the Wine Quality app: loading files, predicting, and styling charts."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "WineQT.csv"
MODEL_PATH = BASE_DIR / "models" / "wine_quality_model.json"
META_PATH = BASE_DIR / "models" / "model_metadata.json"
NOTEBOOK_PATH = BASE_DIR / "models" / "notebook_results.json"

# Colours used in all charts so they match the dark theme
BG = "#1E1E24"
PANEL = "#2A2A32"
TEXT = "#F5F5F5"
RED = "#E5383B"


def style_chart(fig, ax=None):
    """Make a matplotlib figure match the dark theme."""
    fig.patch.set_facecolor(BG)
    axes = [ax] if ax is not None else fig.axes
    for a in axes:
        a.set_facecolor(PANEL)
        a.tick_params(colors=TEXT)
        a.xaxis.label.set_color(TEXT)
        a.yaxis.label.set_color(TEXT)
        a.title.set_color(TEXT)
        for spine in a.spines.values():
            spine.set_color("#555560")
    return fig


@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load the original WineQT.csv (including the 'Id' column, as in the notebook)."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_json(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path) as f:
        return json.load(f)


@st.cache_resource
def load_model():
    """Load the XGBoost model saved in native JSON format."""
    from xgboost import XGBClassifier
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Run `python train_model.py` first.")
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


def predict_quality(model, meta: dict, values: dict):
    """
    Predict wine quality from a dict {feature name: value}.
    Returns (predicted quality score, {quality score: probability}).
    """
    features = meta["features"]
    missing = [f for f in features if f not in values]
    if missing:
        raise ValueError(f"Missing input values: {missing}")
    # Build a 1-row DataFrame in the exact training column order
    row = pd.DataFrame([[float(values[f]) for f in features]], columns=features)
    if row.isna().any().any() or not np.isfinite(row.values).all():
        raise ValueError("All inputs must be finite numbers.")
    proba = model.predict_proba(row)[0]
    if len(proba) != len(meta["classes"]):
        raise ValueError("Model output does not match the saved label mapping.")
    idx = int(np.argmax(proba))
    return meta["classes"][idx], {c: float(p) for c, p in zip(meta["classes"], proba)}
