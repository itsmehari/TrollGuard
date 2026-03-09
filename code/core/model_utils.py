"""
TrollGuard – Text Cleaning and Model Utilities
"""

import re
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .data_loader import get_project_root


def clean_text(text: str) -> str:
    """Clean raw text for TF-IDF (lowercase, remove URLs, mentions, hashtags, non-alpha)."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_artefact_dir() -> str:
    """Return path for saving model (models/ – committed for Streamlit Cloud)."""
    root = get_project_root()
    path = os.path.join(root, "models")
    os.makedirs(path, exist_ok=True)
    return path


def load_model_and_vectoriser():
    """Load tfidf.joblib and logreg_model.joblib. Checks models/ then outputs/."""
    root = get_project_root()
    for folder in ("models", "outputs"):
        tf_path = os.path.join(root, folder, "tfidf.joblib")
        mdl_path = os.path.join(root, folder, "logreg_model.joblib")
        if os.path.exists(tf_path) and os.path.exists(mdl_path):
            try:
                return joblib.load(tf_path), joblib.load(mdl_path)
            except Exception:
                pass
    return None, None


def predict_text(text: str, tfidf, model) -> int:
    """Predict label (0 or 1) for a single text."""
    if not text or (tfidf is None or model is None):
        return 0
    cleaned = clean_text(text)
    if not cleaned.strip():
        return 0
    X = tfidf.transform([cleaned])
    return int(model.predict(X)[0])


def predict_batch(texts: list, tfidf, model) -> np.ndarray:
    """Predict labels for a list of texts."""
    if not texts or (tfidf is None or model is None):
        return np.array([])
    cleaned = [clean_text(t) for t in texts]
    X = tfidf.transform(cleaned)
    return model.predict(X)
