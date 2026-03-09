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


def get_top_words(tfidf, model, top_n: int = 15) -> tuple:
    """
    Get top bullying and non-bullying indicative words from LogisticRegression.
    Returns (bullying_words, non_bullying_words) as lists of (word, coef).
    Guards against vocab/coef length mismatch (e.g. from sklearn version differences).
    """
    if tfidf is None or model is None or not hasattr(model, "coef_"):
        return [], []
    vocab = np.asarray(tfidf.get_feature_names_out())
    coef = np.asarray(model.coef_[0])
    n = min(len(vocab), len(coef))
    if n == 0:
        return [], []
    vocab, coef = vocab[:n], coef[:n]
    idx_sorted = np.argsort(coef)
    non_bullying = [(str(vocab[i]), float(coef[i])) for i in idx_sorted[:top_n]]
    bullying = [(str(vocab[i]), float(coef[i])) for i in idx_sorted[-top_n:][::-1]]
    return bullying, non_bullying
