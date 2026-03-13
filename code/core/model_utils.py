"""
================================================================================
TrollGuard – Text Cleaning and Model Utilities
================================================================================

WHAT THIS FILE DOES (in simple terms):
This is the "engine room" of TrollGuard. It contains:
  1. clean_text() - Prepares raw messages for the model (removes noise)
  2. get_artefact_dir() - Tells us where to save/load the model files
  3. load_model_and_vectoriser() - Loads the trained model from disk
  4. predict_text() - Classifies a single message as 0 (safe) or 1 (bullying)
  5. predict_batch() - Classifies many messages at once (faster)
  6. get_top_words() - Shows which words the model associates with bullying vs safe

ANALOGY: If the model is a "judge", this file contains:
  - The rules for cleaning evidence (clean_text)
  - Where the judge's notes are stored (get_artefact_dir, load_model)
  - The judge's verdict for one or many cases (predict_text, predict_batch)
  - The judge's reasoning (get_top_words)

================================================================================
"""

import re
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .data_loader import get_project_root

# Model registry: id -> (display_name, model_class, **kwargs)
MODEL_REGISTRY = {
    "logreg": ("Logistic Regression", LogisticRegression, {"max_iter": 300}),
    "nb": ("Naive Bayes", MultinomialNB, {}),
    "svm": ("SVM (Linear)", SGDClassifier, {"loss": "hinge", "max_iter": 1000, "random_state": 42}),
    "rf": ("Random Forest", RandomForestClassifier, {"n_estimators": 100, "random_state": 42}),
}


def clean_text(text: str) -> str:
    """
    Clean raw text so the model can process it properly.

    WHY CLEANING MATTERS:
    Real messages have URLs, @mentions, #hashtags, emojis, extra spaces.
    The model was trained on cleaned text. If we feed it "Check out http://spam.com @user"
    without cleaning, those tokens add noise and can hurt accuracy.

    WHAT WE REMOVE:
    - URLs (http://..., www....) - not useful for bullying detection
    - @mentions - usernames don't help; we care about the words
    - #hashtags - same reason
    - Anything that's not a letter or space - punctuation, numbers, emojis

    WHAT WE KEEP:
    - Letters (a-z) converted to lowercase
    - Single spaces between words (multiple spaces collapsed to one)

    Example: "You're SO stupid!! @john http://x.com" -> "you re so stupid"
    """
    text = str(text).lower()
    text = re.sub(r"https?://\S+", "", text)   # Remove http:// and https:// links
    text = re.sub(r"www\.\S+", "", text)       # Remove www. links
    text = re.sub(r"@\w+", "", text)           # Remove @username
    text = re.sub(r"#\w+", "", text)           # Remove #hashtag
    text = re.sub(r"[^a-z ]", " ", text)       # Replace anything not a-z or space with space
    text = re.sub(r"\s+", " ", text).strip()   # Collapse multiple spaces, trim edges
    return text


def get_artefact_dir() -> str:
    """
    Return the path to the models/ folder where we save and load model files.

    "Artefact" = something we produce (the trained model). We create the folder
    if it doesn't exist (makedirs with exist_ok=True).

    Returns: Full path like "C:\\...\\code\\models"
    """
    root = get_project_root()
    path = os.path.join(root, "models")
    os.makedirs(path, exist_ok=True)
    return path


def create_model(model_id: str):
    """
    Create an untrained classifier from model_id.
    Returns sklearn-compatible classifier (has .fit, .predict).
    """
    if model_id not in MODEL_REGISTRY:
        model_id = "logreg"
    _, model_class, kwargs = MODEL_REGISTRY[model_id]
    return model_class(**kwargs)


def load_model_and_vectoriser():
    """
    Load the trained TF-IDF vectoriser and classifier from disk.
    Supports new format (classifier.joblib) and legacy (logreg_model.joblib).
    Returns: (tfidf, model) if found, (None, None) if not found or load fails
    """
    import warnings
    root = get_project_root()
    for folder in ("models", "outputs"):
        tf_path = os.path.join(root, folder, "tfidf.joblib")
        mdl_path = os.path.join(root, folder, "classifier.joblib")
        if not os.path.exists(mdl_path):
            mdl_path = os.path.join(root, folder, "logreg_model.joblib")
        if os.path.exists(tf_path) and os.path.exists(mdl_path):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # Hide sklearn version mismatch warnings
                    tfidf = joblib.load(tf_path)
                    model = joblib.load(mdl_path)
                # CRITICAL: Ensure TF-IDF and model were trained together (same vocabulary size).
                # Mismatch causes "X has N features, but LogisticRegression expects M" on predict.
                try:
                    X_dummy = tfidf.transform(["test"])
                    n_features = X_dummy.shape[1]
                    n_expected = getattr(model, "n_features_in_", None)
                    if n_expected is not None and n_features != n_expected:
                        return None, None
                except Exception:
                    return None, None
                return tfidf, model
            except Exception:
                pass
    return None, None


def predict_text(text: str, tfidf, model) -> int:
    """
    Predict whether a single message is bullying (1) or not (0).

    STEP BY STEP:
    1. Check we have valid input and a loaded model
    2. Clean the text (remove URLs, etc.)
    3. Convert text to numbers using TF-IDF (same way as training)
    4. Ask the model for a prediction
    5. Return 0 or 1

    Returns: 0 = non-bullying (safe), 1 = bullying (flagged)
             Returns 0 for empty text or if model not loaded (fail-safe)
    """
    if not text or (tfidf is None or model is None):
        return 0
    cleaned = clean_text(text)
    if not cleaned.strip():
        return 0
    # transform expects a list; we pass [cleaned] for one item
    X = tfidf.transform([cleaned])
    return int(model.predict(X)[0])


def predict_batch(texts: list, tfidf, model) -> np.ndarray:
    """
    Predict labels for many messages at once.

    WHY BATCH IS FASTER:
    Processing 1000 messages one-by-one means 1000 separate TF-IDF transforms.
    Processing 1000 at once = 1 transform. Much more efficient!

    Returns: Numpy array of 0s and 1s, same length as input list
             Returns empty array if input empty or model not loaded
    """
    if not texts or (tfidf is None or model is None):
        return np.array([])
    cleaned = [clean_text(t) for t in texts]
    X = tfidf.transform(cleaned)
    return model.predict(X)


def get_top_words(tfidf, model, top_n: int = 15) -> tuple:
    """
    Find which words the model associates with bullying vs safe messages.

    Supports: Logistic Regression, SVM (coef_), Naive Bayes (feature_log_prob_),
    Random Forest (feature_importances_).

    Returns: (bullying_words, safe_words)
             Each is a list of (word, score) tuples
    """
    if tfidf is None or model is None:
        return [], []
    # get_feature_names_out (sklearn>=1.0); fallback for older sklearn
    fn = getattr(tfidf, "get_feature_names_out", None) or getattr(tfidf, "get_feature_names", None)
    if fn is None:
        return [], []
    vocab = np.asarray(fn())
    coef = None
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_[0])
    elif hasattr(model, "feature_log_prob_"):
        # NB: log P(word|bullying) - log P(word|safe) as "coefficient"
        lp = model.feature_log_prob_
        if lp.shape[0] >= 2:
            coef = lp[1] - lp[0]
        else:
            coef = lp[0]
    elif hasattr(model, "feature_importances_"):
        coef = np.asarray(model.feature_importances_)
    if coef is None:
        return [], []
    n = min(len(vocab), len(coef))
    if n == 0:
        return [], []
    vocab, coef = vocab[:n], coef[:n]
    idx_sorted = np.argsort(coef)
    non_bullying = [(str(vocab[i]), float(coef[i])) for i in idx_sorted[:top_n]]
    bullying = [(str(vocab[i]), float(coef[i])) for i in idx_sorted[-top_n:][::-1]]
    return bullying, non_bullying
