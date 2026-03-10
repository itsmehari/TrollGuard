"""
================================================================================
TrollGuard – Retrain Model for scikit-learn Version Compatibility
================================================================================

WHAT THIS FILE DOES (in simple terms):
Sometimes a model trained with an older version of scikit-learn fails to load
on a newer version (e.g. on Streamlit Cloud). This script retrains the model
with the CURRENT version of scikit-learn and saves fresh .joblib files.

OPTIMISATION:
We limit training to 30,000 samples (even if you have 540K). This makes
retraining much faster (a few minutes instead of 20+ minutes) while still
producing a usable model.

WHEN TO RUN:
- After deploying to Streamlit Cloud and seeing "model load failed"
- When you get sklearn compatibility warnings
- When you want to refresh the model quickly

Run from the code/ folder:  python scripts/train_and_save.py

================================================================================
"""

import os
import sys

# ----- PATH SETUP -----
# scripts/train_and_save.py -> parent = code/ (project root for our structure)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # So dataset/, models/ resolve correctly

from core.data_loader import load_parsed_datasets, get_sample_fallback
from core.model_utils import clean_text, get_artefact_dir
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load data
df = load_parsed_datasets()
if df.empty:
    df = get_sample_fallback()

# Limit to 30k rows for faster retrain (full dataset can be ~540k)
# sample() randomly picks rows; random_state=42 for reproducibility
if len(df) > 30_000:
    df = df.sample(30_000, random_state=42)

# Preprocess and prepare
df["clean_text"] = df["text"].apply(clean_text)
X = df["clean_text"].values
y = df["label"].values

# 80/20 split, stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF + Logistic Regression (same settings as train_model.py)
vectoriser = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
X_train_vec = vectoriser.fit_transform(X_train)
X_test_vec = vectoriser.transform(X_test)
mdl = LogisticRegression(max_iter=300)
mdl.fit(X_train_vec, y_train)

acc = accuracy_score(y_test, mdl.predict(X_test_vec))
print(f"Accuracy: {acc:.2%}")

# Save to models/
ad = get_artefact_dir()
joblib.dump(vectoriser, os.path.join(ad, "tfidf.joblib"))
joblib.dump(mdl, os.path.join(ad, "logreg_model.joblib"))
print(f"Model saved to {ad}")
