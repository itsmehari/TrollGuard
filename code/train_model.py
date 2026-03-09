"""
TrollGuard – Train and save model (run before first Streamlit use or deployment).

Usage: python train_model.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.data_loader import load_parsed_datasets, get_sample_fallback
from core.model_utils import clean_text, get_artefact_dir
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def main():
    df = load_parsed_datasets()
    if df.empty:
        df = get_sample_fallback()
        print("Using sample fallback (no CSVs found)")
    df["clean_text"] = df["text"].apply(clean_text)
    X = df["clean_text"].values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Training TF-IDF + Logistic Regression...")
    vectoriser = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_train_vec = vectoriser.fit_transform(X_train)
    X_test_vec = vectoriser.transform(X_test)
    model = LogisticRegression(max_iter=300)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(acc, 4))
    print(classification_report(y_test, y_pred))
    ad = get_artefact_dir()
    joblib.dump(vectoriser, os.path.join(ad, "tfidf.joblib"))
    joblib.dump(model, os.path.join(ad, "logreg_model.joblib"))
    print("Model saved to", ad)

if __name__ == "__main__":
    main()
