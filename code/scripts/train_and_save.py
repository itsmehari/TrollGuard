"""
Retrain TrollGuard model with current scikit-learn (fixes version mismatch).
Run: python scripts/train_and_save.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from core.data_loader import load_parsed_datasets, get_sample_fallback
from core.model_utils import clean_text, get_artefact_dir
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = load_parsed_datasets()
if df.empty:
    df = get_sample_fallback()
# Use max 30k for faster retrain (full data = 540k)
if len(df) > 30_000:
    df = df.sample(30_000, random_state=42)
df["clean_text"] = df["text"].apply(clean_text)
X = df["clean_text"].values
y = df["label"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectoriser = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
X_train_vec = vectoriser.fit_transform(X_train)
X_test_vec = vectoriser.transform(X_test)
mdl = LogisticRegression(max_iter=300)
mdl.fit(X_train_vec, y_train)
acc = accuracy_score(y_test, mdl.predict(X_test_vec))
print(f"Accuracy: {acc:.2%}")

ad = get_artefact_dir()
joblib.dump(vectoriser, os.path.join(ad, "tfidf.joblib"))
joblib.dump(mdl, os.path.join(ad, "logreg_model.joblib"))
print(f"Model saved to {ad}")
