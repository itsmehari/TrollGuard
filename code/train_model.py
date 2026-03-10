"""
================================================================================
TrollGuard – Train and Save Model (Standalone Script)
================================================================================

WHAT THIS FILE DOES (in simple terms):
This script trains the bullying-detection model from scratch and saves it to
the models/ folder. Run this BEFORE using the Streamlit app if you don't have
model files yet.

WORKFLOW:
1. Load all datasets from dataset/ (or use sample data if empty)
2. Clean each message (remove URLs, @mentions, etc.)
3. Split data: 80% for training, 20% for testing
4. Convert text to numbers using TF-IDF (Term Frequency - Inverse Document Frequency)
5. Train Logistic Regression (a simple but effective classifier)
6. Evaluate accuracy and print a report
7. Save the vectoriser and model to models/tfidf.joblib and models/logreg_model.joblib

WHEN TO RUN:
- First time setup (no model exists)
- After adding new datasets
- To refresh the model

Run from the code/ folder:  python train_model.py

================================================================================
"""

import os
import sys

# ----- PATH SETUP -----
# Make sure Python can find the core/ package. ROOT = folder containing train_model.py
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
    # ----- STEP 1: Load data -----
    df = load_parsed_datasets()
    if df.empty:
        df = get_sample_fallback()
        print("Using sample fallback (no CSVs found)")

    # ----- STEP 2: Preprocess -----
    # Apply clean_text to each message. The model was trained on cleaned text,
    # so we must clean new data the same way.
    df["clean_text"] = df["text"].apply(clean_text)
    X = df["clean_text"].values  # Features (the messages)
    y = df["label"].values       # Labels (0 or 1)

    # ----- STEP 3: Split into train and test -----
    # 80% for training, 20% for testing. stratify=y keeps the same proportion
    # of bullying vs non-bullying in both sets (important for imbalanced data).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training TF-IDF + Logistic Regression...")

    # ----- STEP 4: TF-IDF Vectorisation -----
    # Converts text to numbers. ngram_range=(1,2) means we use single words
    # and pairs of words (e.g. "shut up"). min_df=2 ignores rare words.
    # max_df=0.95 ignores words that appear in 95%+ of documents (too common).
    vectoriser = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_train_vec = vectoriser.fit_transform(X_train)  # Learn vocabulary + transform
    X_test_vec = vectoriser.transform(X_test)        # Use same vocabulary

    # ----- STEP 5: Train classifier -----
    # Logistic Regression: fast, interpretable, works well for text classification.
    # max_iter=300 allows enough iterations to converge.
    model = LogisticRegression(max_iter=300)
    model.fit(X_train_vec, y_train)

    # ----- STEP 6: Evaluate -----
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(acc, 4))
    print(classification_report(y_test, y_pred))

    # ----- STEP 7: Save to disk -----
    ad = get_artefact_dir()
    joblib.dump(vectoriser, os.path.join(ad, "tfidf.joblib"))
    joblib.dump(model, os.path.join(ad, "logreg_model.joblib"))
    print("Model saved to", ad)


if __name__ == "__main__":
    main()
