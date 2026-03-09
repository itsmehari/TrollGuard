"""
TrollGuard – Cyberbullying Detection (Streamlit App)

Run: streamlit run app.py
"""

import os
import sys

# Ensure project root is on path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
import numpy as np
from core.data_loader import (
    get_datasets_dir,
    load_parsed_datasets,
    get_sample_fallback,
    load_chat_file,
)
from core.model_utils import (
    clean_text,
    load_model_and_vectoriser,
    predict_text,
    predict_batch,
    get_artefact_dir,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

st.set_page_config(page_title="TrollGuard", page_icon="🛡️", layout="wide")

st.title("🛡️ TrollGuard – Cyberbullying Detector")
st.caption("NLP-based text classification for bullying detection")

tfidf, model = load_model_and_vectoriser()

# Sidebar: model status and train
with st.sidebar:
    st.header("Model")
    if tfidf is not None and model is not None:
        st.success("Model loaded")
        if st.button("Retrain model"):
            st.session_state.train_requested = True
    else:
        st.warning("No model found. Train from data.")
        st.session_state.train_requested = True

    st.header("Datasets")
    ds_dir = get_datasets_dir()
    st.text(f"Data: {ds_dir}")
    df_raw = load_parsed_datasets()
    if df_raw.empty:
        df_raw = get_sample_fallback()
        st.info("Using sample fallback data")
    st.metric("Samples", len(df_raw))

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Predict", "💬 Chat analysis", "📊 Train model", "ℹ️ About"])

# Tab 1: Single/batch prediction
with tab1:
    st.subheader("Text prediction")
    text_input = st.text_area("Enter text to classify", placeholder="Type or paste a message...")
    if text_input.strip():
        if not (tfidf and model):
            st.warning("Model not loaded. Go to **Train model** tab to train.")
        else:
            pred = predict_text(text_input, tfidf, model)
            if pred == 1:
                st.error("Prediction: **Bullying** (flagged)")
            else:
                st.success("Prediction: **Non-bullying** (safe)")

    st.divider()
    st.subheader("Batch upload (CSV)")
    csv_file = st.file_uploader("Upload CSV with 'text' or 'Text' column", type="csv")
    if csv_file:
        try:
            up = pd.read_csv(csv_file)
            col = "text" if "text" in up.columns else "Text"
            if col not in up.columns:
                st.error("CSV must have 'text' or 'Text' column")
            else:
                texts = up[col].fillna("").astype(str).tolist()
                if not (tfidf and model):
                    st.warning("Model not loaded. Train in **Train model** tab first.")
                else:
                    preds = predict_batch(texts, tfidf, model)
                    if len(preds) == len(texts):
                        up = up.copy()
                        up["predicted_label"] = preds
                        up["prediction"] = up["predicted_label"].map({0: "Non-bullying", 1: "Bullying"})
                        st.dataframe(up, use_container_width=True)
                        st.download_button("Download results (CSV)", up.to_csv(index=False).encode("utf-8"),
                                           "trollguard_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

# Tab 2: Chat export
with tab2:
    st.subheader("Chat export analysis (WhatsApp format)")
    st.caption("Format: DD/MM/YYYY, HH:MM - Sender: message")
    chat_df = load_chat_file()
    if chat_df.empty:
        st.info("No sample_chat.txt found. Upload a chat export or add one to dataset/sample_chat.txt")
        chat_upload = st.file_uploader("Or upload chat .txt", type=["txt"])
        if chat_upload:
            content = chat_upload.read().decode("utf-8", errors="ignore")
            tmp_path = os.path.join(get_artefact_dir(), "_uploaded_chat.txt")
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(content)
            chat_df = load_chat_file(tmp_path)
    if not chat_df.empty and (tfidf and model):
        chat_df["clean_text"] = chat_df["message_text"].apply(clean_text)
        preds = predict_batch(chat_df["clean_text"].tolist(), tfidf, model)
        chat_df["bullying_label"] = preds
        chat_df["result"] = chat_df["bullying_label"].map({0: "Safe", 1: "Flagged"})
        st.dataframe(chat_df[["timestamp", "sender", "message_text", "result"]], use_container_width=True)
        summary = chat_df.groupby("sender")["bullying_label"].agg(["count", "sum"]).reset_index()
        summary.columns = ["Sender", "Total messages", "Flagged"]
        summary["Flagged %"] = (summary["Flagged"] / summary["Total messages"] * 100).round(1)
        st.subheader("Per-sender summary")
        st.dataframe(summary, use_container_width=True)

# Tab 3: Train
with tab3:
    st.subheader("Train model")
    df = load_parsed_datasets()
    if df.empty:
        df = get_sample_fallback()
        st.info("Using sample fallback data (no CSVs found)")
    df["clean_text"] = df["text"].apply(clean_text)
    X = df["clean_text"].values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    if st.button("Train TF-IDF + Logistic Regression"):
        with st.spinner("Training..."):
            vectoriser = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
            X_train_vec = vectoriser.fit_transform(X_train)
            X_test_vec = vectoriser.transform(X_test)
            mdl = LogisticRegression(max_iter=300)
            mdl.fit(X_train_vec, y_train)
            y_pred = mdl.predict(X_test_vec)
            acc = accuracy_score(y_test, y_pred)
            st.success(f"Accuracy: {acc:.2%}")
            st.text(classification_report(y_test, y_pred))
            cm = confusion_matrix(y_test, y_pred)
            st.write("Confusion matrix:")
            st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))
            ad = get_artefact_dir()
            joblib.dump(vectoriser, os.path.join(ad, "tfidf.joblib"))
            joblib.dump(mdl, os.path.join(ad, "logreg_model.joblib"))
            st.info("Model saved. Refresh the page to use it.")

# Tab 4: About
with tab4:
    st.subheader("About TrollGuard")
    st.write("""
    TrollGuard is a text-based cyberbullying detection system using TF-IDF features and Logistic Regression.
    
    - **Predict**: Enter single text or upload a CSV with a text column.
    - **Chat analysis**: Analyse WhatsApp-style chat exports.
    - **Train**: Train the model on your dataset (`*_parsed_dataset.csv` files).
    
    Dataset format: CSV with columns `Text` and `oh_label` (0 = non-bullying, 1 = bullying).
    """)
    st.caption("B.Sc. Computer Science Final Year Project")
