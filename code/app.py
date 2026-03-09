"""
TrollGuard – Cyberbullying Detection (Streamlit App)

Run: streamlit run app.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import streamlit as st
import pandas as pd
import numpy as np
from core.data_loader import (
    get_datasets_dir,
    load_parsed_datasets,
    get_sample_fallback,
)
from core.chat_parser import parse_chat_from_string
from core.model_utils import (
    clean_text,
    load_model_and_vectoriser,
    predict_text,
    predict_batch,
    get_artefact_dir,
    get_top_words,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Config
MAX_INPUT_CHARS = 50_000
BATCH_CSV_LIMIT = 10_000

st.set_page_config(page_title="TrollGuard", page_icon="🛡️", layout="wide")


@st.cache_resource
def _load_model():
    """Lazy-load model (reduces Streamlit Cloud memory at startup)."""
    return load_model_and_vectoriser()


# Session state for prediction counts
if "pred_total" not in st.session_state:
    st.session_state.pred_total = 0
if "pred_flagged" not in st.session_state:
    st.session_state.pred_flagged = 0

st.title("🛡️ TrollGuard – Cyberbullying Detector")
st.caption("NLP-based text classification for bullying detection")

tfidf, model = _load_model()

# Sidebar: model, datasets, stats
with st.sidebar:
    st.header("Model")
    if tfidf is not None and model is not None:
        st.success("Model loaded")
        if st.button("Retrain model"):
            st.session_state.train_requested = True
    else:
        st.warning("No model found. Train from data.")
        st.session_state.train_requested = True

    st.divider()
    st.header("Session stats")
    st.metric("Predictions (this session)", st.session_state.pred_total)
    st.metric("Flagged", st.session_state.pred_flagged)

    if tfidf and model:
        with st.expander("Feature importance"):
            bull, safe = get_top_words(tfidf, model, 8)
            st.caption("Bullying: " + ", ".join(w for w, _ in bull[:5]))
            st.caption("Safe: " + ", ".join(w for w, _ in safe[:5]))
    st.header("Datasets")
    ds_dir = get_datasets_dir()
    st.text(f"Data: {ds_dir}")
    df_raw = load_parsed_datasets()
    if df_raw.empty:
        df_raw = get_sample_fallback()
        st.info("Using sample fallback data")
    st.metric("Samples", len(df_raw))

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Predict", "💬 Chat analysis", "📊 Train model", "ℹ️ About"])

# Tab 1: Single / paste / batch prediction
with tab1:
    st.subheader("Text prediction")
    text_input = st.text_area("Enter text to classify", placeholder="Type or paste a message...", max_chars=MAX_INPUT_CHARS)
    if text_input.strip():
        text_input = text_input[:MAX_INPUT_CHARS]
        if not (tfidf and model):
            st.warning("Model not loaded. Go to **Train model** tab to train.")
        else:
            pred = predict_text(text_input, tfidf, model)
            st.session_state.pred_total += 1
            if pred == 1:
                st.session_state.pred_flagged += 1
                st.error("Prediction: **Bullying** (flagged)")
            else:
                st.success("Prediction: **Non-bullying** (safe)")

    st.divider()
    st.subheader("Paste text analysis")
    st.caption("Paste multiple lines; each non-empty line will be analyzed separately.")
    paste_input = st.text_area("Paste text to analyze", placeholder="Paste several messages here, one per line...", key="paste_text", max_chars=MAX_INPUT_CHARS)
    if paste_input.strip() and (tfidf and model):
        paste_input = paste_input[:MAX_INPUT_CHARS]
        lines = [ln.strip() for ln in paste_input.strip().splitlines() if ln.strip()]
        if lines:
            preds = predict_batch(lines, tfidf, model)
            st.session_state.pred_total += len(lines)
            st.session_state.pred_flagged += int(sum(preds))
            paste_df = pd.DataFrame({"Text": lines, "Prediction": ["Non-bullying" if p == 0 else "Bullying" for p in preds]})
            with st.expander("View results", expanded=True):
                st.dataframe(paste_df, width="stretch")
            flagged = sum(1 for p in preds if p == 1)
            st.caption(f"Total: {len(lines)} lines · Flagged as bullying: {flagged}")
    elif paste_input.strip() and not (tfidf and model):
        st.warning("Model not loaded. Train in **Train model** tab first.")

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
                n_texts = len(texts)
                if n_texts > BATCH_CSV_LIMIT:
                    st.warning(f"CSV has {n_texts:,} rows. Processing first {BATCH_CSV_LIMIT:,} only.")
                    texts = texts[:BATCH_CSV_LIMIT]
                if not (tfidf and model):
                    st.warning("Model not loaded. Train in **Train model** tab first.")
                else:
                    progress = st.progress(0)
                    chunk = 500
                    preds = []
                    for i in range(0, len(texts), chunk):
                        batch = texts[i:i + chunk]
                        preds.extend(predict_batch(batch, tfidf, model))
                        progress.progress(min(1.0, (i + len(batch)) / len(texts)))
                    progress.empty()
                    st.session_state.pred_total += len(preds)
                    st.session_state.pred_flagged += int(sum(preds))
                    up = up.iloc[:len(preds)].copy()
                    up["predicted_label"] = preds
                    up["prediction"] = up["predicted_label"].map({0: "Non-bullying", 1: "Bullying"})
                    with st.expander("View results", expanded=True):
                        st.dataframe(up, width="stretch")
                    st.download_button("Download results (CSV)", up.to_csv(index=False).encode("utf-8"),
                                       "trollguard_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

# Tab 2: Chat export (upload or paste only – no preload)
with tab2:
    st.subheader("Chat export analysis")
    format_hint = st.selectbox("Chat format", ["auto", "whatsapp", "telegram", "discord"], format_func=lambda x: {"auto": "Auto-detect", "whatsapp": "WhatsApp", "telegram": "Telegram", "discord": "Discord"}[x])

    chat_df = pd.DataFrame(columns=["timestamp", "sender", "message_text"])

    mode = st.radio("Choose input", ["Upload file", "Paste chat text"], horizontal=True)
    if mode == "Upload file":
        chat_upload = st.file_uploader("Upload chat .txt", type=["txt"])
        if chat_upload:
            content = chat_upload.read().decode("utf-8", errors="ignore")[:MAX_INPUT_CHARS]
            chat_df = parse_chat_from_string(content, format_hint)
    else:
        pasted = st.text_area("Paste your chat export below", placeholder="09/03/2025, 14:30 - Alice: Hello...\n09/03/2025, 14:31 - Bob: Hi there", max_chars=MAX_INPUT_CHARS, key="chat_paste")
        if pasted.strip():
            chat_df = parse_chat_from_string(pasted[:MAX_INPUT_CHARS], format_hint)

    if chat_df.empty:
        st.info("Upload a .txt file or paste chat text above to analyze.")
    elif not (tfidf and model):
        st.warning("Model not loaded. Go to **Train model** tab first.")
    else:
        chat_df = chat_df.copy()
        if not chat_df.empty:
            date_col = pd.to_datetime(chat_df["timestamp"], errors="coerce")
            valid_dates = date_col.dropna()
            if not valid_dates.empty:
                col1, col2 = st.columns(2)
                with col1:
                    start_d = st.date_input("From date", value=valid_dates.min().date(), key="chat_start")
                with col2:
                    end_d = st.date_input("To date", value=valid_dates.max().date(), key="chat_end")
                if start_d and end_d:
                    mask = (date_col.dt.date >= start_d) & (date_col.dt.date <= end_d)
                    chat_df = chat_df.loc[mask]
        chat_df["clean_text"] = chat_df["message_text"].apply(clean_text)
        preds = predict_batch(chat_df["clean_text"].tolist(), tfidf, model)
        chat_df["bullying_label"] = preds
        chat_df["result"] = chat_df["bullying_label"].map({0: "Safe", 1: "Flagged"})
        st.session_state.pred_total += len(preds)
        st.session_state.pred_flagged += int(sum(preds))
        with st.expander("Message-level results", expanded=True):
            st.dataframe(chat_df[["timestamp", "sender", "message_text", "result"]], width="stretch")
        summary = chat_df.groupby("sender")["bullying_label"].agg(["count", "sum"]).reset_index()
        summary.columns = ["Sender", "Total messages", "Flagged"]
        summary["Flagged %"] = (summary["Flagged"] / summary["Total messages"] * 100).round(1)
        st.subheader("Per-sender summary")
        st.dataframe(summary, width="stretch")
        export_df = chat_df[["timestamp", "sender", "message_text", "result"]]
        st.download_button("Export chat analysis (CSV)", export_df.to_csv(index=False).encode("utf-8"), "trollguard_chat_analysis.csv", "text/csv", key="chat_export")

# Tab 3: Train
with tab3:
    st.subheader("Train model")
    df = load_parsed_datasets()
    if df.empty:
        df = get_sample_fallback()
        st.info("Using sample fallback data (no CSVs found)")
    with st.expander("Dataset stats"):
        st.metric("Total samples", len(df))
        st.metric("Bullying (1)", int((df["label"] == 1).sum()))
        st.metric("Non-bullying (0)", int((df["label"] == 0).sum()))
        st.dataframe(df["label"].value_counts().rename("count"), width="stretch")
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
            report_str = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            st.success(f"Accuracy: {acc:.2%}")
            st.text(report_str)
            st.write("Confusion matrix:")
            cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
            st.dataframe(cm_df, width="stretch")
            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button("Download report", report_str.encode("utf-8"), "classification_report.txt", "text/plain", key="dl_report")
            with col_b:
                st.download_button("Download confusion matrix (CSV)", cm_df.to_csv().encode("utf-8"), "confusion_matrix.csv", "text/csv", key="dl_cm")
            bull_words, safe_words = get_top_words(vectoriser, mdl, 15)
            with st.expander("Feature importance (top words)"):
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Bullying-indicative**")
                    for w, c in bull_words:
                        st.caption(f"  {w} ({c:.3f})")
                with c2:
                    st.write("**Non-bullying (safe)**")
                    for w, c in safe_words:
                        st.caption(f"  {w} ({c:.3f})")
            ad = get_artefact_dir()
            joblib.dump(vectoriser, os.path.join(ad, "tfidf.joblib"))
            joblib.dump(mdl, os.path.join(ad, "logreg_model.joblib"))
            st.info("Model saved. Refresh the page to use it.")

# Tab 4: About
with tab4:
    st.subheader("About TrollGuard")
    st.write("""
    TrollGuard is a text-based cyberbullying detection system using TF-IDF features and Logistic Regression.
    
    - **Predict**: Enter single text, paste multiple lines for bulk analysis, or upload a CSV with a text column.
    - **Chat analysis**: Upload or paste WhatsApp-style chat exports (no preloaded sample).
    - **Train**: Train the model on your dataset (`*_parsed_dataset.csv` files).
    
    Dataset format: CSV with columns `Text` and `oh_label` (0 = non-bullying, 1 = bullying).
    """)
    st.caption("B.Sc. Computer Science Final Year Project")
