"""
================================================================================
TrollGuard – Main Web Application (Streamlit)
================================================================================

WHAT THIS FILE DOES (in simple terms):
This is the main "front-end" of TrollGuard. When you run `streamlit run app.py`,
a web page opens where users can:
  1. Type a message and see if it's classified as bullying or not
  2. Paste many messages at once and get predictions for each
  3. Upload a CSV file with a text column for bulk analysis
  4. Upload or paste chat exports (WhatsApp, Telegram, Discord) and analyse them
  5. Train a new model using the datasets in the dataset/ folder

Think of this file as the "control centre" that connects the user interface
(buttons, text boxes, tables) with the "brain" (the model that detects bullying).

Run from the code/ folder:  streamlit run app.py
================================================================================
"""

# =============================================================================
# SECTION 1: PATH SETUP (Why we need this)
# =============================================================================
# When you run an app, Python needs to know where to find other files in your
# project (like dataset/, models/, and the core/ package). On Streamlit Cloud,
# the app may run from a different directory. So we:
#   - Find the folder containing app.py (that's our "project root")
#   - Add it to Python's search path (so "from core.xxx" works)
#   - Change the current working directory (so "dataset/" and "models/" paths work)

import os
import sys

# __file__ = path to this app.py file. dirname = the folder containing it.
# abspath makes it a full path (e.g. C:\...\code\app.py -> root = C:\...\code)
ROOT = os.path.dirname(os.path.abspath(__file__))

# sys.path is where Python looks when you write "import something".
# We add ROOT so "from core.data_loader import ..." finds the core folder.
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# chdir = "change directory". Now "dataset/" means ROOT/dataset/, "models/" means ROOT/models/
os.chdir(ROOT)


# =============================================================================
# SECTION 2: IMPORT LIBRARIES
# =============================================================================

import streamlit as st   # Creates the web interface (buttons, text boxes, tabs)
import pandas as pd      # For tables and data (like Excel in Python)
import numpy as np       # For numbers and arrays

# Our own modules (the "backend" logic):
from core.data_loader import (
    get_datasets_dir,      # Returns path to dataset/ folder
    load_parsed_datasets,  # Loads all *_parsed_dataset.csv files
    get_sample_fallback,   # Returns 5 sample messages if no datasets exist
)
from core.chat_parser import parse_chat_from_string  # Parses WhatsApp/Telegram/Discord chat text
from core.model_utils import (
    clean_text,
    load_model_and_vectoriser,
    predict_text,
    predict_batch,
    get_artefact_dir,
    get_top_words,
    create_model,
    MODEL_REGISTRY,
)

# Machine learning libraries:
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split        # Splits data for training
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)  # Measures how well the model performs
import joblib  # Saves and loads Python objects (our model) to/from disk


# =============================================================================
# SECTION 3: SAFETY LIMITS (Prevents abuse and crashes)
# =============================================================================
# Without limits, someone could paste a million characters and crash the app or
# upload a CSV with 1 million rows and freeze the server. These limits protect us.

MAX_INPUT_CHARS = 50_000   # Max characters per text box or paste (50 thousand)
BATCH_CSV_LIMIT = 5_000    # Max rows for batch CSV (Streamlit Cloud memory limit)
MAX_TRAIN_SAMPLES = 5_000   # Max dataset rows (Streamlit Cloud ~1GB limit)


# =============================================================================
# SECTION 4: PAGE CONFIGURATION
# =============================================================================
# Sets the browser tab title, favicon (🛡️), and uses wide layout for tables.
st.set_page_config(page_title="TrollGuard", page_icon="🛡️", layout="wide")


# =============================================================================
# SECTION 5: LAZY MODEL LOADER (Performance optimisation)
# =============================================================================
# Loading the model from disk takes time and memory. @st.cache_resource tells
# Streamlit: "Run this function once and remember the result. Don't run it again
# on every user interaction." This is called "caching" and makes the app faster,
# especially important on Streamlit Cloud where memory is limited.

@st.cache_resource
def _load_model():
    """
    Load the TF-IDF vectoriser and Logistic Regression model from models/ folder.
    Returns (tfidf, model) or (None, None) if files don't exist.
    Cached so we only load once per session.
    """
    return load_model_and_vectoriser()


@st.cache_data(ttl=300)  # Cache 5 min to avoid reloading on every rerun
def _load_data_capped():
    """Load datasets capped at MAX_TRAIN_SAMPLES to stay within Streamlit Cloud memory."""
    df = load_parsed_datasets(max_rows=MAX_TRAIN_SAMPLES)
    if df.empty:
        return get_sample_fallback()
    return df


# =============================================================================
# SECTION 6: SESSION STATE (Remembering things across page reruns)
# =============================================================================
# Streamlit reruns the whole script every time the user clicks something. Without
# session_state, our "prediction count" would reset to 0 each time. session_state
# is like a small memory that persists across reruns for this user's session.

if "pred_total" not in st.session_state:
    st.session_state.pred_total = 0   # How many messages we've analysed this session
if "pred_flagged" not in st.session_state:
    st.session_state.pred_flagged = 0  # How many were flagged as bullying


# =============================================================================
# SECTION 7: MAIN PAGE HEADER
# =============================================================================
st.title("🛡️ TrollGuard – Cyberbullying Detector")
st.caption("NLP-based text classification for bullying detection")

# Load the model (or get None if no model exists yet - user must train first)
tfidf, model = _load_model()


# =============================================================================
# SECTION 8: SIDEBAR (Left panel - model status, stats, feature importance)
# =============================================================================

with st.sidebar:
    st.header("Model")
    # Check if model loaded successfully
    if tfidf is not None and model is not None:
        st.success("Model loaded")  # Green success message
        if st.button("Retrain model"):
            # User can request retraining from sidebar
            st.session_state.train_requested = True
    else:
        st.warning("No model found. Train from data.")  # Yellow warning
        st.session_state.train_requested = True

    st.divider()  # Horizontal line

    # Session statistics - how many predictions and how many flagged
    st.header("Session stats")
    st.metric("Predictions (this session)", st.session_state.pred_total)
    st.metric("Flagged", st.session_state.pred_flagged)

    # Feature importance: Which words does the model associate with bullying vs safe?
    if tfidf and model:
        with st.expander("Feature importance"):
            try:
                bull, safe = get_top_words(tfidf, model, 8)
                st.caption("Bullying: " + ", ".join(w for w, _ in bull[:5]) if bull else "—")
                st.caption("Safe: " + ", ".join(w for w, _ in safe[:5]) if safe else "—")
            except Exception as e:
                st.caption(f"Feature importance unavailable: {e}")

    # Dataset info - where data is and how many samples (cached, capped for memory)
    st.header("Datasets")
    ds_dir = get_datasets_dir()
    st.text(f"Data: {ds_dir}")
    df_raw = _load_data_capped()
    st.metric("Samples", len(df_raw))


# =============================================================================
# SECTION 9: MAIN TABS (Predict | Chat analysis | Train model | About)
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Predict",       # Single text, paste bulk, batch CSV
    "💬 Chat analysis", # Upload/paste chat exports
    "📊 Train model",   # Train a new model
    "ℹ️ About"          # Project info
])


# -----------------------------------------------------------------------------
# TAB 1: PREDICT - Three ways to get predictions
# -----------------------------------------------------------------------------

with tab1:
    # ----- Sub-tab 1a: Single text prediction -----
    st.subheader("Text prediction")
    text_input = st.text_area(
        "Enter text to classify",
        placeholder="Type or paste a message...",
        max_chars=MAX_INPUT_CHARS
    )

    if text_input.strip():
        # User typed something - enforce limit and predict
        text_input = text_input[:MAX_INPUT_CHARS]
        if not (tfidf and model):
            st.warning("Model not loaded. Go to **Train model** tab to train.")
        else:
            pred = predict_text(text_input, tfidf, model)  # Returns 0 or 1
            st.session_state.pred_total += 1
            if pred == 1:
                st.session_state.pred_flagged += 1
                st.error("Prediction: **Bullying** (flagged)")  # Red
            else:
                st.success("Prediction: **Non-bullying** (safe)")  # Green

    st.divider()

    # ----- Sub-tab 1b: Paste multiple lines -----
    st.subheader("Paste text analysis")
    st.caption("Paste multiple lines; each non-empty line will be analyzed separately.")
    paste_input = st.text_area(
        "Paste text to analyze",
        placeholder="Paste several messages here, one per line...",
        key="paste_text",
        max_chars=MAX_INPUT_CHARS
    )

    if paste_input.strip() and (tfidf and model):
        paste_input = paste_input[:MAX_INPUT_CHARS]
        # Split by newlines, remove empty lines
        lines = [ln.strip() for ln in paste_input.strip().splitlines() if ln.strip()]
        if lines:
            preds = predict_batch(lines, tfidf, model)  # Get prediction for each line
            st.session_state.pred_total += len(lines)
            st.session_state.pred_flagged += int(sum(preds))
            # Build a small table: Text | Prediction
            paste_df = pd.DataFrame({
                "Text": lines,
                "Prediction": ["Non-bullying" if p == 0 else "Bullying" for p in preds]
            })
            with st.expander("View results", expanded=True):
                st.dataframe(paste_df)
            flagged = sum(1 for p in preds if p == 1)
            st.caption(f"Total: {len(lines)} lines · Flagged as bullying: {flagged}")
    elif paste_input.strip() and not (tfidf and model):
        st.warning("Model not loaded. Train in **Train model** tab first.")

    st.divider()

    # ----- Sub-tab 1c: Batch CSV upload -----
    st.subheader("Batch upload (CSV)")
    csv_file = st.file_uploader("Upload CSV with 'text' or 'Text' column", type="csv")

    if csv_file:
        try:
            up = pd.read_csv(csv_file)
            # CSV can have "text" or "Text" column (case varies)
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
                    # Process in chunks of 500 to show progress bar
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
                    # Add prediction columns to the uploaded dataframe
                    up = up.iloc[:len(preds)].copy()
                    up["predicted_label"] = preds
                    up["prediction"] = up["predicted_label"].map({0: "Non-bullying", 1: "Bullying"})
                    with st.expander("View results", expanded=True):
                        st.dataframe(up)
                    # Let user download results as CSV
                    st.download_button(
                        "Download results (CSV)",
                        up.to_csv(index=False).encode("utf-8"),
                        "trollguard_predictions.csv",
                        "text/csv"
                    )
        except Exception as e:
            st.error(f"Error: {e}")


# -----------------------------------------------------------------------------
# TAB 2: CHAT ANALYSIS - Upload or paste chat exports
# -----------------------------------------------------------------------------

with tab2:
    st.subheader("Chat export analysis")
    # User selects format: Auto-detect or specific app
    format_hint = st.selectbox(
        "Chat format",
        ["auto", "whatsapp", "telegram", "discord"],
        format_func=lambda x: {
            "auto": "Auto-detect",
            "whatsapp": "WhatsApp",
            "telegram": "Telegram",
            "discord": "Discord"
        }[x]
    )

    chat_df = pd.DataFrame(columns=["timestamp", "sender", "message_text"])

    mode = st.radio("Choose input", ["Upload file", "Paste chat text"], horizontal=True)

    if mode == "Upload file":
        chat_upload = st.file_uploader("Upload chat .txt", type=["txt"])
        if chat_upload:
            content = chat_upload.read().decode("utf-8", errors="ignore")[:MAX_INPUT_CHARS]
            chat_df = parse_chat_from_string(content, format_hint)
    else:
        pasted = st.text_area(
            "Paste your chat export below",
            placeholder="09/03/2025, 14:30 - Alice: Hello...\n09/03/2025, 14:31 - Bob: Hi there",
            max_chars=MAX_INPUT_CHARS,
            key="chat_paste"
        )
        if st.button("Analyze chat", key="chat_analyze_btn") and pasted.strip():
            try:
                chat_df = parse_chat_from_string(pasted[:MAX_INPUT_CHARS], format_hint)
                st.session_state["chat_analysis_result"] = chat_df  # Save for reruns
            except Exception as e:
                st.error(f"Error parsing chat: {e}")
        if "chat_analysis_result" in st.session_state and mode == "Paste chat text":
            chat_df = st.session_state["chat_analysis_result"]

    if chat_df.empty:
        if mode == "Upload file":
            st.info("Upload a .txt file above to analyze.")
        else:
            st.info("Paste your chat export above and click **Analyze chat**.")
    elif not (tfidf and model):
        st.warning("Model not loaded. Go to **Train model** tab first.")
    else:
        try:
            chat_df = chat_df.copy()
            # Optional: Filter by date range
            if not chat_df.empty:
                date_col = pd.to_datetime(chat_df["timestamp"], errors="coerce")
                valid_dates = date_col.dropna()
                if not valid_dates.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        start_d = st.date_input("From date", value=valid_dates.min().date(), key="chat_start")
                    with col2:
                        end_d = st.date_input("To date", value=valid_dates.max().date(), key="chat_end")
                    if start_d is not None and end_d is not None:
                        mask = (date_col.dt.date >= start_d) & (date_col.dt.date <= end_d)
                        chat_df = chat_df.loc[mask]
            # Clean each message and predict
            chat_df["clean_text"] = chat_df["message_text"].apply(clean_text)
            preds = predict_batch(chat_df["clean_text"].tolist(), tfidf, model)
            chat_df["bullying_label"] = preds
            chat_df["result"] = chat_df["bullying_label"].map({0: "Safe", 1: "Flagged"})
            st.session_state.pred_total += len(preds)
            st.session_state.pred_flagged += int(sum(preds))
            with st.expander("Message-level results", expanded=True):
                st.dataframe(chat_df[["timestamp", "sender", "message_text", "result"]])
            # Per-sender summary: total messages, how many flagged, percentage
            summary = chat_df.groupby("sender")["bullying_label"].agg(["count", "sum"]).reset_index()
            summary.columns = ["Sender", "Total messages", "Flagged"]
            total = summary["Total messages"]
            # clip(lower=1) avoids division by zero if someone has 0 messages
            summary["Flagged %"] = (summary["Flagged"] / total.clip(lower=1) * 100).round(1)
            st.subheader("Per-sender summary")
            st.dataframe(summary)
            export_df = chat_df[["timestamp", "sender", "message_text", "result"]]
            st.download_button(
                "Export chat analysis (CSV)",
                export_df.to_csv(index=False).encode("utf-8"),
                "trollguard_chat_analysis.csv",
                "text/csv",
                key="chat_export"
            )
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")


# -----------------------------------------------------------------------------
# TAB 3: TRAIN MODEL - Train a new model from datasets
# -----------------------------------------------------------------------------

with tab3:
    st.subheader("Train model")
    df = _load_data_capped()
    if df.empty:
        df = get_sample_fallback()
        st.info("Using sample fallback data (no CSVs found)")
    with st.expander("Dataset stats"):
        st.caption(f"(Capped at {MAX_TRAIN_SAMPLES:,} for memory)")
        st.metric("Total samples", len(df))
        st.metric("Bullying (1)", int((df["label"] == 1).sum()))
        st.metric("Non-bullying (0)", int((df["label"] == 0).sum()))
        st.dataframe(df["label"].value_counts().rename("count"))

    # Prepare features (X) and labels (y)
    df["clean_text"] = df["text"].apply(clean_text)
    X = df["clean_text"].values
    y = df["label"].values
    n_classes = len(np.unique(y))
    use_stratify = n_classes >= 2 and len(y) >= 10  # Stratify keeps class balance in train/test

    if len(X) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if use_stratify else None,
        )

    # Model selector: lightweight sklearn classifiers (Streamlit Cloud friendly)
    model_choice = st.selectbox(
        "Choose classifier",
        options=list(MODEL_REGISTRY.keys()),
        format_func=lambda k: MODEL_REGISTRY[k][0],
        key="model_select",
    )

    if st.button("Train TF-IDF + Classifier"):
        if len(X) < 2:
            st.error("Need at least 2 samples to train.")
        else:
            with st.spinner(f"Training {MODEL_REGISTRY[model_choice][0]}..."):
                min_df = 1 if len(df) < 20 else 2
                vectoriser = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_df=0.95)
                X_train_vec = vectoriser.fit_transform(X_train)
                X_test_vec = vectoriser.transform(X_test)
                mdl = create_model(model_choice)
                mdl.fit(X_train_vec, y_train)
                y_pred = mdl.predict(X_test_vec)
                acc = accuracy_score(y_test, y_pred)
                report_str = classification_report(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                st.success(f"Accuracy: {acc:.2%}")
                st.text(report_str)
                st.write("Confusion matrix:")
                cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
                st.dataframe(cm_df)
                col_a, col_b = st.columns(2)
                with col_a:
                    st.download_button("Download report", report_str.encode("utf-8"), "classification_report.txt", "text/plain", key="dl_report")
                with col_b:
                    st.download_button("Download confusion matrix (CSV)", cm_df.to_csv().encode("utf-8"), "confusion_matrix.csv", "text/csv", key="dl_cm")
                try:
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
                except Exception as e:
                    with st.expander("Feature importance (top words)"):
                        st.caption(f"Not available for this model: {e}")
                # Save model to disk (classifier.joblib for all models)
                ad = get_artefact_dir()
                joblib.dump(vectoriser, os.path.join(ad, "tfidf.joblib"))
                joblib.dump(mdl, os.path.join(ad, "classifier.joblib"))
                _load_model.clear()  # Clear cache so next load uses new model
                st.info("Model saved. The app will reload with the new model.")
                st.rerun()


# -----------------------------------------------------------------------------
# TAB 4: ABOUT - Project description
# -----------------------------------------------------------------------------

with tab4:
    st.subheader("About TrollGuard")
    st.write("""
    TrollGuard is a text-based cyberbullying detection system using TF-IDF features and multiple classifiers.

    **Models**: Logistic Regression, Naive Bayes, SVM (Linear), Random Forest. Choose one when training.

    - **Predict**: Enter single text, paste multiple lines for bulk analysis, or upload a CSV with a text column.
    - **Chat analysis**: Upload or paste WhatsApp-style chat exports (no preloaded sample).
    - **Train**: Train the model on your dataset (`*_parsed_dataset.csv` files).

    Dataset format: CSV with columns `Text` and `oh_label` (0 = non-bullying, 1 = bullying).
    """)
    st.caption("B.Sc. Computer Science Final Year Project")
