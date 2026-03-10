---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

# TrollGuard
## Cyberbullying Detector with Contextual Understanding

B.Sc. Computer Science Final Year Project

[College Name] | [Academic Year]

---

# Outline

1. Introduction & Problem
2. System Study & Background
3. Requirements & Design
4. Methodology & Modules
5. Implementation & Datasets
6. Results & Evaluation
7. Limitations & Future Work
8. Conclusion

---

# 1. Introduction

- **Problem:** Cyberbullying and online harassment on social media
- **Challenge:** Manual moderation is slow, subjective, and hard to scale
- **Solution:** Automated text-based detection using NLP and ML
- **Project:** TrollGuard – end-to-end pipeline from data to prediction
- **Output:** Binary classification (0 = non-bullying, 1 = bullying) with web UI

---

# 2. Cyberbullying

- Bullying through digital channels: posts, comments, DMs, group chats
- Can be direct insults, threats, exclusion, or subtle harassment
- Impact: depression, anxiety, poor academic performance
- **Need:** Assistive tools for moderators, educators, and parents

---

# 3. Text Classification Evolution

| Approach | Features | Strengths |
|----------|----------|-----------|
| Rule-based | Keyword lists | Simple, fast |
| Classical ML | TF-IDF, BoW | Good baselines |
| Neural | Embeddings, CNN/RNN | Learns patterns |
| Contextual | Transformers | Nuanced understanding |

**TrollGuard:** TF-IDF + Logistic Regression (interpretable baseline)

---

# 4. Requirements

- **Hardware:** Laptop, 8 GB RAM, internet
- **Software:** Python 3.8+, scikit-learn, pandas, joblib, Streamlit
- **Functional:**
  - Load CSV → Clean → Train → Predict
  - Single / paste bulk / batch CSV prediction (10K row limit, progress bar)
  - Multi-format chat (WhatsApp, Telegram, Discord) via upload or paste
  - Date filter and per-sender summary; CSV export
  - Train from app with downloadable report and confusion matrix

---

# 5. System Design

```
Raw Text → Cleaning → TF-IDF → Logistic Regression → Label (0/1)
```

- **Input:** Short messages (tweets, comments, chat); single text, paste bulk, batch CSV, chat export
- **Output:** Binary: 0 = non-bullying, 1 = bullying; per-sender chat summaries
- **Features:** TF-IDF (ngrams 1–2, min_df=2, max_df=0.95)
- **Limits:** 50K chars per text area; 10K rows per batch CSV

---

# 6. Modules

1. **Data Loader** – Load `*_parsed_dataset.csv`; normalise labels; sample fallback when empty
2. **Chat Parser** – WhatsApp, Telegram, Discord; malformed lines → sender="Unknown"
3. **Text Cleaning** – Lowercase; remove URLs, @mentions, #hashtags; keep letters
4. **TF-IDF** – Vectorise text (unigrams + bigrams)
5. **Model** – Logistic Regression; feature importance (top bullying/safe words)
6. **Streamlit App** – Predict (single/paste/batch), Chat analysis (upload/paste, format selector, date filter, export), Train, About
7. **Scripts** – `augment_data.py`, `generate_synthetic_data.py`, `download_datasets.py`, `train_and_save.py`

---

# 7. Datasets

- 10+ parsed datasets combined (~540K+ rows)
- Sources: Kaggle (Jigsaw toxicity), Hugging Face (OLID, hate_speech), Twitter, YouTube, augmented, synthetic
- Format: `Text` + `oh_label` (0/1)
- Class balance: ~87% non-bullying, ~13% bullying
- Scripts: `augment_data.py`, `generate_synthetic_data.py`, `download_datasets.py`

---

# 8. Methodology

1. Load & merge CSVs from `dataset/` or `datasets/`
2. Normalise labels to binary 0/1
3. Clean text (lowercase, remove URLs, etc.)
4. EDA and visualisations
5. Train–test split (80/20, stratified)
6. Fit TF-IDF + Logistic Regression
7. Evaluate: accuracy, precision, recall, F1, confusion matrix
8. Export model (`models/` or `outputs/`) and predictions

---

# 9. Implementation Highlights

- **Streamlit App:** Predict (single, paste bulk, batch CSV with progress bar), Chat analysis (upload/paste, auto/whatsapp/telegram/discord, date filter, per-sender summary, export), Train (stats, feature importance, downloadable report/confusion matrix), About
- **Core modules:** `core/data_loader.py`, `core/model_utils.py`, `core/chat_parser.py`
- **Lazy model loading:** `@st.cache_resource` for Streamlit Cloud
- **Session stats:** Prediction counts in sidebar
- **Deployment:** Streamlit Cloud; GitHub: github.com/itsmehari/TrollGuard

---

# 10. Results

- **Accuracy:** ~85–92% (baseline)
- **Metrics:** Precision, recall, F1; confusion matrix (downloadable)
- **Feature importance:** Top bullying/safe words in sidebar and Train tab
- **Chat export:** Per-sender summary (total, flagged, flagged %); date filter; CSV export

---

# 11. Limitations

- English only
- Message-level only (limited conversation context)
- No image / audio / video
- Dataset bias; sarcasm and coded language challenging
- No real-time API by default
- Input limits (50K chars, 10K batch rows)

---

# 12. Future Work

- Multilingual and code-mixed support
- Real-time REST API
- User feedback loop (moderator corrections for retraining)
- Multimodal analysis (images, voice)
- Transformer-based models (e.g. BERT)
- *Explainability via feature importance already implemented*

---

# 13. Conclusion

- TrollGuard: end-to-end cyberbullying detection pipeline
- TF-IDF + Logistic Regression provides a strong, interpretable baseline
- Streamlit app for prediction, chat analysis, and training
- Suitable for moderation assistance in schools and organisations
- Foundation for transformer-based and multilingual extensions

---

# Thank You

**Questions?**

---

# Appendix – Demo Screenshots

*(Insert screenshots of Streamlit app: Predict tab, Chat analysis with date filter, Train tab with confusion matrix, sidebar feature importance)*
