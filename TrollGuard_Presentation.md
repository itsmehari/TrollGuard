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
- **Challenge:** Manual moderation is slow and hard to scale
- **Solution:** Automated text-based detection using NLP and ML
- **Project:** TrollGuard – end-to-end pipeline from data to prediction

---

# 2. Cyberbullying

- Bullying through digital channels: posts, comments, DMs, group chats
- Can be direct insults, threats, exclusion, or subtle harassment
- Impact: depression, anxiety, poor academic performance
- **Need:** Assistive tools for moderators and educators

---

# 3. Text Classification Evolution

| Approach | Features | Strengths |
|----------|----------|-----------|
| Rule-based | Keyword lists | Simple, fast |
| Classical ML | TF-IDF, BoW | Good baselines |
| Neural | Embeddings, CNN/RNN | Learns patterns |
| Contextual | Transformers | Nuanced understanding |

**TrollGuard:** TF-IDF + Logistic Regression baseline (+ optional contextual)

---

# 4. Requirements

- **Hardware:** Laptop, 8 GB RAM, internet
- **Software:** Python 3, scikit-learn, pandas, Streamlit
- **Functional:** Load CSV → Clean → Train → Predict; Chat export analysis

---

# 5. System Design

```
Raw Text → Cleaning → TF-IDF → Logistic Regression → Label (0/1)
```

- **Input:** Short messages (tweets, comments, chat)
- **Output:** Binary: 0 = non-bullying, 1 = bullying
- **Features:** TF-IDF (ngrams 1–2)

---

# 6. Modules

1. **Data Loader** – Load `*_parsed_dataset.csv` files
2. **Text Cleaning** – Remove URLs, @mentions, #hashtags
3. **EDA** – Class distribution, length analysis
4. **TF-IDF** – Vectorise text
5. **Model** – Logistic Regression
6. **Streamlit App** – Predict, Chat analysis, Train

---

# 7. Datasets

- 8 public datasets combined (~449K rows)
- Sources: Kaggle, Twitter, YouTube, toxicity datasets
- Format: `Text` + `oh_label` (0/1)
- Class balance: ~87% non-bullying, ~13% bullying

---

# 8. Methodology

1. Load & merge CSVs
2. Normalise labels to binary 0/1
3. Clean text (lowercase, remove URLs, etc.)
4. EDA and visualisations
5. Train–test split (80/20)
6. Fit TF-IDF + Logistic Regression
7. Evaluate: accuracy, precision, recall, F1, confusion matrix
8. Export model and predictions

---

# 9. Implementation Highlights

- **Notebook:** Google Colab – full pipeline
- **Streamlit App:** Web UI for prediction, chat export, training
- **Core modules:** `core/data_loader.py`, `core/model_utils.py`, `core/chat_parser.py`
- **Path auto-detection:** Works locally and on Colab/Drive

---

# 10. Results

- **Accuracy:** ~70–85% (baseline)
- **Metrics:** Precision, recall, F1 for bullying class
- **Confusion matrix:** Shows false positives/negatives
- **Chat export:** Per-sender summary (total, flagged, rate)

---

# 11. Limitations

- English only
- Message-level only (limited conversation context)
- No image/audio/video
- Dataset bias
- No real-time deployment by default

---

# 12. Future Work

- Multilingual and code-mixed support
- Real-time API
- User feedback loop
- Multimodal analysis (images, voice)
- Explainability (why flagged?)

---

# 13. Conclusion

- TrollGuard: end-to-end cyberbullying detection pipeline
- TF-IDF + Logistic Regression provides a strong baseline
- Streamlit app for prediction and chat analysis
- Suitable for moderation assistance in schools and organisations
- Foundation for more advanced (e.g. transformer-based) models

---

# Thank You

**Questions?**

---

# Appendix – Demo Screenshots

*(Insert screenshots of Streamlit app, confusion matrix, class distribution here)*
