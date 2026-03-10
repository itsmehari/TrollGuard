# TrollGuard – Cyberbullying Detector with Contextual Understanding

**B.Sc. Computer Science Final Year Project Report**

---

## Front Matter

### Title Page

**Dhanraj Baid Jain College (Autonomous)**  
Thoraipakkam, Chennai – 600097

*(NAAC Accredited / Affiliated to [University Name])*

---

**Project Title:** TrollGuard – Cyberbullying Detector with Contextual Understanding

**One-line description:** AI-assisted text-based cyberbullying detection system using NLP and machine learning.

**Submitted by**  
[Your Full Name]  
Roll / Register No.: [Your Roll No.]

**Under the guidance of**  
[Guide Name]  
[Guide Designation, e.g. Assistant Professor]  
Department of Computer Science

**Degree:** B.Sc. Computer Science  
**Academic Year:** [e.g. 2024 – 2025]

---

### Bonafide Certificate

This is to certify that the project work titled **"TrollGuard – Cyberbullying Detector with Contextual Understanding"** is a bonafide record of work done by **[Your Full Name]** (Roll No.: [Your Roll No.]) in partial fulfilment of the requirements for the award of the degree of **B.Sc. Computer Science** under the guidance of **[Guide Name]** during the academic year [e.g. 2024 – 2025].

**Head of the Department**  
Department of Computer Science  
[College Name]

**Place:**  
**Date:**

---

### Declaration

I hereby declare that the project work titled **"TrollGuard – Cyberbullying Detector with Contextual Understanding"** has been carried out by me as part of my B.Sc. Computer Science final year academic requirement. This work is based on my own study and implementation, except where references are clearly cited. The project has not been submitted to any other institution or university for the award of any degree or diploma.

**Place:**  
**Date:**  
**Signature of the Student**  
**Name:** [Your Full Name]

---

### Acknowledgement

I wish to express my sincere thanks to **[Guide Name]** for providing consistent guidance, valuable suggestions, and encouragement throughout the project. Their support helped me understand both the technical and practical aspects of this work.

I am grateful to the Department of Computer Science and **[College Name]** for the opportunity and resources to complete this project. I also thank my classmates, friends, and family for their moral support, and I acknowledge the creators of the publicly available datasets and open-source tools used in this work.

---

## List of Abbreviations

| Abbreviation | Full Form |
|--------------|-----------|
| API | Application Programming Interface |
| BoW | Bag of Words |
| CNN | Convolutional Neural Network |
| CSV | Comma-Separated Values |
| EDA | Exploratory Data Analysis |
| F1 | F1-Score (harmonic mean of precision and recall) |
| IDF | Inverse Document Frequency |
| LR | Logistic Regression |
| ML | Machine Learning |
| NLP | Natural Language Processing |
| TF | Term Frequency |
| TF-IDF | Term Frequency–Inverse Document Frequency |
| URL | Uniform Resource Locator |
| SVM | Support Vector Machine |

---

## Abstract

The rapid growth of social media and online communication platforms has led to a parallel rise in cyberbullying, harassment, and toxic behaviour. Manual moderation is slow, subjective, and difficult to scale. This project, titled **TrollGuard**, implements a text-based cyberbullying detection system that uses data preprocessing, exploratory data analysis, feature engineering, and classification models to identify harmful messages.

The system focuses on short text messages such as comments, tweets, and chat lines. Publicly available labelled datasets are collected, augmented, and cleaned to remove noise (URLs, emojis, unnecessary symbols). Exploratory Data Analysis (EDA) is performed to understand label distribution, message length, and text characteristics. A baseline classifier is developed using TF-IDF features and Logistic Regression. The model is evaluated using accuracy, precision, recall, F1-score, and confusion matrix analysis.

Results show that the baseline model achieves reasonable performance (approximately 85–92% accuracy). The Streamlit web application supports single-text prediction, paste bulk analysis, batch CSV upload (up to 10,000 rows with progress bar), and multi-format chat export analysis (WhatsApp, Telegram, Discord) with date filtering, per-sender summaries, and CSV export. The system also includes a Train tab for in-app model retraining with downloadable reports and feature importance. The project demonstrates an end-to-end pipeline suitable for moderation assistance in educational or organisational settings.

**Keywords:** Cyberbullying, Text Classification, Toxic Comments, Logistic Regression, TF-IDF, NLP, Machine Learning

---

## Chapter 1 – Introduction

The widespread adoption of smartphones, affordable data, and social networking platforms has fundamentally changed how people communicate. Information flows faster than ever, but a significant downside has emerged in the form of cyberbullying and online harassment.

Cyberbullying refers to bullying behaviour carried out through digital channels: social media posts, comments, messages, and group chats. It can take the form of direct insults, repeated mocking, threats, exclusion, or indirect remarks that damage a person's self-esteem and mental health. Manual monitoring of all online communication is unrealistic. TrollGuard is designed as a B.Sc. final year project that addresses this need by demonstrating how text data can be collected, cleaned, augmented, analysed, and used to train supervised classification models, with a Streamlit web interface for predictions and chat analysis.

---

## Chapter 2 – System Study

### 2.1 Cyberbullying in Online Platforms

Online platforms allow people to post content instantly and anonymously. Studies associate cyberbullying with depression, anxiety, and decreased academic performance. The challenge is to monitor vast amounts of user-generated text and respond quickly when harmful content appears. Automated detection can provide an additional layer of protection.

### 2.2 Evolution of Text Classification

1. **Rule-based / Keyword filters** – Simple but easy to bypass  
2. **Classical ML** – TF-IDF with Naive Bayes, SVM, Logistic Regression  
3. **Neural models** – Word embeddings, CNN, RNN  
4. **Contextual models** – Transformers, contextual embeddings

TrollGuard positions itself at the classical ML approach, with a clear path to more advanced models.

### 2.3 Related Applications

The same techniques apply to: toxicity and hate speech detection, spam detection, sentiment analysis, and content moderation in forums and gaming platforms.

---

## Chapter 3 – Requirements

### 3.1 Hardware Requirements

- Computer or laptop with a web browser  
- Reliable internet connection  
- Recommended RAM: 8 GB or above  
- Optional: GPU via Google Colab for larger models

### 3.2 Software Requirements

- **Python 3.8+** – Programming language  
- **Libraries:** pandas (≥1.3.0), numpy (≥1.20.0), scikit-learn (1.7.2), joblib (≥1.1.0), streamlit (≥1.20.0)  
- **Optional:** Google Colab, Kaggle API, Hugging Face datasets (for data scripts)

### 3.3 Functional Requirements

- Load labelled text datasets (CSV) from `dataset/` or `datasets/`  
- Preprocess and clean text (lowercase, remove URLs, @mentions, #hashtags)  
- Perform EDA and visualisation  
- Train TF-IDF + Logistic Regression model  
- Evaluate using accuracy, precision, recall, F1, confusion matrix  
- Support single-text, paste-text bulk analysis, and batch CSV prediction (progress bar, 10K row limit)  
- Parse multi-format chat exports (WhatsApp, Telegram, Discord) via upload or paste; format auto-detect  
- Optional date filter for chat analysis  
- Per-sender summary (total messages, flagged count, flagged %)  
- Export chat analysis and training reports to CSV  
- Display feature importance (top bullying/safe words) in sidebar and Train tab  
- Session statistics (predictions, flagged count) in sidebar  
- Deploy on Streamlit Cloud with lazy model loading  
- Retrain model from app or via `train_model.py` / `scripts/train_and_save.py`

---

## Chapter 4 – Design

### 4.1 System Architecture

```
Raw Text → Cleaning → TF-IDF → Logistic Regression → Prediction (0/1)
```

### 4.2 Data Flow

1. **Input:** CSV files with `Text` and `oh_label` columns; or chat text (upload/paste)  
2. **Preprocessing:** Lowercase, remove URLs, mentions, hashtags, non-alpha  
3. **Feature engineering:** TF-IDF with n-grams (1, 2), min_df=2, max_df=0.95  
4. **Model:** Logistic Regression (max_iter=300)  
5. **Output:** Binary label (0 = non-bullying, 1 = bullying)

### 4.3 Chat Export Flow

```
Upload / Paste chat (.txt) → Format selector (WhatsApp/Telegram/Discord/Auto) → Parse (timestamp, sender, message) → Optional date filter → Clean → Predict → Aggregate per sender → Export CSV
```

**Chat formats supported:**
- **WhatsApp:** `DD/MM/YYYY, HH:MM - Sender: message`  
- **Telegram:** `DD.MM.YYYY, HH:MM - Sender: message` (dots converted to slashes)  
- **Discord:** `[DD/MM/YYYY HH:MM] Sender: message`

Malformed lines are kept with sender="Unknown", timestamp=1970-01-01.

### 4.4 Configuration Limits

- **MAX_INPUT_CHARS:** 50,000 characters per text area / paste / upload  
- **BATCH_CSV_LIMIT:** 10,000 rows per batch CSV

### 4.5 Deployment

- **Streamlit Cloud:** App URL: https://trollguard-eem9mwmcp2gtqwff95wu7v.streamlit.app/  
- **GitHub:** https://github.com/itsmehari/TrollGuard  
- **Main file:** `code/app.py`  
- **Theme:** Light (`.streamlit/config.toml`)  
- **Model storage:** `models/` (primary) or `outputs/` (fallback)

---

## Chapter 5 – Modules

| Module | Description |
|--------|-------------|
| **Data Loader** | `core/data_loader.py` – Loads `*_parsed_dataset.csv`; normalises labels; supports `datasets/` and `dataset/`; `get_sample_fallback()` when no CSVs |
| **Text Cleaning** | `core/model_utils.py` – `clean_text()`: lowercase, remove URLs, @mentions, #hashtags, non-alpha |
| **Chat Parser** | `core/chat_parser.py` – `parse_chat_from_string()` for WhatsApp, Telegram, Discord; `format_hint`: auto/whatsapp/telegram/discord; malformed lines → Unknown |
| **EDA** | Class distribution, text length histogram, dataset stats |
| **Feature Engineering** | TF-IDF (ngram_range=(1,2), min_df=2, max_df=0.95) |
| **Model Training** | Logistic Regression (max_iter=300); `train_model.py`, `scripts/train_and_save.py` |
| **Feature Importance** | `get_top_words()` – top bullying/safe words from coefficients; guards against vocab/coef length mismatch |
| **Evaluation** | Accuracy, classification report, confusion matrix; downloadable reports |
| **Prediction** | Single text, paste bulk, batch CSV (progress bar); chat analysis with date filter, per-sender summary, CSV export |
| **Streamlit App** | Tabs: Predict, Chat analysis, Train model, About; sidebar: model status, session stats, feature importance, datasets |
| **Scripts** | `augment_data.py` (synonym/swap augmentation), `generate_synthetic_data.py` (template-based), `download_datasets.py` (Kaggle, Hugging Face), `train_and_save.py` (retrain for sklearn version compatibility) |

---

## Chapter 6 – Implementation

### 6.1 Core Implementation

- **Data loading:** `core/data_loader.py` – `load_parsed_datasets()`, `get_datasets_dir()`, `get_project_root()` (Colab/Drive/local), `get_sample_fallback()`  
- **Model utilities:** `core/model_utils.py` – `clean_text`, `load_model_and_vectoriser`, `predict_text`, `predict_batch`, `get_top_words`, `get_artefact_dir`  
- **Chat parser:** `core/chat_parser.py` – `parse_chat_from_string()` for WhatsApp, Telegram, Discord; malformed lines preserved  
- **Streamlit app:** `app.py` – Predict (single, paste bulk, batch CSV), Chat analysis (upload/paste, format selector, date filter, per-sender summary, export), Train model (stats, feature importance, downloadable report/confusion matrix), About  
- **Training:** `train_model.py` – full training; `scripts/train_and_save.py` – retrain (max 30k samples) for sklearn version compatibility

### 6.2 Configuration

- Project root auto-detection for Colab (`/content/drive/MyDrive/TrollGuard_Project*`), Streamlit Cloud, and local runs  
- Supports `dataset/` and `datasets/` folder names  
- Lazy model loading via `@st.cache_resource`  
- Input limits: 50,000 chars; 10,000 rows for batch CSV  
- Sample fallback when no datasets present

### 6.3 Deployment

- **Streamlit Cloud:** https://trollguard-eem9mwmcp2gtqwff95wu7v.streamlit.app/  
- **GitHub:** https://github.com/itsmehari/TrollGuard  
- **Theme:** Light (`.streamlit/config.toml`)  
- Runs with sample fallback data when no datasets are present

---

## Chapter 7 – Datasets

### 7.1 Data Sources

Parsed datasets in `dataset/` or `datasets/` (combined ~540,000+ rows):

- `aggression_parsed_dataset.csv`, `attack_parsed_dataset.csv`, `kaggle_parsed_dataset.csv`  
- `toxicity_parsed_dataset.csv`, `twitter_parsed_dataset.csv`  
- `twitter_racism_parsed_dataset.csv`, `twitter_sexism_parsed_dataset.csv`  
- `youtube_parsed_dataset.csv`  
- `augmented_parsed_dataset.csv` (from `scripts/augment_data.py`)  
- `synthetic_parsed_dataset.csv` (from `scripts/generate_synthetic_data.py`)  
- `olid_parsed_dataset.csv`, `hate_speech_parsed_dataset.csv` (from `scripts/download_datasets.py`)  
- `jigsaw_toxic_parsed_dataset.csv` (Kaggle Jigsaw)

Class distribution approximately 87% non-bullying, 13% bullying.

### 7.2 Dataset Format

- **Required columns:** `Text`, `oh_label` (0 = non-bullying, 1 = bullying)  
- **Label normalisation:** String labels like "none", "normal", "non-toxic" mapped to 0  
- **Chat formats:** See Chapter 4.3

### 7.3 Data Loader Utility

See `core/data_loader.py` and `dataset/DATASET_FORMAT.md` for format documentation.

---

## Chapter 8 – Results

### 8.1 Evaluation Metrics

- **Accuracy:** Proportion of correct predictions  
- **Precision (bullying class):** Of predicted bullying, how many are true  
- **Recall (bullying class):** Of true bullying, how many are detected  
- **F1-Score:** Harmonic mean of precision and recall  
- **Confusion Matrix:** 2×2 grid of true vs predicted

### 8.2 Baseline Results

The Logistic Regression + TF-IDF baseline achieves accuracy in the 85–92% range on held-out test data. F1 for the bullying class is lower due to class imbalance. Feature importance (top bullying/safe words) is available in the sidebar and Train tab.

### 8.3 Chat Export Analysis

The system produces per-message predictions and per-sender summaries (total messages, flagged count, flagged %). Users can filter by date range and export results to CSV.

---

## Chapter 9 – Limitations

1. **Language:** English only; no regional languages or code-mixed text  
2. **Context:** Message-level analysis; limited conversation-level context  
3. **Multimodality:** No image, video, or audio analysis  
4. **Dataset bias:** Performance depends on training data  
5. **Real-time:** Offline/batch processing; no live API by default  
6. **Input limits:** 50K chars, 10K batch rows  
7. **Sarcasm/coded language:** Challenging for TF-IDF + LR

---

## Chapter 10 – Future Work

1. **Multilingual support** – Regional languages and code-mixing  
2. **Real-time API** – REST API for integration  
3. **User feedback loop** – Moderator corrections for retraining  
4. **Multimodal analysis** – Image, voice, video  
5. **Transformer-based models** – BERT or similar

*Note: Explainability via feature importance (top bullying/safe words) is already implemented.*

---

## Chapter 11 – Business Model

TrollGuard can be positioned as:

- **Educational:** Tool for schools and colleges  
- **SaaS:** Subscription-based moderation API  
- **Consulting:** Custom models for organisations  
- **Open source:** Community-driven improvements

---

## Chapter 12 – Conclusion

TrollGuard demonstrates how supervised text classification can be applied to cyberbullying detection. The pipeline—from dataset loading, augmentation, and cleaning to feature engineering, model training, evaluation, and deployment—offers a practical example suitable for a B.Sc. final year project.

The TF-IDF + Logistic Regression baseline shows that simple approaches can flag a significant proportion of harmful messages. The Streamlit app supports single and bulk prediction, multi-format chat analysis with date filtering and per-sender summaries, and in-app retraining with downloadable reports. Responsible deployment will require attention to fairness, privacy, and human-in-the-loop moderation.

---

## References

1. scikit-learn Documentation. *Text feature extraction – TfidfVectorizer.* https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

2. scikit-learn Documentation. *Logistic Regression.* https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

3. Python Software Foundation. *Python 3 Documentation.* https://docs.python.org/3/

4. Streamlit. *Streamlit Documentation.* https://docs.streamlit.io/

5. NAAC / University affiliation documents (as applicable).

6. Research on toxic comment classification and cyberbullying detection.

7. Kaggle / Hugging Face / public dataset sources used.

---

## Appendices

### Appendix A – Sample Code Snippets

**Data loading:**
```python
from core.data_loader import load_parsed_datasets, get_sample_fallback
df = load_parsed_datasets()
if df.empty:
    df = get_sample_fallback()
```

**Text cleaning:**
```python
from core.model_utils import clean_text
cleaned = clean_text("Raw message with @user and http://url.com")
```

**Prediction:**
```python
from core.model_utils import load_model_and_vectoriser, predict_text, predict_batch
tfidf, model = load_model_and_vectoriser()
label = predict_text("Your message here", tfidf, model)
labels_batch = predict_batch(["msg1", "msg2"], tfidf, model)
```

**Chat parsing (multi-format):**
```python
from core.chat_parser import parse_chat_from_string
df = parse_chat_from_string(chat_text, format_hint="auto")  # or "whatsapp", "telegram", "discord"
```

### Appendix B – Screenshots and Diagrams

*(Insert screenshots: class distribution, confusion matrix, Streamlit app tabs, sidebar, chat analysis with date filter, per-sender summary.)*

### Appendix C – Dataset Format

See `dataset/DATASET_FORMAT.md` for CSV and chat export specifications.
