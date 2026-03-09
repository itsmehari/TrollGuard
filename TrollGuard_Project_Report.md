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

The system focuses on short text messages such as comments, tweets, and chat lines. Publicly available labelled datasets are collected and cleaned to remove noise (URLs, emojis, unnecessary symbols). Exploratory Data Analysis (EDA) is performed to understand label distribution, message length, and text characteristics. A baseline classifier is developed using TF-IDF features and Logistic Regression. An optional advanced approach can use contextual sentence representations. The models are evaluated using accuracy, precision, recall, F1-score, and confusion matrix analysis.

Results show that a simple baseline model achieves reasonable performance, while advanced representations can improve sensitivity to subtle bullying. The project demonstrates an end-to-end pipeline that can be extended towards practical moderation tools in educational or organisational settings.

**Keywords:** Cyberbullying, Text Classification, Toxic Comments, Logistic Regression, TF-IDF, NLP, Machine Learning

---

## Chapter 1 – Introduction

The widespread adoption of smartphones, affordable data, and social networking platforms has fundamentally changed how people communicate. Information flows faster than ever, but a significant downside has emerged in the form of cyberbullying and online harassment.

Cyberbullying refers to bullying behaviour carried out through digital channels: social media posts, comments, messages, and group chats. It can take the form of direct insults, repeated mocking, threats, exclusion, or indirect remarks that damage a person’s self-esteem and mental health. Manual monitoring of all online communication is unrealistic. TrollGuard is designed as a B.Sc. final year project that addresses this need by demonstrating how text data can be collected, cleaned, analysed, and used to train supervised classification models.

---

## Chapter 2 – System Study

### 2.1 Cyberbullying in Online Platforms

Online platforms allow people to post content instantly and anonymously. Studies associate cyberbullying with depression, anxiety, and decreased academic performance. The challenge is to monitor vast amounts of user-generated text and respond quickly when harmful content appears. Automated detection can provide an additional layer of protection.

### 2.2 Evolution of Text Classification

1. **Rule-based / Keyword filters** – Simple but easy to bypass  
2. **Classical ML** – TF-IDF with Naive Bayes, SVM, Logistic Regression  
3. **Neural models** – Word embeddings, CNN, RNN  
4. **Contextual models** – Transformers, contextual embeddings

TrollGuard positions itself at the junction between classical and contextual approaches.

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

- **Python 3.x** – Programming language  
- **Google Colab** – Cloud notebook (optional)  
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, streamlit

### 3.3 Functional Requirements

- Load labelled text datasets (CSV)  
- Preprocess and clean text  
- Perform EDA and visualisation  
- Train TF-IDF + Logistic Regression model  
- Evaluate using standard metrics  
- Support single-text and batch prediction  
- Parse WhatsApp-style chat exports and analyse per-sender

---

## Chapter 4 – Design

### 4.1 System Architecture

```
Raw Text → Cleaning → TF-IDF → Logistic Regression → Prediction (0/1)
```

### 4.2 Data Flow

1. **Input:** CSV files with `Text` and `oh_label` columns  
2. **Preprocessing:** Lowercase, remove URLs, mentions, hashtags, non-alpha  
3. **Feature engineering:** TF-IDF with n-grams (1, 2)  
4. **Model:** Logistic Regression  
5. **Output:** Binary label (0 = non-bullying, 1 = bullying)

### 4.3 Chat Export Flow

```
WhatsApp export (.txt) → Parse (timestamp, sender, message) → Clean → Predict → Aggregate per sender
```

---

## Chapter 5 – Modules

| Module | Description |
|--------|-------------|
| **Data Loader** | Loads `*_parsed_dataset.csv` files, normalises labels to binary |
| **Text Cleaning** | Lowercase, remove URLs, @mentions, #hashtags, non-alpha |
| **EDA** | Class distribution, text length histogram, basic statistics |
| **Feature Engineering** | TF-IDF vectorisation (ngram_range=(1,2), min_df=2, max_df=0.95) |
| **Model Training** | Logistic Regression (max_iter=300) |
| **Evaluation** | Accuracy, classification report, confusion matrix |
| **Prediction** | Single text, batch CSV, chat export analysis |
| **Streamlit App** | Web UI for prediction, chat analysis, and training |

---

## Chapter 6 – Implementation

### 6.1 Core Implementation

- **Data loading:** `core/data_loader.py` – loads all `*_parsed_dataset.csv` files  
- **Model utilities:** `core/model_utils.py` – `clean_text`, `predict_text`, `predict_batch`  
- **Chat parser:** `core/chat_parser.py` – WhatsApp format parsing  
- **Streamlit app:** `app.py` – Predict, Chat analysis, Train model, About

### 6.2 Configuration

The system auto-detects project root for both Colab/Drive and local environments. It supports `dataset/` or `datasets/` folder naming.

### 6.3 Deployment Target

Streamlit Cloud (or similar). The app runs with sample fallback data when no datasets are present.

---

## Chapter 7 – Datasets

### 7.1 Data Sources

Eight publicly available parsed datasets:

- `aggression_parsed_dataset.csv`  
- `attack_parsed_dataset.csv`  
- `kaggle_parsed_dataset.csv`  
- `toxicity_parsed_dataset.csv`  
- `twitter_parsed_dataset.csv`  
- `twitter_racism_parsed_dataset.csv`  
- `twitter_sexism_parsed_dataset.csv`  
- `youtube_parsed_dataset.csv`  

Combined size: ~449,000 rows. Class distribution: ~87% non-bullying, ~13% bullying.

### 7.2 Dataset Format

- **Required columns:** `Text`, `oh_label` (0 = non-bullying, 1 = bullying)  
- **Chat export:** `sample_chat.txt` in format `DD/MM/YYYY, HH:MM - Sender: message`

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

The Logistic Regression + TF-IDF baseline typically achieves accuracy in the 70–85% range. F1 for the bullying class is lower due to class imbalance.

### 8.3 Chat Export Analysis

The system produces per-message predictions and per-sender summaries (total messages, flagged count, flagged rate).

---

## Chapter 9 – Limitations

1. **Language:** English only; no regional languages or code-mixed text  
2. **Context:** Message-level analysis; limited conversation-level context  
3. **Multimodality:** No image, video, or audio analysis  
4. **Dataset bias:** Performance depends on training data; may not generalise to all bullying types  
5. **Real-time:** Offline/batch processing; no live API deployment by default  
6. **False positives/negatives:** Sarcasm, coded language, and subtle aggression remain challenging  

---

## Chapter 10 – Future Work

1. **Multilingual support** – Handle regional languages and code-mixing  
2. **Real-time API** – REST API for integration into chat applications  
3. **User feedback loop** – Moderator corrections for retraining  
4. **Multimodal analysis** – Image, voice, and video content  
5. **Explainability** – Show which words/phrases triggered the flag  
6. **Transformer-based models** – BERT or similar for contextual understanding  

---

## Chapter 11 – Business Model

TrollGuard can be positioned as:

- **Educational:** Tool for schools and colleges to monitor group chats and forums  
- **SaaS:** Subscription-based moderation API for small platforms  
- **Consulting:** Custom models for organisations with domain-specific data  
- **Open source:** Community-driven improvements with optional premium support  

---

## Chapter 12 – Conclusion

TrollGuard demonstrates how supervised text classification can be applied to cyberbullying detection. The pipeline—from dataset loading and cleaning to feature engineering, model training, evaluation, and deployment—offers a practical example suitable for a B.Sc. final year project.

The TF-IDF + Logistic Regression baseline shows that simple approaches can flag a significant proportion of harmful messages. More advanced representations can improve performance for subtle cases. Responsible deployment will require attention to fairness, privacy, and human-in-the-loop moderation.

---

## References

1. scikit-learn Documentation. *Text feature extraction – TfidfVectorizer.* [https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

2. scikit-learn Documentation. *Logistic Regression.* [https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)

3. Python Software Foundation. *Python 3 Documentation.* [https://docs.python.org/3/](https://docs.python.org/3/)

4. Streamlit. *Streamlit Documentation.* [https://docs.streamlit.io/](https://docs.streamlit.io/)

5. NAAC / University affiliation documents (as applicable).

6. Research on toxic comment classification and cyberbullying detection (cite specific papers used).

7. Kaggle / public dataset sources used in the project.

---

## Appendices

### Appendix A – Sample Code Snippets

**Data loading:**
```python
from core.data_loader import load_parsed_datasets
df = load_parsed_datasets()
```

**Text cleaning:**
```python
from core.model_utils import clean_text
cleaned = clean_text("Raw message with @user and http://url.com")
```

**Prediction:**
```python
from core.model_utils import load_model_and_vectoriser, predict_text
tfidf, model = load_model_and_vectoriser()
label = predict_text("Your message here", tfidf, model)
```

### Appendix B – Screenshots and Diagrams

*(Insert screenshots: class distribution, confusion matrix, Streamlit app, etc.)*

### Appendix C – Dataset Format

See `dataset/DATASET_FORMAT.md` for CSV and chat export specifications.
