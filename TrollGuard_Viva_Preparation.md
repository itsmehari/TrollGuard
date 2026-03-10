---
title: "TrollGuard – Viva Voce Preparation Document"
subtitle: "B.Sc. Computer Science Final Year Project"
author: "Project Team"
date: "2025"
toc: true
toc-title: "Table of Contents"
numbersections: true
---

# TrollGuard – Viva Voce Preparation Document

**B.Sc. Computer Science Final Year Project**

---

## Project Context Summary

| Item | Detail |
|------|--------|
| **Project Title** | TrollGuard – Cyberbullying Detector with Contextual Understanding |
| **Problem** | Cyberbullying and toxic behaviour on social media; manual moderation is slow and hard to scale |
| **Objective** | Build an NLP-based text classifier to detect bullying in short messages (comments, tweets, chat) |
| **Tech Stack** | Python, Streamlit, scikit-learn, pandas, joblib |
| **Model** | TF-IDF + Logistic Regression |
| **Input** | Single text, pasted bulk, batch CSV, chat export (WhatsApp/Telegram/Discord) |
| **Output** | Binary label (0=non-bullying, 1=bullying); per-sender chat summaries; CSV export |
| **Deployment** | GitHub + Streamlit Cloud |

---

## Architecture Explanation

**Pipeline:** Raw Text → Cleaning → TF-IDF → Logistic Regression → Prediction (0/1)

- **app.py** – Streamlit web UI: Predict, Chat analysis, Train, About
- **core/data_loader.py** – Loads `*_parsed_dataset.csv`, normalises labels
- **core/model_utils.py** – Text cleaning, model load/save, prediction, feature importance
- **core/chat_parser.py** – Multi-format chat parsing
- **train_model.py** – Trains TF-IDF + Logistic Regression, saves to `models/`

---

## CATEGORY 1 — Project Overview (10 Questions)

### Q1: What is the problem statement of your project?

**Answer:**  
Manual moderation of online content is slow, subjective, and hard to scale. Cyberbullying and toxic comments on social media need automated detection to help platforms and schools moderate content.

**Short Answer:**  
Automated detection of cyberbullying in text because manual moderation does not scale.

**Hint:**  
Explain why manual moderation fails at scale and how automation helps.

---

### Q2: What is the main objective of TrollGuard?

**Answer:**  
To build an AI-assisted text classification system that flags bullying or toxic messages in short text (comments, tweets, chat) using NLP and machine learning, with a simple web interface for predictions and chat analysis.

**Short Answer:**  
Detect cyberbullying in text using NLP and provide a Streamlit web app for predictions.

**Hint:**  
Focus on detection, NLP, and the web interface.

---

### Q3: What is the scope of your system?

**Answer:**  
English text only; message-level analysis (no full conversation context); support for single text, bulk paste, batch CSV, and chat exports (WhatsApp, Telegram, Discord). No image, audio, or video analysis.

**Short Answer:**  
English text classification only; no multimodal content.

**Hint:**  
Clarify what the system does and does not support.

---

### Q4: Who are the intended users of TrollGuard?

**Answer:**  
Educators, schools, parents, and platform moderators who need to identify harmful text in comments, group chats, or forums.

**Short Answer:**  
Schools, educators, moderators, and parents.

**Hint:**  
Think of real-world users who need moderation support.

---

### Q5: Why did you choose cyberbullying detection as the project topic?

**Answer:**  
Cyberbullying is a growing concern on social media. Automated detection can help schools and platforms identify harmful content faster and support mental health.

**Short Answer:**  
Important societal issue; automation helps scale moderation.

**Hint:**  
Link to relevance and impact.

---

### Q6: What type of classification does TrollGuard perform?

**Answer:**  
Binary classification: 0 = non-bullying (safe), 1 = bullying (flagged).

**Short Answer:**  
Binary classification (0 or 1).

**Hint:**  
Only two classes, not multi-class.

---

### Q7: What formats of input does TrollGuard accept?

**Answer:**  
Single text in a text area, pasted multiple lines, batch CSV with a text column, and chat export files (.txt) in WhatsApp, Telegram, or Discord format.

**Short Answer:**  
Single text, paste, CSV, and chat .txt (WhatsApp, Telegram, Discord).

**Hint:**  
Recall the Predict and Chat tabs.

---

### Q8: What is the output of the system?

**Answer:**  
Binary label (Bullying/Non-bullying), per-message results, per-sender summary (total messages, flagged count, flagged %), and downloadable CSV for batch and chat analysis.

**Short Answer:**  
Binary prediction, per-sender summary, CSV export.

**Hint:**  
Outputs depend on the tab (Predict vs Chat).

---

### Q9: What datasets does TrollGuard use for training?

**Answer:**  
Combined parsed datasets: aggression, attack, Kaggle toxicity, Twitter (racism, sexism), YouTube, augmented, synthetic. Around 540K+ rows with columns `Text` and `oh_label`.

**Short Answer:**  
~540K rows from Kaggle, Twitter, YouTube, augmented, synthetic; `Text` + `oh_label`.

**Hint:**  
All `*_parsed_dataset.csv` in `dataset/`.

---

### Q10: What is the approximate class distribution in your training data?

**Answer:**  
Roughly 87% non-bullying (0) and 13% bullying (1), i.e. class imbalance.

**Short Answer:**  
~87% safe, ~13% bullying (imbalanced).

**Hint:**  
Bullying is the minority class.

---

## CATEGORY 2 — System Architecture (10 Questions)

### Q11: Describe the high-level architecture of TrollGuard.

**Answer:**  
Raw text → cleaning (lowercase, remove URLs/mentions/hashtags) → TF-IDF vectorisation → Logistic Regression model → prediction (0 or 1). The Streamlit app wraps this pipeline with UI for Predict, Chat analysis, Train, and About.

**Short Answer:**  
Text → Clean → TF-IDF → Logistic Regression → 0/1 prediction.

**Hint:**  
Follow data flow from input to output.

---

### Q12: What are the main modules in your project?

**Answer:**  
`app.py` (UI), `core/data_loader.py` (data loading), `core/model_utils.py` (cleaning, model, prediction), `core/chat_parser.py` (chat parsing), `train_model.py` (training), and scripts for augmentation, synthetic data, and dataset download.

**Short Answer:**  
app.py, data_loader, model_utils, chat_parser, train_model, scripts.

**Hint:**  
Enumerate files under `core/` and `scripts/`.

---

### Q13: What does core/data_loader.py do?

**Answer:**  
Loads all `*_parsed_dataset.csv` files, expects `Text` and `oh_label`, normalises labels to 0/1, and returns a merged DataFrame with columns `text` and `label`. Also provides `get_sample_fallback()` when no CSVs exist.

**Short Answer:**  
Loads and merges parsed CSVs, normalises labels.

**Hint:**  
`load_parsed_datasets()` is the main function.

---

### Q14: What does core/model_utils.py do?

**Answer:**  
Provides `clean_text()`, `load_model_and_vectoriser()`, `predict_text()`, `predict_batch()`, `get_top_words()`, and `get_artefact_dir()`. Handles preprocessing, model loading, prediction, and feature importance.

**Short Answer:**  
Cleaning, model load/save, prediction, feature importance.

**Hint:**  
Central module for model and text processing.

---

### Q15: What does core/chat_parser.py do?

**Answer:**  
Parses chat exports from WhatsApp, Telegram, and Discord. Uses `parse_chat_from_string()` and returns a DataFrame with `timestamp`, `sender`, `message_text`. Malformed lines are kept with sender "Unknown".

**Short Answer:**  
Parses WhatsApp/Telegram/Discord chat formats; handles malformed lines.

**Hint:**  
`parse_chat_from_string()` and supported formats.

---

### Q16: How does data flow from user input to prediction?

**Answer:**  
User enters text or uploads file → Streamlit passes it to backend → `clean_text()` preprocesses → TF-IDF transforms → Logistic Regression predicts → label (0/1) is returned → Streamlit displays result.

**Short Answer:**  
Input → clean → TF-IDF → model → 0/1 → display.

**Hint:**  
Follow one path (e.g. single text prediction).

---

### Q17: Where are the trained model files stored?

**Answer:**  
In `code/models/` as `tfidf.joblib` and `logreg_model.joblib`. `outputs/` is an alternate folder but typically gitignored; the app checks `models/` first.

**Short Answer:**  
`models/tfidf.joblib` and `models/logreg_model.joblib`.

**Hint:**  
`get_artefact_dir()` returns `models/`.

---

### Q18: How does the app load the model at startup?

**Answer:**  
Using `@st.cache_resource` on `_load_model()`, which calls `load_model_and_vectoriser()`. The model is cached across reruns to reduce startup and memory use on Streamlit Cloud.

**Short Answer:**  
Lazy loading via `@st.cache_resource` and `load_model_and_vectoriser()`.

**Hint:**  
Lazy loading reduces memory and startup time.

---

### Q19: What is the purpose of the scripts folder?

**Answer:**  
`augment_data.py` (synonym augmentation), `generate_synthetic_data.py` (template-based synthetic data), `download_datasets.py` (Kaggle/Hugging Face), `train_and_save.py` (retrain for sklearn version compatibility). These run separately from the Streamlit app.

**Short Answer:**  
Data augmentation, synthetic data, dataset download, model retrain.

**Hint:**  
Support scripts, not part of the main app flow.

---

### Q20: How does the Streamlit app organise its tabs?

**Answer:**  
Four tabs: Predict (single, paste bulk, batch CSV), Chat analysis (upload/paste, format selector, date filter, export), Train model (dataset stats, train button, metrics, feature importance), and About (project description).

**Short Answer:**  
Predict, Chat analysis, Train model, About.

**Hint:**  
`st.tabs()` defines the layout.

---

## CATEGORY 3 — Programming Concepts (10 Questions)

### Q21: Why do you use `os.chdir(ROOT)` in app.py?

**Answer:**  
So relative paths like `dataset/` and `models/` resolve correctly when the app runs from different directories (e.g. Streamlit Cloud). ROOT is the directory containing `app.py`.

**Short Answer:**  
To make relative paths work; required for Streamlit Cloud.

**Hint:**  
Deployment and path handling.

---

### Q22: What does `sys.path.insert(0, ROOT)` do?

**Answer:**  
Adds the project root to Python’s module search path so `from core.data_loader import ...` works even when running from another directory.

**Short Answer:**  
Ensures `core` package can be imported.

**Hint:**  
Import path configuration.

---

### Q23: What is the purpose of `st.session_state`?

**Answer:**  
Stores data across Streamlit reruns, e.g. `pred_total`, `pred_flagged`, and `chat_analysis_result`. Without it, these values would reset on each interaction.

**Short Answer:**  
Persists data across app reruns.

**Hint:**  
Session state for counters and cached results.

---

### Q24: What does `pd.to_datetime(..., errors="coerce")` do?

**Answer:**  
Converts values to datetime; invalid values become `NaT` (Not a Time) instead of raising an error, so the rest of the DataFrame is usable.

**Short Answer:**  
Converts to datetime; invalid values become NaT.

**Hint:**  
Handling bad dates in chat parsing.

---

### Q25: Why use `re.sub()` in clean_text?

**Answer:**  
`re.sub()` uses regex to remove URLs, @mentions, #hashtags, and non-alphabetic characters, preparing text for TF-IDF.

**Short Answer:**  
Regex-based removal of noise from text.

**Hint:**  
Preprocessing for NLP.

---

### Q26: What is the purpose of `stratify=y` in train_test_split?

**Answer:**  
Ensures the train and test sets have the same proportion of each class (bullying vs non-bullying), which is important with imbalanced data.

**Short Answer:**  
Keeps class balance in train and test sets.

**Hint:**  
Class imbalance handling.

---

### Q27: What does joblib.dump() do?

**Answer:**  
Serialises (saves) the TF-IDF vectoriser and Logistic Regression model to `.joblib` files for reuse without retraining.

**Short Answer:**  
Saves model objects to disk.

**Hint:**  
Model persistence.

---

### Q28: Why use `try/except` in the chat analysis tab?

**Answer:**  
To catch parsing or prediction errors and show them to the user instead of crashing the app, improving robustness and debugging.

**Short Answer:**  
Error handling for parsing and prediction.

**Hint:**  
Robustness and user feedback.

---

### Q29: What does `total.clip(lower=1)` do in the chat summary?

**Answer:**  
Replaces zeros in `total` with 1 to avoid division by zero when computing `Flagged %` for senders with zero messages.

**Short Answer:**  
Avoids division by zero.

**Hint:**  
Safe percentage calculation.

---

### Q30: What is the purpose of the `format_hint` parameter in parse_chat_from_string?

**Answer:**  
Tells the parser which format to use: "whatsapp", "telegram", "discord", or "auto" (tries multiple formats). Reduces parsing errors when the format is known.

**Short Answer:**  
Specifies chat format; improves parsing accuracy.

**Hint:**  
Multi-format chat support.

---

## CATEGORY 4 — Streamlit Framework (10 Questions)

### Q31: Why did you choose Streamlit for the UI?

**Answer:**  
Streamlit is Python-based, needs no separate frontend, and has built-in widgets. It suits data apps and prototypes and works well for college projects and quick deployment.

**Short Answer:**  
Python-only, fast to build, good for data apps.

**Hint:**  
Simplicity and Python integration.

---

### Q32: What is st.cache_resource used for?

**Answer:**  
Caches the result of `_load_model()` across reruns so the model is loaded once and reused, reducing memory and startup time.

**Short Answer:**  
Caches model loading to avoid reloading.

**Hint:**  
Performance and memory.

---

### Q33: What is the difference between st.text_area and st.text_input?

**Answer:**  
`st.text_area` supports multiple lines; `st.text_input` is single-line. TrollGuard uses `st.text_area` for paste and chat input.

**Short Answer:**  
Text area = multi-line; text input = single line.

**Hint:**  
Use case for paste and chat.

---

### Q34: What does st.expander do?

**Answer:**  
Creates a collapsible section. Used for results, dataset stats, and feature importance to keep the UI compact.

**Short Answer:**  
Collapsible section to save space.

**Hint:**  
UI organisation.

---

### Q35: How does st.file_uploader work?

**Answer:**  
Renders a file upload button. When the user selects a file, Streamlit returns an upload object; we read it with `.read().decode()` and process the bytes.

**Short Answer:**  
Lets user upload files; we read and process them.

**Hint:**  
File upload flow.

---

### Q36: What is the purpose of the key parameter in Streamlit widgets?

**Answer:**  
Each widget needs a unique `key` so Streamlit can track its state across reruns. Reusing keys or omitting them can cause unexpected behaviour.

**Short Answer:**  
Unique ID for widget state.

**Hint:**  
State management.

---

### Q37: Why use st.columns?

**Answer:**  
To place widgets side by side (e.g. date pickers for From/To in chat analysis), improving layout.

**Short Answer:**  
Side-by-side layout.

**Hint:**  
Layout control.

---

### Q38: What does st.download_button do?

**Answer:**  
Adds a button that lets the user download a file. We use it for CSV export of predictions and chat analysis.

**Short Answer:**  
Button to download generated files.

**Hint:**  
Export functionality.

---

### Q39: What is st.spinner used for?

**Answer:**  
Shows a loading indicator while training runs, so the user knows the app is working.

**Short Answer:**  
Shows loading during long operations.

**Hint:**  
User feedback during training.

---

### Q40: What does st.progress do?

**Answer:**  
Displays a progress bar. Used during batch CSV processing to show completion percentage.

**Short Answer:**  
Progress bar for batch processing.

**Hint:**  
Batch CSV processing feedback.

---

## CATEGORY 5 — Data Processing (10 Questions)

### Q41: What preprocessing steps are applied to text?

**Answer:**  
Lowercasing, removing URLs (http, www), @mentions, #hashtags, non-alphabetic characters; collapsing multiple spaces into one; trimming leading/trailing spaces.

**Short Answer:**  
Lowercase; remove URLs, @mentions, #hashtags; keep letters only.

**Hint:**  
`clean_text()` logic.

---

### Q42: What columns must the training CSV have?

**Answer:**  
`Text` (message content) and `oh_label` (0 = non-bullying, 1 = bullying).

**Short Answer:**  
`Text` and `oh_label`.

**Hint:**  
`load_parsed_datasets()` requirement.

---

### Q43: How does the data loader normalise labels?

**Answer:**  
Converts to numeric; maps string labels like "none", "normal" to 0; other values to 1; ensures all labels are 0 or 1.

**Short Answer:**  
Converts to 0/1; maps known safe words to 0.

**Hint:**  
Label consistency for training.

---

### Q44: What is the WhatsApp chat format supported?

**Answer:**  
`DD/MM/YYYY, HH:MM - Sender: message` (e.g. `09/03/2025, 14:30 - Alice: Hello`).

**Short Answer:**  
`DD/MM/YYYY, HH:MM - Sender: message`.

**Hint:**  
chat_parser format.

---

### Q45: What is the Discord chat format supported?

**Answer:**  
`[DD/MM/YYYY HH:MM] Sender: message` (e.g. `[09/03/2025 14:30] Alice: Hello`).

**Short Answer:**  
`[DD/MM/YYYY HH:MM] Sender: message`.

**Hint:**  
chat_parser format.

---

### Q46: What happens to malformed chat lines?

**Answer:**  
They are kept with `sender="Unknown"` and `timestamp=1970-01-01` so no data is lost, and they can still be classified.

**Short Answer:**  
Stored as Unknown sender; still classified.

**Hint:**  
Robust parsing.

---

### Q47: What is MAX_INPUT_CHARS and why is it used?

**Answer:**  
Limit of 50,000 characters per input. Prevents huge inputs that could slow or crash the app and acts as a simple security measure.

**Short Answer:**  
50K character limit; prevents abuse and overload.

**Hint:**  
Security and performance.

---

### Q48: What is BATCH_CSV_LIMIT?

**Answer:**  
Max 10,000 rows for batch CSV uploads. Ensures processing finishes in time and avoids memory issues.

**Short Answer:**  
10K row limit for batch CSV.

**Hint:**  
Resource limits.

---

### Q49: How does the date filter work in chat analysis?

**Answer:**  
`pd.to_datetime()` converts timestamps; user selects From/To dates; a boolean mask filters rows; filtered DataFrame is used for analysis and export.

**Short Answer:**  
Filters chat rows by selected date range.

**Hint:**  
Chat analysis date filter.

---

### Q50: What is get_sample_fallback?

**Answer:**  
Returns a small hardcoded dataset (5 examples) when no `*_parsed_dataset.csv` files are found, so training and demo still work.

**Short Answer:**  
Fallback dataset when no CSVs exist.

**Hint:**  
Robustness when data is missing.

---

## CATEGORY 6 — Machine Learning (10 Questions)

### Q51: What is TF-IDF?

**Answer:**  
Term Frequency–Inverse Document Frequency. Measures word importance: high TF-IDF when a term is frequent in a document but rare across the corpus. Used to turn text into numerical vectors.

**Short Answer:**  
Term importance score; converts text to numbers.

**Hint:**  
Feature representation for text.

---

### Q52: What n-gram range do you use and why?

**Answer:**  
`ngram_range=(1, 2)` for unigrams and bigrams. Bigrams capture phrases like "shut up" or "go away", which matter for bullying detection.

**Short Answer:**  
Unigrams and bigrams; phrases are important.

**Hint:**  
Phrase-level patterns.

---

### Q53: What do min_df and max_df mean in TfidfVectorizer?

**Answer:**  
`min_df=2`: ignore terms in fewer than 2 documents. `max_df=0.95`: ignore terms in more than 95% of documents (too common to be useful).

**Short Answer:**  
min_df = minimum docs; max_df = maximum doc fraction.

**Hint:**  
Vocabulary filtering.

---

### Q54: Why Logistic Regression?

**Answer:**  
Simple, fast, interpretable, works well with sparse TF-IDF features, and provides coefficients for feature importance (top bullying/safe words).

**Short Answer:**  
Simple, interpretable, good for text classification.

**Hint:**  
Baseline model choice.

---

### Q55: What is the train-test split ratio?

**Answer:**  
80% train, 20% test with `test_size=0.2`, `random_state=42`, and `stratify=y` to preserve class balance.

**Short Answer:**  
80-20, stratified.

**Hint:**  
`train_test_split` parameters.

---

### Q56: How does get_top_words work?

**Answer:**  
Uses model coefficients: low coefficients → non-bullying words, high coefficients → bullying words. Returns top N words from each group. Includes handling for vocab/coef length mismatch.

**Short Answer:**  
Sorts by coefficients; low = safe, high = bullying.

**Hint:**  
Feature importance and interpretability.

---

### Q57: What metrics are used for evaluation?

**Answer:**  
Accuracy, precision, recall, F1-score (from classification_report), and confusion matrix. Report and matrix can be downloaded.

**Short Answer:**  
Accuracy, precision, recall, F1, confusion matrix.

**Hint:**  
Train tab metrics.

---

### Q58: What is the typical accuracy range of your model?

**Answer:**  
About 85–92% on test data, depending on dataset size and sampling.

**Short Answer:**  
~85–92%.

**Hint:**  
Reported accuracy in training.

---

### Q59: How does predict_batch differ from predict_text?

**Answer:**  
`predict_text` handles a single string and returns one label. `predict_batch` takes a list of strings, cleans them, transforms with TF-IDF, and returns an array of labels.

**Short Answer:**  
Single vs list; batch is more efficient for many texts.

**Hint:**  
Batch efficiency.

---

### Q60: Why guard against vocab/coef length mismatch in get_top_words?

**Answer:**  
Models saved with one scikit-learn version can have a different number of features than the TF-IDF vocabulary when loaded with another version. Truncating to the smaller length avoids IndexError.

**Short Answer:**  
Sklearn version mismatch can cause different sizes; truncation prevents crash.

**Hint:**  
Streamlit Cloud / version compatibility.

---

## CATEGORY 7 — Deployment (10 Questions)

### Q61: Where is the app deployed?

**Answer:**  
Streamlit Cloud. URL: https://trollguard-eem9mwmcp2gtqwff95wu7v.streamlit.app/

**Short Answer:**  
Streamlit Cloud.

**Hint:**  
Online hosting.

---

### Q62: What is the GitHub repository URL?

**Answer:**  
https://github.com/itsmehari/TrollGuard

**Short Answer:**  
github.com/itsmehari/TrollGuard.

**Hint:**  
Version control and code sharing.

---

### Q63: How does Streamlit Cloud run the app?

**Answer:**  
It clones the GitHub repo, installs dependencies from `requirements.txt`, and runs `streamlit run code/app.py`. Main file path is set to `code/app.py`.

**Short Answer:**  
Clones repo, installs deps, runs streamlit.

**Hint:**  
Deployment flow.

---

### Q64: What is in requirements.txt?

**Answer:**  
pandas, numpy, scikit-learn, joblib, streamlit (with version constraints like `>=1.20.0`).

**Short Answer:**  
pandas, numpy, scikit-learn, joblib, streamlit.

**Hint:**  
Dependencies.

---

### Q65: Why use lazy model loading for deployment?

**Answer:**  
Loading the model only when needed (e.g. first prediction) and caching it reduces initial memory and startup time, helping on Streamlit Cloud’s limited resources.

**Short Answer:**  
Reduces startup memory; loads on first use.

**Hint:**  
Resource limits on cloud.

---

### Q66: What is .streamlit/config.toml used for?

**Answer:**  
Streamlit configuration: theme (e.g. light), port, CORS, XSRF protection, and other server settings.

**Short Answer:**  
Streamlit theme and server config.

**Hint:**  
Configuration.

---

### Q67: What is version control and why use Git?

**Answer:**  
Version control tracks code changes. Git lets you commit, branch, and push to GitHub, and integrate with Streamlit Cloud for deployment.

**Short Answer:**  
Tracks changes; enables collaboration and deployment.

**Hint:**  
Standard software practice.

---

### Q68: What does git push do?

**Answer:**  
Sends local commits to the remote repository (e.g. GitHub). Pushing triggers Streamlit Cloud to redeploy if connected.

**Short Answer:**  
Sends commits to remote; triggers redeploy.

**Hint:**  
Deployment workflow.

---

### Q69: Why are model files committed to the repo?

**Answer:**  
Streamlit Cloud runs from the repo. Committing `tfidf.joblib` and `logreg_model.joblib` in `models/` ensures the app can load the model without training on deploy.

**Short Answer:**  
App needs models in repo to load them on deploy.

**Hint:**  
Model availability in cloud.

---

### Q70: What is scripts/train_and_save.py used for?

**Answer:**  
Retrains the model with the current scikit-learn version (uses up to 30k samples) to avoid version mismatch on Streamlit Cloud. Saves updated `tfidf.joblib` and `logreg_model.joblib`.

**Short Answer:**  
Retrain model for sklearn version compatibility.

**Hint:**  
Version compatibility fix.

---

## CATEGORY 8 — Security and Reliability (10 Questions)

### Q71: How do you handle invalid CSV uploads?

**Answer:**  
Check for `text` or `Text` column; if missing, show an error. Wrap processing in try/except to catch read/parse errors and display them to the user.

**Short Answer:**  
Check column names; use try/except for errors.

**Hint:**  
Input validation and error handling.

---

### Q72: How do you handle parsing errors in chat?

**Answer:**  
try/except around `parse_chat_from_string()`. On error, show a user-friendly message with `st.error()` instead of crashing.

**Short Answer:**  
try/except; show error message.

**Hint:**  
Robust parsing.

---

### Q73: Is user input stored or logged?

**Answer:**  
Input is processed in memory and not stored long-term. Session state holds prediction counts and cached chat results for the session only.

**Short Answer:**  
Processed in memory; not persisted.

**Hint:**  
Privacy and data handling.

---

### Q74: What input limits are enforced?

**Answer:**  
50,000 characters per text/paste/upload; 10,000 rows per batch CSV. Enforced by truncation and limiting before processing.

**Short Answer:**  
50K chars; 10K rows for batch.

**Hint:**  
Security and resource limits.

---

### Q75: How does the app behave when the model is not loaded?

**Answer:**  
Shows a warning to go to the Train tab. Predictions are skipped; no crash.

**Short Answer:**  
Shows warning; no prediction.

**Hint:**  
Graceful handling of missing model.

---

### Q76: What happens if all chat lines are malformed?

**Answer:**  
Each line is stored with sender "Unknown". Classification still runs; results include all messages. Per-sender summary treats "Unknown" as one sender.

**Short Answer:**  
Parsed as Unknown; still classified.

**Hint:**  
Robustness.

---

### Q77: How do you avoid division by zero in summaries?

**Answer:**  
Use `total.clip(lower=1)` so denominators are at least 1 when computing `Flagged %`.

**Short Answer:**  
`clip(lower=1)` to avoid divide by zero.

**Hint:**  
Safe percentage calculation.

---

### Q78: What if the dataset folder has no CSV files?

**Answer:**  
`load_parsed_datasets()` returns an empty DataFrame. The app uses `get_sample_fallback()` to provide minimal data for demo and training.

**Short Answer:**  
Fallback to sample data.

**Hint:**  
Fallback strategy.

---

### Q79: How is the model loading failure handled?

**Answer:**  
`load_model_and_vectoriser()` returns (None, None) on failure. The app checks for None and shows "No model found" with a retrain prompt.

**Short Answer:**  
Returns None; app shows warning.

**Hint:**  
Defensive checks.

---

### Q80: Why use errors="ignore" when decoding uploaded files?

**Answer:**  
Prevents crashes when the file has invalid UTF-8. Invalid bytes are replaced instead of raising an error.

**Short Answer:**  
Handles non-UTF8 characters safely.

**Hint:**  
Encoding robustness.

---

## CATEGORY 9 — Limitations and Improvements (10 Questions)

### Q81: What are the main limitations of TrollGuard?

**Answer:**  
English only; message-level (no full conversation context); no image/audio/video; performance depends on training data; sarcasm and coded language are hard; no real-time API by default.

**Short Answer:**  
English only; message-level; no multimodal; dataset-dependent.

**Hint:**  
Scope and constraints.

---

### Q82: What would you improve next?

**Answer:**  
Add multilingual support, a REST API for integration, transformer-based models (e.g. BERT), and better handling of sarcasm and context.

**Short Answer:**  
Multilingual, API, BERT, context.

**Hint:**  
Future enhancements.

---

### Q83: Why is F1 for the bullying class often lower?

**Answer:**  
Class imbalance: bullying is ~13%. The model may favour the majority class, so recall for bullying can be lower and F1 suffers.

**Short Answer:**  
Class imbalance; minority class harder.

**Hint:**  
Imbalanced data.

---

### Q84: How could you handle sarcasm better?

**Answer:**  
Use contextual models (e.g. BERT), conversation history, or extra labels for sarcasm. Current TF-IDF + LR does not model context.

**Short Answer:**  
Contextual models; conversation history.

**Hint:**  
Context and semantics.

---

### Q85: What is the limitation of TF-IDF for this task?

**Answer:**  
TF-IDF ignores word order and context. Phrases like "not bad" vs "bad" are treated similarly. No semantic understanding.

**Short Answer:**  
No word order or context.

**Hint:**  
Bag-of-words limitation.

---

### Q86: Why not use deep learning (e.g. BERT)?

**Answer:**  
TF-IDF + LR is simpler, faster, and easier to explain for a college project. BERT needs more data, compute, and complexity. It could be a future improvement.

**Short Answer:**  
Simpler for project; BERT as future work.

**Hint:**  
Trade-off between simplicity and power.

---

### Q87: How could you improve chat parsing?

**Answer:**  
Support more formats, handle multi-line messages, and add timezone handling. Current parser covers common WhatsApp/Telegram/Discord formats.

**Short Answer:**  
More formats; multi-line messages.

**Hint:**  
Format diversity.

---

### Q88: What real-world challenges would you face in production?

**Answer:**  
Scale (volume of messages), low latency, fairness across demographics, privacy, and need for human-in-the-loop moderation.

**Short Answer:**  
Scale, speed, fairness, privacy, human review.

**Hint:**  
Production considerations.

---

### Q89: How could you reduce false positives?

**Answer:**  
Use confidence thresholds, add human review for borderline cases, improve training data, and possibly use ensemble or more advanced models.

**Short Answer:**  
Confidence threshold; better data; human review.

**Hint:**  
Precision improvement.

---

### Q90: What ethical considerations apply to cyberbullying detection?

**Answer:**  
 privacy (what data is stored), bias (fair across groups), transparency (explainability), and the need for human review before serious actions.

**Short Answer:**  
Privacy, bias, transparency, human review.

**Hint:**  
Responsible AI.

---

## CATEGORY 10 — Practical Viva Questions (10 Questions)

### Q91: Why did you choose Streamlit over Flask or Django?

**Answer:**  
Streamlit is Python-only and built for data apps. No HTML/CSS/JS or separate frontend. Faster to build and suitable for a prototype/demo.

**Short Answer:**  
Easier for data apps; no separate frontend.

**Hint:**  
Development speed and use case.

---

### Q92: What was the biggest challenge in this project?

**Answer:**  
Examples: scikit-learn version mismatch on Streamlit Cloud (solved by retraining), chat format variations, handling malformed input, and class imbalance.

**Short Answer:**  
Version mismatch; format variations; imbalanced data.

**Hint:**  
Personal experience; be honest.

---

### Q93: How long did the project take?

**Answer:**  
Answer based on actual experience. Mention phases: data collection, preprocessing, model training, app development, deployment, testing.

**Short Answer:**  
[Your estimate] – data, model, app, deploy.

**Hint:**  
Be realistic.

---

### Q94: What would you do differently if you started again?

**Answer:**  
Plan dataset format early, set up version control from the start, test deployment sooner, and possibly try a transformer model for comparison.

**Short Answer:**  
Better planning; earlier deployment testing.

**Hint:**  
Lessons learned.

---

### Q95: How did you test your system?

**Answer:**  
Manual testing: single text, paste bulk, CSV upload, chat analysis. Checked accuracy on test set. Verified deployment on Streamlit Cloud.

**Short Answer:**  
Manual UI testing; test set metrics; deployment check.

**Hint:**  
Validation approach.

---

### Q96: Where did you get the datasets?

**Answer:**  
Public sources: Kaggle (e.g. Jigsaw toxicity), Hugging Face (e.g. OLID, hate speech), plus augmentation and synthetic generation scripts.

**Short Answer:**  
Kaggle, Hugging Face, augmented, synthetic.

**Hint:**  
Data sources.

---

### Q97: Can this system run offline?

**Answer:**  
Yes. After `pip install` and `python train_model.py`, run `streamlit run app.py` locally. No internet needed except for initial install and Streamlit Cloud deployment.

**Short Answer:**  
Yes; run locally with streamlit run.

**Hint:**  
Local vs cloud.

---

### Q98: What is the difference between frontend and backend in your project?

**Answer:**  
Frontend: Streamlit widgets (st.text_area, st.button, st.dataframe, etc.). Backend: core modules (data_loader, model_utils, chat_parser) and prediction logic. Both are in the same Python codebase.

**Short Answer:**  
Frontend = Streamlit UI; Backend = core logic.

**Hint:**  
Separation of concerns.

---

### Q99: How would you explain your project in one minute?

**Answer:**  
"TrollGuard is a cyberbullying detection system. Users enter text or upload chat exports. The system preprocesses text with TF-IDF and classifies it with Logistic Regression as bullying or not. Results and per-sender summaries can be exported. The app is built in Streamlit and deployed on Streamlit Cloud."

**Short Answer:**  
Text → Clean → TF-IDF → LR → Bullying/Non-bullying; Streamlit app on cloud.

**Hint:**  
Elevator pitch.

---

### Q100: What did you learn from this project?

**Answer:**  
NLP basics (TF-IDF, text cleaning), ML workflow (train–test, evaluation), Streamlit for web apps, deployment on Streamlit Cloud, and handling real-world issues (imbalanced data, format variations).

**Short Answer:**  
NLP, ML workflow, Streamlit, deployment, robustness.

**Hint:**  
Personal learning outcomes.

---

## Rapid Fire Questions (10)

| # | Question | Short Answer |
|---|----------|--------------|
| 1 | What is TF-IDF? | Term frequency–inverse document frequency; converts text to numbers |
| 2 | What is Logistic Regression? | Classification algorithm; outputs probability, we use for binary 0/1 |
| 3 | What is Streamlit? | Python library for building data web apps quickly |
| 4 | What is Git? | Version control system for tracking code changes |
| 5 | What is deployment? | Making an app available online for users |
| 6 | What is preprocessing? | Cleaning and transforming raw data before model use |
| 7 | What is a confusion matrix? | 2×2 table of actual vs predicted labels |
| 8 | What is F1-score? | Harmonic mean of precision and recall |
| 9 | What is joblib? | Python library for saving/loading model objects |
| 10 | What is binary classification? | Predict one of two classes (0 or 1) |

---

## Key Concepts Students Must Remember

1. **Pipeline:** Raw Text → Clean → TF-IDF → Logistic Regression → 0/1  
2. **Modules:** app.py, data_loader, model_utils, chat_parser  
3. **Input:** Single text, paste, CSV, chat (WhatsApp/Telegram/Discord)  
4. **Output:** Binary prediction, per-sender summary, CSV export  
5. **Model:** TF-IDF (ngram 1–2) + Logistic Regression  
6. **Deployment:** GitHub + Streamlit Cloud  
7. **Limitations:** English only; message-level; no multimodal  
8. **Class balance:** ~87% non-bullying, ~13% bullying  
