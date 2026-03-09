# TrollGuard – Cyberbullying Detector

AI-assisted text-based cyberbullying detection system using NLP and Machine Learning.

**B.Sc. Computer Science Final Year Project**

## Tech Stack

- Python 3.8+
- scikit-learn (TF-IDF, Logistic Regression)
- pandas, numpy
- Streamlit (web app)

## Quick Start

**Run all commands from the `code/` folder.**

### 1. Install dependencies

```bash
cd code
pip install -r requirements.txt
```

### 2. Train the model (first time)

```bash
python train_model.py
```

### 3. Run Streamlit app

```bash
streamlit run app.py
```

Open http://localhost:8501

## Project Structure

```
current-project/
├── TrollGuard_Project_Report.docx
├── TrollGuard_Project_Report.md
├── TrollGuard_Presentation.md
├── TrollGuard_Presentation_Final_Gloriya_2025.pptx
├── REMAINING_TASKS.md
└── code/
    ├── app.py                 # Streamlit app
├── train_model.py         # Train and save model
├── core/
│   ├── data_loader.py     # Load datasets
│   ├── model_utils.py     # clean_text, predict
│   └── chat_parser.py     # WhatsApp chat parser
├── dataset/               # CSV datasets + sample_chat.txt
├── outputs/               # Saved model (tfidf.joblib, logreg_model.joblib)
└── requirements.txt
```

## Dataset Format

- Place `*_parsed_dataset.csv` files in `dataset/` (or `datasets/`)
- Required columns: `Text`, `oh_label` (0 = non-bullying, 1 = bullying)
- See `dataset/DATASET_FORMAT.md` for details

## Streamlit Cloud Deployment

1. Push project to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in and "New app"
4. Select repo, branch, **Main file path:** `current-project/code/app.py`
5. Add `requirements.txt` in the same repo
6. Deploy

**Note:** If no model exists in the repo, use the **Train model** tab in the app after deploy (or run `train_model.py` locally and commit `outputs/*.joblib`). The app uses sample fallback data when datasets are empty.

## Notebook (Google Colab)

Use `ipynb_file/troll_guard_full_python_code_colab.ipynb`:

1. Upload project folder to Google Drive
2. Mount Drive and run notebook
3. Paths auto-detect Colab vs local

## Report & Presentation

Report and presentation files are in the parent folder (`current-project/`):

- **Report:** `TrollGuard_Project_Report.docx`, `TrollGuard_Project_Report.md`
- **Presentation:** `TrollGuard_Presentation_Final_Gloriya_2025.pptx`, `TrollGuard_Presentation.md` (Marp source)

### Export to DOCX (from current-project folder)

```bash
cd ..
pandoc TrollGuard_Project_Report.md -o TrollGuard_Project_Report.docx
```
