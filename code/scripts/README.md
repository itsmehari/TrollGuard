# TrollGuard – Data Scripts

## 1. Download Datasets

Download more datasets from Kaggle and Hugging Face:

```bash
cd code
pip install kaggle datasets
# For Kaggle: set KAGGLE_USERNAME, KAGGLE_KEY or add ~/.kaggle/kaggle.json
python scripts/download_datasets.py
```

Adds: `olid_parsed_dataset.csv`, `hate_speech_parsed_dataset.csv`, `jigsaw_toxic_parsed_dataset.csv`

---

## 2. Augment Existing Data

Create ~20–50% more training samples via synonym replacement:

```bash
python scripts/augment_data.py --factor 0.3
```

Output: `dataset/augmented_parsed_dataset.csv`

---

## 3. Generate Synthetic Data

Template-based synthetic examples (no API):

```bash
python scripts/generate_synthetic_data.py --count 5000
```

Output: `dataset/synthetic_parsed_dataset.csv`

---

## Full Pipeline (max data)

```bash
python scripts/download_datasets.py
python scripts/augment_data.py --factor 0.4
python scripts/generate_synthetic_data.py --count 10000
python train_model.py
```
