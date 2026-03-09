"""
TrollGuard – Download datasets from Kaggle / Hugging Face.

Usage:
  pip install kaggle datasets
  Set KAGGLE_USERNAME and KAGGLE_KEY (or place kaggle.json in ~/.kaggle/)
  python scripts/download_datasets.py

Output: Saves *_parsed_dataset.csv to dataset/
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
from core.data_loader import get_datasets_dir

OUTPUT_DIR = get_datasets_dir()


def to_binary_label(val):
    v = str(val).strip().lower()
    safe = {"0", "0.0", "none", "normal", "non-toxic", "safe", "neutral"}
    if v in safe:
        return 0
    try:
        return 1 if float(val) > 0 else 0
    except (ValueError, TypeError):
        return 1


def download_huggingface_olid():
    try:
        from datasets import load_dataset
        ds = load_dataset("olid", "olid", split="test")
        df = pd.DataFrame(ds)
        df["Text"] = df["tweet"]
        df["oh_label"] = (df["subtask_a"] == "OFF").astype(int)
        path = os.path.join(OUTPUT_DIR, "olid_parsed_dataset.csv")
        df[["Text", "oh_label"]].to_csv(path, index=False)
        print(f"Saved {len(df)} rows to {path}")
    except Exception as e:
        print("HuggingFace OLID:", e, "- pip install datasets")


def download_huggingface_hate_speech():
    try:
        from datasets import load_dataset
        ds = load_dataset("hate_speech_offensive", split="train")
        df = pd.DataFrame(ds)
        df["Text"] = df["tweet"]
        df["oh_label"] = (df["class"] < 2).astype(int)
        path = os.path.join(OUTPUT_DIR, "hate_speech_parsed_dataset.csv")
        df[["Text", "oh_label"]].to_csv(path, index=False)
        print(f"Saved {len(df)} rows to {path}")
    except Exception as e:
        print("HuggingFace hate_speech:", e)


def download_kaggle_jigsaw():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("julian3833/jigsaw-toxic-comment-train", path=OUTPUT_DIR, unzip=True)
        path = os.path.join(OUTPUT_DIR, "train.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["oh_label"] = (df["toxic"] + df["severe_toxic"] + df["obscene"] + df["threat"] + df["insult"] + df["identity_hate"]).clip(upper=1)
            df["Text"] = df["comment_text"]
            p = os.path.join(OUTPUT_DIR, "jigsaw_toxic_parsed_dataset.csv")
            df[["Text", "oh_label"]].to_csv(p, index=False)
            print(f"Saved {len(df)} rows to {p}")
    except Exception as e:
        print("Kaggle Jigsaw:", e, "- pip install kaggle, set KAGGLE credentials")


def main():
    print("Downloading datasets...")
    download_huggingface_olid()
    download_huggingface_hate_speech()
    download_kaggle_jigsaw()
    print("Done.")


if __name__ == "__main__":
    main()
