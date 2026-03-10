"""
================================================================================
TrollGuard – Download Datasets from Kaggle and Hugging Face
================================================================================

WHAT THIS FILE DOES (in simple terms):
This script downloads publicly available datasets for toxic/offensive/hate speech
detection and converts them into our standard format (Text, oh_label). Saves
them as *_parsed_dataset.csv in the dataset/ folder.

DATASETS:
1. Hugging Face OLID - Offensive Language Identification (tweets)
2. Hugging Face hate_speech_offensive - Hate speech and offensive tweets
3. Kaggle Jigsaw Toxic Comment - Wikipedia comments with toxicity labels

PREREQUISITES:
- For Hugging Face: pip install datasets
- For Kaggle: pip install kaggle, then set KAGGLE_USERNAME and KAGGLE_KEY
  (or place kaggle.json in ~/.kaggle/ or C:\Users\You\.kaggle\)

Usage: python scripts/download_datasets.py

================================================================================
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
    """
    Convert various label formats to 0 (safe) or 1 (bullying/offensive).
    Different datasets use different labels; we normalise to our format.
    """
    v = str(val).strip().lower()
    safe = {"0", "0.0", "none", "normal", "non-toxic", "safe", "neutral"}
    if v in safe:
        return 0
    try:
        return 1 if float(val) > 0 else 0
    except (ValueError, TypeError):
        return 1


def download_huggingface_olid():
    """
    Download OLID (Offensive Language Identification Dataset) from Hugging Face.

    OLID has subtask_a: "OFF" (offensive) or "NOT" (not offensive).
    We map OFF -> 1 (bullying), NOT -> 0 (safe).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("olid", "olid", split="test")
        df = pd.DataFrame(ds)
        df["Text"] = df["tweet"]
        df["oh_label"] = (df["subtask_a"] == "OFF").astype(int)  # OFF = offensive -> 1
        path = os.path.join(OUTPUT_DIR, "olid_parsed_dataset.csv")
        df[["Text", "oh_label"]].to_csv(path, index=False)
        print(f"Saved {len(df)} rows to {path}")
    except Exception as e:
        print("HuggingFace OLID:", e, "- pip install datasets")


def download_huggingface_hate_speech():
    """
    Download hate_speech_offensive dataset from Hugging Face.

    class: 0 = hate, 1 = offensive, 2 = neither.
    We map 0 and 1 -> 1 (bullying), 2 -> 0 (safe).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("hate_speech_offensive", split="train")
        df = pd.DataFrame(ds)
        df["Text"] = df["tweet"]
        df["oh_label"] = (df["class"] < 2).astype(int)  # 0,1 = hate/offensive
        path = os.path.join(OUTPUT_DIR, "hate_speech_parsed_dataset.csv")
        df[["Text", "oh_label"]].to_csv(path, index=False)
        print(f"Saved {len(df)} rows to {path}")
    except Exception as e:
        print("HuggingFace hate_speech:", e)


def download_kaggle_jigsaw():
    """
    Download Jigsaw Toxic Comment dataset from Kaggle.

    Jigsaw has multiple columns: toxic, severe_toxic, obscene, threat, insult, identity_hate.
    If ANY of these is > 0, we label as bullying (1). Otherwise 0.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            "julian3833/jigsaw-toxic-comment-train",
            path=OUTPUT_DIR,
            unzip=True
        )
        path = os.path.join(OUTPUT_DIR, "train.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Sum all toxicity columns; if any > 0, label = 1. clip(upper=1) keeps max 1.
            df["oh_label"] = (
                df["toxic"] + df["severe_toxic"] + df["obscene"] +
                df["threat"] + df["insult"] + df["identity_hate"]
            ).clip(upper=1)
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
