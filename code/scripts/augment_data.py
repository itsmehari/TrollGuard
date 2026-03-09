"""
TrollGuard – Augment existing datasets (synonym replacement, etc.)

Usage: python scripts/augment_data.py [--factor 0.3]

Output: dataset/augmented_parsed_dataset.csv
"""

import os
import sys
import argparse
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
from core.data_loader import load_parsed_datasets, get_datasets_dir

SYNONYMS = {
    "stupid": ["dumb", "idiot", "fool"],
    "ugly": ["hideous", "gross"],
    "hate": ["loathe", "despise"],
    "loser": ["failure", "pathetic"],
    "pathetic": ["pitiful", "weak"],
    "trash": ["garbage", "rubbish"],
    "worthless": ["useless", "pointless"],
}


def augment_synonym(text):
    words = text.lower().split()
    out = [random.choice(SYNONYMS.get(w, [w])) if w in SYNONYMS else w for w in words]
    return " ".join(out)


def augment_swap(text):
    words = text.split()
    if len(words) < 3:
        return text
    i = random.randint(0, len(words) - 2)
    words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


def augment_one(text):
    fns = [augment_synonym, augment_swap]
    return random.choice(fns)(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=float, default=0.3)
    args = ap.parse_args()

    df = load_parsed_datasets()
    if df.empty:
        print("No datasets found.")
        return

    n = int(len(df) * args.factor)
    idx = random.sample(range(len(df)), min(n, len(df)))
    aug_rows = []
    for i in idx:
        row = df.iloc[i]
        new_text = augment_one(str(row["text"]))
        if new_text.strip() and new_text != str(row["text"]):
            aug_rows.append({"Text": new_text, "oh_label": int(row["label"])})

    aug_df = pd.DataFrame(aug_rows)
    out_path = os.path.join(get_datasets_dir(), "augmented_parsed_dataset.csv")
    aug_df.to_csv(out_path, index=False)
    print(f"Generated {len(aug_df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
