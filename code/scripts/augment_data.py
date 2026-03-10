"""
================================================================================
TrollGuard – Data Augmentation Script
================================================================================

WHAT THIS FILE DOES (in simple terms):
Data augmentation = creating new training examples from existing ones.
More diverse training data often leads to a better model. This script:
  1. Takes your existing dataset
  2. Randomly picks a fraction (default 30%) of the rows
  3. For each row, creates a variation by either:
     - Synonym replacement: "stupid" -> "dumb" or "idiot" (random choice)
     - Word swap: randomly swap two adjacent words ("you are" -> "are you")
  4. Saves the new examples to dataset/augmented_parsed_dataset.csv

WHY AUGMENTATION HELPS:
- The model sees more sentence variations
- Reduces overfitting (memorising exact phrases)
- Especially useful when you have limited original data

Usage: python scripts/augment_data.py [--factor 0.3]
  --factor: Fraction of dataset to augment (0.3 = 30%). Default 0.3

================================================================================
"""

import os
import sys
import argparse
import random

# Path setup: scripts/augment_data.py -> code/ = ROOT
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
from core.data_loader import load_parsed_datasets, get_datasets_dir

# Synonym dictionary: for common bullying-related words, we have alternatives.
# When we augment, we randomly pick one of the alternatives (or keep original).
# Example: "You are stupid" might become "You are dumb" or "You are idiot"
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
    """
    Replace words with synonyms where we have them in our dictionary.
    Each word is checked; if it's in SYNONYMS, we randomly pick a replacement.
    """
    words = text.lower().split()
    out = [random.choice(SYNONYMS.get(w, [w])) if w in SYNONYMS else w for w in words]
    return " ".join(out)


def augment_swap(text):
    """
    Randomly swap two adjacent words. Adds word-order variation.
    Example: "you are stupid" -> "are you stupid" (if we swap "you" and "are")
    We need at least 3 words to swap meaningfully.
    """
    words = text.split()
    if len(words) < 3:
        return text
    i = random.randint(0, len(words) - 2)  # Pick position (can't swap last word)
    words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


def augment_one(text):
    """Apply one random augmentation: either synonym replacement OR word swap."""
    fns = [augment_synonym, augment_swap]
    return random.choice(fns)(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=float, default=0.3,
                    help="Fraction of dataset to augment (default 0.3)")
    args = ap.parse_args()

    df = load_parsed_datasets()
    if df.empty:
        print("No datasets found.")
        return

    # How many rows to augment? factor=0.3 means 30% of the dataset
    n = int(len(df) * args.factor)
    # Randomly pick n row indices (without replacement)
    idx = random.sample(range(len(df)), min(n, len(df)))
    aug_rows = []
    for i in idx:
        row = df.iloc[i]
        new_text = augment_one(str(row["text"]))
        # Only add if we got something new and non-empty
        if new_text.strip() and new_text != str(row["text"]):
            aug_rows.append({"Text": new_text, "oh_label": int(row["label"])})

    aug_df = pd.DataFrame(aug_rows)
    out_path = os.path.join(get_datasets_dir(), "augmented_parsed_dataset.csv")
    aug_df.to_csv(out_path, index=False)
    print(f"Generated {len(aug_df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
