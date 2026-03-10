"""
================================================================================
TrollGuard – Synthetic Data Generator
================================================================================

WHAT THIS FILE DOES (in simple terms):
Sometimes you don't have enough labelled data. This script creates NEW
fake-but-realistic messages using templates. For example:
  - Bullying: "you are so {adj}" -> "you are so stupid" (random adjective)
  - Safe: "thanks for your help" (no placeholder, used as-is)

No API or internet required - everything is predefined. Good for:
  - Testing the pipeline with more data
  - Bootstrapping when you have few real examples
  - Creating balanced datasets (half bullying, half safe)

OUTPUT: dataset/synthetic_parsed_dataset.csv

Usage: python scripts/generate_synthetic_data.py [--count 5000] [--output path]
  --count: How many rows to generate (default 5000)
  --output: Custom output path (optional)

================================================================================
"""

import os
import sys
import argparse
import random

# Path setup
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
from core.data_loader import get_datasets_dir

# Bullying templates. {adj}, {verb}, {noun} are placeholders we fill randomly.
# Example: "you are so {adj}" with adj="stupid" -> "you are so stupid"
BULLY_TEMPLATES = [
    "you are so {adj}",
    "you {verb} {noun}",
    "shut up you {noun}",
    "nobody likes you",
    "you will never amount to anything",
    "get lost {noun}",
    "you are such a {noun}",
    "why are you so {adj}",
    "you deserve nothing",
    "you are worthless",
    "everyone hates you",
    "you are a {adj} {noun}",
    "go die somewhere",
    "you ruin everything",
    "you are pathetic",
    "you have no friends",
    "stop being so {adj}",
    "you are a failure",
    "you make me sick",
]

# Safe templates - no placeholders, used as-is
SAFE_TEMPLATES = [
    "thanks for your help",
    "have a good day",
    "that sounds interesting",
    "i agree with you",
    "let me know when you are free",
    "nice to meet you",
    "see you tomorrow",
    "hope you feel better",
    "good luck with the project",
    "we can discuss this later",
    "please share the document",
    "i will check and get back",
    "thanks for the update",
    "looking forward to it",
    "no problem at all",
]

# Options for filling placeholders in bullying templates
ADJECTIVES = ["stupid", "ugly", "dumb", "pathetic", "useless", "annoying", "worthless", "disgusting"]
VERBS = ["ruin", "destroy", "hate", "deserve"]
NOUNS = ["loser", "idiot", "failure", "nobody", "waste", "joke", "freak"]


def fill_template(template: str) -> str:
    """
    Replace {adj}, {verb}, {noun} with random choices from our lists.
    Example: "you are so {adj}" -> "you are so stupid" or "you are so ugly"
    """
    placeholders = {"{adj}": ADJECTIVES, "{verb}": VERBS, "{noun}": NOUNS}
    out = template
    for ph, opts in placeholders.items():
        if ph in out:
            out = out.replace(ph, random.choice(opts))
    return out


def generate_template_based(count: int) -> pd.DataFrame:
    """
    Generate 'count' rows: roughly half bullying (label=1), half safe (label=0).
    Shuffle so they're mixed. Each row has Text and oh_label.
    """
    rows = []
    half = count // 2
    for _ in range(half):
        t = random.choice(BULLY_TEMPLATES)
        rows.append({"Text": fill_template(t), "oh_label": 1})
    for _ in range(count - half):
        t = random.choice(SAFE_TEMPLATES)
        rows.append({"Text": t, "oh_label": 0})
    random.shuffle(rows)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5000, help="Number of rows to generate")
    ap.add_argument("--output", default=None, help="Output CSV path (optional)")
    args = ap.parse_args()

    df = generate_template_based(args.count)
    out = args.output or os.path.join(get_datasets_dir(), "synthetic_parsed_dataset.csv")
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} synthetic rows -> {out}")


if __name__ == "__main__":
    main()
