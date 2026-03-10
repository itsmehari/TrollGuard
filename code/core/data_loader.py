"""
================================================================================
TrollGuard – Data Loader Utility
================================================================================

WHAT THIS FILE DOES (in simple terms):
This module is responsible for loading and preparing the training data. Think of
it as the "librarian" of the project:
  - It finds all the CSV files that contain labelled messages (bullying or not)
  - It reads them and merges them into one big table
  - It makes sure every label is either 0 (non-bullying) or 1 (bullying)
  - If no data exists, it provides 5 sample messages so the app can still demo

Expected CSV format: Each file named *_parsed_dataset.csv with columns:
  - Text: the message content
  - oh_label: 0 = non-bullying (safe), 1 = bullying (flagged)

================================================================================
"""

import os
import glob
import pandas as pd


def get_project_root() -> str:
    """
    Find the project root folder (where dataset/ and models/ live).

    WHY WE NEED THIS:
    The code might run from different places:
      - On your computer (e.g. C:\\...\\code\\)
      - On Google Colab (e.g. /content/drive/MyDrive/TrollGuard_Project/)
      - On Streamlit Cloud (e.g. /app/.../code/)

    We check for Colab paths first, then fall back to the folder containing
    core/ (which is the parent of the parent of this file).

    Returns: Full path to the project root folder
    """
    # Check for Google Colab / Google Drive mount (common for student projects)
    if os.path.exists("/content/drive/MyDrive/TrollGuard_Project"):
        return "/content/drive/MyDrive/TrollGuard_Project"
    if os.path.exists("/content/drive/MyDrive/TrollGuard_Project_Gloryia_2025"):
        return "/content/drive/MyDrive/TrollGuard_Project_Gloryia_2025"

    # Standard case: this file is at code/core/data_loader.py
    # dirname twice: data_loader.py -> core/ -> code/ (project root for our structure)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return root


def get_datasets_dir() -> str:
    """
    Return the path to the folder containing our datasets.

    We support two folder names: 'datasets' or 'dataset'. Some projects use
    plural, some singular. We check both and return whichever exists.
    If neither exists, we return the path where we'd expect 'dataset/'.

    Returns: Full path like "C:\\...\\code\\dataset" or "C:\\...\\code\\datasets"
    """
    root = get_project_root()
    for name in ("datasets", "dataset"):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    # Default to "dataset" even if folder doesn't exist yet
    return os.path.join(root, "dataset")


def load_parsed_datasets(datasets_dir: str = None) -> pd.DataFrame:
    """
    Load all *_parsed_dataset.csv files, merge them into one table, and normalise labels.

    HOW IT WORKS (step by step):
    1. Find all CSV files matching *_parsed_dataset.csv (e.g. twitter_parsed_dataset.csv)
    2. Read each file and check it has 'Text' and 'oh_label' columns
    3. Merge all rows into one DataFrame
    4. Convert labels to 0 or 1 (handle "none", "normal", "non-toxic" etc. -> 0)
    5. Remove any rows with missing data

    Returns: DataFrame with columns [text, label]
             - text: the message
             - label: 0 (safe) or 1 (bullying)
    """
    if datasets_dir is None:
        datasets_dir = get_datasets_dir()

    # glob finds all files matching a pattern (like * in file names)
    pattern = os.path.join(datasets_dir, "*_parsed_dataset.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        return pd.DataFrame(columns=["text", "label"])

    dfs = []
    for path in csv_files:
        try:
            tmp = pd.read_csv(path)
            # Only use files that have the required columns
            if "Text" not in tmp.columns or "oh_label" not in tmp.columns:
                continue
            # Keep only Text and oh_label, rename to lowercase for consistency
            tmp = tmp[["Text", "oh_label"]].rename(
                columns={"Text": "text", "oh_label": "label"}
            )
            dfs.append(tmp)
        except Exception:
            # If a file is corrupted or malformed, skip it
            continue

    if not dfs:
        return pd.DataFrame(columns=["text", "label"])

    # concat = stick all the small tables together into one big table
    df = pd.concat(dfs, ignore_index=True)

    # ---------- LABEL NORMALISATION ----------
    # Different datasets use different labels: 0/1, "none"/"off", "non-toxic"/"toxic", etc.
    # We need everything as 0 (safe) or 1 (bullying).

    try:
        # First try: convert to numbers directly (0.0, 1.0, etc.)
        df["label"] = pd.to_numeric(df["label"])
    except Exception:
        # Second try: handle string labels
        # These words mean "safe" -> map to 0
        NON_BULLY = {
            "none", "normal", "non-toxic", "non_toxic",
            "non-aggressive", "not_cyberbullying", "safe", "neutral"
        }

        def to_binary(x):
            s = str(x).strip().lower()
            if s in NON_BULLY or s == "0":
                return 0
            return 1  # Anything else -> bullying

        df["label"] = df["label"].apply(to_binary)

    # Final check: ensure we only have 0 or 1 (some datasets might have 2, 3, etc.)
    if df["label"].dtype.kind in "iu":  # integer or unsigned int
        uniq = set(df["label"].unique())
        if not (uniq <= {0, 1}):  # If we have values other than 0 and 1
            df["label"] = df["label"].apply(lambda v: 0 if v == 0 else 1)

    df = df[["text", "label"]].dropna()
    return df


def get_sample_fallback() -> pd.DataFrame:
    """
    Return a small built-in dataset when no CSV files exist.

    WHY WE NEED THIS:
    On first run or Streamlit Cloud deploy, the dataset/ folder might be empty.
    Instead of crashing, we provide 5 example messages so the user can:
      - See the app working
      - Train a minimal model
      - Understand the expected format

    Returns: DataFrame with 5 rows: 3 bullying, 2 safe
    """
    return pd.DataFrame({
        "text": [
            "You are so stupid and useless",    # Bullying
            "Thanks for your help today",       # Safe
            "No one likes you, go away",        # Bullying
            "Good morning everyone",            # Safe
            "You will never amount to anything", # Bullying
        ],
        "label": [1, 0, 1, 0, 1],
    })


def load_chat_file(chat_path: str = None) -> pd.DataFrame:
    """
    Load and parse a WhatsApp-style chat from a file.

    If chat_path is not given, we look for dataset/sample_chat.txt.
    Used for loading pre-saved chat files (less common than paste/upload in the app).

    Returns: DataFrame with columns [timestamp, sender, message_text]
             or empty DataFrame if file not found.
    """
    from .chat_parser import parse_whatsapp_chat

    if chat_path is None:
        root = get_project_root()
        for base in ("datasets", "dataset"):
            p = os.path.join(root, base, "sample_chat.txt")
            if os.path.exists(p):
                chat_path = p
                break
        if chat_path is None:
            chat_path = os.path.join(root, "dataset", "sample_chat.txt")

    return parse_whatsapp_chat(chat_path)
