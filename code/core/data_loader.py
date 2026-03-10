"""
TrollGuard – Data Loader Utility

Loads all *_parsed_dataset.csv files from dataset/ or datasets/.
Expects columns: Text, oh_label (0=non-bullying, 1=bullying).
Returns a combined DataFrame with [text, label].
"""

import os
import glob
import pandas as pd


def get_project_root() -> str:
    """
    Detect project root directory (folder containing dataset/, models/).
    Handles Colab, Streamlit Cloud, and local runs.
    """
    if os.path.exists("/content/drive/MyDrive/TrollGuard_Project"):
        return "/content/drive/MyDrive/TrollGuard_Project"
    if os.path.exists("/content/drive/MyDrive/TrollGuard_Project_Gloryia_2025"):
        return "/content/drive/MyDrive/TrollGuard_Project_Gloryia_2025"
    # Standard: code/core/data_loader.py -> root = code/
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return root


def get_datasets_dir() -> str:
    """Return path to datasets directory. Prefers 'datasets', then 'dataset'."""
    root = get_project_root()
    for name in ("datasets", "dataset"):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    return os.path.join(root, "dataset")


def load_parsed_datasets(datasets_dir: str = None) -> pd.DataFrame:
    """
    Load all *_parsed_dataset.csv files and merge into one DataFrame.
    Expects columns: Text, oh_label.
    Normalises labels to binary 0/1.
    Returns DataFrame with [text, label].
    """
    if datasets_dir is None:
        datasets_dir = get_datasets_dir()

    pattern = os.path.join(datasets_dir, "*_parsed_dataset.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        return pd.DataFrame(columns=["text", "label"])

    dfs = []
    for path in csv_files:
        try:
            tmp = pd.read_csv(path)
            if "Text" not in tmp.columns or "oh_label" not in tmp.columns:
                continue
            tmp = tmp[["Text", "oh_label"]].rename(
                columns={"Text": "text", "oh_label": "label"}
            )
            dfs.append(tmp)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame(columns=["text", "label"])

    df = pd.concat(dfs, ignore_index=True)

    # Normalise labels to binary 0/1
    try:
        df["label"] = pd.to_numeric(df["label"])
    except Exception:
        # Handle string labels
        NON_BULLY = {
            "none", "normal", "non-toxic", "non_toxic",
            "non-aggressive", "not_cyberbullying", "safe", "neutral"
        }
        def to_binary(x):
            s = str(x).strip().lower()
            if s in NON_BULLY or s == "0":
                return 0
            return 1
        df["label"] = df["label"].apply(to_binary)

    # Ensure labels are strictly 0 or 1
    if df["label"].dtype.kind in "iu":
        uniq = set(df["label"].unique())
        if not (uniq <= {0, 1}):
            df["label"] = df["label"].apply(lambda v: 0 if v == 0 else 1)

    df = df[["text", "label"]].dropna()
    return df


def get_sample_fallback() -> pd.DataFrame:
    """Return minimal sample dataset when no CSV files exist."""
    return pd.DataFrame({
        "text": [
            "You are so stupid and useless",
            "Thanks for your help today",
            "No one likes you, go away",
            "Good morning everyone",
            "You will never amount to anything",
        ],
        "label": [1, 0, 1, 0, 1],
    })


def load_chat_file(chat_path: str = None) -> pd.DataFrame:
    """
    Load and parse WhatsApp-style chat from file.
    If chat_path is None, looks for dataset/sample_chat.txt.
    Returns empty DataFrame if not found.
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
