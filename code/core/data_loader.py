"""
TrollGuard – Data Loader Utility

Loads *_parsed_dataset.csv files from dataset/datasets folder.
Expects columns: Text, oh_label.
Returns a combined DataFrame with [text, label] (binary 0/1).
"""

import os
import glob
import pandas as pd


def get_project_root() -> str:
    """Detect project root (works locally and in Colab)."""
    # Try Colab / Google Drive paths first
    if os.path.exists("/content/drive/MyDrive/TrollGuard_Project"):
        return "/content/drive/MyDrive/TrollGuard_Project"
    if os.path.exists("/content/drive/MyDrive/TrollGuard_Project_Gloryia_2025"):
        return "/content/drive/MyDrive/TrollGuard_Project_Gloryia_2025"
    # Local path: use directory of this file
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_datasets_dir() -> str:
    """Return datasets directory (datasets or dataset)."""
    root = get_project_root()
    for name in ("datasets", "dataset"):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    return os.path.join(root, "dataset")


def load_parsed_datasets(datasets_dir: str = None) -> pd.DataFrame:
    """
    Load all *_parsed_dataset.csv files into a single DataFrame.

    Returns:
        DataFrame with columns [text, label] (label: 0 = non-bullying, 1 = bullying).
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

    if df["label"].dtype.kind in "iu":
        uniq = set(df["label"].unique())
        if not (uniq <= {0, 1}):
            df["label"] = df["label"].apply(lambda v: 0 if v == 0 else 1)

    df = df[["text", "label"]].dropna()
    return df


def get_sample_fallback() -> pd.DataFrame:
    """Return a minimal sample dataset when no CSV files exist (fallback)."""
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
    """Load and parse WhatsApp-style chat export. Returns empty DataFrame if not found."""
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
