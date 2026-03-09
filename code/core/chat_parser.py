"""
TrollGuard – WhatsApp Chat Export Parser

Expects format: DD/MM/YYYY, HH:MM - Sender: message
"""

import os
import pandas as pd
from datetime import datetime


def parse_whatsapp_chat(chat_path: str) -> pd.DataFrame:
    """
    Parse WhatsApp-style chat export.

    Format: DD/MM/YYYY, HH:MM - Sender: message
    Returns DataFrame with [timestamp, sender, message_text].
    """
    rows = []
    if not chat_path or not os.path.exists(chat_path):
        return pd.DataFrame(columns=["timestamp", "sender", "message_text"])

    with open(chat_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                date_part, rest = line.split(" - ", 1)
                ts = datetime.strptime(date_part.strip(), "%d/%m/%Y, %H:%M")
                sender_part, msg = rest.split(": ", 1)
                rows.append({
                    "timestamp": ts,
                    "sender": sender_part.strip(),
                    "message_text": msg.strip(),
                })
            except (ValueError, IndexError):
                continue

    return pd.DataFrame(rows)
