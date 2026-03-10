"""
================================================================================
TrollGuard – Multi-Format Chat Export Parser
================================================================================

WHAT THIS FILE DOES (in simple terms):
When you export a chat from WhatsApp, Telegram, or Discord, you get a .txt file
with lines like "09/03/2025, 14:30 - Alice: Hello everyone". This module reads
that raw text and extracts:
  - Who sent each message (sender)
  - When they sent it (timestamp)
  - What they said (message_text)

Each app has a slightly different format. We support:
  - WhatsApp: DD/MM/YYYY, HH:MM - Sender: message
  - Telegram: DD.MM.YYYY, HH:MM - Sender: message (uses dots instead of slashes)
  - Discord:  [DD/MM/YYYY HH:MM] Sender: message (uses brackets)

Malformed lines (that don't match any format) are kept with sender="Unknown"
so we don't lose any text - we can still classify them.

================================================================================
"""

import os
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional


def _parse_date_whatsapp(s: str) -> Optional[datetime]:
    """
    Parse a date string in WhatsApp/Telegram style.

    FORMATS WE SUPPORT:
    - DD/MM/YYYY, HH:MM       (e.g. 09/03/2025, 14:30)
    - DD/MM/YYYY HH:MM        (no comma)
    - DD/MM/YYYY, HH:MM:SS    (with seconds)
    - DD.MM.YYYY, HH:MM       (Telegram uses dots; we convert to slashes)

    Returns: datetime object or None if parsing fails
    """
    s = s.strip().replace(".", "/")  # Telegram: 09.03.2025 -> 09/03/2025
    for fmt in [
        ("%d/%m/%Y, %H:%M", 16),      # DD/MM/YYYY, HH:MM
        ("%d/%m/%Y %H:%M", 16),       # DD/MM/YYYY HH:MM
        ("%d/%m/%Y, %H:%M:%S", 19),   # With seconds
        ("%d/%m/%Y %H:%M:%S", 19),
    ]:
        try:
            return datetime.strptime(s[:fmt[1]], fmt[0])
        except (ValueError, TypeError):
            continue
    return None


def _parse_date_discord(s: str) -> Optional[datetime]:
    """
    Parse a date string in Discord style.

    Discord format: [DD/MM/YYYY HH:MM] or [DD/MM/YYYY HH:MM:SS]
    Note: Discord exports can vary; we handle the common format.

    Returns: datetime object or None if parsing fails
    """
    s = s.strip()[:24]
    for fmt in ("%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(s[:19], fmt)
        except ValueError:
            continue
    return None


def _parse_line_whatsapp_telegram(line: str) -> Optional[Tuple[datetime, str, str]]:
    """
    Parse one line in WhatsApp or Telegram format.

    EXPECTED: DD/MM/YYYY, HH:MM - Sender: message
    Example: 09/03/2025, 14:30 - Alice: Hi there!

    We split by " - " to get [date_part, rest], then by ": " to get [sender, message].

    Returns: (timestamp, sender, message_text) or None if line doesn't match
    """
    if " - " not in line or ": " not in line:
        return None
    part0, rest = line.split(" - ", 1)  # Split only at first " - "
    ts = _parse_date_whatsapp(part0)
    if not ts:
        return None
    idx = rest.find(": ")
    if idx < 0:
        return None
    sender, msg = rest[:idx], rest[idx + 2:]
    return ts, sender.strip(), msg.strip()


def _parse_line_discord(line: str) -> Optional[Tuple[datetime, str, str]]:
    """
    Parse one line in Discord format.

    EXPECTED: [DD/MM/YYYY HH:MM] Sender: message
    Example: [09/03/2025 14:30] Bob: Hello!

    Discord uses square brackets around the timestamp. We extract the part
    between [ and ], parse the date, then extract sender and message.

    Returns: (timestamp, sender, message_text) or None if line doesn't match
    """
    if not line.strip().startswith("[") or "] " not in line:
        return None
    end = line.index("] ")
    date_str = line[1:end].strip()  # Remove [ and ]
    ts = _parse_date_discord(date_str)
    if not ts:
        return None
    rest = line[end + 2:]  # Text after "] "
    if ": " not in rest:
        return None
    idx = rest.find(": ")
    sender, msg = rest[:idx], rest[idx + 2:]
    return ts, sender.strip(), msg.strip()


def parse_chat_from_string(content: str, format_hint: str = "auto") -> pd.DataFrame:
    """
    Parse a full chat export (raw text) into a structured table.

    PARAMETERS:
    - content: The raw chat text (e.g. from pasting or reading a file)
    - format_hint: "whatsapp" | "telegram" | "discord" | "auto"
      - "auto" tries both WhatsApp/Telegram and Discord parsers for each line
      - Specific hint uses only that parser (faster, more accurate if you know the source)

    MALFORMED LINES:
    Lines that don't match any format (e.g. "---" or continuation lines) get:
    - sender = "Unknown"
    - timestamp = 1970-01-01 (placeholder)
    - message_text = the raw line
    We keep them so no data is lost; they can still be classified.

    Returns: DataFrame with columns [timestamp, sender, message_text]
    """
    rows = []
    if format_hint == "whatsapp" or format_hint == "telegram":
        parsers = [_parse_line_whatsapp_telegram]
    elif format_hint == "discord":
        parsers = [_parse_line_discord]
    else:
        parsers = [_parse_line_whatsapp_telegram, _parse_line_discord]

    fallback_ts = datetime(1970, 1, 1)
    for line in content.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = None
        for p in parsers:
            parsed = p(line)
            if parsed:
                break
        if parsed:
            ts, sender, msg = parsed
            rows.append({"timestamp": ts, "sender": sender, "message_text": msg})
        else:
            rows.append({"timestamp": fallback_ts, "sender": "Unknown", "message_text": line})
    return pd.DataFrame(rows)


def parse_whatsapp_chat_from_string(content: str) -> pd.DataFrame:
    """
    Backward-compatible wrapper: parse chat as WhatsApp format.

    Same as parse_chat_from_string(content, "whatsapp").
    """
    return parse_chat_from_string(content, "whatsapp")


def parse_whatsapp_chat(chat_path: str) -> pd.DataFrame:
    """
    Load and parse a WhatsApp chat from a file path.

    Reads the file line by line and parses each line. Used when you have
    a .txt file on disk (e.g. sample_chat.txt).

    Returns: DataFrame with [timestamp, sender, message_text], or empty if
    path invalid or file not found. Skips lines that fail to parse.
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
