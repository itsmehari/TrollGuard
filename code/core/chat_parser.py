"""
TrollGuard – Multi-format Chat Export Parser

Supports: WhatsApp, Telegram, Discord. Falls back for malformed lines.
"""

import os
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional

# WhatsApp: DD/MM/YYYY, HH:MM - Sender: message
# Telegram: DD.MM.YYYY, HH:MM - Sender: message
# Discord:  [DD/MM/YYYY HH:MM] Sender: message


def _parse_date_whatsapp(s: str) -> Optional[datetime]:
    """Parse WhatsApp/Telegram style date."""
    s = s.strip().replace(".", "/")
    for fmt in [("%d/%m/%Y, %H:%M", 16), ("%d/%m/%Y %H:%M", 16), ("%d/%m/%Y, %H:%M:%S", 19), ("%d/%m/%Y %H:%M:%S", 19)]:
        try:
            return datetime.strptime(s[:fmt[1]], fmt[0])
        except (ValueError, TypeError):
            continue
    return None


def _parse_date_discord(s: str) -> Optional[datetime]:
    s = s.strip()[:24]
    for fmt in ("%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(s[:19], fmt)
        except ValueError:
            continue
    return None


def _parse_line_whatsapp_telegram(line: str) -> Optional[Tuple[datetime, str, str]]:
    """DD/MM/YYYY, HH:MM - Sender: message  or  DD.MM.YYYY, HH:MM - Sender: message"""
    if " - " not in line or ": " not in line:
        return None
    part0, rest = line.split(" - ", 1)
    ts = _parse_date_whatsapp(part0)
    if not ts:
        return None
    idx = rest.find(": ")
    if idx < 0:
        return None
    sender, msg = rest[:idx], rest[idx + 2:]
    return ts, sender.strip(), msg.strip()


def _parse_line_discord(line: str) -> Optional[Tuple[datetime, str, str]]:
    """[DD/MM/YYYY HH:MM] Sender: message"""
    if not line.strip().startswith("[") or "] " not in line:
        return None
    end = line.index("] ")
    date_str = line[1:end].strip()
    ts = _parse_date_discord(date_str)
    if not ts:
        return None
    rest = line[end + 2:]
    if ": " not in rest:
        return None
    idx = rest.find(": ")
    sender, msg = rest[:idx], rest[idx + 2:]
    return ts, sender.strip(), msg.strip()


def parse_chat_from_string(content: str, format_hint: str = "auto") -> pd.DataFrame:
    """
    Parse chat from string. Supports WhatsApp, Telegram, Discord.
    Malformed lines are kept as (timestamp=min, sender="Unknown", message_text=line).

    format_hint: "whatsapp" | "telegram" | "discord" | "auto"
    """
    rows = []
    parsers = []
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
    """Backward-compat alias for WhatsApp format."""
    return parse_chat_from_string(content, "whatsapp")


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
