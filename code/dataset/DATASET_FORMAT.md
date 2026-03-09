# TrollGuard Dataset Format

## Training datasets (CSV)

Place files matching `*_parsed_dataset.csv` in the `dataset/` (or `datasets/`) folder.

**Required columns:**

| Column | Description |
|--------|-------------|
| `Text` | Raw message text (string) |
| `oh_label` | Binary label: 0 = non-bullying, 1 = bullying (numeric or string) |

**Optional columns:** Any other columns are ignored.

**Example:**

```csv
index,Text,oh_label
0,"This is a normal message",0
1,"You are stupid and useless",1
```

## Chat export (sample_chat.txt)

WhatsApp-style export for the **Chat analysis** tab:

```
DD/MM/YYYY, HH:MM - SenderName: message text
```

**Example:**

```
01/08/2025, 09:00 - Alice: Hey everyone
01/08/2025, 09:01 - Bob: You are so slow at this
```
