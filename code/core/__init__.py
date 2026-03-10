"""
================================================================================
TrollGuard Core Package
================================================================================

This folder contains the "backend" logic of TrollGuard - the parts that run
behind the scenes. When you write "from core.data_loader import ...", Python
looks in this package.

AVAILABLE MODULES:
  - data_loader: Load and merge CSV datasets; find project paths
  - model_utils: Clean text, load model, predict, feature importance
  - chat_parser: Parse WhatsApp, Telegram, and Discord chat exports

You typically import from these modules directly, e.g.:
  from core.data_loader import load_parsed_datasets
  from core.model_utils import predict_text, clean_text
  from core.chat_parser import parse_chat_from_string
================================================================================
"""
