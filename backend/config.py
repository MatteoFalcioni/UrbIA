from __future__ import annotations

import os
from pathlib import Path


# ---------- LLM Configuration ----------
# Default LLM configuration (can be overridden per-thread via configs table)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "30000"))  # Default 30k