"""Ensure project ``src/`` root is on ``sys.path`` for script-style imports."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent


def ensure_src_path() -> None:
    path = str(_SRC_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
