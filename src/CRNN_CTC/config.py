"""
config.py
=========
Centralised paths and settings for the CRNN-CTC pipeline.

Only contains values that correspond to the *current* pipeline state:
PrIMuS → jazz-styled rendering → augmentation → LMX conversion → CRNN input.

Model architecture, training, and decoding hyperparameters will be added
here once they are justified through experimentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Pipeline configuration (paths and data settings only)."""

    # ── Paths ──────────────────────────────────────────────────────────────
    data_dir: Path = Path("data/realbook_primus_aa")
    scanned_dir: Path = Path("data/realbook_primus_aa_scanned")
    model_dir: Path = Path("models")
    vocab_path: Path = Path("src/CRNN_CTC/vocabulary.txt")

    # ── Reproducibility ────────────────────────────────────────────────────
    seed: int = 42

    def __post_init__(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
