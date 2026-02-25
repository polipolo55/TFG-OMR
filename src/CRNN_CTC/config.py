"""
config.py
=========
Centralised configuration for every stage of the CRNN-CTC pipeline.

All hyper-parameters live here so that experiments are fully reproducible
from a single ``Config`` instance (serialised inside every checkpoint).
Defaults are conservative enough for an RTX 3060 (12 GB VRAM).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Pipeline configuration — paths, data, model, and training settings."""

    # ── Paths ──────────────────────────────────────────────────────────────
    data_dir: Path = Path("data/realbook_primus_aa")
    scanned_dir: Path = Path("data/realbook_primus_aa_scanned")
    model_dir: Path = Path("models")
    vocab_path: Path = Path("src/CRNN_CTC/vocabulary.txt")

    # ── Reproducibility ────────────────────────────────────────────────────
    seed: int = 42

    # ── Data ───────────────────────────────────────────────────────────────
    img_height: int = 128          # resize all images to this height
    use_scanned: bool = False      # train on augmented scanned images
    val_frac: float = 0.10         # fraction held out for validation
    test_frac: float = 0.10        # fraction held out for final test

    # ── Model — CNN ────────────────────────────────────────────────────────
    cnn_out_channels: int = 256    # feature maps at the CNN output

    # ── Model — RNN ────────────────────────────────────────────────────────
    rnn_hidden: int = 256          # hidden size per LSTM direction
    rnn_layers: int = 2            # stacked LSTM layers
    dropout: float = 0.3           # dropout between LSTM layers

    # ── Training ───────────────────────────────────────────────────────────
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-3               # peak learning rate (OneCycleLR)
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05      # fraction of total steps for LR warm-up
    num_workers: int = 4           # DataLoader workers

    def __post_init__(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
