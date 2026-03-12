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
    data_dir: Path = Path("data/realbook_primus/package_aa")
    scanned_dir: Path = Path("data/realbook_primus_augmented/package_aa")
    extra_data_dirs: list[Path] = field(default_factory=list)
    extra_scanned_dirs: list[Path] = field(default_factory=list)
    model_dir: Path = Path("models")
    vocab_path: Path = Path("src/CRNN_CTC/vocabulary.txt")

    # ── Reproducibility ────────────────────────────────────────────────────
    seed: int = 42

    # ── Data ───────────────────────────────────────────────────────────────
    img_height: int = 128          # resize all images to this height
    max_image_width: int = 2048    # clamp width after height-resize (OOM guard)
    use_scanned: bool = True           # train on augmented scanned images
    val_frac: float = 0.10         # fraction held out for validation
    test_frac: float = 0.10        # fraction held out for final test

    # ── Data filtering ────────────────────────────────────────────────────
    # filter_rest_heavy: drop samples where >80% of tokens are structural
    # rest/measure tokens and the total length exceeds 50.  These are
    # multi-bar tacet passages from orchestral PrIMuS pieces — the image
    # shows a long string of whole-measure rests, a rare pattern that is
    # irrelevant for jazz lead sheets and inflates CTC edit distance.
    filter_rest_heavy: bool = True
    # filter_unwanted_clefs: drop samples containing C1, C2, or F3 clef tokens.
    # C1/C2 (soprano / mezzo-soprano) are visually similar to tenor (C4) but
    # shifted by one staff line — the model confuses them and the resulting
    # pitch cascade accounts for ~9 000 substitution errors in Run 4.
    # F3 (baritone) looks like the standard bass clef (F4) shifted one line up,
    # causing the same cascade issue for bass-register samples.
    # None of these clefs appear in jazz lead sheets.
    filter_unwanted_clefs: bool = True
    # filter_multi_staff: drop images whose original height exceeds
    # max_source_height.  LilyPond occasionally wraps a long excerpt onto two
    # staff lines in a single PNG — the resulting image is ~2-3× taller than
    # a normal single staff.  Height distribution for realbook_primus_aa:
    #   normal range  84–152 px (p5–p95), median 133 px
    #   multi-staff   ≥200 px (hard gap, no values 153–199 px)
    # A threshold of 180 px removes all 1 132 / 43 563 multi-staff images
    # (2.6 %) while retaining every legitimate single-staff sample.
    filter_multi_staff: bool = True
    max_source_height: int = 180   # px — upper bound for single-staff check

    # ── Model — backbone ───────────────────────────────────────────────────
    backbone: str = "resnet18"         # CNN backbone: "resnet18" or "vgg"

    # ── Model — CNN ────────────────────────────────────────────────────────
    cnn_out_channels: int = 256    # feature maps at the CNN output (VGG only)
    cnn_dropout: float = 0.25      # Dropout2d after CNN blocks

    # ── Model — RNN ────────────────────────────────────────────────────────
    rnn_hidden: int = 256          # hidden size per LSTM direction
    rnn_layers: int = 2            # stacked LSTM layers
    dropout: float = 0.3           # dropout between LSTM layers

    # ── Training ───────────────────────────────────────────────────────────
    epochs: int = 50
    batch_size: int = 16
    lr: float = 5e-4               # peak learning rate (OneCycleLR)
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05      # fraction of total steps for LR warm-up
    num_workers: int = 10          # DataLoader workers
    early_stopping_patience: int = 10  # stop if val SER stalls N epochs (0 = off)

    def __post_init__(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
