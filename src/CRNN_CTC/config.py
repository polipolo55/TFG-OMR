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
    # Clean / scanned data follow the new unified layout:
    #   data/raw/primus/...
    #   data/processed/primus/clean/...
    #   data/processed/primus/scanned/...
    data_dir: Path = Path("data/processed/primus/clean")
    scanned_dir: Path = Path("data/processed/primus/scanned")
    extra_data_dirs: list[Path] = field(default_factory=list)
    extra_scanned_dirs: list[Path] = field(default_factory=list)
    model_dir: Path = Path("models")
    vocab_path: Path = Path("data/vocab/primus_lmx.txt")

    # ── Reproducibility ────────────────────────────────────────────────────
    seed: int = 42

    # ── Data ───────────────────────────────────────────────────────────────
    img_height: int = 128          # resize all images to this height
    max_image_width: int = 2048    # clamp width after height-resize (OOM guard)
    use_scanned: bool = True           # train on augmented scanned images
    val_frac: float = 0.10         # fraction held out for validation
    test_frac: float = 0.10        # fraction held out for final test

    # ── Data filtering ────────────────────────────────────────────────────
    # The filters below define the lead-sheet domain (see
    # ``docs/overview.md`` → "Domain Specification").  They are conceptually
    # not "exclusions" but the contract of the system: monophonic + treble
    # clef + single staff + jazz common-time meters.  Disabling them on a
    # PrIMuS-trained run will not generalise the model — it only forces it
    # to spend capacity on patterns that do not appear in the Real Book.
    #
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

    # ── Data augmentation ──────────────────────────────────────────────────
    # strip_header_prob: during training, randomly remove the visual header
    # (clef + key + time region) from this fraction of samples so the model
    # learns to recognise continuation lines that lack a header — as they
    # appear on lines 2+ of a Real Book page.  Applied to training only.
    strip_header_prob: float = 0.4

    # online_aug_prob: probability of applying lightweight per-sample jitter
    # (brightness ±5%, contrast bias ±3%, gaussian noise σ≈0.01, ±2 px
    # horizontal shift) on top of the offline-augmented scanned PNG.  Without
    # this, the dataloader returns identical pixel grids every epoch and the
    # model overfits the specific augmentations baked into ``scanned/``.
    # Applied to training only; 0 disables.  Cost: ~50 µs per sample.
    online_aug_prob: float = 0.5

    # rare_lmx_oversample: training indices for samples whose .lmx contains
    # any token in rare_lmx_tokens are repeated (N-1) extra times so each
    # epoch sees them N× as often.  1 = disabled.  Default 2 up-weights:
    #
    #   * ``tied:start`` / ``tied:stop`` — visually subtle, under-predicted on scans.
    #
    # Note: ``key:fifths:0`` (C major) was previously listed here because
    # ``semantic_to_lmx.py`` did not emit it for samples with no explicit
    # ``keySignature-`` token, leaving only 8 C-major training labels.
    # That bug is now fixed — the converter always injects ``key:fifths:0``
    # when no key is specified, so ~45 % of the corpus is C major and no
    # oversampling is needed or desired (adding it back would make C major
    # ~59 % of the virtual training set).
    rare_lmx_oversample: int = 2
    rare_lmx_tokens: tuple[str, ...] = (
        "tied:start", "tied:stop",
    )

    # filter_non_leadsheet_clef: drop samples containing any clef token except
    # clef:G2.  C3 (alto), C4 (tenor), G1 (French violin), and F4 (bass) appear
    # in orchestral PrIMuS excerpts but never in jazz lead sheets.  Keeping them
    # wastes model capacity on visual patterns it will never encounter at inference.
    filter_non_leadsheet_clef: bool = True

    # filter_unusual_time: drop samples whose time signature is not in the common
    # jazz set {4/4, 3/4, 2/4, 2/2, 6/8, 6/4, 5/4, 12/8}.  Exotic meters like
    # 7/4 or 11/8 appear in classical PrIMuS but not in Real Book.  grammar_fix
    # already rejects these at inference so there is no point training on them.
    filter_unusual_time: bool = True

    # ── Fine-tuning (optional target-domain data) ─────────────────────────
    # Extra clean-image directories merged into the training split only.
    finetune_data_dirs: list[Path] = field(default_factory=list)
    finetune_scanned_dirs: list[Path] = field(default_factory=list)

    # ── Training ───────────────────────────────────────────────────────────
    # Defaults tuned for a full PrIMuS → scanned run on ~12 GB VRAM (OneCycleLR).
    epochs: int = 60
    batch_size: int = 16
    lr: float = 1e-3               # peak LR — higher than 5e-4 for faster convergence
    weight_decay: float = 1e-4
    warmup_frac: float = 0.08      # slightly longer warm-up for large loaders
    num_workers: int = 10          # DataLoader workers
    early_stopping_patience: int = 12  # val SER plateau (0 = off)
    max_grad_norm: float = 5.0         # gradient clipping max norm

    def __post_init__(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
