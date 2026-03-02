#!/usr/bin/env python3
"""
cli.py — Unified command-line interface for the TFG-OMR pipeline.
=================================================================

Subcommands
-----------
render      Render PrIMuS samples → LilyJAZZ PNGs + LMX annotations.
convert     Convert PrIMuS .semantic → monophonic LMX (standalone).
augment     Apply scan-simulation augmentations to clean images.
vocab       Build LMX vocabulary from converted data.
train       Train the CRNN-CTC model.
evaluate    Evaluate a trained model checkpoint.

Usage examples::

    poetry run python src/cli.py render  --source data/primus/package_aa --output data/realbook_primus_aa
    poetry run python src/cli.py convert --source data/realbook_primus_aa --workers 8
    poetry run python src/cli.py augment --source data/realbook_primus_aa --output data/realbook_primus_aa_scanned
    poetry run python src/cli.py vocab   --data-dir data/realbook_primus_aa
    poetry run python src/cli.py train   --epochs 50 --batch-size 16 --lr 1e-3
    poetry run python src/cli.py evaluate --checkpoint models/best_model.pt --split test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure src/ is on sys.path so imports work when the script is invoked
# directly (``python src/cli.py …``).
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parent            # …/TFG-OMR/src
for _p in (str(_SRC), str(_SRC.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

log = logging.getLogger("omr.cli")


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand handlers
# ═══════════════════════════════════════════════════════════════════════════

# ── render ────────────────────────────────────────────────────────────────

def cmd_render(args: argparse.Namespace) -> None:
    """Render PrIMuS samples with LilyJAZZ and optionally generate LMX."""
    from data_processing.generate_realbook import main as _render_main

    argv: list[str] = [
        "--source", str(args.source),
        "--output", str(args.output),
        "--dpi", str(args.dpi),
    ]
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]
    if args.workers is not None:
        argv += ["--workers", str(args.workers)]
    if args.force:
        argv.append("--force")
    if args.no_lmx:
        argv.append("--no-lmx")
    if args.verbose:
        argv.append("--verbose")

    old_argv = sys.argv
    sys.argv = ["generate_realbook"] + argv
    try:
        _render_main()
    finally:
        sys.argv = old_argv


# ── convert ───────────────────────────────────────────────────────────────

def cmd_convert(args: argparse.Namespace) -> None:
    """Convert PrIMuS .semantic annotations → monophonic LMX via music21."""
    from data_processing.semantic_to_lmx import main as _convert_main

    # Build argv for the converter's own argparse
    argv: list[str] = ["--source", str(args.source)]
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]
    if args.workers is not None:
        argv += ["--workers", str(args.workers)]
    if args.verbose:
        argv.append("--verbose")
    if args.keep_visual:
        argv.append("--keep-visual")

    old_argv = sys.argv
    sys.argv = ["semantic_to_lmx"] + argv
    try:
        _convert_main()
    finally:
        sys.argv = old_argv


# ── vocab ─────────────────────────────────────────────────────────────────

def cmd_vocab(args: argparse.Namespace) -> None:
    """Build the LMX vocabulary file from a directory of .lmx files."""
    from CRNN_CTC.vocab import Vocabulary

    data_dir = Path(args.data_dir)
    out_path = Path(args.output)

    log.info("Building vocabulary from %s …", data_dir)
    vocab = Vocabulary.build_from_lmx_dir(data_dir)
    vocab.save(out_path)
    log.info("Saved vocabulary (%d tokens incl. blank+pad) → %s", len(vocab), out_path)


# ── train ─────────────────────────────────────────────────────────────────

def _build_config_from_args(args: argparse.Namespace):
    """Construct a Config from CLI flags, overriding only what was set."""
    from CRNN_CTC.config import Config

    overrides: dict = {}
    # Map CLI flag names → Config field names
    flag_map = {
        "data_dir": "data_dir",
        "scanned_dir": "scanned_dir",
        "model_dir": "model_dir",
        "vocab_path": "vocab_path",
        "seed": "seed",
        "img_height": "img_height",
        "use_scanned": "use_scanned",
        "val_frac": "val_frac",
        "test_frac": "test_frac",
        "cnn_out_channels": "cnn_out_channels",
        "cnn_dropout": "cnn_dropout",
        "rnn_hidden": "rnn_hidden",
        "rnn_layers": "rnn_layers",
        "dropout": "dropout",
        "epochs": "epochs",
        "batch_size": "batch_size",
        "lr": "lr",
        "weight_decay": "weight_decay",
        "warmup_frac": "warmup_frac",
        "num_workers": "num_workers",
        "early_stopping_patience": "early_stopping_patience",
        "max_source_height": "max_source_height",
    }
    # Boolean filter flags use store_false with default=None (only
    # override when the user explicitly passes the --no-... flag)
    for bflag in ("filter_rest_heavy", "filter_unwanted_clefs", "filter_multi_staff"):
        val = getattr(args, bflag, None)
        if val is not None:
            overrides[bflag] = val
    for flag, field in flag_map.items():
        val = getattr(args, flag, None)
        if val is not None:
            overrides[field] = val

    # Convert string paths to Path objects
    for key in ("data_dir", "scanned_dir", "model_dir", "vocab_path"):
        if key in overrides and isinstance(overrides[key], str):
            overrides[key] = Path(overrides[key])

    return Config(**overrides)


def cmd_train(args: argparse.Namespace) -> None:
    """Launch CRNN-CTC training."""
    from CRNN_CTC.train import train

    cfg = _build_config_from_args(args)
    log.info("Config: %s", cfg)
    best_ckpt = train(cfg)
    log.info("Best checkpoint saved to %s", best_ckpt)


# ── evaluate ──────────────────────────────────────────────────────────────

def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a trained model checkpoint."""
    from CRNN_CTC.evaluate import evaluate

    cfg = _build_config_from_args(args)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        log.error("Checkpoint not found: %s", checkpoint)
        sys.exit(1)

    ser = evaluate(
        cfg,
        checkpoint,
        split=args.split,
        per_sample=args.per_sample,
    )
    print(f"SER ({args.split}): {ser:.4f}")


# ── augment ───────────────────────────────────────────────────────────────

def cmd_augment(args: argparse.Namespace) -> None:
    """Apply scan-simulation augmentations to clean dataset images."""
    from data_processing.augment_scanned import main as _augment_main

    argv: list[str] = [
        "--source", str(args.source),
        "--output", str(args.output),
    ]
    if args.copies is not None:
        argv += ["--copies", str(args.copies)]
    if args.seed is not None:
        argv += ["--seed", str(args.seed)]
    if args.workers is not None:
        argv += ["--workers", str(args.workers)]
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]

    old_argv = sys.argv
    sys.argv = ["augment_scanned"] + argv
    try:
        _augment_main()
    finally:
        sys.argv = old_argv


# ═══════════════════════════════════════════════════════════════════════════
# Argument parser
# ═══════════════════════════════════════════════════════════════════════════

def _add_common_data_args(parser: argparse.ArgumentParser) -> None:
    """Add flags shared by train / evaluate."""
    g = parser.add_argument_group("data")
    g.add_argument("--data-dir", type=str, default=None,
                   help="Root sample directory (default: data/realbook_primus_aa)")
    g.add_argument("--scanned-dir", type=str, default=None,
                   help="Scanned-image directory (default: data/realbook_primus_aa_scanned)")
    g.add_argument("--vocab-path", type=str, default=None,
                   help="Vocabulary file (default: src/CRNN_CTC/vocabulary.txt)")
    g.add_argument("--img-height", type=int, default=None,
                   help="Resize images to this height (default: 128)")
    g.add_argument("--use-scanned", action="store_true", default=None,
                   help="Use augmented scanned images instead of clean originals")
    g.add_argument("--val-frac", type=float, default=None,
                   help="Validation split fraction (default: 0.10)")
    g.add_argument("--test-frac", type=float, default=None,
                   help="Test split fraction (default: 0.10)")
    g.add_argument("--num-workers", type=int, default=None,
                   help="DataLoader workers (default: 4)")
    g.add_argument("--seed", type=int, default=None,
                   help="Random seed (default: 42)")
    g.add_argument("--no-filter-rest-heavy", dest="filter_rest_heavy",
                   action="store_false", default=None,
                   help="Disable filtering of rest-heavy samples")
    g.add_argument("--no-filter-unwanted-clefs", dest="filter_unwanted_clefs",
                   action="store_false", default=None,
                   help="Disable filtering of C1/C2 clef samples")
    g.add_argument("--no-filter-multi-staff", dest="filter_multi_staff",
                   action="store_false", default=None,
                   help="Disable filtering of multi-staff (tall) images")
    g.add_argument("--max-source-height", type=int, default=None,
                   help="Max original image height for single-staff filter (default: 180 px)")


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add model-architecture flags shared by train / evaluate."""
    g = parser.add_argument_group("model")
    g.add_argument("--cnn-out-channels", type=int, default=None,
                   help="CNN output feature maps (default: 256)")
    g.add_argument("--cnn-dropout", type=float, default=None,
                   help="Dropout2d after each CNN block (default: 0.2)")
    g.add_argument("--rnn-hidden", type=int, default=None,
                   help="LSTM hidden size per direction (default: 256)")
    g.add_argument("--rnn-layers", type=int, default=None,
                   help="Stacked LSTM layers (default: 2)")
    g.add_argument("--dropout", type=float, default=None,
                   help="Dropout between LSTM layers (default: 0.3)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omr",
        description="TFG-OMR: End-to-end Optical Music Recognition pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable DEBUG-level logging")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── render ────────────────────────────────────────────────────────
    p_rend = sub.add_parser(
        "render",
        help="Render PrIMuS → LilyJAZZ PNGs (+ LMX annotations)",
        description="Re-render PrIMuS samples with LilyJAZZ styling and "
                    "optionally generate LMX labels inline.",
    )
    p_rend.add_argument("--source", type=Path,
                        default=Path("data/primus/package_aa"),
                        help="PrIMuS source directory (default: data/primus/package_aa)")
    p_rend.add_argument("--output", type=Path,
                        default=Path("data/realbook_primus_aa"),
                        help="Output dataset directory (default: data/realbook_primus_aa)")
    p_rend.add_argument("--dpi", type=int, default=200,
                        help="Rendering resolution (default: 200)")
    p_rend.add_argument("--limit", type=int, default=None,
                        help="Process at most N samples (for testing)")
    p_rend.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default: CPU count)")
    p_rend.add_argument("--force", action="store_true",
                        help="Re-render even if output PNG already exists")
    p_rend.add_argument("--no-lmx", action="store_true",
                        help="Skip inline LMX generation")
    p_rend.set_defaults(func=cmd_render)

    # ── convert ───────────────────────────────────────────────────────
    p_conv = sub.add_parser(
        "convert",
        help="Convert PrIMuS .semantic → monophonic LMX",
        description="Run the semantic → LMX conversion pipeline.",
    )
    p_conv.add_argument("--source", type=Path,
                        default=Path("data/realbook_primus_aa"),
                        help="Root directory of PrIMuS samples")
    p_conv.add_argument("--limit", type=int, default=None,
                        help="Process only N samples (for smoke tests)")
    p_conv.add_argument("--workers", type=int, default=None,
                        help="Parallel workers for conversion")
    p_conv.add_argument("--verbose", action="store_true",
                        help="Per-sample conversion logging")
    p_conv.add_argument("--keep-visual", action="store_true",
                        help="Retain visual tokens (beam, stem, staff)")
    p_conv.set_defaults(func=cmd_convert)

    # ── augment ───────────────────────────────────────────────────────
    p_aug = sub.add_parser(
        "augment",
        help="Apply scan-simulation augmentations to clean images",
        description="Distort clean LilyJAZZ PNGs to simulate physical scans.",
    )
    p_aug.add_argument("--source", type=Path,
                       default=Path("data/realbook_primus_aa"),
                       help="Clean dataset root (default: data/realbook_primus_aa)")
    p_aug.add_argument("--output", type=Path,
                       default=Path("data/realbook_primus_aa_scanned"),
                       help="Output root (default: data/realbook_primus_aa_scanned)")
    p_aug.add_argument("--copies", type=int, default=None,
                       help="Augmented copies per sample (default: 1)")
    p_aug.add_argument("--seed", type=int, default=None,
                       help="Global random seed (default: 42)")
    p_aug.add_argument("--workers", type=int, default=None,
                       help="Parallel workers (default: half CPU count)")
    p_aug.add_argument("--limit", type=int, default=None,
                       help="Process at most N samples (for testing)")
    p_aug.set_defaults(func=cmd_augment)

    # ── vocab ─────────────────────────────────────────────────────────
    p_vocab = sub.add_parser(
        "vocab",
        help="Build LMX vocabulary file from .lmx data",
        description="Scan .lmx files and produce a sorted vocabulary.",
    )
    p_vocab.add_argument("--data-dir", type=str,
                         default="data/realbook_primus_aa",
                         help="Directory with .lmx files (searched recursively)")
    p_vocab.add_argument("--output", type=str,
                         default="src/CRNN_CTC/vocabulary.txt",
                         help="Output vocabulary file path")
    p_vocab.set_defaults(func=cmd_vocab)

    # ── train ─────────────────────────────────────────────────────────
    p_train = sub.add_parser(
        "train",
        help="Train the CRNN-CTC model",
        description="Run the full training loop with CTC loss and AMP.",
    )
    _add_common_data_args(p_train)
    _add_model_args(p_train)
    g_train = p_train.add_argument_group("training")
    g_train.add_argument("--epochs", type=int, default=None,
                         help="Training epochs (default: 50)")
    g_train.add_argument("--batch-size", type=int, default=None,
                         help="Mini-batch size (default: 16)")
    g_train.add_argument("--lr", type=float, default=None,
                         help="Peak learning rate (default: 1e-3)")
    g_train.add_argument("--weight-decay", type=float, default=None,
                         help="AdamW weight decay (default: 1e-4)")
    g_train.add_argument("--warmup-frac", type=float, default=None,
                         help="LR warm-up fraction (default: 0.05)")
    g_train.add_argument("--early-stopping-patience", type=int, default=None,
                         help="Stop after N epochs without val SER improvement (0=off, default: 10)")
    g_train.add_argument("--model-dir", type=str, default=None,
                         help="Directory for checkpoints (default: models/)")
    p_train.set_defaults(func=cmd_train)

    # ── evaluate ──────────────────────────────────────────────────────
    p_eval = sub.add_parser(
        "evaluate",
        help="Evaluate a model checkpoint",
        description="Load a checkpoint, decode a split, and report SER.",
    )
    p_eval.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    p_eval.add_argument("--split", choices=["train", "val", "test"],
                        default="test", help="Split to evaluate (default: test)")
    p_eval.add_argument("--per-sample", action="store_true",
                        help="Log per-sample SER (worst first)")
    _add_common_data_args(p_eval)
    _add_model_args(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args.func(args)


if __name__ == "__main__":
    main()
