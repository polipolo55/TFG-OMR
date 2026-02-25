#!/usr/bin/env python3
"""
cli.py — Unified command-line interface for the TFG-OMR pipeline.
=================================================================

Subcommands
-----------
convert     Convert PrIMuS .semantic → LMX (Route A).
vocab       Build LMX vocabulary from converted data.
train       Train the CRNN-CTC model.
evaluate    Evaluate a trained model on test split.

Usage::

    poetry run python src/cli.py convert --source data/realbook_primus_aa --workers 8
    poetry run python src/cli.py vocab
    poetry run python src/cli.py train --epochs 40
    poetry run python src/cli.py evaluate --checkpoint models/best_model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so relative imports work when the
# script is invoked directly (``python src/cli.py …``).
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent  # …/TFG-OMR
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Subcommand handlers ───────────────────────────────────────────────────

def cmd_convert(args: argparse.Namespace) -> None:
    """Run the semantic → LMX conversion."""
    from src.data_processing.semantic_to_lmx import main as _convert_main

    # Build the argv list that semantic_to_lmx.main() expects.
    argv: list[str] = ["--source", str(args.source)]
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]
    if args.workers is not None:
        argv += ["--workers", str(args.workers)]
    if args.verbose:
        argv.append("--verbose")
    if args.keep_visual:
        argv.append("--keep-visual")

    # Patch sys.argv so argparse inside semantic_to_lmx sees the flags.
    old_argv = sys.argv
    sys.argv = ["semantic_to_lmx"] + argv
    try:
        _convert_main()
    finally:
        sys.argv = old_argv


def cmd_vocab(args: argparse.Namespace) -> None:
    """Build the LMX vocabulary file from a directory of .lmx files."""
    from src.CRNN_CTC.vocab import Vocabulary

    data_dir = Path(args.data_dir)
    out_path = Path(args.output)

    print(f"Building vocabulary from {data_dir} …")
    vocab = Vocabulary.build_from_lmx_dir(data_dir)
    vocab.save(out_path)
    print(f"Saved vocabulary ({len(vocab)} tokens) → {out_path}")


def cmd_train(args: argparse.Namespace) -> None:
    """Launch CRNN-CTC training (not yet implemented)."""
    from src.CRNN_CTC.config import Config
    from src.CRNN_CTC.train import train

    cfg = Config()
    train(cfg)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a trained model checkpoint (not yet implemented)."""
    print("Evaluation not yet implemented — model training required first.")
    sys.exit(1)


# ── CLI definition ────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omr",
        description="TFG-OMR: Optical Music Recognition pipeline CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── convert ───────────────────────────────────────────────────────
    p_conv = sub.add_parser(
        "convert",
        help="Convert PrIMuS .semantic files → monophonic LMX",
    )
    p_conv.add_argument(
        "--source", type=Path, default=Path("data/realbook_primus_aa"),
        help="Root directory of PrIMuS samples (default: data/realbook_primus_aa)",
    )
    p_conv.add_argument("--limit", type=int, default=None, help="Process N samples only")
    p_conv.add_argument("--workers", type=int, default=None, help="Parallel workers")
    p_conv.add_argument("--verbose", action="store_true", help="Per-sample logging")
    p_conv.add_argument("--keep-visual", action="store_true", help="Keep visual tokens")
    p_conv.set_defaults(func=cmd_convert)

    # ── vocab ─────────────────────────────────────────────────────────
    p_vocab = sub.add_parser("vocab", help="Build LMX vocabulary file")
    p_vocab.add_argument(
        "--data-dir", type=str, default="data/realbook_primus_aa",
        help="Directory with .lmx files (searched recursively)",
    )
    p_vocab.add_argument(
        "--output", type=str,
        default="src/CRNN_CTC/vocabulary.txt",
        help="Output vocabulary file path",
    )
    p_vocab.set_defaults(func=cmd_vocab)

    # ── train ─────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train the CRNN-CTC model (stub)")
    p_train.set_defaults(func=cmd_train)

    # ── evaluate ──────────────────────────────────────────────────────
    p_eval = sub.add_parser("evaluate", help="Evaluate a model checkpoint (stub)")
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
