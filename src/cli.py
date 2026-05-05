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
evaluate-ab Compare SER for multiple beam widths on one split.

Usage examples::

    poetry run python src/cli.py render  --source data/raw/primus --output data/processed/primus/clean
    poetry run python src/cli.py convert --source data/processed/primus/clean --workers 8
    poetry run python src/cli.py augment --source data/processed/primus/clean --output data/processed/primus/scanned
    poetry run python src/cli.py vocab   --data-dir data/processed/primus/clean
    poetry run python src/cli.py train   --epochs 50 --batch-size 16 --lr 1e-3
    poetry run python src/cli.py evaluate --checkpoint models/latest/best_model.pt --split test
    poetry run python src/cli.py api
    poetry run python src/cli.py pipeline
    poetry run python src/cli.py pipeline-train
"""

from __future__ import annotations

import argparse
import logging
import os
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

# Full pipeline defaults (render DPI list matches generate_realbook).
# Multiple values trigger per-sample DPI jitter, which improves robustness
# to scanner-resolution mismatch between training renders and real PDFs.
_FULL_RUN_RENDER_DPI: tuple[int, ...] = (200, 250, 300)
_FULL_RUN_AUGMENT_SEED = 42
_FULL_RUN_AUGMENT_COPIES = 1  # scanned tree mirrors clean sample ids (one PNG per id)


def _get_default_workers() -> int:
    """Compute default number of workers based on CPU core count."""
    cpu_count = os.cpu_count() or 4  # Default to 4 if cpu_count() returns None
    # Leave at least 2 cores for the OS and other processes
    return max(1, cpu_count - 2)


def _discover_packages(root: Path) -> list[str]:
    """Return sorted package_* directory names inside ``root``."""
    if not root.is_dir():
        return []
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and p.name.startswith("package_")
    )


def _resolve_packages(primus_dir: Path, requested: list[str] | None) -> list[str]:
    """Resolve package list, auto-discovering from ``primus_dir`` when omitted."""
    if requested:
        return requested

    discovered = _discover_packages(primus_dir)
    if not discovered:
        log.error(
            "No package_* directories found under %s. "
            "Pass --packages explicitly or check the data path.",
            primus_dir,
        )
        sys.exit(1)
    log.info("Auto-discovered packages: %s", ", ".join(discovered))
    return discovered


def _wire_training_data_from_packages(args: argparse.Namespace, packages: list[str]) -> None:
    """Populate train args using clean+augmented dirs for all selected packages."""
    clean_packages = [p for p in packages if (args.output_dir / p).is_dir()]
    augmented_packages = [p for p in packages if (args.augmented_dir / p).is_dir()]

    usable_packages = [p for p in clean_packages if p in set(augmented_packages)]
    if not usable_packages:
        log.error(
            "No package had both clean and augmented data. "
            "Checked %s and %s for packages: %s",
            args.output_dir,
            args.augmented_dir,
            ", ".join(packages),
        )
        sys.exit(1)

    args.data_dir = str(args.output_dir / usable_packages[0])
    args.scanned_dir = str(args.augmented_dir / usable_packages[0])
    args.extra_data_dir = [str(args.output_dir / p) for p in usable_packages[1:]] or None
    args.extra_scanned_dir = [str(args.augmented_dir / p) for p in usable_packages[1:]] or None

    log.info(
        "Training data packages: %s",
        ", ".join(usable_packages),
    )


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
        "--dpi", *(str(d) for d in args.dpi),
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
    """Convert PrIMuS .semantic annotations → monophonic LMX (direct token remapping)."""
    from data_processing.semantic_to_lmx import main as _convert_main

    # Build argv for the converter's own argparse
    argv: list[str] = ["--source", str(args.source)]
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]
    if args.workers is not None:
        argv += ["--workers", str(args.workers)]
    if args.verbose:
        argv.append("--verbose")

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

    data_dirs = [Path(args.data_dir)]
    if getattr(args, "extra_data_dir", None):
        data_dirs.extend(Path(p) for p in args.extra_data_dir)

    out_path = Path(args.output)

    log.info("Building vocabulary from %d directories …", len(data_dirs))
    for d in data_dirs:
        log.info("  - %s", d)
    try:
        vocab = Vocabulary.build_from_lmx_dirs(data_dirs, workers=args.workers)
    except RuntimeError as exc:
        log.error("Vocabulary build failed: %s", exc)
        sys.exit(1)

    vocab.save(out_path)
    log.info("Saved vocabulary (%d tokens incl. blank+pad+unk) → %s", len(vocab), out_path)


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
        "backbone": "backbone",
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
        "rare_lmx_oversample": "rare_lmx_oversample",
        "strip_header_prob": "strip_header_prob",
        "online_aug_prob": "online_aug_prob",
    }
    # Boolean filter flags use store_false with default=None (only
    # override when the user explicitly passes the --no-... flag).
    for bflag in ("filter_multi_staff",):
        val = getattr(args, bflag, None)
        if val is not None:
            overrides[bflag] = val
    for flag, field in flag_map.items():
        val = getattr(args, flag, None)
        if val is not None:
            overrides[field] = val

    # Repeatable list flags
    extra_data = getattr(args, "extra_data_dir", None)
    if extra_data:
        overrides["extra_data_dirs"] = [Path(p) for p in extra_data]
    extra_scanned = getattr(args, "extra_scanned_dir", None)
    if extra_scanned:
        overrides["extra_scanned_dirs"] = [Path(p) for p in extra_scanned]

    ft_clean = getattr(args, "finetune_data_dir", None)
    if ft_clean:
        overrides["finetune_data_dirs"] = [Path(p) for p in ft_clean]
    ft_scanned = getattr(args, "finetune_scanned_dir", None)
    if ft_scanned:
        overrides["finetune_scanned_dirs"] = [Path(p) for p in ft_scanned]

    rlx = getattr(args, "rare_lmx_tokens", None)
    if rlx is not None:
        overrides["rare_lmx_tokens"] = tuple(
            t.strip() for t in rlx.split(",") if t.strip()
        )

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

    # Validate that data directories exist before committing to a training run
    _missing: list[str] = []
    if not cfg.data_dir.is_dir():
        _missing.append(f"data_dir: {cfg.data_dir}")
    if cfg.use_scanned and not cfg.scanned_dir.is_dir():
        _missing.append(f"scanned_dir: {cfg.scanned_dir}")
    for extra in cfg.extra_data_dirs or []:
        if not Path(extra).is_dir():
            _missing.append(f"extra_data_dir: {extra}")
    if cfg.use_scanned:
        for extra in cfg.extra_scanned_dirs or []:
            if not Path(extra).is_dir():
                _missing.append(f"extra_scanned_dir: {extra}")
    for ft in cfg.finetune_data_dirs or []:
        if not Path(ft).is_dir():
            _missing.append(f"finetune_data_dir: {ft}")
    if cfg.use_scanned:
        for ft in cfg.finetune_scanned_dirs or []:
            if not Path(ft).is_dir():
                _missing.append(f"finetune_scanned_dir: {ft}")
    if _missing:
        log.error("Data directories not found:\n  %s", "\n  ".join(_missing))
        sys.exit(1)

    resume_from: Path | None = None
    resume_arg = getattr(args, "resume", None)
    if resume_arg is not None:
        # --resume without a value → auto-detect from latest run
        if resume_arg == "":
            # Try new layout: models/latest/latest_checkpoint.pt
            auto = cfg.model_dir / "latest" / "latest_checkpoint.pt"
            if not auto.exists():
                # Fallback: legacy flat layout
                auto = cfg.model_dir / "latest_checkpoint.pt"
            if not auto.exists():
                log.error(
                    "No checkpoint found in %s. Start a fresh run first.",
                    cfg.model_dir,
                )
                sys.exit(1)
            resume_from = auto.resolve()  # resolve symlinks
        else:
            resume_from = Path(resume_arg)

    best_ckpt = train(cfg, resume_from=resume_from)
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
        beam_width=getattr(args, "beam_width", 1) or 1,
    )
    print(f"SER ({args.split}): {ser:.4f}")


def cmd_evaluate_ab(args: argparse.Namespace) -> None:
    """Compare greedy vs beam search SER on the same split (CRNN dataloader path)."""
    from CRNN_CTC.evaluate import evaluate

    cfg = _build_config_from_args(args)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        log.error("Checkpoint not found: %s", checkpoint)
        sys.exit(1)

    beams = [int(x.strip()) for x in args.beams.split(",") if x.strip()]
    if not beams:
        log.error("No beam widths parsed from %r", args.beams)
        sys.exit(1)

    print(f"{'beam':>6}  SER ({args.split})")
    for bw in beams:
        ser = evaluate(cfg, checkpoint, split=args.split, beam_width=max(1, bw))
        print(f"{bw:>6d}  {ser:.4f}")


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


# ── api ────────────────────────────────────────────────────────────────────

def cmd_api(args: argparse.Namespace) -> None:
    """Start the OMR web API server."""
    import uvicorn
    # Run from project root; api lives in src/api
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from api.main import app
    uvicorn.run(app, host=args.host, port=args.port)


# ── pipeline ──────────────────────────────────────────────────────────────

def cmd_pipeline(args: argparse.Namespace) -> None:
    """Run the full data pipeline (render → convert → augment → vocab).

    New layout (no package-specific wiring):
        raw PrIMuS:          data/raw/primus/...
        clean rendered:      data/processed/primus/clean/...
        scanned/augmented:   data/processed/primus/scanned/...
        vocabulary:          data/vocab/primus_lmx.txt
    """
    extra_vocab = getattr(args, "extra_vocab_data_dir", None) or None

    # 1. Render raw PrIMuS → clean dataset (PNG + .semantic/.agnostic/.mid + .lmx)
    log.info("--- Rendering PrIMuS → clean dataset ---")
    render_args = argparse.Namespace(
        source=args.raw_primus_dir,
        output=args.clean_dir,
        dpi=getattr(args, "render_dpi", _FULL_RUN_RENDER_DPI),
        limit=args.limit,
        workers=args.workers,
        force=getattr(args, "force_render", False),
        no_lmx=False,
        verbose=args.verbose,
    )
    cmd_render(render_args)

    # 2. (Optional) Re-run semantic → LMX conversion over clean dataset
    #    This is safe even if generate_realbook already produced .lmx files.
    log.info("--- Converting .semantic → .lmx in clean dataset ---")
    convert_args = argparse.Namespace(
        source=args.clean_dir,
        limit=args.limit,
        workers=args.workers,
        verbose=args.verbose,
    )
    cmd_convert(convert_args)

    # 3. Augment clean images → scanned/augmented dataset (labels copied over)
    log.info("--- Augmenting clean images → scanned dataset ---")
    augment_args = argparse.Namespace(
        source=args.clean_dir,
        output=args.scanned_dir,
        copies=getattr(args, "augment_copies", _FULL_RUN_AUGMENT_COPIES),
        seed=getattr(args, "augment_seed", _FULL_RUN_AUGMENT_SEED),
        workers=args.workers,
        limit=args.limit,
    )
    cmd_augment(augment_args)

    # 4. Build unified vocabulary over the clean dataset
    log.info("--- Building unified LMX vocabulary ---")
    vocab_args = argparse.Namespace(
        data_dir=str(args.clean_dir),
        extra_data_dir=extra_vocab,
        output=args.vocab_path,
        workers=args.workers,
    )
    cmd_vocab(vocab_args)


# ── pipeline-train ────────────────────────────────────────────────────────

def cmd_pipeline_train(args: argparse.Namespace) -> None:
    """Run full pipeline followed by training."""
    cmd_pipeline(args)
    log.info("--- Starting Training ---")

    # Wire pipeline-produced dirs into training args for Config.
    args.data_dir = str(args.clean_dir)
    args.scanned_dir = str(args.scanned_dir)
    args.vocab_path = str(args.vocab_path)
    # Full-run default: train on scanned aug unless explicitly disabled
    if getattr(args, "use_scanned", None) is None:
        args.use_scanned = True
    cmd_train(args)


# ═══════════════════════════════════════════════════════════════════════════
# Argument parser
# ═══════════════════════════════════════════════════════════════════════════

def _add_common_data_args(parser: argparse.ArgumentParser) -> None:
    """Add flags shared by train / evaluate."""
    g = parser.add_argument_group("data")
    g.add_argument("--data-dir", type=str, default=None,
                   help="Root clean sample directory "
                        "(default: data/processed/primus/clean)")
    g.add_argument("--scanned-dir", type=str, default=None,
                   help="Scanned-image directory "
                        "(default: data/processed/primus/scanned)")
    g.add_argument("--vocab-path", type=str, default=None,
                   help="Vocabulary file (default: data/vocab/primus_lmx.txt)")
    g.add_argument("--img-height", type=int, default=None,
                   help="Resize images to this height (default: 128)")
    scan_group = g.add_mutually_exclusive_group()
    scan_group.add_argument(
        "--use-scanned",
        dest="use_scanned",
        action="store_true",
        default=None,
        help="Use augmented scanned images instead of clean originals (default: True)",
    )
    scan_group.add_argument(
        "--no-use-scanned",
        dest="use_scanned",
        action="store_false",
        help="Force use of clean originals only (override default True).",
    )
    g.add_argument("--val-frac", type=float, default=None,
                   help="Validation split fraction (default: 0.10)")
    g.add_argument("--test-frac", type=float, default=None,
                   help="Test split fraction (default: 0.10)")
    g.add_argument("--num-workers", type=int, default=None,
                   help="DataLoader workers (default: 10)")
    g.add_argument("--seed", type=int, default=None,
                   help="Random seed (default: 42)")
    g.add_argument("--no-filter-multi-staff", dest="filter_multi_staff",
                   action="store_false", default=None,
                   help="Disable filtering of multi-staff (tall) images (default: enabled)")
    g.add_argument("--max-source-height", type=int, default=None,
                   help="Max original image height for single-staff filter (default: 180 px)")
    g.add_argument("--extra-data-dir", type=str, action="append", default=None,
                   help="Additional data directory (repeatable, for package_ab etc.)")
    g.add_argument("--extra-scanned-dir", type=str, action="append", default=None,
                   help="Additional scanned-image directory (repeatable)")


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add model-architecture flags shared by train / evaluate."""
    g = parser.add_argument_group("model")
    g.add_argument("--backbone", type=str, default=None,
                   choices=["resnet18", "vgg"],
                   help="CNN backbone (default: resnet18)")
    g.add_argument("--cnn-out-channels", type=int, default=None,
                   help="CNN output feature maps — VGG only (default: 256)")
    g.add_argument("--cnn-dropout", type=float, default=None,
                   help="Dropout2d after CNN blocks (default: 0.25)")
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
                        default=Path("data/raw/primus"),
                        help="PrIMuS source root (default: data/raw/primus)")
    p_rend.add_argument("--output", type=Path,
                        default=Path("data/processed/primus/clean"),
                        help="Clean rendered dataset root "
                             "(default: data/processed/primus/clean)")
    p_rend.add_argument("--dpi", type=int, nargs="+",
                        default=[200, 250, 300],
                        help="Rendering resolution(s).  Pass one int for a "
                             "fixed DPI or several for per-sample uniform "
                             "jitter (default: 200 250 300).")
    p_rend.add_argument("--limit", type=int, default=None,
                        help="Process at most N samples (for testing)")
    p_rend.add_argument("--workers", type=int, default=_get_default_workers(),
                        help="Parallel workers (default: cpu_count - 2)")
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
                        default=Path("data/processed/primus/clean"),
                        help="Root directory of clean rendered samples "
                             "(default: data/processed/primus/clean)")
    p_conv.add_argument("--limit", type=int, default=None,
                        help="Process only N samples (for smoke tests)")
    p_conv.add_argument("--workers", type=int, default=_get_default_workers(),
                        help="Parallel workers for conversion (default: cpu_count - 2)")
    p_conv.add_argument("--verbose", action="store_true",
                        help="Per-sample conversion logging")
    p_conv.set_defaults(func=cmd_convert)

    # ── augment ───────────────────────────────────────────────────────
    p_aug = sub.add_parser(
        "augment",
        help="Apply scan-simulation augmentations to clean images",
        description="Distort clean LilyJAZZ PNGs to simulate physical scans.",
    )
    p_aug.add_argument("--source", type=Path,
                       default=Path("data/processed/primus/clean"),
                       help="Clean dataset root (default: data/processed/primus/clean)")
    p_aug.add_argument("--output", type=Path,
                       default=Path("data/processed/primus/scanned"),
                       help="Scanned/augmented dataset root "
                            "(default: data/processed/primus/scanned)")
    p_aug.add_argument("--copies", type=int, default=None,
                       help="Augmented copies per sample (default: 1)")
    p_aug.add_argument("--seed", type=int, default=None,
                       help="Global random seed (default: 42)")
    p_aug.add_argument("--workers", type=int, default=_get_default_workers(),
                       help="Parallel workers (default: cpu_count - 2)")
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
                         default="data/processed/primus/clean",
                         help="Directory with .lmx files (searched recursively, "
                              "default: data/processed/primus/clean)")
    p_vocab.add_argument("--extra-data-dir", type=str, action="append", default=None,
                         help="Additional data directory (repeatable, for package_ab etc.)")
    p_vocab.add_argument("--output", type=str,
                         default="data/vocab/primus_lmx.txt",
                         help="Output vocabulary file path "
                              "(default: data/vocab/primus_lmx.txt)")
    p_vocab.add_argument(
        "--workers",
        type=int,
        default=_get_default_workers(),
        help="Parallel workers when scanning .lmx files (default: cpu_count - 2)",
    )
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
                   help="Number of training epochs (default: 60)")
    g_train.add_argument("--batch-size", type=int, default=None,
                   help="Training batch size (default: 16)")
    g_train.add_argument("--lr", type=float, default=None,
                   help="Peak learning rate — OneCycleLR (default: 1e-3)")
    g_train.add_argument("--weight-decay", type=float, default=None,
                   help="AdamW weight decay (default: 1e-4)")
    g_train.add_argument("--warmup-frac", type=float, default=None,
                   help="Fraction of steps for LR warm-up (default: 0.08)")
    g_train.add_argument("--early-stopping-patience", type=int, default=None,
                   help="Epochs without val SER improvement to wait before stopping (default: 12)")
    g_train.add_argument("--model-dir", type=str, default=None,
                         help="Directory for checkpoints (default: models/)")
    g_train.add_argument("--resume", nargs="?", const="", default=None, metavar="CHECKPOINT",
                         help="Resume from a checkpoint. Omit a path to auto-use "
                              "the latest run's checkpoint, or supply an explicit .pt path.")
    g_train.add_argument(
        "--rare-lmx-oversample",
        type=int,
        default=None,
        help="Repeat training indices for samples containing rare LMX tokens "
             "(default: 2; 1 disables). Tokens: --rare-lmx-tokens.",
    )
    g_train.add_argument(
        "--strip-header-prob",
        type=float,
        default=None,
        help="Probability of stripping clef+key+time visual header during "
             "training (default: 0.4; 0 disables).",
    )
    g_train.add_argument(
        "--online-aug-prob",
        type=float,
        default=None,
        help="Probability of applying lightweight online jitter (brightness, "
             "noise, ±2 px shift) per training sample (default: 0.5; 0 disables).",
    )
    g_train.add_argument(
        "--rare-lmx-tokens",
        type=str,
        default=None,
        help="Comma-separated LMX tokens to up-weight (default: tied:start,tied:stop,key:fifths:0). "
             "Pass empty string to disable token-based oversampling.",
    )
    g_train.add_argument(
        "--finetune-data-dir",
        type=str,
        action="append",
        default=None,
        help="Extra clean dataset root for fine-tuning (repeatable); merged into train split.",
    )
    g_train.add_argument(
        "--finetune-scanned-dir",
        type=str,
        action="append",
        default=None,
        help="Extra scanned-image root for fine-tuning (repeatable); used when --use-scanned.",
    )
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
    p_eval.add_argument("--beam-width", type=int, default=1,
                        help="Beam search width (1=greedy, default: 1)")
    _add_common_data_args(p_eval)
    _add_model_args(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    p_eval_ab = sub.add_parser(
        "evaluate-ab",
        help="Compare SER across multiple beam widths (same split)",
        description="Runs evaluate() once per beam width for A/B comparison.",
    )
    p_eval_ab.add_argument("--checkpoint", type=str, required=True,
                           help="Path to model checkpoint (.pt)")
    p_eval_ab.add_argument("--split", choices=["train", "val", "test"],
                           default="test", help="Split (default: test)")
    p_eval_ab.add_argument(
        "--beams",
        type=str,
        default="1,5,10",
        help="Comma-separated beam widths (default: 1,5,10)",
    )
    _add_common_data_args(p_eval_ab)
    _add_model_args(p_eval_ab)
    p_eval_ab.set_defaults(func=cmd_evaluate_ab)

    # ── api ───────────────────────────────────────────────────────────
    p_api = sub.add_parser(
        "api",
        help="Start the OMR web API server",
    )
    p_api.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p_api.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p_api.set_defaults(func=cmd_api)

    # ── pipeline ──────────────────────────────────────────────────────
    p_pipe = sub.add_parser(
        "pipeline",
        help="Run full data pipeline (render → convert → augment → vocab)",
    )
    p_pipe.add_argument("--raw-primus-dir", type=Path, default=Path("data/raw/primus"),
                        help="PrIMuS source root (default: data/raw/primus)")
    p_pipe.add_argument("--clean-dir", type=Path, default=Path("data/processed/primus/clean"),
                        help="Rendered clean output root "
                             "(default: data/processed/primus/clean)")
    p_pipe.add_argument("--scanned-dir", type=Path, default=Path("data/processed/primus/scanned"),
                        help="Scanned/augmented output root "
                             "(default: data/processed/primus/scanned)")
    p_pipe.add_argument("--vocab-path", type=str, default="data/vocab/primus_lmx.txt",
                        help="Output vocabulary path "
                             "(default: data/vocab/primus_lmx.txt)")
    p_pipe.add_argument(
        "--extra-vocab-data-dir",
        type=str,
        action="append",
        default=None,
        help="Extra directory to scan for .lmx when building vocab (repeatable), "
             "e.g. an additional realbook_primus package.",
    )
    p_pipe.add_argument(
        "--render-dpi", type=int, nargs="+",
        default=list(_FULL_RUN_RENDER_DPI),
        help=(
            "LilyPond render DPI(s).  Pass one int for a fixed DPI or several "
            "for per-sample uniform jitter (default: %s)."
            % " ".join(str(d) for d in _FULL_RUN_RENDER_DPI)
        ),
    )
    p_pipe.add_argument(
        "--force-render",
        action="store_true",
        help="Re-render every sample even if PNG exists (use after pipeline code changes).",
    )
    p_pipe.add_argument(
        "--augment-copies",
        type=int,
        default=_FULL_RUN_AUGMENT_COPIES,
        help=f"Augmented PNGs per clean sample; keep 1 so scanned/ mirrors clean ids "
             f"(default: {_FULL_RUN_AUGMENT_COPIES}).",
    )
    p_pipe.add_argument(
        "--augment-seed",
        type=int,
        default=_FULL_RUN_AUGMENT_SEED,
        help=f"Augmentation RNG seed (default: {_FULL_RUN_AUGMENT_SEED}).",
    )
    p_pipe.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples processed (for testing)")
    p_pipe.add_argument("--workers", type=int, default=_get_default_workers(),
                        help="Parallel workers (default: cpu_count - 2)")
    p_pipe.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    p_pipe.set_defaults(func=cmd_pipeline)

    # ── pipeline-train ────────────────────────────────────────────────
    p_ptrain = sub.add_parser(
        "pipeline-train",
        help="Run full data pipeline followed by training",
    )
    # Pipeline-specific args (dedicated to pipeline; train uses wired values)
    p_ptrain.add_argument("--raw-primus-dir", type=Path, default=Path("data/raw/primus"))
    p_ptrain.add_argument("--clean-dir", type=Path, default=Path("data/processed/primus/clean"))
    p_ptrain.add_argument("--scanned-dir", type=Path, default=Path("data/processed/primus/scanned"))
    p_ptrain.add_argument("--vocab-path", type=str, default="data/vocab/primus_lmx.txt")
    p_ptrain.add_argument(
        "--extra-vocab-data-dir",
        type=str,
        action="append",
        default=None,
        help="Extra .lmx roots for vocab (repeatable); same as pipeline.",
    )
    p_ptrain.add_argument(
        "--render-dpi", type=int, nargs="+",
        default=list(_FULL_RUN_RENDER_DPI),
        help=(
            "LilyPond render DPI(s); same semantics as `pipeline --render-dpi` "
            "(default: %s)."
            % " ".join(str(d) for d in _FULL_RUN_RENDER_DPI)
        ),
    )
    p_ptrain.add_argument(
        "--force-render",
        action="store_true",
        help="Re-render all samples (see pipeline).",
    )
    p_ptrain.add_argument("--augment-copies", type=int, default=_FULL_RUN_AUGMENT_COPIES)
    p_ptrain.add_argument("--augment-seed", type=int, default=_FULL_RUN_AUGMENT_SEED)
    p_ptrain.add_argument("--limit", type=int, default=None)
    p_ptrain.add_argument("--workers", type=int, default=_get_default_workers())
    p_ptrain.add_argument("--verbose", action="store_true")
    
    # Training-specific args (inherited, excluding generic data paths)
    _add_model_args(p_ptrain)
    
    # Training-specific flags not covered by common groups
    g_train = p_ptrain.add_argument_group("training")
    g_train.add_argument("--epochs", type=int, default=None,
                         help="Training epochs (default: from Config, 60)")
    g_train.add_argument("--batch-size", type=int, default=None,
                         help="Batch size (default: 16)")
    g_train.add_argument("--lr", type=float, default=None,
                         help="Peak learning rate / OneCycleLR (default: 1e-3)")
    g_train.add_argument("--weight-decay", type=float, default=None)
    g_train.add_argument("--warmup-frac", type=float, default=None)
    g_train.add_argument("--early-stopping-patience", type=int, default=None)
    g_train.add_argument("--model-dir", type=str, default=None)
    g_train.add_argument("--resume", nargs="?", const="", default=None, metavar="CHECKPOINT")
    # Data-related training overrides (optional; by default pipeline dirs are used)
    g_data = p_ptrain.add_argument_group("data")
    g_data.add_argument("--img-height", type=int, default=None,
                        help="Resize images to this height (default: 128)")
    scan_group = g_data.add_mutually_exclusive_group()
    scan_group.add_argument(
        "--use-scanned",
        dest="use_scanned",
        action="store_true",
        default=None,
        help="Use scanned/augmented images (default: True)",
    )
    scan_group.add_argument(
        "--no-use-scanned",
        dest="use_scanned",
        action="store_false",
        help="Force use of clean originals only (override default True).",
    )
    g_data.add_argument("--val-frac", type=float, default=None,
                        help="Validation split fraction (default: 0.10)")
    g_data.add_argument("--test-frac", type=float, default=None,
                        help="Test split fraction (default: 0.10)")
    g_data.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader workers (default: 10)")
    g_data.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: 42)")
    g_data.add_argument("--no-filter-multi-staff", dest="filter_multi_staff",
                        action="store_false", default=None,
                        help="Disable filtering of multi-staff (tall) images (default: enabled)")
    g_data.add_argument("--max-source-height", type=int, default=None,
                        help="Max original image height for single-staff filter (default: 180 px)")
    g_data.add_argument("--extra-data-dir", type=str, action="append", default=None,
                        help="Additional clean data directory (repeatable)")
    g_data.add_argument("--extra-scanned-dir", type=str, action="append", default=None,
                        help="Additional scanned data directory (repeatable)")
    
    p_ptrain.set_defaults(func=cmd_pipeline_train)

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
