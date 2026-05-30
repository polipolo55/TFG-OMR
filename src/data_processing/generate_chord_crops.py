"""
generate_chord_crops.py
=======================
Generate a synthetic Real Book chord-strip dataset for the chord CRNN.

Each sample is a horizontal strip of N (random) jazz chords rendered with
LilyPond + LilyJAZZ font and Real Book-style chord naming overrides (see
``chord_render.py``).  The label is the space-joined visual chord string.

Output layout::

    {output}/
        train/
            00000000.png
            00000001.png
            …
        val/
            10000000.png
            …
        train_labels.csv          # filename,label
        val_labels.csv

Usage::

    poetry run python src/data_processing/generate_chord_crops.py \
        --output data/chord_synth --num-train 30000 --num-val 2000 --workers 8
"""

from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing
import random
import sys
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Make src/ importable when run from project root
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from CRNN_CTC.lilypond_render import crop_content, run_lilypond
from data_processing.chord_render import (
    choose_halfdim_style,
    make_ly_source,
    sample_progression,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

# DPI range — matches the effective resolution of inference-time chord crops
# (page PDFs render at 300dpi → chord strips upscaled/downscaled to 64-256 px tall).
_DPI_CHOICES = (140, 160, 180, 200, 220, 240, 260)

# Staff-size jitter — varies chord text size relative to the strip; mirrors the
# music renderer's staff-size augmentation.  Smaller sizes produce denser text.
_STAFF_SIZES = (14, 16, 17, 18, 19, 20, 21, 22, 24)

# Chords per strip — Real Book inter-staff strips typically carry 2–8 chords.
_CHORDS_RANGE = (2, 8)


def render_one(
    seed: int,
    out_dir: Path,
) -> tuple[str, str] | None:
    """Render one chord strip; return (filename, label) or None on failure.

    Pure function suitable for ``multiprocessing.Pool``.  Uses the seed to
    drive deterministic sampling so a given seed always produces the same
    chord progression — useful for reproducibility and resuming.
    """
    rng = random.Random(seed)
    n_chords = rng.randint(*_CHORDS_RANGE)
    chords = sample_progression(rng, n_chords)
    body = " ".join(c.lily for c in chords)
    label = " ".join(c.label for c in chords)

    dpi = rng.choice(_DPI_CHOICES)
    size = rng.choice(_STAFF_SIZES)
    # Paper width scales with chord count so chords aren't cramped.  LilyPond
    # uses ragged-right, so this is an upper bound — actual ink width is
    # whatever the chords take up.
    paper_width = 50 + 30 * n_chords
    staff_size_directive = f"#(set-global-staff-size {size})"
    ly_source = make_ly_source(
        body,
        paper_width=paper_width,
        staff_size_directive=staff_size_directive,
        halfdim_style=choose_halfdim_style(rng),
    )

    name = f"{seed:08d}"
    with tempfile.TemporaryDirectory(prefix="chord_synth_") as tmp:
        tmp_dir = Path(tmp)
        png_path = run_lilypond(ly_source, name, tmp_dir, dpi=dpi, timeout=20)
        if png_path is None:
            return None
        try:
            raw = np.array(Image.open(png_path).convert("L"))
        except Exception:
            return None
        cropped = crop_content(raw, pad=8)
        if cropped.size == 0 or np.all(cropped == 255):
            return None
        # Defensively skip strips that are absurdly thin/short — these
        # indicate a render glitch.
        if cropped.shape[0] < 12 or cropped.shape[1] < 20:
            return None
        out_png = out_dir / f"{name}.png"
        Image.fromarray(cropped).save(out_png, optimize=True)

    return name + ".png", label


# ---------------------------------------------------------------------------
# Multiprocess driver
# ---------------------------------------------------------------------------


def generate_split(
    split_name: str,
    seeds: range,
    out_root: Path,
    workers: int,
) -> int:
    """Generate one split (train or val); return number of successful samples."""
    split_dir = out_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    labels_csv = out_root / f"{split_name}_labels.csv"

    n_ok = 0
    worker = partial(render_one, out_dir=split_dir)

    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])

        if workers <= 1:
            iterator = (worker(s) for s in seeds)
        else:
            pool = multiprocessing.Pool(workers)
            iterator = pool.imap_unordered(worker, seeds, chunksize=8)

        try:
            for result in tqdm(iterator, total=len(seeds), desc=f"{split_name}"):
                if result is None:
                    continue
                filename, label = result
                writer.writerow([filename, label])
                n_ok += 1
        finally:
            if workers > 1:
                pool.close()
                pool.join()

    log.info("%s: %d/%d samples rendered → %s", split_name, n_ok, len(seeds), labels_csv)
    return n_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", default="data/chord_synth", help="Root output directory.")
    parser.add_argument("--num-train", type=int, default=30000, help="Number of training samples to attempt.")
    parser.add_argument("--num-val", type=int, default=2000, help="Number of validation samples to attempt.")
    parser.add_argument(
        "--workers", type=int, default=max(1, (multiprocessing.cpu_count() or 4) - 1), help="Multiprocessing pool size."
    )
    parser.add_argument(
        "--seed-base", type=int, default=0, help="Base seed for training data; val uses seed-base + 10_000_000."
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    log.info("Generating chord training set (%d samples, %d workers)", args.num_train, args.workers)
    train_seeds = range(args.seed_base, args.seed_base + args.num_train)
    generate_split("train", train_seeds, out_root, args.workers)

    log.info("Generating chord validation set (%d samples)", args.num_val)
    val_seeds = range(
        args.seed_base + 10_000_000,
        args.seed_base + 10_000_000 + args.num_val,
    )
    generate_split("val", val_seeds, out_root, args.workers)

    log.info("Done.  Output → %s", out_root)


if __name__ == "__main__":
    main()
