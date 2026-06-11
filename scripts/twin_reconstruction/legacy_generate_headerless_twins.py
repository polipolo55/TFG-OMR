"""
generate_headerless_twins.py
============================
Generate first-class header-less ("continuation staff") twin samples for an
already rendered LilyJAZZ dataset.

Real Real Book pages contain continuation systems that do not reprint the
leading clef and time signature. To teach the recogniser to read them, for a
fraction of in-scope (treble) samples we render a TWIN: the same music with the
clef and time-signature glyphs hidden (LilyPond ``\\omit``), paired with a label
that drops the corresponding tokens. Because the twin is a real render matched
to a real label, image and label are aligned by construction — unlike the old
training-time crop, which could not locate the header boundary reliably.

The key signature is kept (shown): hiding it would make LilyPond print explicit
accidentals on key-altered notes, changing the in-body accidental tokens. With
the key kept, the twin label is exactly the parent label minus the clef and
time tokens, and the body is untouched.

Each twin is written as a sibling sample directory ``<id>__nh/`` containing
``<id>__nh.png`` and ``<id>__nh.lmx`` inside the same data directory, so it is
discovered like any other sample and picked up by the scanned-augmentation pass
(re-run ``augment_scanned`` after this so the twins get scanned variants too).

Only ``clef:G2`` (treble) samples get twins: a non-treble twin would have its
clef token stripped and then wrongly pass the leadsheet-clef filter while its
noteheads sit at non-treble positions, corrupting training.

Idempotent / resumable: a sample whose twin dir already exists is skipped
unless ``--force``. Membership in the twin fraction is a deterministic function
of (seed, sample id), so reruns are stable.

Usage::

    poetry run python src/data_processing/generate_headerless_twins.py \
        --data-dir data/processed/primus/clean --fraction 0.35 --workers 12
    # then regenerate scanned variants so twins are augmented too:
    poetry run python src/data_processing/augment_scanned.py ...
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import multiprocessing
import sys
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from CRNN_CTC.lilypond_render import crop_content, run_lilypond
from legacy_generate_realbook import (
    headerless_label_tokens,
    omit_header_in_ly,
)

log = logging.getLogger(__name__)

_TWIN_SUFFIX = "__nh"


def _in_fraction(sample_id: str, seed: int, fraction: float) -> bool:
    """Deterministic per-sample inclusion test (parallel-safe)."""
    h = hashlib.md5(f"{seed}:{sample_id}".encode()).hexdigest()
    return (int(h[:8], 16) % 10_000) < fraction * 10_000


def _dpi_for(sample_id: str, choices: tuple[int, ...]) -> int:
    h = hashlib.md5(f"dpi:{sample_id}".encode()).hexdigest()
    return choices[int(h[:8], 16) % len(choices)]


def process_one(
    sample_dir: Path,
    seed: int,
    fraction: float,
    dpi_choices: tuple[int, ...],
    force: bool,
) -> str:
    """Return one of: 'ok', 'skip-fraction', 'skip-exists', 'skip-nontreble', 'fail'."""
    sid = sample_dir.name
    if sid.endswith(_TWIN_SUFFIX):
        return "skip-fraction"  # never make a twin of a twin
    if not _in_fraction(sid, seed, fraction):
        return "skip-fraction"

    ly_path = sample_dir / f"{sid}.ly"
    lmx_path = sample_dir / f"{sid}.lmx"
    if not ly_path.exists() or not lmx_path.exists():
        return "fail"

    tokens = lmx_path.read_text(encoding="utf-8").split()
    if "clef:G2" not in tokens:
        return "skip-nontreble"

    twin_id = f"{sid}{_TWIN_SUFFIX}"
    twin_dir = sample_dir.parent / twin_id
    twin_png = twin_dir / f"{twin_id}.png"
    if twin_png.exists() and not force:
        return "skip-exists"

    ly = ly_path.read_text(encoding="utf-8")
    twin_ly = omit_header_in_ly(ly)
    if twin_ly == ly:
        return "fail"  # no \new Staff{ found — unexpected

    dpi = _dpi_for(sid, dpi_choices)
    with tempfile.TemporaryDirectory(prefix="twin_") as tmp:
        png = run_lilypond(twin_ly, twin_id, Path(tmp), dpi=dpi, timeout=30)
        if png is None:
            return "fail"
        try:
            cropped = crop_content(np.array(Image.open(png).convert("L")))
        except Exception:
            return "fail"
        if cropped.size == 0 or np.all(cropped == 255):
            return "fail"
        twin_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(cropped).save(twin_png)

    twin_tokens = headerless_label_tokens(tokens)
    (twin_dir / f"{twin_id}.lmx").write_text(" ".join(twin_tokens), encoding="utf-8")
    return "ok"


def run_headerless_twins(
    data_dir: Path,
    *,
    fraction: float = 0.35,
    dpi_choices: tuple[int, ...] = (180, 200, 220),
    workers: int | None = None,
    seed: int = 42,
    force: bool = False,
    limit: int | None = None,
) -> dict[str, int]:
    """Generate ``__nh`` twin samples under *data_dir*. Returns per-status counts."""
    if workers is None:
        workers = max(1, (multiprocessing.cpu_count() or 2) - 2)

    dirs = sorted(d for d in data_dir.iterdir() if d.is_dir() and not d.name.endswith(_TWIN_SUFFIX))
    if limit:
        dirs = dirs[:limit]
    log.info(
        "Header-less twins: scanning %d dirs (fraction=%.2f, dpi=%s, workers=%d, force=%s)",
        len(dirs),
        fraction,
        dpi_choices,
        workers,
        force,
    )

    worker = partial(
        process_one,
        seed=seed,
        fraction=fraction,
        dpi_choices=dpi_choices,
        force=force,
    )
    counts: dict[str, int] = {}
    if workers <= 1:
        it = (worker(d) for d in dirs)
        for status in tqdm(it, total=len(dirs), desc="twins"):
            counts[status] = counts.get(status, 0) + 1
    else:
        with multiprocessing.Pool(workers) as pool:
            for status in tqdm(
                pool.imap_unordered(worker, dirs, chunksize=16),
                total=len(dirs),
                desc="twins",
            ):
                counts[status] = counts.get(status, 0) + 1

    log.info(
        "Done. created=%d  skipped(not in fraction)=%d  skipped(exists)=%d  "
        "skipped(non-treble)=%d  failed=%d",
        counts.get("ok", 0),
        counts.get("skip-fraction", 0),
        counts.get("skip-exists", 0),
        counts.get("skip-nontreble", 0),
        counts.get("fail", 0),
    )
    return counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Generate header-less continuation-staff twin samples.")
    p.add_argument("--data-dir", type=Path, default=Path("data/processed/primus/clean"))
    p.add_argument("--fraction", type=float, default=0.35, help="Share of treble samples to twin (default 0.35).")
    p.add_argument("--dpi", type=int, nargs="+", default=[180, 200, 220], help="DPI choices (one picked per sample).")
    p.add_argument("--workers", type=int, default=max(1, (multiprocessing.cpu_count() or 2) - 2))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Process at most N dirs (0 = all).")
    args = p.parse_args()

    run_headerless_twins(
        args.data_dir,
        fraction=args.fraction,
        dpi_choices=tuple(args.dpi),
        workers=args.workers,
        seed=args.seed,
        force=args.force,
        limit=args.limit or None,
    )


if __name__ == "__main__":
    main()
