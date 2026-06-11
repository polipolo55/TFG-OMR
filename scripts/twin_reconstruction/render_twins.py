"""Re-render the deleted __nh twin samples into a scratch dir (forensic reconstruction).

Replicates generate_headerless_twins.process_one (git 7a5dc1c) byte-for-byte in
behavior, except twins are written under --output instead of into the live
data/processed/primus/clean tree. Resumable: existing twin PNGs are skipped.
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing
import sys
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO / "src"))

from legacy_generate_headerless_twins import _dpi_for, _in_fraction  # noqa: E402
from legacy_generate_realbook import headerless_label_tokens, omit_header_in_ly  # noqa: E402
from CRNN_CTC.lilypond_render import crop_content, run_lilypond  # noqa: E402

SEED = 42
FRACTION = 0.35
DPI_CHOICES = (180, 200, 220)


def process_one(sample_dir: Path, out_root: Path) -> str:
    sid = sample_dir.name
    if sid.endswith("__nh") or not _in_fraction(sid, SEED, FRACTION):
        return "skip-fraction"
    ly_path = sample_dir / f"{sid}.ly"
    lmx_path = sample_dir / f"{sid}.lmx"
    if not ly_path.exists() or not lmx_path.exists():
        return "fail"
    tokens = lmx_path.read_text(encoding="utf-8").split()
    if "clef:G2" not in tokens:
        return "skip-nontreble"

    twin_id = f"{sid}__nh"
    twin_dir = out_root / twin_id
    twin_png = twin_dir / f"{twin_id}.png"
    if twin_png.exists():
        return "ok"  # resumable re-run

    ly = ly_path.read_text(encoding="utf-8")
    twin_ly = omit_header_in_ly(ly)
    if twin_ly == ly:
        return "fail"
    dpi = _dpi_for(sid, DPI_CHOICES)
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
    (twin_dir / f"{twin_id}.lmx").write_text(
        " ".join(headerless_label_tokens(tokens)), encoding="utf-8"
    )
    return "ok"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=_REPO / "data/processed/primus/clean")
    p.add_argument("--output", type=Path, default=_REPO / "data/scratch/twin_recon/clean")
    p.add_argument("--workers", type=int, default=max(1, (multiprocessing.cpu_count() or 2) - 2))
    p.add_argument("--limit", type=int, default=0, help="Process at most N dirs (smoke test).")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    dirs = sorted(d for d in args.data_dir.iterdir() if d.is_dir())
    if args.limit:
        dirs = dirs[: args.limit]
    worker = partial(process_one, out_root=args.output)
    counts: dict[str, int] = {}
    with multiprocessing.Pool(args.workers) as pool:
        for status in tqdm(pool.imap_unordered(worker, dirs, chunksize=16), total=len(dirs)):
            counts[status] = counts.get(status, 0) + 1
    print(counts)


if __name__ == "__main__":
    main()
