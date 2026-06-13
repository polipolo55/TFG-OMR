"""build_reject_label_set.py
============================
Build (and later export) the staff-reject calibration set via hand labeling.

The CTC mean-logprob gate in ``staff_reject.py`` is checkpoint-dependent
(CLAUDE.md hard constraint #7) and must be recalibrated after every CRNN
re-train.  ``cli.py calibrate-reject`` needs a labelled fixture set laid out as

    data/staff_reject/music/      ← real single staves the gate must ACCEPT
    data/staff_reject/non_music/  ← detector false-positives it must REJECT

This script produces that set in two phases:

``harvest`` (default)
    Run the same staff detector the pipeline uses over one or more PDFs and
    dump every detected candidate strip into ``data/staff_reject_label/strips/``,
    seeding ``labels.jsonl`` with one ``pending`` record per strip.  Then a human
    sorts them music / non_music in the browser tool at ``/reject-labeler``
    (served by ``cli.py api`` — see ``static/reject_labeler.html``).

``export``
    Read ``labels.jsonl`` and copy each ``done`` strip into
    ``data/staff_reject/{music,non_music}/`` ready for ``calibrate-reject``.

Both phases are idempotent and resumable.

Usage::

    poetry run python scripts/build_reject_label_set.py harvest \\
        --pdfs 'data/real_book/*.pdf' --pages 40
    # label in the browser at http://localhost:8000/reject-labeler
    poetry run python scripts/build_reject_label_set.py export
    poetry run python src/cli.py calibrate-reject   # then recalibrate
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import shutil
import sys
from pathlib import Path

import cv2

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from omr_pipeline.preprocess import load_pdf_page, pdf_load_dpi, preprocess_page  # noqa: E402
from omr_pipeline.staff_detect import detect_systems  # noqa: E402

log = logging.getLogger("build_reject_label_set")

LABEL_ROOT = _REPO / "data" / "staff_reject_label"
STRIPS_DIR = LABEL_ROOT / "strips"
LABELS_PATH = LABEL_ROOT / "labels.jsonl"
CALIB_ROOT = _REPO / "data" / "staff_reject"
VALID_LABELS = ("music", "non_music")


def _load_records() -> list[dict]:
    if not LABELS_PATH.exists():
        return []
    with open(LABELS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _save_records(records: list[dict]) -> None:
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = LABELS_PATH.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(LABELS_PATH)


def harvest(pdf_globs: list[str], pages: int) -> None:
    """Detect candidate strips from PDFs and seed pending label records."""
    STRIPS_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for pat in pdf_globs:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        log.error("no PDFs matched %s", pdf_globs)
        return

    records = _load_records()
    known = {r["filename"] for r in records}
    dpi = pdf_load_dpi()
    added = 0

    for p in paths:
        try:
            blob = Path(p).read_bytes()
        except OSError as exc:
            log.warning("skip %s: %s", p, exc)
            continue
        for page_idx in range(max(1, pages)):
            try:
                img = load_pdf_page(blob, page=page_idx, dpi=dpi)
            except Exception:
                break  # ran past the last page
            page = preprocess_page(img)
            systems = detect_systems(page.grayscale, page.binary)
            for s_idx, s in enumerate(systems):
                if s.music_image is None or s.music_image.size == 0:
                    continue
                name = f"{Path(p).stem}_p{page_idx:03d}_s{s_idx:03d}.png"
                if name in known:
                    continue
                out = STRIPS_DIR / name
                if not out.exists():
                    cv2.imwrite(str(out), s.music_image)
                records.append({"filename": name, "label": None, "status": "pending"})
                known.add(name)
                added += 1

    _save_records(records)
    pend = sum(1 for r in records if r["status"] == "pending")
    log.info(
        "Harvest done: +%d new strips (%d pending / %d total). Label at /reject-labeler.",
        added, pend, len(records),
    )


def export() -> None:
    """Copy labelled strips into data/staff_reject/{music,non_music}/."""
    records = _load_records()
    if not records:
        log.error("no labels.jsonl at %s — run harvest first", LABELS_PATH)
        return
    counts = {lbl: 0 for lbl in VALID_LABELS}
    skipped = 0
    for r in records:
        if r.get("status") != "done":
            skipped += 1
            continue
        lbl = r.get("label")
        if lbl not in VALID_LABELS:
            log.warning("strip %s has done status but invalid label %r — skipping", r["filename"], lbl)
            skipped += 1
            continue
        dst_dir = CALIB_ROOT / lbl
        dst_dir.mkdir(parents=True, exist_ok=True)
        src = STRIPS_DIR / r["filename"]
        if not src.exists():
            log.warning("strip file missing: %s", src)
            continue
        shutil.copy2(src, dst_dir / r["filename"])
        counts[lbl] += 1
    log.info(
        "Export done → %s : music=%d  non_music=%d  (skipped %d not-done/invalid)",
        CALIB_ROOT, counts["music"], counts["non_music"], skipped,
    )
    if counts["music"] == 0 or counts["non_music"] == 0:
        log.warning(
            "calibrate-reject needs BOTH classes non-empty; label more strips of the empty class.",
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest", help="Detect candidate strips from PDFs and seed labels.jsonl")
    h.add_argument("--pdfs", nargs="+", default=["data/real_book/*.pdf"], help="Glob(s) of PDFs")
    h.add_argument("--pages", type=int, default=40, help="Max pages per PDF (default 40)")

    sub.add_parser("export", help="Copy labelled strips into data/staff_reject/{music,non_music}/")

    args = ap.parse_args()
    if args.cmd == "harvest":
        harvest(args.pdfs, args.pages)
    elif args.cmd == "export":
        export()


if __name__ == "__main__":
    main()
