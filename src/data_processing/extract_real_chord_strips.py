"""
extract_real_chord_strips.py
============================
Extract chord-strip crops from real Real Book PDF pages and pre-label them
with the current chord CRNN so the user can review/correct in the labeling
UI instead of typing every label from scratch.

Output layout::

    {output_root}/
        strips/
            page0010_staff0.png
            page0010_staff1.png
            …
        labels.jsonl        # one line per strip:
                            #   {"filename": "...", "predicted": "...",
                            #    "label": null, "status": "pending"}

The labeling UI (``static/chord_labeler.html`` + endpoints in
``src/api/main.py``) reads/writes this file in place.

Usage::

    poetry run python src/data_processing/extract_real_chord_strips.py \
        --pdf data/real_book/full_realbook.pdf \
        --output data/chord_real \
        --page-step 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from omr_pipeline.chord_recognizer import recognize_chords_crnn
from omr_pipeline.preprocess import load_pdf_page, preprocess_page
from omr_pipeline.staff_detect import detect_systems

log = logging.getLogger(__name__)


def extract_one_page(
    pdf_bytes: bytes,
    page_idx: int,
    out_strips: Path,
    dpi: int = 200,
) -> list[dict]:
    """Extract chord strips from a single PDF page; return label records."""
    try:
        raw = load_pdf_page(pdf_bytes, page=page_idx, dpi=dpi)
    except Exception as exc:
        log.warning("page %d: load failed: %s", page_idx, exc)
        return []
    page = preprocess_page(raw)
    systems = detect_systems(page.grayscale, page.binary)
    if not systems:
        return []

    chord_imgs: list[np.ndarray] = []
    keep_indices: list[int] = []
    for i, s in enumerate(systems):
        if s.chord_image is None or s.chord_image.size == 0:
            continue
        if s.chord_image.shape[0] < 12 or s.chord_image.shape[1] < 40:
            continue
        chord_imgs.append(s.chord_image)
        keep_indices.append(i)
    if not chord_imgs:
        return []

    predictions = recognize_chords_crnn(chord_imgs)

    records: list[dict] = []
    for img, idx, pred in zip(chord_imgs, keep_indices, predictions):
        filename = f"page{page_idx:04d}_staff{idx}.png"
        Image.fromarray(img).save(out_strips / filename, optimize=True)
        records.append(
            {
                "filename": filename,
                "predicted": pred,
                "label": None,  # filled in by labeling UI
                "status": "pending",  # pending | done | skip
            }
        )
    return records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pdf", default="data/real_book/full_realbook.pdf")
    p.add_argument("--output", default="data/chord_real")
    p.add_argument("--page-step", type=int, default=10, help="Extract every Nth page (default: 10)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None, help="Last page (exclusive); default: full document")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pdf_path = Path(args.pdf)
    out_root = Path(args.output)
    out_strips = out_root / "strips"
    out_strips.mkdir(parents=True, exist_ok=True)
    labels_path = out_root / "labels.jsonl"

    # Resume support: skip pages that already have entries
    seen_pages: set[int] = set()
    if labels_path.exists():
        with open(labels_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    pg = int(rec["filename"][4:8])
                    seen_pages.add(pg)
                except Exception:
                    pass
        log.info("Resuming — already have records for %d pages", len(seen_pages))

    import fitz

    doc = fitz.open(pdf_path)
    n_pages = doc.page_count
    doc.close()
    end = args.end if args.end is not None else n_pages
    page_ids = list(range(args.start, end, args.page_step))

    pdf_bytes = pdf_path.read_bytes()

    new_records: list[dict] = []
    for page_idx in tqdm(page_ids, desc="pages"):
        if page_idx in seen_pages:
            continue
        records = extract_one_page(pdf_bytes, page_idx, out_strips, dpi=args.dpi)
        new_records.extend(records)

    # Append new records to labels.jsonl
    with open(labels_path, "a", encoding="utf-8") as f:
        for rec in new_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info(
        "Extracted %d strips from %d pages (every %dth, %d–%d) → %s",
        len(new_records),
        len(page_ids),
        args.page_step,
        args.start,
        end,
        labels_path,
    )


if __name__ == "__main__":
    main()
