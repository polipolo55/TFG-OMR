"""
extract_real_music_strips.py
============================
Extract music-staff crops from real Real Book PDF pages and pre-label them
with the current note CRNN + grammar_fix so the user can review/correct in
the music labeling UI (static/music_labeler.html).

Output layout::

    {output_root}/
        strips/
            page0010_staff0.png
            page0010_staff1.png
            …
        labels.jsonl   # one line per strip:
                       #   {"filename": "…", "predicted": "measure clef:G2 …",
                       #    "label": null, "status": "pending"}

After labeling, use --export to convert the done records into the
--finetune-data-dir layout expected by the training CLI::

    poetry run python src/data_processing/extract_real_music_strips.py \\
        --export data/music_real --output-finetune data/finetune/realbook/clean

Usage::

    # Step 1 — extract and pre-label
    poetry run python src/data_processing/extract_real_music_strips.py \\
        --pdf data/real_book/full_realbook.pdf \\
        --output data/music_real \\
        --page-step 10

    # Step 2 — label via the UI, then export for fine-tuning
    poetry run python src/data_processing/extract_real_music_strips.py \\
        --export data/music_real \\
        --output-finetune data/finetune/realbook/clean
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from omr_pipeline.grammar_fix import fix_sequence
from omr_pipeline.inference import recognize_music
from omr_pipeline.preprocess import load_pdf_page, preprocess_page
from omr_pipeline.staff_detect import detect_systems

log = logging.getLogger(__name__)


def _likely_has_header(music_strip_gray: np.ndarray) -> bool:
    """Heuristic: does this strip start with a visible clef (= header staff)?

    Without a clef, the leftmost columns of the strip contain only the 5
    staff lines crossing — a low, predictable ink density. A clef glyph
    (and key/time signature) adds substantial extra ink concentrated in
    the leftmost ~30–40 px.

    Returns True if the leftmost band has clearly more ink than what
    staff lines alone would contribute.
    """
    if music_strip_gray is None or music_strip_gray.size == 0:
        return True  # default: assume header present (don't strip)

    h, w = music_strip_gray.shape
    if w < 100 or h < 30:
        return True

    _, binary = cv2.threshold(music_strip_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = (binary > 0).astype(np.uint8)

    left_w = max(30, int(w * 0.05))
    if left_w >= w:
        return True

    left_density = float(binary[:, :left_w].mean())
    # 5 staff lines, each ~1 px thick → baseline density ≈ 5 / h
    baseline = 5.0 / h
    return left_density > baseline * 2.5


def _strip_header_tokens(lmx_str: str) -> str:
    """Remove all clef:, key:fifths:, and time/beats/beat-type tokens.

    For Real Book continuation staves these tokens are hallucinated by a
    model trained to always emit a header, so the correct label for fine-
    tuning is to drop them entirely.
    """
    tokens = lmx_str.split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("clef:") or tok.startswith("key:fifths:"):
            i += 1
            continue
        if tok == "time":
            i += 1
            if i < len(tokens) and tokens[i].startswith("beats:"):
                i += 1
            if i < len(tokens) and tokens[i].startswith("beat-type:"):
                i += 1
            continue
        out.append(tok)
        i += 1
    return " ".join(out)


def extract_one_page(
    pdf_bytes: bytes,
    page_idx: int,
    out_strips: Path,
    checkpoint_path: Path | None,
    dpi: int = 200,
) -> list[dict]:
    """Extract music strips from one PDF page; return label records."""
    try:
        raw = load_pdf_page(pdf_bytes, page=page_idx, dpi=dpi)
    except Exception as exc:
        log.warning("page %d: load failed: %s", page_idx, exc)
        return []

    page = preprocess_page(raw)
    systems = detect_systems(page.grayscale, page.binary)
    if not systems:
        log.debug("page %d: no systems detected", page_idx)
        return []

    music_imgs: list[np.ndarray] = []
    keep_indices: list[int] = []
    for i, s in enumerate(systems):
        if s.music_image is None or s.music_image.size == 0:
            continue
        if s.music_image.shape[0] < 20 or s.music_image.shape[1] < 100:
            continue
        music_imgs.append(s.music_image)
        keep_indices.append(i)

    if not music_imgs:
        return []

    raw_preds = recognize_music(music_imgs, checkpoint_path)

    # Apply grammar_fix per-staff with NO cross-staff propagation.
    # This is intentionally different from the API path: for fine-tune labels
    # we want each staff to reflect only what is visually present in its image.
    # The API's _propagate_time would inject the first staff's time signature
    # into every continuation staff even though the image doesn't show it,
    # which would teach the model the wrong behavior.
    fixed_preds: list[str] = []
    for raw_pred in raw_preds:
        fixed, _, _ = fix_sequence(
            raw_pred,
            global_key=None,
            global_time=None,
            force_clef=True,
        )
        fixed_preds.append(fixed)

    # For each strip, decide if it has a visible header. If not, strip the
    # hallucinated clef/key/time tokens — the model always emits them due to
    # its training contract, but for a continuation staff they are wrong.
    records: list[dict] = []
    for img, idx, pred in zip(music_imgs, keep_indices, fixed_preds):
        filename = f"page{page_idx:04d}_staff{idx}.png"
        cv2.imwrite(str(out_strips / filename), img)

        if not _likely_has_header(img):
            pred = _strip_header_tokens(pred)

        records.append(
            {
                "filename": filename,
                "predicted": pred,
                "label": None,
                "status": "pending",
            }
        )
    return records


def export_finetune(labels_jsonl: Path, out_dir: Path) -> int:
    """Convert labeled records into --finetune-data-dir layout.

    Each done record becomes::

        out_dir/{stem}/{stem}.png
        out_dir/{stem}/{stem}.lmx

    Returns the number of exported samples.
    """
    if not labels_jsonl.exists():
        log.error("labels.jsonl not found: %s", labels_jsonl)
        return 0

    strips_dir = labels_jsonl.parent / "strips"
    out_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    with open(labels_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status") != "done" or not rec.get("label"):
                continue

            stem = Path(rec["filename"]).stem
            sample_dir = out_dir / stem
            sample_dir.mkdir(exist_ok=True)

            src_png = strips_dir / rec["filename"]
            dst_png = sample_dir / f"{stem}.png"
            if src_png.exists():
                shutil.copy2(src_png, dst_png)
            else:
                log.warning("Strip not found: %s", src_png)
                continue

            (sample_dir / f"{stem}.lmx").write_text(rec["label"].strip(), encoding="utf-8")
            exported += 1

    log.info("Exported %d fine-tune samples → %s", exported, out_dir)
    return exported


def _re_predict(data_root: Path, checkpoint_arg: str | None, dpi: int) -> None:
    """Re-run model predictions on all pending strips; update labels.jsonl in place.

    Done/skip records are left untouched so no labeling work is lost.
    """
    labels_path = data_root / "labels.jsonl"
    strips_dir = data_root / "strips"
    if not labels_path.exists():
        log.error("labels.jsonl not found: %s", labels_path)
        return

    checkpoint_path = Path(checkpoint_arg) if checkpoint_arg else Path("models/latest/best_model.pt")
    if not checkpoint_path.exists():
        log.error("Checkpoint not found: %s", checkpoint_path)
        return

    with open(labels_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    pending = [r for r in records if r.get("status") == "pending"]
    log.info("Re-predicting %d pending strips (keeping %d done/skip)", len(pending), len(records) - len(pending))

    batch_size = 16
    updated = 0
    for i in range(0, len(pending), batch_size):
        batch_recs = pending[i : i + batch_size]
        imgs: list[np.ndarray] = []
        for rec in batch_recs:
            path = strips_dir / rec["filename"]
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.exists() else None
            imgs.append(img if img is not None else np.zeros((64, 128), dtype=np.uint8))

        raw_preds = recognize_music(imgs, checkpoint_path)
        for rec, raw_pred, img in zip(batch_recs, raw_preds, imgs):
            fixed, _, _ = fix_sequence(raw_pred, global_key=None, global_time=None, force_clef=True)
            if not _likely_has_header(img):
                fixed = _strip_header_tokens(fixed)
            rec["predicted"] = fixed
            updated += 1

        log.info("Re-predicted %d / %d", min(i + batch_size, len(pending)), len(pending))

    tmp = labels_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(labels_path)
    log.info("Updated predictions for %d pending strips → %s", updated, labels_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])

    # Extraction mode
    p.add_argument("--pdf", default="data/real_book/full_realbook.pdf", help="Path to Real Book PDF")
    p.add_argument("--output", default="data/music_real", help="Output root for strips/ and labels.jsonl")
    p.add_argument("--page-step", type=int, default=10, help="Extract every Nth page (default: 10)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None, help="Last page (exclusive); default: full document")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--checkpoint", default=None, help="CRNN checkpoint path (default: models/latest/best_model.pt)")

    # Export mode
    p.add_argument(
        "--export", default=None, help="If set: export labeled records from this data root instead of extracting"
    )
    p.add_argument(
        "--output-finetune", default="data/finetune/realbook/clean", help="Destination for fine-tune data export"
    )

    # Re-predict mode
    p.add_argument(
        "--re-predict",
        default=None,
        help="If set: re-run predictions on all pending strips in this data root "
        "(updates labels.jsonl without touching done/skip records)",
    )

    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.export:
        labels_path = Path(args.export) / "labels.jsonl"
        n = export_finetune(labels_path, Path(args.output_finetune))
        print(f"Exported {n} samples to {args.output_finetune}")
        return

    if args.re_predict:
        _re_predict(Path(args.re_predict), args.checkpoint, args.dpi)
        return

    # --- Extraction mode ---
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        log.error("PDF not found: %s", pdf_path)
        sys.exit(1)

    out_root = Path(args.output)
    out_strips = out_root / "strips"
    out_strips.mkdir(parents=True, exist_ok=True)
    labels_path = out_root / "labels.jsonl"

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path("models/latest/best_model.pt")
    if not checkpoint_path.exists():
        log.error("Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)

    # Resume: skip pages already in labels.jsonl
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

    try:
        import fitz

        doc = fitz.open(pdf_path)
        n_pages = doc.page_count
        doc.close()
    except ImportError:
        log.error("PyMuPDF (fitz) required: pip install pymupdf")
        sys.exit(1)

    end = args.end if args.end is not None else n_pages
    page_ids = list(range(args.start, end, args.page_step))
    pdf_bytes = pdf_path.read_bytes()

    try:
        from tqdm import tqdm

        pages_iter = tqdm(page_ids, desc="pages")
    except ImportError:
        pages_iter = page_ids

    new_records: list[dict] = []
    for page_idx in pages_iter:
        if page_idx in seen_pages:
            continue
        records = extract_one_page(pdf_bytes, page_idx, out_strips, checkpoint_path, dpi=args.dpi)
        new_records.extend(records)

    with open(labels_path, "a", encoding="utf-8") as f:
        for rec in new_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info(
        "Extracted %d strips from %d new pages → %s",
        len(new_records),
        len(page_ids) - len(seen_pages),
        labels_path,
    )


if __name__ == "__main__":
    main()
