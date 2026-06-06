"""
_gen_satin_figures.py
=====================
Generate the Satin Doll qualitative figures for Ch. 6:

  §6.8  this system on a genuine Real Book scan
        - satin_annotated.png        full page, staff bboxes (green=recognised,
                                     red=rejected) + recognised chords per staff
        - satin_staff{A,B}_input.png / _pred.png   input strip + LMX re-render
                                     for two representative recognised staves
  §6.6  side-by-side vs Audiveris
        - satin_ours_stacked.png     our recognised staves rendered + stacked
                                     (Audiveris half is rendered separately from
                                     its exported MusicXML)

Run:  poetry run python scripts/_gen_satin_figures.py
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO / "src"))

from CRNN_CTC.lilypond_render import render_tokens  # noqa: E402
from omr_pipeline.pipeline import run_pipeline  # noqa: E402

PDF = REPO / "latex_documents/main/figures/sheet_pdfs/Satin Doll.pdf"
OUT = REPO / "latex_documents/main/figures/real_page"
OUT.mkdir(parents=True, exist_ok=True)
SHOW = {"A": 1, "B": 4}  # representative recognised segments


def _decode_page(data_url: str) -> np.ndarray:
    b64 = data_url.split(",", 1)[1]
    buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def _stack(images: list[np.ndarray], gap: int = 30, pad: int = 20) -> np.ndarray:
    """Vertically stack variable-width grayscale renders, left-aligned on white."""
    w = max(im.shape[1] for im in images) + 2 * pad
    rows = []
    sep = np.full((gap, w), 255, np.uint8)
    for im in images:
        canvas = np.full((im.shape[0] + 2 * pad, w), 255, np.uint8)
        canvas[pad:pad + im.shape[0], pad:pad + im.shape[1]] = im
        rows.append(canvas)
        rows.append(sep)
    return np.vstack(rows[:-1])


def main():
    res = run_pipeline(PDF.read_bytes(), "Satin Doll.pdf")
    page = _decode_page(res["pages"][0]["page_image_data_url"])
    segs = res["pages"][0]["segments"]
    canvas = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
    GREEN, RED = (40, 160, 40), (40, 40, 200)

    for i, s in enumerate(segs):
        x, y, w, h = s["staff_bbox"]
        rejected = s.get("rejected")
        colour = RED if rejected else GREEN
        cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, 4)
        label = (f"seg{i}: REJECTED ({rejected})" if rejected
                 else f"seg{i}: " + (" ".join(s.get("chords", [])) or "(no chords)"))
        cv2.putText(canvas, label, (x + 6, max(30, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2, cv2.LINE_AA)
    ann = OUT / "satin_annotated.png"
    cv2.imwrite(str(ann), canvas)
    print(f"wrote {ann.name} ({canvas.shape[1]}x{canvas.shape[0]})")

    # Per-staff input strips + rendered predictions for two examples
    for tag, idx in SHOW.items():
        s = segs[idx]
        x, y, w, h = s["staff_bbox"]
        cv2.imwrite(str(OUT / f"satin_staff{tag}_input.png"), page[y:y + h, x:x + w])
        r = render_tokens(s["lmx_tokens"], name=f"satin_{tag}")
        if r is not None:
            cv2.imwrite(str(OUT / f"satin_staff{tag}_pred.png"), r)
            print(f"  staff {tag} (seg{idx}): chords={s.get('chords')} -> rendered")

    # Stacked "our transcription" of all recognised staves (for §6.6)
    renders = []
    for i, s in enumerate(segs):
        if s.get("rejected") or not s.get("lmx_tokens"):
            continue
        r = render_tokens(s["lmx_tokens"], name=f"satin_stack_{i}")
        if r is not None:
            renders.append(r)
    if renders:
        stacked = _stack(renders)
        cv2.imwrite(str(OUT / "satin_ours_stacked.png"), stacked)
        print(f"  stacked {len(renders)} staves -> satin_ours_stacked.png "
              f"({stacked.shape[1]}x{stacked.shape[0]})")

    rec = [i for i, s in enumerate(segs) if not s.get("rejected")]
    rej = [(i, s.get("rejected")) for i, s in enumerate(segs) if s.get("rejected")]
    nchords = sum(len(s.get("chords", [])) for s in segs)
    print(f"\nsystems={len(segs)} recognised={len(rec)} chords_total={nchords} rejected={rej}")


if __name__ == "__main__":
    main()
