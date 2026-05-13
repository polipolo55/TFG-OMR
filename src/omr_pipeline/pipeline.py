"""
OMR pipeline — load → preprocess → detect staves → recognise music + chords.

Flow per upload:
  1. Decode bytes (image or PDF) to a grayscale page.
  2. Deskew + binarise.
  3. Detect staff systems (morphological staff-line finder).
  4. For each system, recognise music with the CRNN and chords with OCR.
  5. Apply LMX grammar correction across all systems.
  6. Assemble one segment per staff (staff_bbox, chord_bbox, lmx_tokens, chords).

Set ``OMR_DEBUG_DIR`` to save intermediate crops for inspection.
PDF rasterisation DPI follows ``OMR_PDF_DPI`` (default 300).
"""
from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from .preprocess import load_image, load_pdf_page, pdf_load_dpi, preprocess_page
from .staff_detect import System, detect_systems
from .inference import recognize_music
from .chord_recognizer import recognize_chords_crnn
from .grammar_fix import fix_sequence

log = logging.getLogger(__name__)

_DEBUG_DIR = os.environ.get("OMR_DEBUG_DIR", "")


def _is_pdf(data: bytes) -> bool:
    return data[:4] == b"%PDF"


def _page_data_url(grayscale: np.ndarray) -> str | None:
    try:
        ok, buf = cv2.imencode(".png", grayscale)
        if ok:
            return f"data:image/png;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"
    except Exception:  # noqa: BLE001
        log.exception("PNG encode failed")
    return None


def _save_debug(music_imgs: list[np.ndarray], chord_imgs: list[np.ndarray]) -> None:
    if not _DEBUG_DIR:
        return
    dbg = Path(_DEBUG_DIR)
    dbg.mkdir(parents=True, exist_ok=True)
    for idx, img in enumerate(music_imgs):
        if img is not None and img.size > 0:
            cv2.imwrite(str(dbg / f"music_{idx:02d}.png"), img)
    for idx, img in enumerate(chord_imgs):
        if img is not None and img.size > 0:
            cv2.imwrite(str(dbg / f"chord_{idx:02d}.png"), img)
    log.info("Debug crops saved to %s", dbg)


# ---------------------------------------------------------------------------
# Per-page recognition
# ---------------------------------------------------------------------------

def _process_systems(
    systems: list[System],
    checkpoint_path: Path | None,
) -> list[dict]:
    """One segment per staff.

    Each segment: ``{staff_bbox, chord_bbox, lmx_tokens, chords}``.
    Segments are ordered top-to-bottom.
    """
    music_imgs: list[np.ndarray] = []
    chord_imgs: list[np.ndarray] = []

    for sys in systems:
        if sys.music_image is not None and sys.music_image.size > 0:
            music_imgs.append(sys.music_image)
        else:
            music_imgs.append(np.zeros((10, 10), dtype=np.uint8))
        if sys.chord_image is not None and sys.chord_image.size > 0:
            chord_imgs.append(sys.chord_image)
        else:
            chord_imgs.append(np.zeros((10, 10), dtype=np.uint8))

    _save_debug(music_imgs, chord_imgs)

    music_preds = recognize_music(music_imgs, checkpoint_path)
    chord_preds = recognize_chords_crnn(chord_imgs)

    # LMX grammar correction with cross-system key + time propagation
    global_key: str | None = None
    global_time: tuple[str, str, str] | None = None
    fixed_music: list[str] = []
    for pred in music_preds:
        fixed, global_key, global_time = fix_sequence(
            pred, global_key=global_key, global_time=global_time,
            force_clef=True,
        )
        fixed_music.append(fixed)

    segments: list[dict] = []
    for i, sys in enumerate(systems):
        lmx_str = fixed_music[i] if i < len(fixed_music) else ""
        chord_str = chord_preds[i] if i < len(chord_preds) else ""

        mx, my, mw, mh = sys.music_bbox
        chord_bbox = list(sys.chord_bbox) if sys.chord_bbox is not None else None

        segments.append({
            "staff_bbox": [mx, my, mw, mh],
            "chord_bbox": chord_bbox,
            "lmx_tokens": lmx_str.split() if lmx_str else [],
            "chords": chord_str.split() if chord_str else [],
        })

    return segments


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    file_data: bytes,
    filename: str,
    checkpoint_path: Path | None = None,
) -> dict:
    """Run OMR on a single uploaded file.  Returns a JSON-serialisable dict."""

    # 1. Load
    pdf_render_dpi: int | None = None
    try:
        if _is_pdf(file_data):
            pdf_render_dpi = pdf_load_dpi()
            img = load_pdf_page(file_data, page=0, dpi=pdf_render_dpi)
        else:
            img = load_image(file_data)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Load failed: {exc}", "pages": []}

    # 2. Preprocess
    try:
        page = preprocess_page(img)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Preprocess failed: {exc}", "pages": []}

    page_url = _page_data_url(page.grayscale)

    base_meta = {
        "filename": filename,
        "page_height": page.meta["height"],
        "page_width": page.meta["width"],
        "deskew_angle_deg": page.meta.get("deskew_angle_deg", 0.0),
        "pdf_render_dpi": pdf_render_dpi,
    }

    # 3. Staff detection
    systems = detect_systems(page.grayscale, page.binary)
    if not systems:
        log.warning("No staff systems detected")
        return {
            "error": "No staff systems detected in the image.",
            "pages": [{
                "index": 0,
                "page_image_data_url": page_url,
                "segments": [],
            }],
            "meta": {**base_meta, "num_systems": 0},
        }

    log.info("Detected %d staff system(s)", len(systems))

    # 4. Recognise + grammar fix
    segments = _process_systems(systems, checkpoint_path)

    return {
        "error": None,
        "pages": [{
            "index": 0,
            "page_image_data_url": page_url,
            "segments": segments,
        }],
        "meta": {**base_meta, "num_systems": len(systems)},
    }
