"""
OMR pipeline — load → preprocess → detect staves → recognise → output.

Single flow:
  1. Load image / PDF and convert to grayscale.
  2. Deskew and binarise.
  3. Detect staff systems (morphological staff-line finder).
  4. For each system: recognise music (CRNN) and chords (OCR).
  5. Apply music-theory grammar corrections across all systems.
  6. Assemble the JSON result.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path

import cv2
import numpy as np

from .preprocess import PageImage, load_image, load_pdf_page, preprocess_page
from .staff_detect import System, detect_systems
from .inference import recognize_music
from .ocr_chords import recognize_chords
from .grammar_fix import fix_sequence

log = logging.getLogger(__name__)


def _is_pdf(data: bytes) -> bool:
    return data[:4] == b"%PDF"


def _page_data_url(grayscale: np.ndarray) -> str | None:
    try:
        ok, buf = cv2.imencode(".png", grayscale)
        if ok:
            return f"data:image/png;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# System-aware recognition
# ---------------------------------------------------------------------------

def _process_systems(
    systems: list[System],
    checkpoint_path: Path | None,
) -> list[dict]:
    """Recognise music and chords for every detected staff system.

    Returns a flat list of segment dicts (type, bbox, content) ordered
    top-to-bottom on the page.
    """
    music_imgs: list[np.ndarray] = []
    staff_positions: list[list[int] | None] = []
    bbox_y0s: list[int] = []
    chord_imgs: list[np.ndarray] = []
    chord_bins: list[np.ndarray | None] = []

    for sys in systems:
        if sys.music_image is not None and sys.music_image.size > 0:
            music_imgs.append(sys.music_image)
            staff_positions.append(sys.staff.line_ys)
            bbox_y0s.append(sys.music_bbox[1])
        else:
            music_imgs.append(np.zeros((10, 10), dtype=np.uint8))
            staff_positions.append(None)
            bbox_y0s.append(0)

        if sys.chord_image is not None and sys.chord_image.size > 0:
            chord_imgs.append(sys.chord_image)
            chord_bins.append(sys.chord_binary)
        else:
            chord_imgs.append(np.zeros((10, 10), dtype=np.uint8))
            chord_bins.append(None)

    music_preds = recognize_music(
        music_imgs, checkpoint_path,
        staff_line_positions=staff_positions,
        music_bbox_y0s=bbox_y0s,
    )
    chord_preds = recognize_chords(chord_imgs, chord_bins)

    # Grammar correction with cross-system key propagation
    global_key: str | None = None
    fixed_music: list[str] = []
    for pred in music_preds:
        fixed, global_key = fix_sequence(pred, global_key=global_key, force_clef=True)
        fixed_music.append(fixed)

    # Assemble segments
    segments: list[dict] = []
    for i, sys in enumerate(systems):
        if sys.chord_bbox is not None:
            cx, cy, cw, ch = sys.chord_bbox
            segments.append({
                "type": "text",
                "bbox": [cx, cy, cw, ch],
                "content": chord_preds[i] if i < len(chord_preds) else "",
            })

        mx, my, mw, mh = sys.music_bbox
        segments.append({
            "type": "music",
            "bbox": [mx, my, mw, mh],
            "content": fixed_music[i] if i < len(fixed_music) else "",
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
    """Process an uploaded file and return a JSON-serialisable result dict."""

    # 1. Load
    try:
        if _is_pdf(file_data):
            img = load_pdf_page(file_data, page=0, dpi=200)
        else:
            img = load_image(file_data)
    except Exception as exc:
        return {"error": f"Load failed: {exc}", "pages": []}

    # 2. Preprocess (deskew + binarise)
    try:
        page = preprocess_page(img)
    except Exception as exc:
        return {"error": f"Preprocess failed: {exc}", "pages": []}

    page_url = _page_data_url(page.grayscale)

    # 3. Detect staff systems
    systems = detect_systems(page.grayscale, page.binary)
    if not systems:
        log.warning("No staff systems detected — returning empty result")
        return {
            "error": "No staff systems detected in the image.",
            "pages": [{
                "index": 0,
                "segments": [],
                "page_image_data_url": page_url,
            }],
            "meta": {
                "filename": filename,
                "page_height": page.meta["height"],
                "page_width": page.meta["width"],
                "deskew_angle_deg": page.meta.get("deskew_angle_deg", 0.0),
            },
        }

    log.info("%d system(s) detected", len(systems))

    # 4. Recognise + correct
    segments = _process_systems(systems, checkpoint_path)

    return {
        "error": None,
        "pages": [{
            "index": 0,
            "segments": segments,
            "page_image_data_url": page_url,
        }],
        "meta": {
            "filename": filename,
            "page_height": page.meta["height"],
            "page_width": page.meta["width"],
            "deskew_angle_deg": page.meta.get("deskew_angle_deg", 0.0),
            "num_systems": len(systems),
        },
    }
