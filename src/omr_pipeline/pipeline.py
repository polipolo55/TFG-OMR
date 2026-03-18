"""
Pipeline orchestrator — preprocess → slice → route → recognize → aggregate.
"""
from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

from .preprocess import PageImage, load_image, load_pdf_page, preprocess_page
from .slicer import Strip, extract_strips
from .router import route_strip
from .inference import recognize_music
from .ocr_chords import recognize_chords


def is_pdf(data: bytes) -> bool:
    return data[:4] == b"%PDF"


_OCR_PAD_Y = 25  # extra rows above/below text strip when feeding OCR


def _ocr_crop(s: Strip, grayscale: np.ndarray) -> np.ndarray:
    """Return a full-width, vertically padded crop for OCR.

    Using the tight Strip.image (17-30 px) gives EasyOCR almost nothing to
    work with.  Instead we take the full page width plus generous vertical
    padding so characters have room and the model has horizontal context.
    """
    H, W = grayscale.shape[:2]
    y0 = max(0, s.y_start - _OCR_PAD_Y)
    y1 = min(H, s.y_end + _OCR_PAD_Y)
    return grayscale[y0:y1, :]


def _merge_adjacent_text_strips(
    strips: list[Strip],
    types: list[str],
    grayscale: np.ndarray,
    binary: np.ndarray,
    max_gap: int = 8,
) -> tuple[list[Strip], list[str]]:
    """Merge consecutive text strips that are close together vertically."""
    if len(strips) <= 1:
        return strips, types

    merged_strips: list[Strip] = []
    merged_types: list[str] = []
    i = 0
    while i < len(strips):
        s = strips[i]
        t = types[i]
        if t != "text":
            merged_strips.append(s)
            merged_types.append(t)
            i += 1
            continue
        # Accumulate consecutive text strips
        y0 = s.y_start
        y1 = s.y_end
        x0 = s.x_start
        x1 = s.x_end
        j = i + 1
        while j < len(strips) and types[j] == "text":
            nxt = strips[j]
            if nxt.y_start - y1 <= max_gap:
                y1 = max(y1, nxt.y_end)
                x0 = min(x0, nxt.x_start)
                x1 = max(x1, nxt.x_end)
                j += 1
            else:
                break
        # Build merged strip
        band_gray = grayscale[y0:y1, x0:x1]
        ink = binary > 0
        band_bin = ink[y0:y1, x0:x1].astype(np.uint8)
        density = float(np.mean(band_bin > 0)) if band_bin.size else 0.0
        merged_strips.append(Strip(
            x_start=x0, x_end=x1,
            y_start=y0, y_end=y1,
            height=y1 - y0,
            image=band_gray,
            binary=band_bin,
            ink_density=density,
        ))
        merged_types.append("text")
        i = j

    return merged_strips, merged_types


def run_pipeline(
    file_data: bytes,
    filename: str,
    checkpoint_path: Path | None = None,
) -> dict:
    """Process uploaded file and return JSON-serializable result."""
    # 1) Load
    if is_pdf(file_data):
        try:
            img = load_pdf_page(file_data, page=0)
        except Exception as e:
            return {"error": f"PDF load failed: {e}", "pages": []}
    else:
        try:
            img = load_image(file_data)
        except Exception as e:
            return {"error": f"Image load failed: {e}", "pages": []}

    # 2) Preprocess
    try:
        page = preprocess_page(img)
    except Exception as e:
        return {"error": f"Preprocess failed: {e}", "pages": []}

    # Page image for visualization
    try:
        ok, buf = cv2.imencode(".png", page.grayscale)
        page_image_data_url = f"data:image/png;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}" if ok else None
    except Exception:
        page_image_data_url = None

    # 3) Slice
    strips = extract_strips(page.grayscale, page.binary)

    # 4) Route
    types = [route_strip(s) for s in strips]

    # 5) Merge adjacent text strips
    strips, types = _merge_adjacent_text_strips(strips, types, page.grayscale, page.binary)

    # 6) Recognize
    music_indices = [i for i, t in enumerate(types) if t == "music"]
    text_indices = [i for i, t in enumerate(types) if t == "text"]

    music_images = [strips[i].image for i in music_indices]

    # For OCR pass both grayscale and binary images.
    # The binary image (pre-binarized by the slicer) gives EasyOCR cleaner
    # edges at small strip heights (17-30 px) than anti-aliased grayscale.
    text_images   = [strips[i].image  for i in text_indices]
    text_binaries = [strips[i].binary for i in text_indices]

    music_preds = recognize_music(music_images, checkpoint_path) if music_images else []
    text_preds = recognize_chords(text_images, text_binaries) if text_images else []

    # 7) Aggregate — skip empty text segments
    music_map = {idx: pred for idx, pred in zip(music_indices, music_preds)}
    text_map = {idx: pred for idx, pred in zip(text_indices, text_preds)}

    segments = []
    for i, (s, t) in enumerate(zip(strips, types)):
        bbox = [s.x_start, s.y_start, s.x_end - s.x_start, s.y_end - s.y_start]
        if t == "music":
            content = music_map.get(i, "")
            segments.append({"type": "music", "bbox": bbox, "content": content})
        else:
            content = text_map.get(i, "")
            if not content.strip():
                # Still include in overlay (for bbox visualization) but mark as empty
                segments.append({"type": "text", "bbox": bbox, "content": ""})
            else:
                segments.append({"type": "text", "bbox": bbox, "content": content})

    return {
        "error": None,
        "pages": [{"index": 0, "segments": segments, "page_image_data_url": page_image_data_url}],
        "meta": {
            "filename": filename,
            "page_height": page.meta["height"],
            "page_width": page.meta["width"],
            "deskew_angle_deg": page.meta.get("deskew_angle_deg", 0.0),
        },
    }
