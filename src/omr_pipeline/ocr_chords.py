"""
Chord OCR — read jazz chord symbols from the text strips above each staff.

Three backends, selectable via the *backend* argument or env vars:

  contour   (default) — connected-component isolation → per-symbol EasyOCR.
  easyocr             — whole-strip EasyOCR (legacy, less accurate).
  vlm                 — vision-language model (GPT-4o / Gemini).

Preprocessing pipeline for every strip:
  1. Ensure light background (auto-invert if needed).
  2. Upscale so the shortest dimension reaches a usable OCR height.
  3. CLAHE + unsharp-mask for contrast.
  4. Connected-component isolation into individual chord groups.
  5. Per-group OCR.
  6. Jazz chord grammar post-processor (chord_postprocess.py).
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Literal

import cv2
import numpy as np

from .chord_postprocess import clean_chord_line

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHORD_ALLOWLIST = (
    "ABCDEFGabcdefg#b0123456789"
    "mMajdimaugsusoø+-/(). ,"
)
_MIN_OCR_HEIGHT = 200
_MIN_GRAY_HEIGHT = 140
_MAX_OCR_WIDTH = 6500

_VLM_MODEL = os.environ.get("OMR_VLM_MODEL", "gpt-4o")

_VLM_PROMPT = (
    "Read the jazz chord symbols in this image from left to right. "
    "Return ONLY the chord symbols separated by double spaces. "
    "Use standard jazz notation: root (A-G), accidentals (# b), "
    "quality (maj - dim aug + sus m), extensions (7 9 11 13), "
    "alterations (b5 #9 etc), slash bass (/C). "
    "Example output: Am7  D7  Gmaj7  Cmaj7  F#m7b5  B7  Em"
)

# ---------------------------------------------------------------------------
# Lazy EasyOCR reader
# ---------------------------------------------------------------------------

_easyocr_reader = None


def _easyocr(lazy: bool = True):
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _easyocr_reader


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _ensure_light_bg(gray: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(gray) if float(np.median(gray)) < 128 else gray


def _upscale(img: np.ndarray, min_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    if h >= min_h:
        return img
    scale = min_h / h
    new_w = min(int(w * scale), _MAX_OCR_WIDTH)
    new_h = min_h if new_w < _MAX_OCR_WIDTH else int(h * _MAX_OCR_WIDTH / w)
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def _clamp_width(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= _MAX_OCR_WIDTH:
        return img
    return cv2.resize(img, (_MAX_OCR_WIDTH, int(h * _MAX_OCR_WIDTH / w)),
                      interpolation=cv2.INTER_AREA)


def _enhance(gray: np.ndarray) -> np.ndarray:
    """CLAHE + unsharp mask + adaptive threshold cleanup."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    out = clahe.apply(gray)
    # Mild denoise to reduce background noise from scans
    out = cv2.fastNlMeansDenoising(out, h=8, templateWindowSize=7, searchWindowSize=21)
    # Unsharp mask for crisp edges
    blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.2)
    out = cv2.addWeighted(out, 1.6, blur, -0.6, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _prep_gray(gray: np.ndarray) -> np.ndarray:
    """Full preprocessing chain for grayscale chord images."""
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY) if gray.shape[2] == 3 else gray[:, :, 0]
    return _enhance(_clamp_width(_upscale(_ensure_light_bg(gray), _MIN_GRAY_HEIGHT)))


def _prep_binary(binary: np.ndarray) -> np.ndarray:
    """Preprocessing for binarised images fed to EasyOCR."""
    img = cv2.bitwise_not((binary > 0).astype(np.uint8) * 255)
    img = _upscale(img, _MIN_OCR_HEIGHT)
    return _clamp_width(img)


# ---------------------------------------------------------------------------
# Connected-component chord isolation
# ---------------------------------------------------------------------------

def _isolate_groups(
    binary: np.ndarray,
    gray: np.ndarray,
    min_area: int = 15,
) -> list[tuple[int, np.ndarray]]:
    """Segment a chord strip into individual chord-symbol groups via CCs.

    Returns (x_centre, cropped_gray) pairs sorted left → right.  Horizontally
    close components are merged (a chord like "Gmaj7" spans multiple CCs).
    The merge gap is computed adaptively from the median component height.
    """
    ink = (binary > 0).astype(np.uint8) * 255
    if ink.max() == 0:
        return []

    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)

    bboxes: list[tuple[int, int, int, int]] = []
    heights: list[int] = []
    for i in range(1, n_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        bboxes.append((x, y, x + w, y + h))
        heights.append(h)

    if not bboxes:
        return []

    # Adaptive merge gap: characters within a chord symbol are close together
    # relative to the font size.  Use ~40% of median character height.
    median_h = float(np.median(heights)) if heights else 12.0
    merge_gap = max(4, int(median_h * 0.4))

    bboxes.sort(key=lambda b: b[0])

    # First pass: merge CCs that are within merge_gap of each other
    merged: list[tuple[int, int, int, int]] = [bboxes[0]]
    for x0, y0, x1, y1 in bboxes[1:]:
        gx0, gy0, gx1, gy1 = merged[-1]
        if x0 - gx1 <= merge_gap:
            merged[-1] = (min(gx0, x0), min(gy0, y0), max(gx1, x1), max(gy1, y1))
        else:
            merged.append((x0, y0, x1, y1))

    # Second pass: split groups that are too wide (likely two separate chords
    # that got merged).  "Too wide" = wider than 3× median height.
    max_chord_w = max(int(median_h * 5), 80)
    groups: list[tuple[int, int, int, int]] = []
    for gx0, gy0, gx1, gy1 in merged:
        if gx1 - gx0 > max_chord_w:
            # Find natural gaps within this group and split
            sub_bbs = [b for b in bboxes if b[0] >= gx0 and b[2] <= gx1]
            if len(sub_bbs) >= 2:
                sub_bbs.sort(key=lambda b: b[0])
                gaps = [(sub_bbs[j + 1][0] - sub_bbs[j][2], j)
                        for j in range(len(sub_bbs) - 1)]
                gaps.sort(key=lambda g: g[0], reverse=True)
                # Split at the widest internal gap
                split_idx = gaps[0][1] + 1
                left = sub_bbs[:split_idx]
                right = sub_bbs[split_idx:]
                lx0 = min(b[0] for b in left)
                ly0 = min(b[1] for b in left)
                lx1 = max(b[2] for b in left)
                ly1 = max(b[3] for b in left)
                rx0 = min(b[0] for b in right)
                ry0 = min(b[1] for b in right)
                rx1 = max(b[2] for b in right)
                ry1 = max(b[3] for b in right)
                groups.append((lx0, ly0, lx1, ly1))
                groups.append((rx0, ry0, rx1, ry1))
                continue
        groups.append((gx0, gy0, gx1, gy1))

    h_img, w_img = gray.shape[:2]
    pad = 4
    results: list[tuple[int, np.ndarray]] = []
    for gx0, gy0, gx1, gy1 in groups:
        rx0 = max(0, gx0 - pad)
        rx1 = min(w_img, gx1 + pad)
        ry0 = max(0, gy0 - pad)
        ry1 = min(h_img, gy1 + pad)
        crop = gray[ry0:ry1, rx0:rx1]
        if crop.size > 0:
            results.append(((gx0 + gx1) // 2, crop))
    return results


# ---------------------------------------------------------------------------
# Per-image EasyOCR runner (shared by contour and easyocr backends)
# ---------------------------------------------------------------------------

def _ocr_image(reader, img: np.ndarray, min_confidence: float = 0.15) -> str:
    """Run EasyOCR on a single preprocessed image, return raw text."""
    try:
        dets = reader.readtext(
            img, detail=1, paragraph=False,
            allowlist=_CHORD_ALLOWLIST,
            text_threshold=0.20, low_text=0.10, width_ths=0.7,
        )
        dets.sort(key=lambda d: d[0][0][0])
        parts = [d[1].strip() for d in dets if d[2] >= min_confidence and d[1].strip()]
        return " ".join(parts)
    except Exception as exc:
        log.warning("EasyOCR readtext failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Backend: contour (default)
# ---------------------------------------------------------------------------

def _backend_contour(gray: np.ndarray, binary: np.ndarray | None) -> str:
    """Isolate chord groups via CCs, then OCR each group individually.

    Per-group OCR on small tightly-cropped images is far more reliable than
    running EasyOCR on a full-width strip.
    """
    if binary is None:
        _, bw = cv2.threshold(
            _ensure_light_bg(gray), 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        binary = (bw > 0).astype(np.uint8)

    groups = _isolate_groups(binary, gray)
    if not groups:
        return ""

    reader = _easyocr()
    parts: list[str] = []
    for _cx, crop in groups:
        text = _ocr_image(reader, _prep_gray(crop), min_confidence=0.10)
        if text:
            parts.append(text)

    return clean_chord_line("  ".join(parts))


# ---------------------------------------------------------------------------
# Backend: easyocr (whole-strip, legacy)
# ---------------------------------------------------------------------------

def _backend_easyocr(gray: np.ndarray, binary: np.ndarray | None) -> str:
    reader = _easyocr()
    result = ""
    if binary is not None:
        result = _ocr_image(reader, _prep_binary(binary))
    if not result:
        result = _ocr_image(reader, _prep_gray(gray))
    return clean_chord_line(result)


# ---------------------------------------------------------------------------
# Backend: VLM
# ---------------------------------------------------------------------------

def _backend_vlm(gray: np.ndarray, _binary: np.ndarray | None) -> str:
    processed = _prep_gray(gray)
    _, buf = cv2.imencode(".png", processed)
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    model = _VLM_MODEL
    if "gpt" in model.lower() or "openai" in model.lower():
        return _vlm_openai(b64, model)
    if "gemini" in model.lower():
        return _vlm_gemini(b64, model)
    log.warning("Unknown VLM model %r", model)
    return ""


def _vlm_openai(b64_img: str, model: str) -> str:
    try:
        from openai import OpenAI
        resp = OpenAI().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": _VLM_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
            ]}],
            max_tokens=256,
            temperature=0.0,
        )
        return clean_chord_line((resp.choices[0].message.content or "").strip())
    except Exception as exc:
        log.warning("OpenAI VLM: %s", exc)
        return ""


def _vlm_gemini(b64_img: str, model: str) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
        resp = genai.GenerativeModel(model).generate_content([
            _VLM_PROMPT,
            {"mime_type": "image/png", "data": base64.b64decode(b64_img)},
        ])
        return clean_chord_line((resp.text or "").strip())
    except Exception as exc:
        log.warning("Gemini VLM: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ChordBackend = Literal["contour", "easyocr", "vlm"]


def recognize_chords(
    strip_images: list[np.ndarray],
    strip_binaries: list[np.ndarray | None] | None = None,
    backend: ChordBackend | None = None,
) -> list[str]:
    """OCR chord-symbol strips.  Returns one cleaned chord string per strip.

    *backend* selects the strategy.  ``None`` auto-selects: VLM if an API key
    is set, otherwise contour.
    """
    if not strip_images:
        return []

    if strip_binaries is None:
        strip_binaries = [None] * len(strip_images)

    if backend is None:
        if os.environ.get("OPENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            backend = "vlm"
        else:
            backend = "contour"

    log.info("Chord OCR backend: %s", backend)

    dispatch = {
        "contour": _backend_contour,
        "easyocr": _backend_easyocr,
        "vlm": _backend_vlm,
    }
    fn = dispatch[backend]

    results: list[str] = []
    for gray, binary in zip(strip_images, strip_binaries):
        if gray is None or gray.size == 0 or gray.shape[0] < 4 or gray.shape[1] < 4:
            results.append("")
            continue
        results.append(fn(gray, binary))
    return results
