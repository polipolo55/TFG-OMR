"""Chord OCR — read chord symbols from text strips using EasyOCR.

Preprocessing pipeline for small chord strips:
  1. Ensure light background (auto-invert if needed).
  2. Upscale to at least 100 px tall with Lanczos interpolation.
  3. CLAHE contrast enhancement.
  4. Unsharp-mask sharpening.
  5. EasyOCR with a chord-relevant allowlist.
  6. Jazz chord grammar post-processor to clean up OCR noise.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from .chord_postprocess import clean_chord_line

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded EasyOCR reader
# ---------------------------------------------------------------------------

_reader = None

# Restrict EasyOCR to characters that appear in jazz chord notation.
# Keeping the set tight prevents the model from hallucinating letters.
_CHORD_ALLOWLIST = (
    "ABCDEFGabcdefg"  # note names
    "#b"              # accidentals
    "0123456789"      # extensions / alterations
    "mMajdimaugsusoø+-/"  # quality words and symbols
    "().,"            # occasional parentheses around alterations
    " "
)

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

_MIN_OCR_HEIGHT = 160   # Target ~5× upscale from a typical 28-40px chord strip
_MIN_GRAY_HEIGHT = 100

# EasyOCR can stall on very wide images.  We cap the processed width at this
# value.  At 200 DPI a full-page chord strip scaled to 160px height reaches
# ~7 000-8 000 px; 6 500 px keeps the widest strips intact while staying
# within EasyOCR's comfortable range.
_MAX_OCR_WIDTH = 6500


def _ensure_light_background(gray: np.ndarray) -> np.ndarray:
    """Invert the image if the background appears to be dark."""
    if float(np.median(gray)) < 128:
        return cv2.bitwise_not(gray)
    return gray


def _preprocess_binary(binary: np.ndarray) -> np.ndarray:
    """Upscale a pre-binarized strip for EasyOCR.

    Uses INTER_NEAREST so binary edges stay sharp without gray anti-aliasing.
    Inverts the strip so background is white and text is black.
    """
    img = cv2.bitwise_not((binary > 0).astype(np.uint8) * 255)
    h, w = img.shape
    if h == 0 or w == 0:
        return img

    if h < _MIN_OCR_HEIGHT:
        scale = _MIN_OCR_HEIGHT / h
        new_h = _MIN_OCR_HEIGHT
        new_w = int(w * scale)
        if new_w > _MAX_OCR_WIDTH:
            new_w = _MAX_OCR_WIDTH
            new_h = int(h * _MAX_OCR_WIDTH / w)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        h, w = new_h, new_w

    if w > _MAX_OCR_WIDTH:
        img = cv2.resize(img, (_MAX_OCR_WIDTH, int(h * _MAX_OCR_WIDTH / w)),
                         interpolation=cv2.INTER_NEAREST)
    return img


def _preprocess_gray(gray: np.ndarray) -> np.ndarray:
    """Upscale a grayscale strip for EasyOCR using Cubic + CLAHE + sharpen."""
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY) if gray.shape[2] == 3 else gray[:, :, 0]
    img = _ensure_light_background(gray)
    h, w = img.shape
    if h == 0 or w == 0:
        return img

    if h < _MIN_GRAY_HEIGHT:
        scale = _MIN_GRAY_HEIGHT / h
        new_h = _MIN_GRAY_HEIGHT
        new_w = int(w * scale)
        if new_w > _MAX_OCR_WIDTH:
            new_w = _MAX_OCR_WIDTH
            new_h = int(h * _MAX_OCR_WIDTH / w)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = new_h, new_w

    if w > _MAX_OCR_WIDTH:
        img = cv2.resize(img, (_MAX_OCR_WIDTH, int(h * _MAX_OCR_WIDTH / w)),
                         interpolation=cv2.INTER_AREA)
        h, w = img.shape

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    img = clahe.apply(img)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def recognize_chords(
    strip_images: list[np.ndarray],
    strip_binaries: list[np.ndarray] | None = None,
) -> list[str]:
    """OCR chord symbol strips. Returns one cleaned chord string per strip.

    Args:
        strip_images: grayscale crop of each text strip (used as fallback).
        strip_binaries: pre-binarized crops (same size as strip_images).
            When provided, the binary image is used for OCR — cleaner edges
            give EasyOCR a better chance on small handwritten chord symbols.
    """
    if not strip_images:
        return []

    if strip_binaries is None:
        strip_binaries = [None] * len(strip_images)  # type: ignore[list-item]

    try:
        reader = _get_reader()
    except Exception as e:
        log.warning("EasyOCR init failed: %s", e)
        return [""] * len(strip_images)

    results: list[str] = []
    for gray, binary in zip(strip_images, strip_binaries):
        if gray is None or gray.size == 0 or gray.shape[0] < 4 or gray.shape[1] < 4:
            results.append("")
            continue

        # Primary: binary-based (crisp edges from the pre-binarized strip)
        result = _run_ocr_on_image(reader, _preprocess_binary(binary))

        # Fallback: grayscale-based OCR (Cubic + CLAHE) when binary misses
        if not result:
            result = _run_ocr_on_image(reader, _preprocess_gray(gray))

        results.append(result)

    return results


def _run_ocr_on_image(reader, processed: np.ndarray) -> str:
    """Run EasyOCR on a preprocessed image, return cleaned chord string."""
    try:
        detections = reader.readtext(
            processed,
            detail=1,
            paragraph=False,
            allowlist=_CHORD_ALLOWLIST,
            text_threshold=0.3,
            low_text=0.2,
            width_ths=0.7,
        )
        detections.sort(key=lambda d: d[0][0][0])
        parts = [
            d[1].strip()
            for d in detections
            if d[2] >= 0.25 and d[1].strip()
        ]
        return clean_chord_line(" ".join(parts))
    except Exception as e:
        log.warning("EasyOCR failed: %s", e)
        return ""
