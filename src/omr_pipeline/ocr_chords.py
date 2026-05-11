"""
Chord OCR — read jazz chord symbols from the strip above each staff.

Two backends, simple:
  vlm      — GPT-4o / Gemini Vision (recommended; handles stylised jazz fonts).
  easyocr  — local EasyOCR fallback (no API key required).

Backend selection:
  1. Explicit `backend=` parameter
  2. `OMR_CHORD_BACKEND` env var: "vlm" or "easyocr"
  3. Auto: "vlm" if OPENAI_API_KEY / GOOGLE_API_KEY is set, otherwise "easyocr"

Preprocessing is a single chain: ensure light background → upscale → CLAHE.
No connected-component isolation (the previous CC pipeline produced unstable
crops; whole-strip OCR is more robust on Real Book chord rows).
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

ChordBackend = Literal["vlm", "easyocr"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHORD_ALLOWLIST = (
    "ABCDEFGabcdefg#b0123456789"
    "mMajdimaugsusoø+-/(). ,"
)
_MIN_OCR_HEIGHT = 220
_MAX_OCR_WIDTH = 4000

_VLM_PROMPT = (
    "Read the jazz chord symbols in this image strictly left to right. "
    "Return ONLY the chord symbols, separated by single spaces. "
    "Do not add any explanatory text, prefixes, or punctuation. "
    "Use standard jazz notation: root (A-G), accidental (# or b), "
    "quality (maj, m, dim, aug, sus, ø), extensions (7, 9, 11, 13), "
    "alterations like b5 #9 b13, and slash bass (/C). "
    "If you see no chord symbols, return an empty response. "
    "Example: Am7 D7 Gmaj7 Cmaj7 F#m7b5 B7 Em"
)


# ---------------------------------------------------------------------------
# EasyOCR — lazy initialisation, error-safe
# ---------------------------------------------------------------------------

_easyocr_reader = None
_easyocr_failed = False


def _easyocr():
    """Return a cached EasyOCR Reader, or None if initialisation has failed."""
    global _easyocr_reader, _easyocr_failed
    if _easyocr_failed:
        return None
    if _easyocr_reader is None:
        try:
            import easyocr
            log.info("Initialising EasyOCR (may download ~200 MB on first run)")
            _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            log.info("EasyOCR ready")
        except Exception as exc:
            log.error("EasyOCR initialisation failed: %s", exc)
            _easyocr_failed = True
            return None
    return _easyocr_reader


# ---------------------------------------------------------------------------
# Image preprocessing — single chain, no CC magic
# ---------------------------------------------------------------------------

def _to_uint8_gray(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("None image")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else img[:, :, 0]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _ensure_light_bg(gray: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(gray) if float(np.median(gray)) < 128 else gray


def _prep(gray: np.ndarray) -> np.ndarray:
    """Light-background → upscale to min OCR height → CLAHE."""
    gray = _ensure_light_bg(_to_uint8_gray(gray))

    h, w = gray.shape
    if h < _MIN_OCR_HEIGHT:
        scale = _MIN_OCR_HEIGHT / h
        new_w = min(int(w * scale), _MAX_OCR_WIDTH)
        new_h = int(h * (new_w / w))  # keep aspect after width clamp
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    elif w > _MAX_OCR_WIDTH:
        scale = _MAX_OCR_WIDTH / w
        gray = cv2.resize(gray, (_MAX_OCR_WIDTH, int(h * scale)),
                          interpolation=cv2.INTER_AREA)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    return clahe.apply(gray)


# ---------------------------------------------------------------------------
# Backend: EasyOCR (whole strip)
# ---------------------------------------------------------------------------

def _backend_easyocr(gray: np.ndarray) -> str:
    reader = _easyocr()
    if reader is None:
        return ""
    img = _prep(gray)
    try:
        dets = reader.readtext(
            img, detail=1, paragraph=False,
            allowlist=_CHORD_ALLOWLIST,
            text_threshold=0.20, low_text=0.10, width_ths=0.7,
        )
    except Exception as exc:
        log.warning("EasyOCR readtext failed: %s", exc)
        return ""
    dets.sort(key=lambda d: d[0][0][0])
    parts = [d[1].strip() for d in dets if d[2] >= 0.10 and d[1].strip()]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Backend: VLM
# ---------------------------------------------------------------------------

def _backend_vlm(gray: np.ndarray) -> str:
    img = _prep(gray)
    _, buf = cv2.imencode(".png", img)
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    if os.environ.get("OPENAI_API_KEY"):
        return _vlm_openai(b64)
    if os.environ.get("GOOGLE_API_KEY"):
        return _vlm_gemini(b64)
    log.warning("VLM backend requested but no OPENAI_API_KEY / GOOGLE_API_KEY set")
    return ""


def _vlm_openai(b64: str) -> str:
    model = os.environ.get("OMR_VLM_MODEL", "gpt-4o-mini")
    try:
        from openai import OpenAI
        resp = OpenAI().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": _VLM_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
            max_tokens=256,
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        log.warning("OpenAI VLM call failed: %s", exc)
        return ""


def _vlm_gemini(b64: str) -> str:
    model = os.environ.get("OMR_VLM_MODEL", "gemini-2.0-flash")
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        resp = genai.GenerativeModel(model).generate_content([
            _VLM_PROMPT,
            {"mime_type": "image/png", "data": base64.b64decode(b64)},
        ])
        return (resp.text or "").strip()
    except Exception as exc:
        log.warning("Gemini VLM call failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _resolve_backend(explicit: ChordBackend | None) -> ChordBackend:
    if explicit in ("vlm", "easyocr"):
        return explicit
    env = os.environ.get("OMR_CHORD_BACKEND", "").strip().lower()
    if env in ("vlm", "easyocr"):
        return env  # type: ignore[return-value]
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "vlm"
    return "easyocr"


def recognize_chords(
    strip_images: list[np.ndarray],
    backend: ChordBackend | None = None,
) -> list[str]:
    """OCR a list of chord strips.  Returns one cleaned chord string per strip.

    The output of each strip is a single string of canonical chord tokens
    separated by double spaces (see ``clean_chord_line``).  Empty strips and
    strips that yield no recognisable chords return ``""``.
    """
    if not strip_images:
        return []

    resolved = _resolve_backend(backend)
    log.info("Chord OCR backend: %s", resolved)
    fn = _backend_vlm if resolved == "vlm" else _backend_easyocr

    results: list[str] = []
    for img in strip_images:
        if img is None or img.size == 0 or img.shape[0] < 4 or img.shape[1] < 4:
            results.append("")
            continue
        raw = fn(img)
        results.append(clean_chord_line(raw))
    return results
