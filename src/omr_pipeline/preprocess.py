"""
Preprocessing — load, grayscale, binarize, deskew.

Deskew uses projection-profile variance: rotate binary image at many
candidate angles, pick the angle whose horizontal projection has the
sharpest peaks (highest variance). This is the gold-standard method
for document/score images with dominant horizontal lines.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


@dataclass
class PageImage:
    """Single page as grayscale and binary versions."""
    grayscale: np.ndarray  # (H, W) uint8
    binary: np.ndarray     # (H, W) 0/255
    meta: dict


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(source: bytes | Path | np.ndarray) -> np.ndarray:
    """Load image as grayscale. Accepts raw bytes, path, or numpy array."""
    if isinstance(source, np.ndarray):
        if len(source.shape) == 3:
            return cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
        return source
    if isinstance(source, bytes):
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def load_pdf_page(pdf_bytes: bytes, page: int = 0, dpi: int = 150) -> np.ndarray:
    """Render a PDF page to grayscale image."""
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) required for PDF support")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page_obj = doc[page]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page_obj.get_pixmap(matrix=mat, alpha=False)
        n = pix.n
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, n)
        if n == 1:
            return img[:, :, 0]
        if n == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if n == 4:
            return cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
        raise ValueError(f"Unsupported pixmap channels: {n}")
    finally:
        doc.close()


def pdf_load_dpi() -> int:
    """DPI for PyMuPDF rasterisation before deskew / staff detection.

    Higher values give taller staff crops so, after resize to ``img_height``,
    symbols stay sharper (same idea as a tight manual crop).  Override with
    env ``OMR_PDF_DPI`` (clamped to 72–600).  Default ``300``.
    """
    raw = os.environ.get("OMR_PDF_DPI", "").strip()
    if raw:
        try:
            return max(72, min(600, int(raw)))
        except ValueError:
            pass
    return 300


# ---------------------------------------------------------------------------
# Binarization
# ---------------------------------------------------------------------------

def binarize(img: np.ndarray) -> np.ndarray:
    """Otsu binarization → ink=255, bg=0."""
    if img.dtype in (np.float32, np.float64):
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


# ---------------------------------------------------------------------------
# Deskew — projection-profile variance method
# ---------------------------------------------------------------------------

def _rotate_small(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Fast rotation for small images (no bound expansion — edges clipped)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _projection_variance(binary: np.ndarray) -> float:
    """Variance of the horizontal projection profile (ink=255)."""
    proj = np.sum(binary > 0, axis=1, dtype=np.float64)
    return float(np.var(proj))


def _estimate_skew_projection(binary_ink: np.ndarray,
                               angle_range: float = 5.0,
                               coarse_step: float = 0.5,
                               fine_step: float = 0.1) -> float:
    """Find the rotation angle that maximises projection-profile variance.

    Two-pass search: coarse sweep, then fine sweep around the best coarse angle.
    """
    best_angle = 0.0
    best_var = _projection_variance(binary_ink)

    # Coarse sweep
    for a10 in range(int(-angle_range / coarse_step), int(angle_range / coarse_step) + 1):
        angle = a10 * coarse_step
        rotated = _rotate_small(binary_ink, angle)
        v = _projection_variance(rotated)
        if v > best_var:
            best_var = v
            best_angle = angle

    # Fine sweep around best coarse angle
    fine_best = best_angle
    for a10 in range(int(-1.0 / fine_step), int(1.0 / fine_step) + 1):
        angle = best_angle + a10 * fine_step
        if abs(angle) > angle_range:
            continue
        rotated = _rotate_small(binary_ink, angle)
        v = _projection_variance(rotated)
        if v > best_var:
            best_var = v
            fine_best = angle

    return fine_best


def _rotate_bound(img: np.ndarray, angle_deg: float, border_value: int = 255) -> np.ndarray:
    """Rotate image around center, expanding canvas so nothing is cropped."""
    h, w = img.shape[:2]
    cX, cY = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cX, cY), angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nW = int(h * sin + w * cos)
    nH = int(h * cos + w * sin)
    M[0, 2] += (nW / 2.0) - cX
    M[1, 2] += (nH / 2.0) - cY
    return cv2.warpAffine(
        img, M, (nW, nH),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def deskew(img_gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Deskew a grayscale page image. Returns (rotated_image, angle_degrees)."""
    h, w = img_gray.shape[:2]

    # Downscale for fast angle estimation
    max_dim = 800
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        small = cv2.resize(img_gray, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = img_gray

    small_bin = binarize(small)
    angle = _estimate_skew_projection(small_bin)

    if abs(angle) < 0.15:
        return img_gray.copy(), 0.0

    rotated = _rotate_bound(img_gray, angle, border_value=255)
    return rotated, float(angle)


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_page(img: np.ndarray) -> PageImage:
    """Deskew, then produce grayscale + binary versions."""
    img_gray = img.astype(np.uint8) if img.dtype != np.uint8 else img
    img_deskewed, angle = deskew(img_gray)
    binary = binarize(img_deskewed)
    return PageImage(
        grayscale=img_deskewed,
        binary=binary,
        meta={
            "deskew_angle_deg": angle,
            "height": img_deskewed.shape[0],
            "width": img_deskewed.shape[1],
        },
    )
