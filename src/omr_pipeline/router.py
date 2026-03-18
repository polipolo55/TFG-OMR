"""
Router skeleton — classify strip as music or text.

Heuristic router: height/density + staff-line evidence (long horizontal runs).
"""
from __future__ import annotations

import numpy as np

from .slicer import Strip

# Rough staff height in pixels (typical single staff ~60–100 px at 150 DPI)
MIN_STAFF_HEIGHT = 40
MAX_TEXT_HEIGHT = 80


def _staff_line_evidence(strip: Strip) -> float:
    """Return a score in [0,1] indicating staff-like horizontal line presence."""
    b = strip.binary > 0
    if b.size == 0:
        return 0.0
    h, w = b.shape[:2]
    if h < 8 or w < 50:
        return 0.0

    # Morphological extraction of horizontal lines (robust to broken scans)
    import cv2

    img = (b.astype(np.uint8) * 255)
    k_w = max(25, w // 12)  # wide horizontal kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
    lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)

    # Score: how much horizontal-line ink remains, normalized by area
    line_density = float(np.mean(lines > 0))

    # Also keep a lightweight run-length fraction (helps when lines are very thin)
    longest = np.zeros((h,), dtype=np.int32)
    for i in range(h):
        row = b[i].astype(np.int8)
        diff = row[1:] - row[:-1]
        starts = np.nonzero(diff == 1)[0] + 1
        ends = np.nonzero(diff == -1)[0] + 1
        if row[0] == 1:
            starts = np.r_[0, starts]
        if row[-1] == 1:
            ends = np.r_[ends, w]
        if starts.size:
            longest[i] = int((ends - starts).max())
    frac_rows = float((longest >= 0.35 * w).mean())

    # Combine; morphological term dominates
    return max(line_density * 6.0, frac_rows)


def route_strip(strip: Strip) -> str:
    """Return 'music' or 'text'."""
    staff_e = _staff_line_evidence(strip)

    # Low ink density means it's almost certainly a chord/text line regardless
    # of height.  Staff systems have ink_density >= ~0.15 at 150 DPI.
    if strip.ink_density < 0.06:
        return "text"

    # Strong staff evidence + tall enough → music
    if staff_e >= 0.06 and strip.height >= 40:
        return "music"

    # Tall-and-dense bands: require *both* height and decent ink density so
    # we don't sweep up sparse mid-staff spacer rows.
    if strip.height >= 70 and strip.ink_density > 0.08:
        return "music"

    # Medium height with both staff evidence and ink density
    if strip.height >= MIN_STAFF_HEIGHT and strip.ink_density > 0.08 and staff_e >= 0.03:
        return "music"

    # Very short strips are almost always chord/text lines
    if strip.height <= 35:
        return "text"

    # Default
    return "text"
