"""
Slicer skeleton — horizontal projection profile, extract strips.

Now includes smoothing, robust region extraction, and basic strip merging.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Strip:
    """Single horizontal strip (music or text region)."""
    x_start: int
    x_end: int
    y_start: int
    y_end: int
    height: int
    image: np.ndarray   # grayscale crop
    binary: np.ndarray  # binary crop (ink=1/True)
    ink_density: float


def horizontal_projection(binary: np.ndarray) -> np.ndarray:
    """Sum ink pixels per row. Binary may be {0,255} or bool."""
    ink = binary > 0
    return np.sum(ink, axis=1).astype(np.float32)


def _smooth_1d(x: np.ndarray, win: int) -> np.ndarray:
    """Simple moving-average smoothing."""
    if win <= 1:
        return x
    win = int(win)
    if win % 2 == 0:
        win += 1
    k = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x, k, mode="same").astype(np.float32)


def _find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous True runs as (start, end) with end exclusive."""
    runs: list[tuple[int, int]] = []
    i = 0
    n = int(mask.size)
    while i < n:
        if bool(mask[i]):
            s = i
            i += 1
            while i < n and bool(mask[i]):
                i += 1
            runs.append((s, i))
        else:
            i += 1
    return runs


def _merge_close_runs(runs: list[tuple[int, int]], *, max_gap: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [runs[0]]
    for s, e in runs[1:]:
        ps, pe = merged[-1]
        if s - pe <= max_gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged


def _tight_bbox(binary_band: np.ndarray, y0: int, y1: int, x_pad: int = 2, y_pad: int = 1) -> tuple[int, int, int, int]:
    """Compute tight bbox of ink in [y0:y1]. Returns (x0,x1,y0,y1) in page coords with x1/y1 exclusive."""
    band = binary_band[y0:y1] > 0
    if not np.any(band):
        h, w = binary_band.shape[:2]
        return 0, w, y0, y1
    ys, xs = np.where(band)
    x0 = int(xs.min()) - x_pad
    x1 = int(xs.max()) + 1 + x_pad
    yy0 = y0 + int(ys.min()) - y_pad
    yy1 = y0 + int(ys.max()) + 1 + y_pad
    h, w = binary_band.shape[:2]
    x0 = max(0, x0)
    x1 = min(w, x1)
    yy0 = max(0, yy0)
    yy1 = min(h, yy1)
    return x0, x1, yy0, yy1


def _staff_row_mask(binary_crop: np.ndarray, *, run_frac: float = 0.45) -> np.ndarray:
    """Rows that look like staff lines via long horizontal ink runs."""
    b = binary_crop > 0
    if b.size == 0:
        return np.zeros((0,), dtype=bool)
    h, w = b.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((h,), dtype=bool)

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
    return longest >= int(run_frac * w)


def _split_music_text_within_run(
    grayscale: np.ndarray,
    binary: np.ndarray,
    *,
    y0: int,
    y1: int,
) -> list[tuple[int, int]]:
    """Given a vertical run [y0,y1), split into sub-runs around staff evidence.

    Returns list of sub-runs (yy0,yy1) in page coords. This tends to separate
    chord text above the staff from the staff itself when they end up in the
    same 'active' region.
    """
    y0 = int(max(0, y0))
    y1 = int(min(binary.shape[0], y1))
    if y1 <= y0:
        return []

    crop = binary[y0:y1]
    staff_rows = _staff_row_mask(crop)
    if staff_rows.size == 0 or not np.any(staff_rows):
        return [(y0, y1)]

    # Grow staff rows slightly so the staff region includes notes/rests
    grow = max(6, (y1 - y0) // 40)
    kernel = np.ones((2 * grow + 1,), dtype=np.int8)
    staff_grown = np.convolve(staff_rows.astype(np.int8), kernel, mode="same") > 0

    # Build music core run as the bounding box of grown staff rows
    ys = np.nonzero(staff_grown)[0]
    m0 = y0 + int(ys.min())
    m1 = y0 + int(ys.max()) + 1

    # Pad music region to include notes/stems but avoid swallowing chord text above staff
    pad = max(6, (y1 - y0) // 25)
    m0 = max(y0, m0 - pad)
    m1 = min(y1, m1 + pad)

    # Anything above/below with ink becomes separate sub-runs (text)
    parts: list[tuple[int, int]] = []
    # Above
    if m0 - y0 > 6:
        top = binary[y0:m0]
        if np.any(top > 0):
            parts.append((y0, m0))
    # Music
    parts.append((m0, m1))
    # Below
    if y1 - m1 > 6:
        bot = binary[m1:y1]
        if np.any(bot > 0):
            parts.append((m1, y1))

    # If split created tiny slivers, merge them back
    min_h = max(10, (y1 - y0) // 25)
    merged: list[tuple[int, int]] = []
    for a, b in parts:
        if not merged:
            merged.append((a, b))
            continue
        if (b - a) < min_h:
            ps, pe = merged[-1]
            merged[-1] = (ps, b)
        else:
            merged.append((a, b))
    return merged


def extract_strips(grayscale: np.ndarray, binary: np.ndarray) -> list[Strip]:
    """Split page into horizontal strips using projection profile."""
    h, w = grayscale.shape[:2]
    prof = horizontal_projection(binary)
    if prof.max() <= 0:
        bb = (0, w, 0, h)
        return [Strip(bb[0], bb[1], bb[2], bb[3], h, grayscale, (binary > 0).astype(np.uint8), 0.0)]

    # Normalise by width to make thresholds more stable across DPI / page sizes
    prof_frac = (prof / float(max(w, 1))).astype(np.float32)
    prof_s = _smooth_1d(prof_frac, win=max(9, h // 180))  # ~1% of height, odd

    # Adaptive threshold: slightly above the low-percentile baseline
    baseline = float(np.percentile(prof_s, 20))
    peak = float(np.percentile(prof_s, 99))
    # If the page is very sparse, fall back to a small absolute threshold
    thr = max(0.0025, baseline + 0.15 * max(peak - baseline, 0.0))
    # Lower threshold for text-like bumps (chord symbols, titles)
    thr_text = max(0.0015, baseline + 0.05 * max(peak - baseline, 0.0))

    # Staff-ish dense regions
    active_staff = prof_s >= thr
    runs_staff = _find_runs(active_staff)
    runs_staff = _merge_close_runs(runs_staff, max_gap=8)

    # Text-ish sparse regions (between thr_text and thr)
    active_text = (prof_s >= thr_text) & (prof_s < thr)
    runs_text = _find_runs(active_text)
    runs_text = _merge_close_runs(runs_text, max_gap=6)

    # Enforce minimum strip height by merging tiny runs into neighbours where possible
    min_h = max(14, h // 140)  # ~0.7% of height
    merged: list[tuple[int, int]] = []
    for s, e in runs_staff:
        if not merged:
            merged.append((s, e))
            continue
        if (e - s) < min_h:
            # Merge into previous if close, otherwise keep (will be filtered later)
            ps, pe = merged[-1]
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    # If we still got nothing (threshold too high), return whole page
    if not merged:
        merged = [(0, h)]

    # Merge/clean text runs similarly, but allow smaller heights (chords can be short)
    merged_text: list[tuple[int, int]] = []
    min_text_h = max(10, h // 220)
    for s, e in runs_text:
        if e - s < min_text_h:
            continue
        if not merged_text:
            merged_text.append((s, e))
            continue
        ps, pe = merged_text[-1]
        if s - pe <= 6:
            merged_text[-1] = (ps, e)
        else:
            merged_text.append((s, e))

    strips: list[Strip] = []
    ink = (binary > 0)
    # 1) Staff/music candidate strips from dense regions
    for y0, y1 in merged:
        for yy0, yy1 in _split_music_text_within_run(grayscale, binary, y0=y0, y1=y1):
            if yy1 <= yy0:
                continue
            x0, x1, yb0, yb1 = _tight_bbox(binary, yy0, yy1)
            band_gray = grayscale[yb0:yb1, x0:x1]
            band_bin = ink[yb0:yb1, x0:x1].astype(np.uint8)
            density = float(np.mean(band_bin > 0)) if band_bin.size else 0.0
            strips.append(
                Strip(
                    x_start=x0,
                    x_end=x1,
                    y_start=yb0,
                    y_end=yb1,
                    height=yb1 - yb0,
                    image=band_gray,
                    binary=band_bin,
                    ink_density=density,
                )
            )

    # 2) Text candidate strips from sparse regions.
    # Reject if they overlap at all with an already-placed strip.  Text strips
    # live in the white space *between* staves — any overlap means the slicer
    # already accounted for that region in the first pass.
    existing_spans = [(s.y_start, s.y_end) for s in strips]
    for y0, y1 in merged_text:
        x0, x1, yb0, yb1 = _tight_bbox(binary, y0, y1, x_pad=3, y_pad=2)
        if yb1 <= yb0:
            continue
        # Reject if any overlap with an existing strip
        too_much = False
        for ys, ye in existing_spans:
            inter = max(0, min(ye, yb1) - max(ys, yb0))
            if inter > 0:
                too_much = True
                break
        if too_much:
            continue
        band_gray = grayscale[yb0:yb1, x0:x1]
        band_bin = ink[yb0:yb1, x0:x1].astype(np.uint8)
        density = float(np.mean(band_bin > 0)) if band_bin.size else 0.0
        strips.append(
            Strip(
                x_start=x0,
                x_end=x1,
                y_start=yb0,
                y_end=yb1,
                height=yb1 - yb0,
                image=band_gray,
                binary=band_bin,
                ink_density=density,
            )
        )
        existing_spans.append((yb0, yb1))

    if not strips:
        strips.append(
            Strip(0, w, 0, h, h, grayscale, ink.astype(np.uint8), float(np.mean(ink)))
        )

    # Sort just in case
    strips.sort(key=lambda s: (s.y_start, s.x_start))
    return strips
