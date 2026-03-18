"""
Staff detection — morphological staff-line finder and system grouper.

Uses a wide horizontal morphological opening to isolate staff lines from the
full binarized page, clusters them into 5-line groups, and pairs each staff
with its chord text region (the gap above it).

This is the sole segmentation strategy; it replaces the old projection-profile
slicer and heuristic router.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Staff:
    """Five detected staff lines with their page-level y-coordinates."""
    line_ys: list[int]
    staff_space: float
    x_start: int
    x_end: int

    @property
    def top(self) -> int:
        return self.line_ys[0]

    @property
    def bottom(self) -> int:
        return self.line_ys[-1]


@dataclass
class System:
    """One staff system on the page: chord region above + music staff."""
    staff: Staff
    chord_bbox: tuple[int, int, int, int] | None   # (x, y, w, h) or None
    music_bbox: tuple[int, int, int, int]           # (x, y, w, h)
    chord_image: np.ndarray | None = field(default=None, repr=False)
    music_image: np.ndarray | None = field(default=None, repr=False)
    chord_binary: np.ndarray | None = field(default=None, repr=False)
    music_binary: np.ndarray | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Staff-line extraction
# ---------------------------------------------------------------------------

def _extract_line_mask(binary: np.ndarray) -> np.ndarray:
    """Return a mask of long horizontal staff-line pixels."""
    h, w = binary.shape[:2]
    ink = (binary > 0).astype(np.uint8) * 255
    kernel_w = max(25, w // 6)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel, iterations=1)


def _line_row_centroids(line_mask: np.ndarray, merge_dist: int = 3) -> list[int]:
    """Find centroid y of each horizontal line in the mask.

    Adjacent active rows within *merge_dist* are fused (thick lines span
    2-3 rows in a real scan).
    """
    proj = np.sum(line_mask > 0, axis=1).astype(np.float32)
    if proj.max() == 0:
        return []

    threshold = proj.max() * 0.15
    active = proj >= threshold

    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i in range(len(active)):
        if active[i] and not in_run:
            start = i
            in_run = True
        elif not active[i] and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, len(active)))

    merged: list[tuple[int, int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= merge_dist:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    return [(s + e) // 2 for s, e in merged]


# ---------------------------------------------------------------------------
# Grouping lines into staves
# ---------------------------------------------------------------------------

def _group_into_staves(
    line_ys: list[int],
    page_width: int,
    spacing_tolerance: float = 0.35,
) -> list[Staff]:
    """Cluster line rows into 5-line staff groups.

    Slides a window of 5 consecutive lines and accepts groups whose inter-line
    spacings are within *spacing_tolerance* of their median.
    """
    if len(line_ys) < 5:
        return []

    used: set[int] = set()
    staves: list[Staff] = []
    i = 0
    while i <= len(line_ys) - 5:
        if i in used:
            i += 1
            continue

        cand = line_ys[i:i + 5]
        gaps = [cand[j + 1] - cand[j] for j in range(4)]
        med = float(np.median(gaps))
        if med < 3:
            i += 1
            continue

        if all(abs(g - med) / med <= spacing_tolerance for g in gaps):
            staves.append(Staff(
                line_ys=cand,
                staff_space=med,
                x_start=0,
                x_end=page_width,
            ))
            used.update(range(i, i + 5))
            i += 5
        else:
            i += 1

    return staves


def _refine_x_bounds(staves: list[Staff], binary: np.ndarray) -> None:
    """Tighten each staff's x-bounds to the actual ink extent."""
    _, w = binary.shape[:2]
    for staff in staves:
        y0 = max(0, staff.top - int(staff.staff_space * 0.5))
        y1 = min(binary.shape[0], staff.bottom + int(staff.staff_space * 0.5))
        cols = np.any(binary[y0:y1] > 0, axis=0)
        if np.any(cols):
            xs = np.where(cols)[0]
            staff.x_start = int(xs[0])
            staff.x_end = int(xs[-1]) + 1
        else:
            staff.x_start = 0
            staff.x_end = w


# ---------------------------------------------------------------------------
# Building systems (chord + music region pairs)
# ---------------------------------------------------------------------------

def _build_systems(
    staves: list[Staff],
    grayscale: np.ndarray,
    binary: np.ndarray,
    stem_margin: float = 1.8,
    chord_min_h: int = 10,
) -> list[System]:
    """Pair each staff with the chord-text region above it.

    Music region = top_line - margin … bottom_line + margin
    Chord region = previous system's bottom … current music region's top
    """
    H, W = grayscale.shape[:2]
    systems: list[System] = []

    for idx, staff in enumerate(staves):
        margin = int(staff.staff_space * stem_margin)

        music_y0 = max(0, staff.top - margin)
        music_y1 = min(H, staff.bottom + margin)
        music_x0 = max(0, staff.x_start - 5)
        music_x1 = min(W, staff.x_end + 5)

        # Chord region: gap between previous system's bottom and this music_y0
        if idx > 0:
            prev = staves[idx - 1]
            prev_bottom = min(H, prev.bottom + int(prev.staff_space * stem_margin))
        else:
            prev_bottom = 0
        chord_y0 = prev_bottom
        chord_y1 = music_y0

        music_gray = grayscale[music_y0:music_y1, music_x0:music_x1]
        music_bin = (binary[music_y0:music_y1, music_x0:music_x1] > 0).astype(np.uint8)

        # Chord crop — tight-bbox around actual ink in the chord band
        chord_bbox = None
        chord_gray = None
        chord_bin = None

        if chord_y1 - chord_y0 >= chord_min_h:
            band = binary[chord_y0:chord_y1, :]
            if np.any(band > 0):
                cols_ink = np.any(band > 0, axis=0)
                rows_ink = np.any(band > 0, axis=1)
                if np.any(cols_ink) and np.any(rows_ink):
                    cx0 = max(0, int(np.where(cols_ink)[0][0]) - 3)
                    cx1 = min(W, int(np.where(cols_ink)[0][-1]) + 4)
                    ry0 = max(chord_y0, chord_y0 + int(np.where(rows_ink)[0][0]) - 3)
                    ry1 = min(chord_y1, chord_y0 + int(np.where(rows_ink)[0][-1]) + 4)
                    chord_bbox = (cx0, ry0, cx1 - cx0, ry1 - ry0)
                    chord_gray = grayscale[ry0:ry1, cx0:cx1]
                    chord_bin = (binary[ry0:ry1, cx0:cx1] > 0).astype(np.uint8)

        systems.append(System(
            staff=staff,
            chord_bbox=chord_bbox,
            music_bbox=(music_x0, music_y0, music_x1 - music_x0, music_y1 - music_y0),
            chord_image=chord_gray,
            music_image=music_gray,
            chord_binary=chord_bin,
            music_binary=music_bin,
        ))

    return systems


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_systems(grayscale: np.ndarray, binary: np.ndarray) -> list[System]:
    """Detect staff systems and their chord regions on a preprocessed page.

    Returns an ordered list of System objects (top → bottom).  Returns an
    empty list only when fewer than 5 horizontal lines can be found (i.e.
    the image contains no recognisable staff).
    """
    H, W = grayscale.shape[:2]

    line_mask = _extract_line_mask(binary)
    line_ys = _line_row_centroids(line_mask)

    log.info("Staff-line detection: %d horizontal lines found", len(line_ys))

    if len(line_ys) < 5:
        return []

    staves = _group_into_staves(line_ys, W)
    if not staves:
        return []

    _refine_x_bounds(staves, binary)
    systems = _build_systems(staves, grayscale, binary)

    log.info(
        "Detected %d staff system(s); staff-space ≈ %.1f px",
        len(systems),
        float(np.mean([s.staff.staff_space for s in systems])),
    )
    return systems
