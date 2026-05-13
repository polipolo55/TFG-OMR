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


def _line_row_centroids(
    line_mask: np.ndarray,
    merge_dist: int = 3,
    *,
    ink_threshold_frac: float = 0.15,
) -> list[int]:
    """Find centroid y of each horizontal line in the mask.

    Adjacent active rows within *merge_dist* are fused (thick lines span
    2-3 rows in a real scan).
    """
    proj = np.sum(line_mask > 0, axis=1).astype(np.float32)
    if proj.max() == 0:
        return []

    threshold = proj.max() * ink_threshold_frac
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
    """Tighten each staff's x-bounds toward notation ink, not bare staff lines.

    Projecting ``binary`` in the staff band marks every column that contains a
    staff-line pixel, so ``x_start``/``x_end`` often span the full page.  We
    subtract a horizontal morphological opening (long horizontal runs) to drop
    most staff lines while keeping noteheads, stems, and barlines, then pad.
    Falls back to full-band ink if no non-staff content is found.
    """
    h_full, w = binary.shape[:2]
    hh = max(25, min(w // 10, 120))
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (hh, 1))

    for staff in staves:
        y0 = max(0, staff.top - int(staff.staff_space * 0.5))
        y1 = min(h_full, staff.bottom + int(staff.staff_space * 0.5))
        band = binary[y0:y1, :]
        if band.size == 0:
            staff.x_start, staff.x_end = 0, w
            continue

        ink = (band > 0).astype(np.uint8) * 255
        horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, hk)
        content = cv2.subtract(ink, horiz)
        cols = np.any(content > 0, axis=0)
        pad = max(8, int(staff.staff_space * 2))

        if np.any(cols):
            xs = np.where(cols)[0]
            staff.x_start = max(0, int(xs[0]) - pad)
            staff.x_end = min(w, int(xs[-1]) + 1 + pad)
            continue

        cols_all = np.any(ink > 0, axis=0)
        if np.any(cols_all):
            xs2 = np.where(cols_all)[0]
            staff.x_start = int(xs2[0])
            staff.x_end = int(xs2[-1]) + 1
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
    stem_margin: float = 2.2,
    chord_min_h: int = 10,
) -> list[System]:
    """Pair each staff with the chord-text region above it.

    Music region = top_line - margin … bottom_line + margin
    Chord region = a thin band *immediately above* the staff (excludes page
    titles / headers that sit far above the first system).
    """
    H, W = grayscale.shape[:2]
    systems: list[System] = []
    pad_x = max(15, int(W * 0.015))

    for idx, staff in enumerate(staves):
        margin = int(staff.staff_space * stem_margin)

        music_y0 = max(0, staff.top - margin)
        music_y1 = min(H, staff.bottom + margin)
        music_x0 = max(0, staff.x_start - pad_x)
        music_x1 = min(W, staff.x_end + pad_x)

        # Chord band: the gap between the previous system's music crop and this
        # staff's first line.  Chord symbols sit in this gap, often very close
        # to (or inside the margin of) the top staff line.
        #
        # chord_y1 = staff.top  (not music_y0) so symbols in the margin zone
        # above line-1 are captured.  The first system uses a shallow cap so
        # the page title / composer header is excluded.
        if idx > 0:
            prev = staves[idx - 1]
            prev_bottom = min(H, prev.bottom + int(prev.staff_space * stem_margin))
            chord_y0 = prev_bottom
        else:
            # First system: look up to 7 staff-spaces above.  Chord text sits
            # ~3-4 spaces above the staff; the page title is typically 10+
            # spaces away so this cap excludes it.
            chord_y0 = max(0, staff.top - max(int(staff.staff_space * 7), 80))
        chord_y1 = staff.top

        music_gray = grayscale[music_y0:music_y1, music_x0:music_x1]
        music_bin = (binary[music_y0:music_y1, music_x0:music_x1] > 0).astype(np.uint8)

        # Chord crop — take the full inter-system band bounded by the music
        # x-extent.  EasyOCR's own detector finds text anywhere in the band,
        # which is more robust than trying to pick a row-cluster here (cluster
        # detection is defeated by binder holes and page borders).
        chord_bbox = None
        chord_gray = None
        chord_bin = None

        if chord_y1 - chord_y0 >= chord_min_h:
            chord_gray = grayscale[chord_y0:chord_y1, music_x0:music_x1]
            chord_bin = (binary[chord_y0:chord_y1, music_x0:music_x1] > 0).astype(np.uint8)
            chord_bbox = (music_x0, chord_y0, music_x1 - music_x0, chord_y1 - chord_y0)

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
# Per-crop staff validation (CRNN input quality)
# ---------------------------------------------------------------------------

def local_primary_staff_lines(music_binary: np.ndarray) -> list[int] | None:
    """Re-detect five staff lines inside a music strip (crop coordinates).

    Page-level grouping can occasionally mis-assign regions; the CRNN was
    trained on crops that always contain a real staff.  If this returns
    *None*, the strip is unlikely to be readable music.
    """
    if music_binary.size == 0 or music_binary.shape[0] < 12:
        return None
    line_mask = _extract_line_mask(music_binary)
    line_ys = _line_row_centroids(line_mask)
    w = music_binary.shape[1]
    staves_local = _group_into_staves(line_ys, w)
    if not staves_local:
        return None
    if len(staves_local) == 1:
        return staves_local[0].line_ys
    hc = music_binary.shape[0] / 2.0

    def centre_dist(st: Staff) -> float:
        c = (st.line_ys[0] + st.line_ys[-1]) / 2.0
        return abs(c - hc)

    best = min(staves_local, key=centre_dist)
    return best.line_ys


def music_strip_has_valid_staff(music_binary: np.ndarray) -> bool:
    """True if *music_binary* contains a plausible five-line staff."""
    return local_primary_staff_lines(music_binary) is not None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _detect_systems_one_pass(
    grayscale: np.ndarray,
    binary: np.ndarray,
    *,
    spacing_tolerance: float,
    line_merge_dist: int,
    ink_threshold_frac: float,
) -> tuple[list[System], list[int]]:
    """One staff-detection attempt; returns (systems, line_ys)."""
    _H, W = grayscale.shape[:2]
    line_mask = _extract_line_mask(binary)
    line_ys = _line_row_centroids(
        line_mask,
        merge_dist=line_merge_dist,
        ink_threshold_frac=ink_threshold_frac,
    )
    if len(line_ys) < 5:
        return [], line_ys

    staves = _group_into_staves(line_ys, W, spacing_tolerance=spacing_tolerance)
    if not staves:
        return [], line_ys

    _refine_x_bounds(staves, binary)
    systems = _build_systems(staves, grayscale, binary)
    return systems, line_ys


def detect_systems(grayscale: np.ndarray, binary: np.ndarray) -> list[System]:
    """Detect staff systems and their chord regions on a preprocessed page.

    Returns an ordered list of System objects (top → bottom).  Returns an
    empty list only when fewer than 5 horizontal lines can be found (i.e.
    the image contains no recognisable staff).

    Retries with looser line clustering and spacing tolerance when wavy or
    thick hand-drawn staves fail the default morphological pass.
    """
    attempts: list[tuple[float, int, float]] = [
        (0.35, 3, 0.15),
        (0.48, 5, 0.11),
        (0.55, 6, 0.085),
    ]

    last_line_count = 0
    fallback_systems: list[System] | None = None

    for spacing_tol, merge_dist, ink_thr in attempts:
        systems, line_ys = _detect_systems_one_pass(
            grayscale,
            binary,
            spacing_tolerance=spacing_tol,
            line_merge_dist=merge_dist,
            ink_threshold_frac=ink_thr,
        )
        last_line_count = len(line_ys)
        log.info(
            "Staff-line pass (tol=%.2f merge=%d ink=%.3f): %d lines → %d raw system(s)",
            spacing_tol, merge_dist, ink_thr, len(line_ys), len(systems),
        )

        if not systems:
            continue

        validated = [s for s in systems if music_strip_has_valid_staff(s.music_binary)]
        if validated:
            if len(validated) < len(systems):
                log.info(
                    "Dropped %d region(s) without a local five-line staff",
                    len(systems) - len(validated),
                )
            systems = validated
            log.info(
                "Detected %d staff system(s); staff-space ≈ %.1f px",
                len(systems),
                float(np.mean([s.staff.staff_space for s in systems])),
            )
            return systems

        fallback_systems = systems

    if fallback_systems:
        log.warning(
            "No strip passed local staff validation — keeping all %d page-level system(s)",
            len(fallback_systems),
        )
        log.info(
            "Detected %d staff system(s); staff-space ≈ %.1f px",
            len(fallback_systems),
            float(np.mean([s.staff.staff_space for s in fallback_systems])),
        )
        return fallback_systems

    log.info("Staff-line detection: %d horizontal lines found", last_line_count)
    return []
