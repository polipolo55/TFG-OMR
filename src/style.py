"""
tfg_omr.style
=============
Central styling module for TFG-OMR.

Usage (notebooks / scripts)
---------------------------
    import sys
    sys.path.insert(0, "../src")   # or adjust to your relative path
    import style
    style.apply()

All colour names defined here are mirrored in docs/main/tfg.sty for LaTeX figures.
"""

from __future__ import annotations

import matplotlib as mpl

# ── Palette ────────────────────────────────────────────────────────────────────
# Role-based names make it easy to swap the whole palette in one place.

PALETTE: dict[str, str] = {
    # Primary accent — used for the "main" or "best" series
    "primary": "#1565C0",  # deep blue
    "primary_light": "#42A5F5",  # light blue
    # Secondary accent — used for contrast / "second" series
    "secondary": "#E65100",  # deep orange
    "secondary_light": "#FFA726",  # light orange
    # Tertiary — additional series, e.g. third model variant
    "tertiary": "#2E7D32",  # deep green
    "tertiary_light": "#81C784",  # light green
    # Highlight — warnings, errors, attention
    "highlight": "#880E4F",  # deep pink / magenta
    "highlight_light": "#F06292",  # light pink
    # Neutral tones
    "neutral_dark": "#212121",  # near-black
    "neutral_mid": "#757575",  # medium grey
    "neutral_light": "#E0E0E0",  # light grey (grid lines, borders)
    # Background helpers
    "bg_paper": "#FFFFFF",
    "bg_figure": "#FAFAFA",
}

# Ordered sequence for multi-series plots (cycle through these by default)
COLOR_CYCLE: list[str] = [
    PALETTE["primary"],
    PALETTE["secondary"],
    PALETTE["tertiary"],
    PALETTE["highlight"],
    PALETTE["primary_light"],
    PALETTE["secondary_light"],
    PALETTE["tertiary_light"],
    PALETTE["highlight_light"],
]

# Convenience aliases
C = PALETTE  # short handle:  style.C["primary"]


# ── rcParams ───────────────────────────────────────────────────────────────────

_RC: dict = {
    # Figure
    "figure.dpi": 150,
    "figure.facecolor": PALETTE["bg_paper"],
    "figure.edgecolor": PALETTE["bg_paper"],
    # Axes
    "axes.facecolor": PALETTE["bg_figure"],
    "axes.edgecolor": PALETTE["neutral_mid"],
    "axes.labelcolor": PALETTE["neutral_dark"],
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": mpl.cycler(color=COLOR_CYCLE),  # type: ignore[attr-defined]
    "axes.grid": True,
    "axes.axisbelow": True,
    # Grid
    "grid.color": PALETTE["neutral_light"],
    "grid.linewidth": 0.8,
    "grid.alpha": 0.7,
    # Lines & markers
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    # Fonts  (LaTeX-compatible, no full LaTeX engine required)
    "font.family": "serif",
    "font.size": 9,
    "mathtext.fontset": "cm",
    # Legend
    "legend.fontsize": 8,
    "legend.framealpha": 0.85,
    "legend.edgecolor": PALETTE["neutral_light"],
    # Ticks
    "xtick.color": PALETTE["neutral_dark"],
    "ytick.color": PALETTE["neutral_dark"],
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # Saving
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": PALETTE["bg_paper"],
}


def apply() -> None:
    """Apply TFG-OMR rcParams to the current matplotlib session."""
    mpl.rcParams.update(_RC)


def reset() -> None:
    """Restore matplotlib defaults (useful at the end of a notebook section)."""
    mpl.rcParams.update(mpl.rcParamsDefault)


# ── Bar-chart helper ───────────────────────────────────────────────────────────


def bar_colors(n: int, *, start: str = "primary") -> list[str]:
    """
    Return *n* colours from the cycle, starting at *start* role.

    Example
    -------
        bars = ax.bar(x, values, color=style.bar_colors(4))
    """
    idx = list(PALETTE.keys()).index(start) if start in PALETTE else 0
    colours = list(PALETTE.values())
    return [colours[(idx + i) % len(colours)] for i in range(n)]
