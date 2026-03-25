"""
lilypond_render.py
==================
LMX token → LilyPond → PNG rendering utilities.

Shared rendering back-end used by:

* **Evaluation notebook** — render model predictions to engraved images.
* **Dataset generation** (``generate_realbook.py``) — re-render PrIMuS
  samples with LilyJAZZ styling.

All clef / key / duration look-up tables and the LilyPond subprocess
pipeline live here so there is a **single source of truth**.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# ── Shared look-up tables ──────────────────────────────────────────────────────

CLEF_LY: dict[str, str] = {
    "G2": "treble",
    "G2/8": "treble_8",
    "G1": "french",
    "F4": "bass",
    "F3": "varbaritone",
    "C1": "soprano",
    "C2": "mezzosoprano",
    "C3": "alto",
    "C4": "tenor",
    "C5": "baritone",    # rare PrIMuS token; LilyPond has no exact C5 clef
    "F5": "subbass",
}
"""Map PrIMuS / LMX clef identifiers to LilyPond clef names."""

# Same IDs as ``filter_unwanted_clefs`` in ``CRNN_CTC.dataset`` — soprano /
# mezzo / varbaritone clefs that are rare in jazz and confuse the CRNN.
# ``generate_realbook`` / ``semantic_to_lmx`` map these to G2 so PNG and LMX
# stay matched (absolute pitches unchanged; LilyPond uses ledger lines).
CLEF_IDS_NORMALIZE_TO_G2: frozenset[str] = frozenset({"C1", "C2", "F3"})


def normalize_clef_id_for_lead_sheet(clef_id: str) -> str:
    """Return ``G2`` for clefs we drop from raw training; otherwise *clef_id*."""
    return "G2" if clef_id in CLEF_IDS_NORMALIZE_TO_G2 else clef_id

KEY_LY: dict[int, str] = {
    -7: r"\key ces \major", -6: r"\key ges \major", -5: r"\key des \major",
    -4: r"\key aes \major", -3: r"\key ees \major", -2: r"\key bes \major",
    -1: r"\key f \major",    0: r"\key c \major",    1: r"\key g \major",
     2: r"\key d \major",    3: r"\key a \major",    4: r"\key e \major",
     5: r"\key b \major",    6: r"\key fis \major",  7: r"\key cis \major",
}
"""Map *fifths* count (LMX ``key:fifths:N``) to a LilyPond ``\\key`` command."""

DUR_LY: dict[str, str] = {
    "whole": "1", "half": "2", "quarter": "4", "eighth": "8",
    "16th": "16", "32nd": "32", "64th": "64", "128th": "128",
    "256th": "256", "1024th": "1024", "breve": r"\breve",
    "longa": r"\longa",
}
"""Map LMX duration token names to LilyPond duration strings."""

_SHARP_ORDER = list("FCGDAEB")
_FLAT_ORDER  = list("BEADGCF")
_PITCH_STEPS = set("ABCDEFG")
_DURATIONS   = set(DUR_LY)

LY_TEMPLATE = r"""
\version "2.24.0"
\include "lilyjazz.ily"
\header {{ tagline = ##f }}
\paper {{
  indent = 0
  ragged-right = ##t
  top-margin = 6\mm
  bottom-margin = 6\mm
  left-margin = 8\mm
  right-margin = 8\mm
  paper-height = 55\mm
}}
\score {{
  \new Staff {{ {music} }}
  \layout {{ \context {{ \Score \omit BarNumber }} }}
}}
""".strip()
"""Default LilyPond template.  Use ``LY_TEMPLATE.format(music=...)``."""


# ── LMX tokens → LilyPond music body ──────────────────────────────────────────

def lmx_to_lilypond(tokens: list[str]) -> str:
    """Convert a list of LMX tokens to a LilyPond music body string.

    Raises :class:`ValueError` for unknown clefs or key signatures so that
    data-quality problems surface immediately rather than being silently
    papered over.
    """
    key_sharps: set[str] = set()
    key_flats:  set[str] = set()
    lily: list[str] = []
    beats_val: str | None = None
    beat_type_val: str | None = None
    pending: dict | None = None
    cur_beats: int = 4          # numerator of current time signature
    cur_beat_type: int = 4      # denominator of current time signature

    def _flush() -> None:
        nonlocal pending
        if pending is None:
            return
        kind = pending["kind"]
        dur  = pending.get("dur")
        if not dur:
            pending = None
            return
        dots = "." * pending.get("dots", 0)

        if kind == "rest":
            lily.append(f"r{dur}{dots}")
        elif kind == "forward":
            lily.append(f"s{dur}{dots}")
        elif kind == "note":
            step    = pending.get("step")
            octave  = pending.get("octave")
            if step is None or octave is None:
                # Incomplete note — skip
                pending = None
                return
            letter  = step.lower()
            acc     = pending.get("acc")

            if acc == "flat":
                letter += "es"
            elif acc == "sharp":
                letter += "is"
            elif acc == "natural":
                pass  # no pitch suffix — but we add '!' after octave marks below
            else:
                # Implicit: apply key-signature accidentals
                if step in key_sharps:
                    letter += "is"
                elif step in key_flats:
                    letter += "es"

            oct_off = octave - 3
            if oct_off > 0:
                letter += "'" * oct_off
            elif oct_off < 0:
                letter += "," * (-oct_off)

            # '!' after octave marks forces an explicit natural in LilyPond,
            # overriding whatever the key signature would impose.
            force_natural = pending.get("acc") == "natural"
            tok_str = f"{letter}{'!' if force_natural else ''}{dur}{dots}"
            if pending.get("tie"):
                tok_str += "~"
            lily.append(tok_str)
        pending = None

    for tok in tokens:
        if tok == "measure":
            _flush()
            if lily and lily[-1] != "|":
                lily.append("|")

        elif tok.startswith("key:fifths:"):
            _flush()
            fifths = int(tok.split(":")[-1])
            fifths = max(-7, min(7, fifths))
            if fifths > 0:
                key_sharps = set(_SHARP_ORDER[:fifths]); key_flats = set()
            elif fifths < 0:
                key_flats = set(_FLAT_ORDER[:abs(fifths)]); key_sharps = set()
            else:
                key_sharps = set(); key_flats = set()
            if fifths not in KEY_LY:
                raise ValueError(f"Unsupported fifths value: {fifths}")
            lily.append(KEY_LY[fifths])

        elif tok == "time":
            _flush()

        elif tok.startswith("beats:"):
            beats_val = tok.split(":")[1]
            if beats_val and beat_type_val:
                cur_beats = int(beats_val)
                cur_beat_type = int(beat_type_val)
                lily.append(rf"\time {beats_val}/{beat_type_val}")
                beats_val = beat_type_val = None

        elif tok.startswith("beat-type:"):
            beat_type_val = tok.split(":")[1]
            if beats_val and beat_type_val:
                cur_beats = int(beats_val)
                cur_beat_type = int(beat_type_val)
                lily.append(rf"\time {beats_val}/{beat_type_val}")
                beats_val = beat_type_val = None

        elif tok.startswith("clef:"):
            _flush()
            clef_id = tok.split(":")[1]
            if clef_id not in CLEF_LY:
                raise ValueError(
                    f"Unknown clef {clef_id!r} in token {tok!r}. "
                    f"Add it to CLEF_LY in lilypond_render.py."
                )
            lily.append(rf"\clef {CLEF_LY[clef_id]}")

        elif tok.startswith("pitch:"):
            _flush()
            step = tok.split(":")[1]
            if step in _PITCH_STEPS:
                pending = {"kind": "note", "step": step, "octave": None,
                           "dur": None, "acc": None, "dots": 0, "tie": False}

        elif tok.startswith("octave:"):
            if pending and pending["kind"] == "note" and pending.get("octave") is None:
                try:
                    pending["octave"] = int(tok.split(":")[1])
                except ValueError:
                    pass

        elif tok in _DURATIONS:
            if pending and pending.get("dur") is None:
                pending["dur"] = DUR_LY[tok]

        elif tok == "rest":
            _flush()
            pending = {"kind": "rest", "dur": None, "dots": 0}

        elif tok == "rest:measure":
            # Cancel any preceding bare `rest` token (it was just a marker).
            # Emit a full-measure rest scaled to the current time signature.
            pending = None
            lily.append(f"R1*{cur_beats}/{cur_beat_type}")

        elif tok == "dot":
            if pending:
                pending["dots"] = pending.get("dots", 0) + 1

        elif tok in ("flat", "sharp", "natural"):
            if pending and pending["kind"] == "note":
                pending["acc"] = tok

        elif tok == "tied:start":
            if pending and pending["kind"] == "note":
                pending["tie"] = True

        elif tok == "forward":
            _flush()
            pending = {"kind": "forward", "dur": None, "dots": 0}

        elif tok == "tuplet:start":
            _flush()
            lily.append(r"\tuplet 3/2 {")

        elif tok == "tuplet:stop":
            _flush()
            lily.append("}")

    _flush()

    # Clean consecutive / leading / trailing barlines
    cleaned: list[str] = []
    for part in lily:
        if part == "|" and cleaned and cleaned[-1] == "|":
            continue
        cleaned.append(part)
    if cleaned and cleaned[-1] == "|":
        cleaned.pop()
    if cleaned and cleaned[0] == "|":
        cleaned.pop(0)

    return " ".join(cleaned)


# ── Image utilities ────────────────────────────────────────────────────────────

def crop_content(img: np.ndarray, pad: int = 6) -> np.ndarray:
    """Crop a white-background grayscale image to its ink bounding box.

    Parameters
    ----------
    img : np.ndarray
        Grayscale uint8 image (H × W).
    pad : int
        Pixels of white margin to keep around the ink.
    """
    ink = img < 250
    rows_mask = np.any(ink, axis=1)
    cols_mask = np.any(ink, axis=0)
    if not rows_mask.any():
        return img
    r0, r1 = np.where(rows_mask)[0][[0, -1]]
    c0, c1 = np.where(cols_mask)[0][[0, -1]]
    r0 = max(0, r0 - pad); r1 = min(img.shape[0] - 1, r1 + pad)
    c0 = max(0, c0 - pad); c1 = min(img.shape[1] - 1, c1 + pad)
    return img[r0:r1 + 1, c0:c1 + 1]


# ── LilyPond rendering pipeline ───────────────────────────────────────────────

_DEFAULT_TMPDIR: Path | None = None


def _get_tmpdir() -> Path:
    """Lazily create a module-level temporary directory."""
    global _DEFAULT_TMPDIR
    if _DEFAULT_TMPDIR is None:
        _DEFAULT_TMPDIR = Path(tempfile.mkdtemp(prefix="omr_ly_"))
    return _DEFAULT_TMPDIR


def run_lilypond(
    ly_source: str,
    name: str,
    out_dir: Path,
    *,
    dpi: int = 200,
    timeout: int = 30,
) -> Path | None:
    """Write *ly_source* to ``out_dir/name.ly``, run LilyPond, return PNG path.

    Returns *None* on render failure or timeout.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ly_path  = out_dir / f"{name}.ly"
    png_path = out_dir / f"{name}.png"
    ly_path.write_text(ly_source, encoding="utf-8")
    try:
        result = subprocess.run(
            ["lilypond", f"-dresolution={dpi}", "--png",
             "-o", str(out_dir / name), str(ly_path)],
            capture_output=True, text=True, cwd=str(out_dir), timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        log.debug("LilyPond timeout for %s", name)
        return None
    if result.returncode != 0:
        log.debug("LilyPond failed for %s: %s", name, result.stderr[-500:])
        return None
    if not png_path.exists():
        # Multi-page output sometimes uses name-1.png, etc.
        candidates = sorted(out_dir.glob(f"{name}*.png"))
        if candidates:
            return candidates[0]
        return None
    return png_path


def render_ly(
    ly_source: str,
    name: str,
    *,
    out_dir: Path | None = None,
    dpi: int = 200,
    timeout: int = 15,
) -> np.ndarray | None:
    """Render *ly_source* via LilyPond, crop, return grayscale array or *None*."""
    out_dir = out_dir or _get_tmpdir()
    png = run_lilypond(ly_source, name, out_dir, dpi=dpi, timeout=timeout)
    if png is None:
        return None
    return crop_content(np.array(Image.open(png).convert("L")))


def render_tokens(
    tokens: list[str],
    name: str = "pred",
    *,
    out_dir: Path | None = None,
    template: str = LY_TEMPLATE,
    dpi: int = 200,
) -> np.ndarray | None:
    """Render LMX tokens via LilyPond.  Returns cropped grayscale or *None*."""
    try:
        music = lmx_to_lilypond(tokens)
    except ValueError:
        log.debug("lmx_to_lilypond failed for %s", name, exc_info=True)
        return None
    if not music.strip():
        return None
    return render_ly(template.format(music=music), name, out_dir=out_dir, dpi=dpi)
