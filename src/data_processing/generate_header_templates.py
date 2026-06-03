# src/data_processing/generate_header_templates.py
"""
generate_header_templates.py
============================
Offline script: prerender 120 header-strip PNG templates (15 key signatures ×
8 time signatures) using LilyPond + LilyJAZZ.  The resulting images are stored
in ``data/header_templates/`` and loaded at inference time by
``src/omr_pipeline/header_injector.py`` to prepend clef+key+time glyphs to
continuation staff images before the CRNN.

Run once before inference (included in ``cli.py pipeline`` stage 3)::

    poetry run python src/cli.py generate-header-templates
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from CRNN_CTC.lilypond_render import KEY_LY, crop_content, run_lilypond

log = logging.getLogger(__name__)

# All key signatures the system supports (fifths -7 … +7)
_ALL_FIFTHS: tuple[int, ...] = tuple(range(-7, 8))  # 15 keys

# All time signatures (must match grammar_fix._COMMON_TIME_SIGS)
_ALL_TIMES: tuple[tuple[int, int], ...] = (
    (4, 4), (3, 4), (2, 4), (2, 2),
    (6, 8), (6, 4), (5, 4), (12, 8),
)

_DPI = 200
_STAFF_SIZE = 17  # matches training renders

_LY_TEMPLATE = r"""
\version "2.26.0"
#(set-global-staff-size {staff_size})
\include "lilyjazz.ily"
\header {{ tagline = ##f }}
\paper {{
  indent = 0
  ragged-right = ##t
  top-margin = 6\mm
  bottom-margin = 6\mm
  left-margin = 4\mm
  right-margin = 4\mm
  paper-height = 55\mm
}}
\score {{
  \new Staff {{
    \clef treble
    {key_cmd}
    \time {beats}/{beat_type}
    s1
  }}
  \layout {{ \context {{ \Score \omit BarNumber }} }}
}}
""".strip()


def template_filename(fifths: int, beats: int, beat_type: int) -> str:
    """Canonical filename for a given key+time combination."""
    return f"key_{fifths}_time_{beats}_{beat_type}.png"


def _render_one(fifths: int, beats: int, beat_type: int, output_dir: Path) -> bool:
    """Render a single template. Returns True on success."""
    out_path = output_dir / template_filename(fifths, beats, beat_type)
    key_cmd = KEY_LY[fifths]
    ly_src = _LY_TEMPLATE.format(
        staff_size=_STAFF_SIZE,
        key_cmd=key_cmd,
        beats=beats,
        beat_type=beat_type,
    )
    with tempfile.TemporaryDirectory(prefix="tmpl_") as tmp:
        png = run_lilypond(ly_src, f"tmpl_{fifths}_{beats}_{beat_type}", Path(tmp), dpi=_DPI)
        if png is None:
            log.warning("LilyPond failed for key=%d time=%d/%d", fifths, beats, beat_type)
            return False
        try:
            img = np.array(Image.open(png).convert("L"))
            cropped = crop_content(img)
        except Exception as exc:
            log.warning("Crop failed for key=%d time=%d/%d: %s", fifths, beats, beat_type, exc)
            return False
        if cropped.size == 0 or np.all(cropped == 255):
            log.warning("Empty render for key=%d time=%d/%d", fifths, beats, beat_type)
            return False
        Image.fromarray(cropped).save(out_path)
    return True


def generate_all_templates(
    output_dir: Path = Path("data/header_templates"),
    force: bool = False,
) -> dict[str, int]:
    """Generate all 120 templates. Returns {'ok': N, 'skip': N, 'fail': N}."""
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"ok": 0, "skip": 0, "fail": 0}
    total = len(_ALL_FIFTHS) * len(_ALL_TIMES)
    done = 0
    for fifths in _ALL_FIFTHS:
        for beats, beat_type in _ALL_TIMES:
            path = output_dir / template_filename(fifths, beats, beat_type)
            if path.exists() and not force:
                counts["skip"] += 1
            elif _render_one(fifths, beats, beat_type, output_dir):
                counts["ok"] += 1
            else:
                counts["fail"] += 1
            done += 1
            if done % 20 == 0 or done == total:
                log.info(
                    "Templates: %d/%d (ok=%d skip=%d fail=%d)",
                    done, total, counts["ok"], counts["skip"], counts["fail"],
                )
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    generate_all_templates(force="--force" in sys.argv)
