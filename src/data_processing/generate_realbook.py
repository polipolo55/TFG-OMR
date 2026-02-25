"""
generate_realbook.py
====================
Re-render PrIMuS monophonic staff lines using LilyPond + LilyJAZZ to produce
a Real Book-styled dataset.

Output structure mirrors PrIMuS:
    data/realbook_primus/{sample_id}/{sample_id}.png
    data/realbook_primus/{sample_id}/{sample_id}.semantic   (copied from PrIMuS)
    data/realbook_primus/{sample_id}/{sample_id}.agnostic   (copied from PrIMuS)
    data/realbook_primus/{sample_id}/{sample_id}.ly         (LilyPond source, debug)

Usage:
    poetry run python src/data_processing/generate_realbook.py --limit 10
    poetry run python src/data_processing/generate_realbook.py --source data/primus --output data/realbook_primus
"""

import argparse
import logging
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PrIMuS semantic → LilyPond conversion
#
# The semantic file is the ground truth for pitch spelling.  It already
# encodes the correct accidentals (e.g. "Bb5", "Eb5") so we never need to
# re-derive them from the key signature.  This avoids the music21 MEI pitch
# parsing bug where all notes are stored as absolute concert pitches and
# LilyPond then prints spurious naturals for every key-signature note.
# ---------------------------------------------------------------------------

_STEP_LILY   = {"C": "c", "D": "d", "E": "e", "F": "f",
                "G": "g", "A": "a", "B": "b"}
_ACC_LILY    = {"b": "es", "bb": "eses", "#": "is", "x": "isis", "": ""}
_DUR_LILY    = {"breve": r"\breve", "whole": "1", "half": "2", "quarter": "4",
                "eighth": "8", "sixteenth": "16", "32nd": "32", "64th": "64"}
_CLEF_LILY   = {"G2": "treble", "G2/8": "treble_8",
                "F4": "bass",   "C3": "alto", "C4": "tenor"}


def _parse_pitch(pitch_str: str) -> str:
    """
    Convert a PrIMuS semantic pitch string (e.g. 'Bb5', 'F#4', 'C6') to a
    LilyPond pitch token (e.g. 'bes''', 'fis\'', 'c\'\'\'').
    """
    step = pitch_str[0]
    # octave is the trailing digit(s)
    m = re.match(r"^([A-G])(b{1,2}|#{1,2}|x?)(\d+)$", pitch_str)
    if not m:
        raise ValueError(f"Cannot parse pitch: {pitch_str!r}")
    step, acc_str, oct_str = m.group(1), m.group(2), m.group(3)

    # normalise double-sharp written as ## → x
    if acc_str == "##":
        acc_str = "x"
    # normalise double-flat written as ## is impossible; bb is the convention

    lily_step = _STEP_LILY[step]
    lily_acc  = _ACC_LILY.get(acc_str, "")
    octave    = int(oct_str)

    # LilyPond middle C = c'  (octave 4)
    # c   = C3, c'  = C4, c'' = C5, c, = C2, c,, = C1
    oct_offset = octave - 3
    if oct_offset > 0:
        oct_marks = "'" * oct_offset
    elif oct_offset < 0:
        oct_marks = "," * (-oct_offset)
    else:
        oct_marks = ""

    return lily_step + lily_acc + oct_marks


def _parse_duration(dur_str: str) -> str:
    """
    Convert a PrIMuS duration string (e.g. 'quarter', 'eighth.', 'half..') to
    a LilyPond duration string (e.g. '4', '8.', '2..').
    """
    # Strip trailing dots
    stripped = dur_str.rstrip(".")
    dots     = "." * (len(dur_str) - len(stripped))
    lily_dur = _DUR_LILY.get(stripped)
    if lily_dur is None:
        raise ValueError(f"Unknown duration: {dur_str!r}")
    return lily_dur + dots


def _parse_key(ks_str: str) -> str:
    """
    Convert a PrIMuS key string (e.g. 'EbM', 'F#m', 'C') to a LilyPond
    \\key statement.
    """
    if ks_str.endswith("M"):
        mode, root = "major", ks_str[:-1]
    elif ks_str.endswith("m"):
        mode, root = "minor", ks_str[:-1]
    else:
        mode, root = "major", ks_str   # bare 'C' = C major

    step    = root[0].lower()
    acc_str = root[1:]
    acc     = _ACC_LILY.get(acc_str, "")
    return rf"\key {step}{acc} \{mode}"


def semantic_to_lily_music(semantic_path: Path) -> str:
    """
    Parse a PrIMuS .semantic file and return the LilyPond music body string
    (everything inside \\new Staff { ... }).

    Uses the semantic file as single source of truth for pitch spelling so
    accidentals always match the original PrIMuS engraving.
    """
    text   = semantic_path.read_text(encoding="utf-8")
    tokens = text.split()

    lily_tokens: list[str] = []
    pending_tie = False   # append ~ to the next note

    for tok in tokens:
        try:
            if tok.startswith("clef-"):
                clef_id  = tok[len("clef-"):]
                lily_clef = _CLEF_LILY.get(clef_id, "treble")
                lily_tokens.append(rf"\clef {lily_clef}")

            elif tok.startswith("keySignature-"):
                ks_str = tok[len("keySignature-"):]
                lily_tokens.append(_parse_key(ks_str))

            elif tok.startswith("timeSignature-"):
                ts_str = tok[len("timeSignature-"):]
                if ts_str == "C":
                    lily_tokens.append(r"\time 4/4")
                elif ts_str in ("C/", "C|"):
                    lily_tokens.append(r"\time 2/2")
                else:
                    lily_tokens.append(rf"\time {ts_str}")

            elif tok.startswith("note-"):
                # format: note-{Pitch}_{duration}
                inner      = tok[len("note-"):]
                pitch_str, dur_str = inner.split("_", 1)
                lily_pitch = _parse_pitch(pitch_str)
                lily_dur   = _parse_duration(dur_str)
                note_tok   = lily_pitch + lily_dur
                if pending_tie:
                    lily_tokens[-1] += "~"
                    pending_tie = False
                lily_tokens.append(note_tok)

            elif tok.startswith("rest-"):
                dur_str  = tok[len("rest-"):]
                lily_dur = _parse_duration(dur_str)
                lily_tokens.append("r" + lily_dur)

            elif tok == "barline":
                lily_tokens.append("|")

            elif tok == "tie":
                # tie connects the most-recently emitted note to the next one
                pending_tie = True

            elif tok.startswith("gracenote-"):
                # grace notes: render as acciaccatura
                inner     = tok[len("gracenote-"):]
                pitch_str = inner.split("_")[0]
                lily_pitch = _parse_pitch(pitch_str)
                lily_tokens.append(rf"\acciaccatura {lily_pitch}8")

            # anything else (multirest, fermata markers, etc.) is skipped

        except Exception as exc:
            log.debug("Skipping token %r: %s", tok, exc)
            continue

    return " ".join(lily_tokens)


# ---------------------------------------------------------------------------
# LilyPond template
# ---------------------------------------------------------------------------

LILY_TEMPLATE = r"""
\version "2.24.0"
\include "lilyjazz.ily"

\header {{
  tagline = ##f
}}

\paper {{
  indent = 0
  ragged-right = ##t
  top-margin    = 6\mm
  bottom-margin = 6\mm
  left-margin   = 8\mm
  right-margin  = 8\mm
  paper-height  = 55\mm
}}

\score {{
  \new Staff {{
    {music}
  }}
  \layout {{
    \context {{
      \Score
      \omit BarNumber
    }}
  }}
}}
""".strip()


def make_lily_source(music_body: str) -> str:
    return LILY_TEMPLATE.format(music=music_body)


# ---------------------------------------------------------------------------
# Render & crop
# ---------------------------------------------------------------------------

def render_lily(ly_source: str, work_dir: Path, basename: str, dpi: int = 200) -> Path | None:
    """
    Write `ly_source` to `work_dir/{basename}.ly`, run LilyPond, and return
    the path to the generated PNG.  Returns None on failure.
    """
    ly_path  = work_dir / f"{basename}.ly"
    png_path = work_dir / f"{basename}.png"

    ly_path.write_text(ly_source, encoding="utf-8")

    result = subprocess.run(
        ["lilypond", f"-dresolution={dpi}", "--png", "-o", str(work_dir / basename), str(ly_path)],
        capture_output=True,
        text=True,
        cwd=str(work_dir),
    )

    if result.returncode != 0:
        log.debug("LilyPond stderr: %s", result.stderr[-500:])
        return None

    if not png_path.exists():
        # LilyPond sometimes appends -1.png for multi-page output
        candidates = sorted(work_dir.glob(f"{basename}*.png"))
        if candidates:
            return candidates[0]
        return None

    return png_path


def crop_content(png_path: Path, padding: int = 6) -> np.ndarray:
    """
    Load a white-background PNG rendered by LilyPond and crop to the
    non-white bounding box, adding `padding` pixels on each side.
    Returns the cropped image as a NumPy array (H×W, uint8 grayscale).
    """
    img = np.array(Image.open(png_path).convert("L"))  # grayscale

    # Mask of ink pixels (not white)
    ink = img < 250
    rows = np.any(ink, axis=1)
    cols = np.any(ink, axis=0)

    if not rows.any():
        return img  # blank page — return as-is

    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]

    r0 = max(0, r0 - padding)
    r1 = min(img.shape[0] - 1, r1 + padding)
    c0 = max(0, c0 - padding)
    c1 = min(img.shape[1] - 1, c1 + padding)

    return img[r0 : r1 + 1, c0 : c1 + 1]


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------

def process_sample(sample_dir: Path, output_dir: Path, dpi: int = 200) -> bool:
    """
    Process one PrIMuS sample directory.
    Returns True on success, False on failure.
    """
    sample_id    = sample_dir.name
    sem_path     = sample_dir / f"{sample_id}.semantic"

    if not sem_path.exists():
        log.warning("No .semantic file in %s — skipping", sample_dir)
        return False

    out_sample = output_dir / sample_id
    out_png    = out_sample / f"{sample_id}.png"

    if out_png.exists():
        return True  # already processed

    # Parse semantic → LilyPond music body
    try:
        music_body = semantic_to_lily_music(sem_path)
    except Exception as exc:
        log.warning("Semantic parse failed for %s: %s", sample_id, exc)
        return False

    if not music_body.strip():
        log.warning("Empty music body for %s — skipping", sample_id)
        return False

    ly_source = make_lily_source(music_body)

    # Render in a temp directory, then move outputs
    with tempfile.TemporaryDirectory(prefix="realbook_") as tmp:
        tmp_dir = Path(tmp)
        png_path = render_lily(ly_source, tmp_dir, sample_id, dpi=dpi)
        if png_path is None:
            log.warning("LilyPond render failed for %s", sample_id)
            return False

        # Crop to staff content
        try:
            cropped = crop_content(png_path)
        except Exception as exc:
            log.warning("Crop failed for %s: %s", sample_id, exc)
            return False

        out_sample.mkdir(parents=True, exist_ok=True)

        # Save cropped PNG
        Image.fromarray(cropped).save(out_png)

        # Copy LilyPond source (useful for debugging/regeneration)
        shutil.copy(tmp_dir / f"{sample_id}.ly", out_sample / f"{sample_id}.ly")

    # Copy annotations from PrIMuS unchanged (same symbolic content)
    for ext in (".semantic", ".agnostic", ".mid"):
        src = sample_dir / f"{sample_id}{ext}"
        if src.exists():
            shutil.copy(src, out_sample / f"{sample_id}{ext}")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-render PrIMuS lines with LilyJAZZ into a Real Book-styled dataset."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/primus"),
        help="Root of the PrIMuS dataset (default: data/primus)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/realbook_primus"),
        help="Output dataset root (default: data/realbook_primus)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N samples (for testing)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Rendering resolution (default: 200)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show DEBUG messages",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Collect all sample directories (immediate children of package_* dirs)
    sample_dirs = sorted(
        d for d in args.source.rglob("*")
        if d.is_dir() and not d.name.startswith(".")
        and (d / f"{d.name}.semantic").exists()
    )

    if not sample_dirs:
        log.error("No PrIMuS samples found in %s", args.source)
        sys.exit(1)

    if args.limit:
        sample_dirs = sample_dirs[: args.limit]

    log.info("Found %d samples → output: %s (workers: %d)", len(sample_dirs), args.output, args.workers)
    args.output.mkdir(parents=True, exist_ok=True)

    ok = fail = 0
    _worker = partial(process_sample, output_dir=args.output, dpi=args.dpi)
    with multiprocessing.Pool(processes=args.workers) as pool:
        for i, success in enumerate(pool.imap_unordered(_worker, sample_dirs), 1):
            if success:
                ok += 1
            else:
                fail += 1
            if i % 100 == 0 or i == len(sample_dirs):
                log.info("Progress %d/%d  ✓ %d  ✗ %d", i, len(sample_dirs), ok, fail)

    log.info("Done. Success: %d  Failed: %d", ok, fail)


if __name__ == "__main__":
    main()
