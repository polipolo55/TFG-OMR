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
import sys
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# Shared rendering back-end (single source of truth for clef maps, template,
# LilyPond invocation, and image cropping).
from CRNN_CTC.lilypond_render import (
    CLEF_LY,
    LY_TEMPLATE,
    crop_content,
    run_lilypond,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
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
                "eighth": "8", "sixteenth": "16", "32nd": "32", "64th": "64",
                # PrIMuS aliases — some samples spell these out
                "thirty_second": "32", "sixty_fourth": "64",
                # longa (4 whole notes); LilyPond calls it \longa
                "quadruple_whole": r"\longa"}


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


def _is_valid_png(path: Path) -> bool:
    """Return True when path exists and can be decoded as a non-empty PNG."""
    if not path.exists():
        return False
    try:
        if path.stat().st_size == 0:
            return False
        with Image.open(path) as img:
            img.load()
    except (OSError, ValueError, UnidentifiedImageError):
        return False
    return True


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
                if clef_id not in CLEF_LY:
                    raise ValueError(
                        f"Unknown clef token: {tok!r}. "
                        f"Add it to CLEF_LY in CRNN_CTC/lilypond_render.py."
                    )
                lily_clef = CLEF_LY[clef_id]
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
                # format: note-{Pitch}_{duration}  or  note-{Pitch}_{duration}_fermata
                inner      = tok[len("note-"):]
                pitch_str, dur_str = inner.split("_", 1)
                # Handle fermata suffix (e.g. "eighth_fermata" → "eighth")
                has_fermata = dur_str.endswith("_fermata")
                if has_fermata:
                    dur_str = dur_str[: -len("_fermata")]
                lily_pitch = _parse_pitch(pitch_str)
                lily_dur   = _parse_duration(dur_str)
                note_tok   = lily_pitch + lily_dur
                if has_fermata:
                    note_tok += r"\fermata"
                if pending_tie:
                    lily_tokens[-1] += "~"
                    pending_tie = False
                lily_tokens.append(note_tok)

            elif tok.startswith("rest-"):
                dur_str  = tok[len("rest-"):]
                # Handle fermata suffix on rests too
                has_fermata = dur_str.endswith("_fermata")
                if has_fermata:
                    dur_str = dur_str[: -len("_fermata")]
                lily_dur = _parse_duration(dur_str)
                rest_tok = "r" + lily_dur
                if has_fermata:
                    rest_tok += r"\fermata"
                lily_tokens.append(rest_tok)

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


def make_lily_source(music_body: str) -> str:
    """Fill the shared LY_TEMPLATE with a music body string."""
    return LY_TEMPLATE.format(music=music_body)


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------

def process_sample(
    sample_dir: Path,
    output_dir: Path,
    dpi: int = 200,
    force: bool = False,
    with_lmx: bool = True,
) -> bool:
    """
    Process one PrIMuS sample directory.
    Returns True on success, False on failure.
    """
    sample_id    = sample_dir.name
    sem_path     = sample_dir / f"{sample_id}.semantic"

    if not sem_path.exists():
        # Using returning a string instead of False to pass an error message to the parent
        return f"No .semantic file in {sample_dir.name}"

    out_sample = output_dir / sample_id
    out_png    = out_sample / f"{sample_id}.png"

    if out_png.exists() and not force:
        if _is_valid_png(out_png):
            return True  # already processed and healthy
        log.warning("Invalid existing PNG for %s (%s); re-rendering", sample_id, out_png)

    # Parse semantic → LilyPond music body
    try:
        music_body = semantic_to_lily_music(sem_path)
    except Exception as exc:
        return f"Semantic parse failed for {sample_id}: {exc}"

    if not music_body.strip():
        return f"Empty music body for {sample_id}"

    ly_source = make_lily_source(music_body)

    # Render in a temp directory, then move outputs
    with tempfile.TemporaryDirectory(prefix="realbook_") as tmp:
        tmp_dir = Path(tmp)
        # Timeout is 15 seconds by default in run_lilypond, we can bump it to 30 for safety
        png_path = run_lilypond(ly_source, sample_id, tmp_dir, dpi=dpi, timeout=30)
        if png_path is None:
            return f"LilyPond render failed or timed out for {sample_id}"

        # Crop to staff content
        try:
            raw = np.array(Image.open(png_path).convert("L"))
            cropped = crop_content(raw)
            if cropped.size == 0 or np.all(cropped == 255):
                return f"LilyPond output was empty/blank for {sample_id}"
        except Exception as exc:
            return f"Crop failed for {sample_id}: {exc}"

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

    # Optionally generate LMX from the copied .semantic file
    if with_lmx:
        try:
            from data_processing.semantic_to_lmx import convert_sample
            result = convert_sample(out_sample)
            if not result:
                return f"LMX generation failed to produce output for {sample_id}"
        except Exception as exc:
            return f"LMX generation exception for {sample_id}: {exc}"

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
        default=max(1, (os.cpu_count() or 4) - 2),
        help="Parallel workers (default: cpu_count - 2)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-render even if output PNG already exists",
    )
    parser.add_argument(
        "--no-lmx",
        action="store_true",
        help="Skip inline LMX generation (use if running semantic_to_lmx.py separately)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show DEBUG messages",
    )
    args = parser.parse_args()

    # Configure logging only if the root logger has no handlers yet
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    elif args.verbose:
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

    error_log = args.output / "errors_render.log"
    if error_log.exists():
        error_log.unlink()

    ok = fail = 0
    _worker = partial(process_sample, output_dir=args.output, dpi=args.dpi,
                      force=args.force, with_lmx=not args.no_lmx)
    
    with multiprocessing.Pool(processes=args.workers) as pool:
        with tqdm(total=len(sample_dirs), desc="Rendering") as pbar:
            # imap returns results in order, imap_unordered is faster
            for sd, result in zip(sample_dirs, pool.imap(_worker, sample_dirs)):
                if result is True:
                    ok += 1
                else:
                    fail += 1
                    with open(error_log, "a") as err_f:
                        err_f.write(f"FAILED: {sd.name} - {result}\n")
                pbar.update(1)

    log.info("Done. Success: %d  Failed: %d", ok, fail)
    if fail > 0:
        log.warning(f"See {error_log} for list of failed samples.")


if __name__ == "__main__":
    main()
