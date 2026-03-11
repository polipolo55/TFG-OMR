"""
semantic_to_lmx.py
==================
Convert PrIMuS .semantic annotations → monophonic LMX (.lmx) via music21
and the ``linearized-musicxml`` package.

Pipeline per sample:
    .semantic  ──parse──▶  music21 Score  ──export──▶  MusicXML  ──lmx──▶  .lmx

The monophonic LMX output strips purely visual tokens (beam, stem, staff,
voice) that carry no musical semantics for a CRNN-CTC model.

The music21 intermediate representation is kept because it correctly computes
written accidentals (natural signs, cautionary flats/sharps) from the key
signature and intra-measure pitch history — logic that would be complex and
error-prone to reimplement.  The sole responsibility of this module is to
build a structurally correct Score so that music21's MusicXML exporter does
not inject phantom fill rests that have no counterpart in the LilyPond-rendered
image.

Usage:
    poetry run python src/data_processing/semantic_to_lmx.py \\
        --source data/realbook_primus_aa --workers 8

    # Test on a small subset:
    poetry run python src/data_processing/semantic_to_lmx.py \\
        --source data/realbook_primus_aa --limit 10 --verbose
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from functools import partial
from pathlib import Path
from typing import Optional

from tqdm import tqdm
import music21

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
# Tokens to strip for monophonic output
# ---------------------------------------------------------------------------
_STRIP_PREFIXES = (
    "voice:",
    "staff:",
    "stem:",
    "beam:",
    "print-object:",
)

# ---------------------------------------------------------------------------
# Conversion tables
# ---------------------------------------------------------------------------

# music21 pitch names: flat = "-", double-flat = "--", sharp = "#", double-sharp = "##"
_ACC_M21: dict[str, str] = {
    "":   "",
    "b":  "-",
    "bb": "--",
    "#":  "#",
    "x":  "##",   # double-sharp (×)
}

_DUR_QL: dict[str, float] = {
    "breve":            8.0,
    "whole":            4.0,
    "half":             2.0,
    "quarter":          1.0,
    "eighth":           0.5,
    "sixteenth":        0.25,
    "32nd":             0.125,
    "64th":             0.0625,
    # legacy PrIMuS aliases
    "thirty_second":    0.125,
    "sixty_fourth":     0.0625,
    # longa (4 whole notes)
    "quadruple_whole":  16.0,
}

_CLEF_MAP: dict[str, type] = {
    "G2": music21.clef.TrebleClef,
    "G1": music21.clef.FrenchViolinClef,
    "F4": music21.clef.BassClef,
    "F3": music21.clef.FBaritoneClef,
    "C1": music21.clef.SopranoClef,
    "C2": music21.clef.MezzoSopranoClef,
    "C3": music21.clef.AltoClef,
    "C4": music21.clef.TenorClef,
    "C5": music21.clef.AltoClef,   # no exact music21 equivalent; alto is closest
}

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_pitch_m21(pitch_str: str) -> str:
    """Convert PrIMuS pitch ``'Bb5'`` → music21 pitch string ``'B-5'``."""
    m = re.match(r"^([A-G])(b{1,2}|#{1,2}|x?)(\d+)$", pitch_str)
    if not m:
        raise ValueError(f"Cannot parse pitch: {pitch_str!r}")
    step, acc, octave = m.group(1), m.group(2), m.group(3)
    return step + _ACC_M21.get(acc, acc) + octave


def _parse_duration_ql(dur_str: str) -> float:
    """Convert PrIMuS duration string ``'quarter.'`` → quarterLength ``1.5``."""
    stripped = dur_str.rstrip(".")
    dots = len(dur_str) - len(stripped)
    base = _DUR_QL.get(stripped)
    if base is None:
        raise ValueError(f"Unknown duration: {dur_str!r}")
    ql, add = base, base
    for _ in range(dots):
        add /= 2.0
        ql += add
    return ql


def _parse_key_m21(ks_str: str) -> music21.key.Key:
    """Convert PrIMuS key-signature string (e.g. ``'EbM'``) → music21 Key."""
    if ks_str.endswith("M"):
        mode, root = "major", ks_str[:-1]
    elif ks_str.endswith("m"):
        mode, root = "minor", ks_str[:-1]
    else:
        mode, root = "major", ks_str
    root_m21 = root[0] + root[1:].replace("b", "-")
    return music21.key.Key(root_m21, mode)


def _parse_time_m21(ts_str: str) -> music21.meter.TimeSignature:
    """Convert PrIMuS time-signature string (e.g. ``'C'``) → music21 TimeSignature."""
    if ts_str == "C":
        ts_str = "4/4"
    elif ts_str in ("C/", "C|"):
        ts_str = "2/2"
    return music21.meter.TimeSignature(ts_str)


# ---------------------------------------------------------------------------
# PrIMuS semantic → music21 Score
# ---------------------------------------------------------------------------

def semantic_to_score(tokens: list[str]) -> music21.stream.Score:
    """
    Build a ``music21.stream.Score`` from PrIMuS semantic tokens.

    Design invariants
    -----------------
    * ``multirest-N`` tokens are skipped entirely, matching ``generate_realbook.py``
      which also does not render these bars in the LilyPond image.
    * A barline that would flush a measure with no notes/rests is also skipped
      (phantom measure produced after a skipped multirest).  The header tokens
      (clef, key, time) stay in ``current_measure`` and are carried forward to
      the first real content.
    * Every incomplete measure is padded before being appended to the Part:
      ``paddingLeft`` for measures flushed at an interior barline (anacrusis /
      pickup bar), ``paddingRight`` for the final measure.  Both flags tell
      music21's MusicXML exporter that the measure is intentionally incomplete,
      preventing it from injecting phantom fill rests that have no counterpart
      in the rendered image.
    """
    score = music21.stream.Score()
    part = music21.stream.Part()

    current_measure = music21.stream.Measure(number=1)
    measure_num = 1
    pending_tie = False

    # Tracks the most-recently-seen TimeSignature across measure boundaries so
    # we can compute bar_ql for any measure, not just the first.
    current_ts: Optional[music21.meter.TimeSignature] = None

    def _flush_measure(m: music21.stream.Measure, is_last: bool) -> None:
        """Apply incomplete-measure padding to *m* then append it to *part*."""
        nonlocal current_ts
        # Update the running TimeSignature from any new TS inside this measure.
        ts_in_measure = list(m.getElementsByClass(music21.meter.TimeSignature))
        if ts_in_measure:
            current_ts = ts_in_measure[-1]

        if current_ts is not None:
            bar_ql = current_ts.barDuration.quarterLength
            notes_ql = sum(n.duration.quarterLength for n in m.notesAndRests)
            remainder = bar_ql - notes_ql
            if 0 < remainder < bar_ql:
                # paddingLeft  → anacrusis / pickup bar (notes at END of bar)
                # paddingRight → final partial bar   (notes at START of bar)
                if is_last:
                    m.paddingRight = remainder
                else:
                    m.paddingLeft = remainder

        part.append(m)

    for tok in tokens:
        try:
            # ── Clef / key / time-signature ───────────────────────────────────
            if tok.startswith("clef-"):
                clef_id = tok[5:]
                if clef_id not in _CLEF_MAP:
                    raise ValueError(f"Unknown clef: {clef_id!r}. Add it to _CLEF_MAP.")
                current_measure.append(_CLEF_MAP[clef_id]())

            elif tok.startswith("keySignature-"):
                current_measure.append(_parse_key_m21(tok[13:]))

            elif tok.startswith("timeSignature-"):
                current_measure.append(_parse_time_m21(tok[14:]))

            # ── Notes ─────────────────────────────────────────────────────────
            elif tok.startswith("note-"):
                inner = tok[5:]
                pitch_str, dur_str = inner.split("_", 1)
                has_fermata = dur_str.endswith("_fermata")
                if has_fermata:
                    dur_str = dur_str[: -len("_fermata")]
                n = music21.note.Note(
                    _parse_pitch_m21(pitch_str),
                    quarterLength=_parse_duration_ql(dur_str),
                )
                if has_fermata:
                    n.expressions.append(music21.expressions.Fermata())
                if pending_tie:
                    n.tie = music21.tie.Tie("stop")
                    pending_tie = False
                current_measure.append(n)

            # ── Rests ─────────────────────────────────────────────────────────
            elif tok.startswith("rest-"):
                dur_str = tok[5:]
                has_fermata = dur_str.endswith("_fermata")
                if has_fermata:
                    dur_str = dur_str[: -len("_fermata")]
                r = music21.note.Rest(quarterLength=_parse_duration_ql(dur_str))
                if has_fermata:
                    r.expressions.append(music21.expressions.Fermata())
                current_measure.append(r)

            # ── Grace notes ───────────────────────────────────────────────────
            elif tok.startswith("gracenote-"):
                inner = tok[10:]
                pitch_str, dur_str = inner.split("_", 1)
                gn = music21.note.Note(
                    _parse_pitch_m21(pitch_str),
                    quarterLength=_parse_duration_ql(dur_str),
                )
                gn.duration.isGrace = True
                current_measure.append(gn)

            # ── Ties ──────────────────────────────────────────────────────────
            elif tok == "tie":
                elements = list(current_measure.notesAndRests)
                if elements and isinstance(elements[-1], music21.note.Note):
                    elements[-1].tie = music21.tie.Tie("start")
                pending_tie = True

            # ── Barlines ──────────────────────────────────────────────────────
            elif tok == "barline":
                if not list(current_measure.notesAndRests):
                    # Empty measure — produced when multirest-N is followed
                    # immediately by a barline.  Skip the flush; header tokens
                    # (clef/key/time) that are already in current_measure will
                    # be carried forward to the first real content token.
                    continue
                _flush_measure(current_measure, is_last=False)
                measure_num += 1
                current_measure = music21.stream.Measure(number=measure_num)

            # ── Multirest ─────────────────────────────────────────────────────
            elif tok.startswith("multirest-"):
                # generate_realbook.py renders the image WITHOUT multirest bars.
                # Skip these tokens so the LMX label matches the image.
                log.debug("Skipping %s (not rendered in image)", tok)

            # ── Unknown tokens (silently ignore) ──────────────────────────────
            else:
                log.debug("Unhandled token %r — skipping", tok)

        except Exception as exc:
            log.debug("Error processing token %r: %s", tok, exc)

    # Flush the final measure if it contains notes/rests or header tokens.
    has_notes = bool(list(current_measure.notesAndRests))
    has_headers = bool(list(current_measure.getElementsByClass(
        (music21.clef.Clef, music21.key.Key, music21.meter.TimeSignature)
    )))
    if has_notes or has_headers:
        _flush_measure(current_measure, is_last=True)

    score.append(part)
    return score


# ---------------------------------------------------------------------------
# music21 Score → LMX tokens
# ---------------------------------------------------------------------------

def score_to_lmx(score: music21.stream.Score) -> list[str]:
    """
    Directly linearize a music21 Score into LMX tokens, bypassing the brittle
    external `linearized-musicxml` package and avoiding intermediate XML export.
    """
    tokens = []

    # Map music21 clefs back to LMX clef tokens
    _CLEF_REVERSE_MAP = {
        music21.clef.TrebleClef: "G2",
        music21.clef.FrenchViolinClef: "G1",
        music21.clef.BassClef: "F4",
        music21.clef.FBaritoneClef: "F3",
        music21.clef.SopranoClef: "C1",
        music21.clef.MezzoSopranoClef: "C2",
        music21.clef.AltoClef: "C3",
        music21.clef.TenorClef: "C4",
    }

    # Ensure accidentals are computed based on key signature
    score.makeAccidentals(inPlace=True, overrideStatus=True)

    for part in score.parts:
        for m in part.getElementsByClass(music21.stream.Measure):
            tokens.append("measure")

            for el in m.elements:
                if isinstance(el, music21.clef.Clef):
                    clef_t = _CLEF_REVERSE_MAP.get(type(el), "G2")
                    tokens.append(f"clef:{clef_t}")

                elif isinstance(el, music21.key.Key):
                    # Key signature is represented by number of fifths (sharps>0, flats<0)
                    tokens.append(f"key:fifths:{el.sharps}")

                elif isinstance(el, music21.meter.TimeSignature):
                    tokens.append("time")
                    tokens.append(f"beats:{el.numerator}")
                    tokens.append(f"beat-type:{el.denominator}")

                elif isinstance(el, music21.note.GeneralNote):
                    # Pitch or Rest
                    if isinstance(el, music21.note.Note):
                        tokens.append(f"{el.pitch.step}{el.pitch.octave}")
                    elif isinstance(el, music21.note.Rest):
                        tokens.append("rest")
                    else:
                        continue

                    # Base duration
                    tok_type = el.duration.type
                    # Handle specific overrides/aliases
                    if tok_type == "complex":
                        # Fallback for weird tuplets or non-standard durations
                        # We try to represent it with the closest single note type
                        if el.duration.quarterLength >= 4:
                            tok_type = "whole"
                        elif el.duration.quarterLength >= 2:
                            tok_type = "half"
                        elif el.duration.quarterLength >= 1:
                            tok_type = "quarter"
                        elif el.duration.quarterLength >= 0.5:
                            tok_type = "eighth"
                        else:
                            tok_type = "sixteenth"
                            
                    tokens.append(tok_type)

                    # Tuplets (Time-modification)
                    for tuplet in el.duration.tuplets:
                        tokens.append(f"{tuplet.numberNotesActual}in{tuplet.numberNotesNormal}")

                    # Dots
                    for _ in range(el.duration.dots):
                        tokens.append("dot")

                    # Accidentals
                    if isinstance(el, music21.note.Note) and el.pitch.accidental and el.pitch.accidental.displayStatus:
                        acc = el.pitch.accidental.name
                        # LMX names: flat, sharp, natural, double-sharp, flat-flat (not natively in music21 usually but similar)
                        if acc == "double-sharp":
                            tokens.append("double-sharp")
                        elif acc == "double-flat":
                            tokens.append("flat-flat")
                        else:
                            tokens.append(acc)  # flat, sharp, natural

                    # Beams
                    if getattr(el, "beams", None):
                        for b in el.beams:
                            if b.type == "start":
                                tokens.append("beam:begin")
                            elif b.type == "stop":
                                tokens.append("beam:end")
                            elif b.type == "continue":
                                tokens.append("beam:continue")
                            elif "partial" in b.type:
                                # We treat forward/backward hooks implicitly or skip them,
                                # semantic dataset doesn't generate them complexly
                                pass

                    # Ties
                    if getattr(el, "tie", None):
                        if el.tie.type == "start":
                            tokens.append("tied:start")
                        elif el.tie.type == "stop":
                            tokens.append("tied:stop")
                        elif el.tie.type == "continue":
                            tokens.append("tied:stop")
                            tokens.append("tied:start")

                    # Articulations & Expressions (Fermata, Staccato, etc.)
                    if getattr(el, "expressions", None):
                        has_fermata = any(isinstance(e, music21.expressions.Fermata) for e in el.expressions)
                        if has_fermata:
                            tokens.append("fermata")

    return tokens


def filter_monophonic(tokens: list[str]) -> list[str]:
    """
    Strip tokens that are irrelevant for a monophonic CRNN-CTC model
    (voice, staff, stem, beam markers).
    """
    return [t for t in tokens if not any(t.startswith(p) for p in _STRIP_PREFIXES)]


# ---------------------------------------------------------------------------
# Full conversion: .semantic → .lmx
# ---------------------------------------------------------------------------

def convert_sample(
    sample_dir: Path,
    *,
    strip_visual: bool = True,
) -> list[str] | None:
    """
    Read the ``.semantic`` file from *sample_dir*, convert to LMX, and write
    a ``.lmx`` file alongside the other annotations.

    Returns the LMX token list on success, ``None`` on failure.
    """
    sample_id = sample_dir.name
    sem_path = sample_dir / f"{sample_id}.semantic"
    lmx_path = sample_dir / f"{sample_id}.lmx"

    if not sem_path.exists():
        log.warning("No .semantic file in %s — skipping", sample_dir)
        return None

    # Read semantic tokens
    text = sem_path.read_text(encoding="utf-8")
    tokens = text.split()

    if not tokens:
        log.warning("Empty semantic file: %s", sem_path)
        return None

    # Parse into music21 Score
    try:
        score = semantic_to_score(tokens)
    except Exception as exc:
        log.warning("Score construction failed for %s: %s", sample_id, exc)
        return None

    # Linearize to LMX
    try:
        lmx_tokens = score_to_lmx(score)
    except Exception as exc:
        log.warning("LMX linearization failed for %s: %s", sample_id, exc)
        return None

    if not lmx_tokens:
        log.warning("Empty LMX output for %s", sample_id)
        return None

    if strip_visual:
        lmx_tokens = filter_monophonic(lmx_tokens)

    # Write .lmx file (space-separated, single line)
    lmx_path.write_text(" ".join(lmx_tokens) + "\n", encoding="utf-8")

    return lmx_tokens


def _convert_worker(sample_dir: Path, strip_visual: bool = True) -> bool:
    """Wrapper for multiprocessing (returns success bool)."""
    result = convert_sample(sample_dir, strip_visual=strip_visual)
    return result is not None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PrIMuS .semantic annotations to monophonic LMX."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/realbook_primus_aa"),
        help="Dataset root containing sample subdirectories (default: data/realbook_primus_aa)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N samples (for testing)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Parallel workers (default: half CPU count)",
    )
    parser.add_argument(
        "--keep-visual",
        action="store_true",
        help="Keep visual tokens (beam, stem, voice, staff) in the output",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show DEBUG messages",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Collect sample directories
    sample_dirs = sorted(
        d for d in args.source.iterdir()
        if d.is_dir() and (d / f"{d.name}.semantic").exists()
    )

    if not sample_dirs:
        log.error("No samples found in %s", args.source)
        sys.exit(1)

    if args.limit:
        sample_dirs = sample_dirs[: args.limit]

    strip = not args.keep_visual
    log.info(
        "Converting %d samples → LMX (strip_visual=%s, workers=%d)",
        len(sample_dirs), strip, args.workers,
    )

    error_log = args.source / "errors_convert.log"
    # wipe previous
    if error_log.exists():
        error_log.unlink()

    ok = fail = 0

    if args.workers <= 1:
        # Single-process (easier to debug)
        with tqdm(total=len(sample_dirs), desc="Converting") as pbar:
            for sd in sample_dirs:
                if _convert_worker(sd, strip_visual=strip):
                    ok += 1
                else:
                    fail += 1
                    with open(error_log, "a") as err_f:
                        err_f.write(f"FAILED: {sd.name}\n")
                pbar.update(1)
    else:
        worker = partial(_convert_worker, strip_visual=strip)
        with multiprocessing.Pool(processes=args.workers) as pool:
            with tqdm(total=len(sample_dirs), desc="Converting") as pbar:
                for sd, success in zip(sample_dirs, pool.imap(worker, sample_dirs)):
                    if success:
                        ok += 1
                    else:
                        fail += 1
                        with open(error_log, "a") as err_f:
                            err_f.write(f"FAILED: {sd.name}\n")
                    pbar.update(1)

    log.info("Done. Converted: %d  Failed: %d", ok, fail)
    if fail > 0:
        log.warning(f"See {error_log} for list of failed samples.")


if __name__ == "__main__":
    main()
