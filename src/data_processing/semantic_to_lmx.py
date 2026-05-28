"""
semantic_to_lmx.py
==================
Convert PrIMuS .semantic annotations → monophonic LMX (.lmx) **directly**,
without an intermediate music21 or MusicXML representation.

Rare soprano / mezzo / varbaritone clefs (C1, C2, F3) are emitted as
``clef:G2`` to match ``generate_realbook.py`` LilyPond output.

Pipeline per sample:
    .semantic  ──tokenise──▶  LMX token list  ──write──▶  .lmx

The converter walks the PrIMuS semantic token stream and emits the
corresponding LMX tokens using simple lookup tables.  Accidental display
is based on the semantic pitch spelling plus the current key signature
(to emit ``natural`` signs when a note cancels a key-signature accidental).

Usage:
    poetry run python src/data_processing/semantic_to_lmx.py \\
        --source data/realbook_primus/package_aa --workers 8

    # Test on a small subset:
    poetry run python src/data_processing/semantic_to_lmx.py \\
        --source data/realbook_primus/package_aa --limit 10 --verbose
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import re
import sys
from pathlib import Path

from tqdm import tqdm

from CRNN_CTC.lilypond_render import normalize_clef_id_for_lead_sheet

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conversion tables
# ---------------------------------------------------------------------------

# PrIMuS duration names → LMX duration token
_DUR_LMX: dict[str, str] = {
    "breve": "breve",
    "whole": "whole",
    "half": "half",
    "quarter": "quarter",
    "eighth": "eighth",
    "sixteenth": "16th",
    "32nd": "32nd",
    "64th": "64th",
    # legacy PrIMuS aliases
    "thirty_second": "32nd",
    "sixty_fourth": "64th",
    # longa (4 whole notes)
    "quadruple_whole": "longa",
    # double whole (breve alias sometimes seen)
    "double_whole": "breve",
}

# Key-signature root → number of fifths.
# The PrIMuS key format is e.g. "EbM", "F#m", "C", "AbM".
_KEY_FIFTHS: dict[str, int] = {
    "Cb": -7,
    "Gb": -6,
    "Db": -5,
    "Ab": -4,
    "Eb": -3,
    "Bb": -2,
    "F": -1,
    "C": 0,
    "G": 1,
    "D": 2,
    "A": 3,
    "E": 4,
    "B": 5,
    "F#": 6,
    "C#": 7,
}

# Steps altered by N fifths.  Positive fifths → sharps in FCGDAEB order;
# negative fifths → flats in BEADGCF order.
_SHARP_ORDER = "FCGDAEB"
_FLAT_ORDER = "BEADGCF"


def _key_altered_steps(fifths: int) -> set[str]:
    """Return the set of step letters altered by the key signature.

    E.g. fifths=-3 (E♭ major) → {"B", "E", "A"} (these are flatted).
    """
    if fifths > 0:
        return set(_SHARP_ORDER[i] for i in range(min(fifths, 7)))
    elif fifths < 0:
        return set(_FLAT_ORDER[i] for i in range(min(-fifths, 7)))
    return set()


# PrIMuS accidental suffixes → LMX accidental token
_ACC_SEMANTIC_TO_LMX: dict[str, str] = {
    "b": "flat",
    "bb": "flat-flat",
    "#": "sharp",
    "x": "double-sharp",
    "##": "double-sharp",
}

# Pitch regex: step (A-G), optional accidental (b, bb, #, ##, x), octave digit(s)
_PITCH_RE = re.compile(r"^([A-G])(b{1,2}|#{1,2}|x?)(\d+)$")


# ---------------------------------------------------------------------------
# Core converter: .semantic tokens → LMX tokens
# ---------------------------------------------------------------------------


def semantic_to_lmx_tokens(
    tokens: list[str],
    *,
    verbose: bool = False,
) -> list[str]:
    """Convert a list of PrIMuS semantic tokens directly to LMX tokens.

    Design invariants
    -----------------
    * ``multirest-N`` tokens are skipped entirely, matching
      ``generate_realbook.py`` which does not render these bars.
    * ``gracenote-*`` tokens are skipped (not in the LMX vocabulary).
    * Barlines reset the measure — the next content starts a new ``measure``.
    * The first token emitted is always ``measure`` (start of first measure).
    * Ties produce ``tied:start`` on the preceding note and ``tied:stop``
      on the following note.
    * Accidental display: show on first occurrence of a step in the measure,
      suppress on consecutive same-step notes, re-show after a different
      step intervenes.  A ``natural`` is emitted when a step cancels a
      key-signature or in-measure accidental.
    """
    out: list[str] = ["measure"]  # first measure always starts immediately

    # State
    key_altered: set[str] = set()  # steps altered by current key sig
    key_emitted: bool = False  # guard: emit key:fifths:0 if no explicit key seen
    pending_tie_stop: bool = False
    in_multirest: bool = False  # suppress barline after multirest

    # Per-measure accidental tracking:
    # acc_shown[step] = the accidental token last shown for this step
    # last_step = step of the most recent note (to detect "consecutive")
    acc_shown: dict[str, str] = {}
    last_step: str | None = None

    for tok in tokens:
        try:
            # ── Clef ──────────────────────────────────────────────────────
            if tok.startswith("clef-"):
                clef_id = normalize_clef_id_for_lead_sheet(tok[5:])
                out.append(f"clef:{clef_id}")

            # ── Key signature ─────────────────────────────────────────────
            elif tok.startswith("keySignature-"):
                ks_str = tok[13:]
                # Strip mode suffix (M/m) to get root
                if ks_str.endswith("M") or ks_str.endswith("m"):
                    root = ks_str[:-1]
                else:
                    root = ks_str  # bare "C" = C major

                fifths = _KEY_FIFTHS.get(root)
                if fifths is None:
                    log.warning("Unknown key root %r in token %r", root, tok)
                    continue
                key_altered = _key_altered_steps(fifths)
                out.append(f"key:fifths:{fifths}")
                key_emitted = True

            # ── Time signature ────────────────────────────────────────────
            elif tok.startswith("timeSignature-"):
                # PrIMuS omits the keySignature token for C major (no accidentals).
                # Inject key:fifths:0 here so the label matches the rendered image,
                # which always shows a (blank) key signature area.
                if not key_emitted:
                    out.append("key:fifths:0")
                    key_emitted = True
                ts_str = tok[14:]
                if ts_str == "C":
                    out.extend(["time", "beats:4", "beat-type:4"])
                elif ts_str in ("C/", "C|"):
                    out.extend(["time", "beats:2", "beat-type:2"])
                else:
                    parts = ts_str.split("/")
                    if len(parts) == 2:
                        out.extend(["time", f"beats:{parts[0]}", f"beat-type:{parts[1]}"])
                    else:
                        log.warning("Cannot parse time signature: %r", tok)

            # ── Notes ─────────────────────────────────────────────────────
            elif tok.startswith("note-"):
                inner = tok[5:]
                pitch_str, dur_str = inner.split("_", 1)
                has_fermata = dur_str.endswith("_fermata")
                if has_fermata:
                    dur_str = dur_str[: -len("_fermata")]

                # Parse pitch
                m = _PITCH_RE.match(pitch_str)
                if not m:
                    if verbose:
                        log.debug("Cannot parse pitch %r — skipping", pitch_str)
                    continue
                step, sem_acc, octave = m.group(1), m.group(2), m.group(3)

                # Parse duration
                stripped_dur = dur_str.rstrip(".")
                dots = len(dur_str) - len(stripped_dur)
                lmx_dur = _DUR_LMX.get(stripped_dur)
                if lmx_dur is None:
                    if verbose:
                        log.debug("Unknown duration %r — skipping", dur_str)
                    continue

                # Emit: pitch, octave, duration, [dots], [accidental], [tied:stop], [fermata]
                out.append(f"pitch:{step}")
                out.append(f"octave:{octave}")
                out.append(lmx_dur)

                for _ in range(dots):
                    out.append("dot")

                # Accidental display
                #
                # Rules (clean, consistent):
                # 1. Semantic has explicit accidental → display it
                # 2. No semantic accidental, but key sig alters this step
                #    → display "natural" (cancels key sig)
                # 3. No semantic accidental, but a previous note in this
                #    measure had an accidental on this step → display
                #    "natural" (cancels in-measure deviation)
                # 4. Suppress if same step as immediately preceding note
                #    AND same accidental was already shown (consecutive dup)
                if sem_acc:
                    lmx_acc = _ACC_SEMANTIC_TO_LMX.get(sem_acc, "")
                elif step in key_altered or step in acc_shown:
                    # Need natural: either key sig says altered, or a
                    # previous note in this measure was explicitly altered
                    lmx_acc = "natural"
                else:
                    lmx_acc = ""

                if lmx_acc:
                    # Suppress consecutive duplicate on same step
                    if step == last_step and acc_shown.get(step) == lmx_acc:
                        pass  # suppress
                    else:
                        out.append(lmx_acc)
                        acc_shown[step] = lmx_acc

                last_step = step

                # Tie stop
                if pending_tie_stop:
                    out.append("tied:stop")
                    pending_tie_stop = False

                # Fermata
                if has_fermata:
                    out.append("fermata")

            # ── Rests ─────────────────────────────────────────────────────
            elif tok.startswith("rest-"):
                dur_str = tok[5:]
                has_fermata = dur_str.endswith("_fermata")
                if has_fermata:
                    dur_str = dur_str[: -len("_fermata")]

                stripped_dur = dur_str.rstrip(".")
                dots = len(dur_str) - len(stripped_dur)
                lmx_dur = _DUR_LMX.get(stripped_dur)
                if lmx_dur is None:
                    if verbose:
                        log.debug("Unknown rest duration %r — skipping", dur_str)
                    continue

                out.append("rest")
                out.append(lmx_dur)

                for _ in range(dots):
                    out.append("dot")

                if has_fermata:
                    out.append("fermata")

            # ── Grace notes ───────────────────────────────────────────────
            elif tok.startswith("gracenote-"):
                # Grace notes are skipped in LMX output
                # They don't affect last_step tracking
                pass

            # ── Ties ──────────────────────────────────────────────────────
            elif tok == "tie":
                # Mark the previous note with tied:start
                # and flag the next note for tied:stop
                out.append("tied:start")
                pending_tie_stop = True

            # ── Barlines ──────────────────────────────────────────────────
            elif tok == "barline":
                if in_multirest:
                    # Suppress the barline after a multirest — the multirest
                    # was skipped so this barline's measure would be empty.
                    in_multirest = False
                else:
                    # Start the next measure
                    out.append("measure")
                # Reset per-measure state
                acc_shown.clear()
                last_step = None

            # ── Multirest ─────────────────────────────────────────────────
            elif tok.startswith("multirest-"):
                # Skipped — not rendered in the image
                in_multirest = True
                log.debug("Skipping %s (not rendered in image)", tok)

            # ── Unknown ──────────────────────────────────────────────────
            else:
                log.debug("Unhandled token %r — skipping", tok)

        except Exception as exc:
            log.debug("Error processing token %r: %s", tok, exc)

    # Strip trailing "measure" token if it's the last thing (from a
    # trailing barline with no content after)
    while out and out[-1] == "measure":
        out.pop()

    return out


# ---------------------------------------------------------------------------
# Full conversion: .semantic → .lmx
# ---------------------------------------------------------------------------


def convert_sample(sample_dir: Path) -> list[str] | None:
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

    # Read semantic tokens (tab-separated in PrIMuS files)
    text = sem_path.read_text(encoding="utf-8")
    tokens = text.split()

    if not tokens:
        log.warning("Empty semantic file: %s", sem_path)
        return None

    # Convert
    try:
        lmx_tokens = semantic_to_lmx_tokens(tokens)
    except Exception as exc:
        log.warning("Conversion failed for %s: %s", sample_id, exc)
        return None

    if not lmx_tokens:
        log.warning("Empty LMX output for %s", sample_id)
        return None

    # Write .lmx file (space-separated, single line)
    lmx_path.write_text(" ".join(lmx_tokens) + "\n", encoding="utf-8")

    return lmx_tokens


def _convert_worker(sample_dir: Path) -> bool:
    """Wrapper for multiprocessing (returns success bool)."""
    return convert_sample(sample_dir) is not None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PrIMuS .semantic annotations to monophonic LMX.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/processed/primus/clean"),
        help="Dataset root containing rendered sample subdirectories (default: data/processed/primus/clean)",
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
        default=max(1, (os.cpu_count() or 4) - 2),
        help="Parallel workers (default: cpu_count - 2)",
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

    # Collect sample directories recursively
    sample_dirs = sorted(d for d in args.source.rglob("*") if d.is_dir() and (d / f"{d.name}.semantic").exists())

    if not sample_dirs:
        log.error("No samples found in %s", args.source)
        sys.exit(1)

    if args.limit:
        sample_dirs = sample_dirs[: args.limit]

    log.info(
        "Converting %d samples → LMX (workers=%d)",
        len(sample_dirs),
        args.workers,
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
                if _convert_worker(sd):
                    ok += 1
                else:
                    fail += 1
                    with open(error_log, "a") as err_f:
                        err_f.write(f"FAILED: {sd.name}\n")
                pbar.update(1)
    else:
        with multiprocessing.Pool(processes=args.workers) as pool:
            with tqdm(total=len(sample_dirs), desc="Converting") as pbar:
                for sd, success in zip(sample_dirs, pool.imap(_convert_worker, sample_dirs)):
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
