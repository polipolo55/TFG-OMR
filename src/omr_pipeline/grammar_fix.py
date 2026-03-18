"""
LMX structural validator — parse and repair CRNN output.

The CRNN produces a flat sequence of LMX tokens.  Due to domain gap, many
sequences contain structurally impossible patterns.  This module walks through
the token sequence, keeps only tokens that fit the LMX grammar, and discards
everything else.

Expected LMX grammar (PrIMuS format):
  SEQUENCE := HEADER (ELEMENT | BARLINE)*
  HEADER   := 'measure' [CLEF] [KEY] [TIME]
  BARLINE  := 'measure' [KEY] [TIME]
  ELEMENT  := NOTE | REST
  NOTE     := 'pitch:X' 'octave:Y' DURATION [DOT...] [ACCIDENTAL] [TIE]
  REST     := 'rest' DURATION [DOT...]
  DURATION := whole | half | quarter | eighth | 16th | 32nd | 64th
  DOT      := 'dot'
  ACCIDENTAL := flat | sharp | natural
  TIE      := 'tied:start' | 'tied:stop'
  CLEF     := 'clef:G2' | ...
  KEY      := 'key:fifths:N'
  TIME     := 'time' 'beats:N' 'beat-type:N'
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token categories
# ---------------------------------------------------------------------------

_DURATIONS = frozenset({
    "whole", "half", "quarter", "eighth", "16th", "32nd", "64th",
    "breve", "longa",
})

_ACCIDENTALS = frozenset({"flat", "sharp", "natural"})

_LEAD_SHEET_CLEF = "clef:G2"

_COMMON_TIME_SIGS = {
    ("beats:4", "beat-type:4"),
    ("beats:3", "beat-type:4"),
    ("beats:2", "beat-type:4"),
    ("beats:2", "beat-type:2"),
    ("beats:6", "beat-type:8"),
    ("beats:6", "beat-type:4"),
    ("beats:5", "beat-type:4"),
    ("beats:12", "beat-type:8"),
}

_OCTAVE_MIN = 3
_OCTAVE_MAX = 6


def _is_pitch(tok: str) -> bool:
    return tok.startswith("pitch:")


def _is_octave(tok: str) -> bool:
    return tok.startswith("octave:")


def _is_duration(tok: str) -> bool:
    return tok in _DURATIONS


def _is_clef(tok: str) -> bool:
    return tok.startswith("clef:")


def _is_key(tok: str) -> bool:
    return tok.startswith("key:fifths:")


def _is_beats(tok: str) -> bool:
    return tok.startswith("beats:")


def _is_beat_type(tok: str) -> bool:
    return tok.startswith("beat-type:")


# ---------------------------------------------------------------------------
# Structural parser
# ---------------------------------------------------------------------------

def _parse_and_repair(tokens: list[str], force_clef: bool) -> list[str]:
    """Walk through the token stream and keep only structurally valid tokens.

    This is a single-pass greedy parser.  At each position it decides what
    tokens are expected given the current state, and skips anything that
    doesn't fit.
    """
    out: list[str] = []
    i = 0
    n = len(tokens)
    seen_header = False
    tie_open = False
    last_had_duration = False   # tracks whether last note/rest has its duration

    while i < n:
        tok = tokens[i]

        # ── measure (barline or start-of-line header) ──
        if tok == "measure":
            # Collapse consecutive measures
            if out and out[-1] == "measure":
                i += 1
                continue

            out.append("measure")
            i += 1

            # After measure: optionally consume clef → key → time
            if i < n and _is_clef(tokens[i]):
                out.append(_LEAD_SHEET_CLEF if force_clef else tokens[i])
                i += 1
            elif not seen_header:
                out.append(_LEAD_SHEET_CLEF)

            if i < n and _is_key(tokens[i]):
                out.append(tokens[i])
                i += 1

            if i < n and tokens[i] == "time":
                beats_tok = None
                bt_tok = None
                j = i + 1
                while j < min(i + 4, n):
                    if _is_beats(tokens[j]) and beats_tok is None:
                        beats_tok = tokens[j]
                        j += 1
                    elif _is_beat_type(tokens[j]) and bt_tok is None:
                        bt_tok = tokens[j]
                        j += 1
                    else:
                        break

                if beats_tok and bt_tok and (beats_tok, bt_tok) in _COMMON_TIME_SIGS:
                    out.append("time")
                    out.append(beats_tok)
                    out.append(bt_tok)
                i = j

            seen_header = True
            last_had_duration = False
            continue

        # ── pitch:X → expect octave:Y DURATION [dot] [accidental] [tie] ──
        if _is_pitch(tok):
            octave_tok = None
            dur_tok = None
            dots: list[str] = []
            acc_tok = None
            tie_tok = None

            j = i + 1
            # Find octave within next 2 tokens
            if j < n and _is_octave(tokens[j]):
                octave_tok = tokens[j]
                j += 1
            elif j + 1 < n and _is_octave(tokens[j + 1]):
                octave_tok = tokens[j + 1]
                j += 2
            else:
                i += 1
                continue

            # Scan ahead for duration (CRNN sometimes outputs dots before
            # the duration, e.g. "dot dot whole" instead of "whole dot dot")
            scan_end = min(j + 5, n)
            pre_dots = 0
            dur_idx = None
            for k in range(j, scan_end):
                if tokens[k] == "dot":
                    pre_dots += 1
                elif _is_duration(tokens[k]):
                    dur_idx = k
                    break
                elif _is_pitch(tokens[k]) or tokens[k] in ("measure", "rest"):
                    break
                else:
                    # skip one garbage token
                    continue

            if dur_idx is not None:
                dur_tok = tokens[dur_idx]
                j = dur_idx + 1
                dots = ["dot"] * pre_dots
            else:
                dur_tok = "quarter"

            # Consume trailing dots
            while j < n and tokens[j] == "dot":
                dots.append("dot")
                j += 1

            # Consume accidental
            if j < n and tokens[j] in _ACCIDENTALS:
                acc_tok = tokens[j]
                j += 1

            # Consume tie
            if j < n and tokens[j] in ("tied:start", "tied:stop"):
                tie_tok = tokens[j]
                j += 1

            # Clamp octave
            try:
                oval = int(octave_tok.split(":")[1])
                oval = max(_OCTAVE_MIN, min(_OCTAVE_MAX, oval))
                octave_tok = f"octave:{oval}"
            except ValueError:
                pass

            # Emit the note
            out.append(tok)          # pitch:X
            out.append(octave_tok)   # octave:Y
            out.append(dur_tok)      # duration
            out.extend(dots)
            if acc_tok:
                out.append(acc_tok)
            if tie_tok:
                if tie_tok == "tied:start":
                    tie_open = True
                    out.append(tie_tok)
                elif tie_tok == "tied:stop" and tie_open:
                    tie_open = False
                    out.append(tie_tok)

            i = j
            last_had_duration = True
            continue

        # ── rest → expect DURATION [dot...] ──
        if tok == "rest":
            j = i + 1
            dur_tok = None
            dots: list[str] = []

            if j < n and _is_duration(tokens[j]):
                dur_tok = tokens[j]
                j += 1
            else:
                dur_tok = "quarter"

            while j < n and tokens[j] == "dot":
                dots.append("dot")
                j += 1

            out.append("rest")
            out.append(dur_tok)
            out.extend(dots)
            i = j
            last_had_duration = True
            continue

        # ── tied:start / tied:stop (standalone) ──
        if tok == "tied:start" and not tie_open:
            tie_open = True
            out.append(tok)
            i += 1
            continue
        if tok == "tied:stop" and tie_open:
            tie_open = False
            out.append(tok)
            i += 1
            continue

        # ── fermata (attaches to previous note) ──
        if tok == "fermata" and last_had_duration:
            out.append(tok)
            i += 1
            continue

        # ── dot after a duration (if grammar was note ... dur ... dot) ──
        if tok == "dot" and out and (out[-1] in _DURATIONS or out[-1] == "dot"):
            out.append(tok)
            i += 1
            continue

        # ── anything else: skip (orphan duration, orphan beat-type, etc.) ──
        log.debug("Dropping orphan token at pos %d: %r", i, tok)
        i += 1

    # Clean up dangling tied:start at the very end
    while out and out[-1] == "tied:start":
        out.pop()

    return out


# ---------------------------------------------------------------------------
# Key propagation
# ---------------------------------------------------------------------------

def _propagate_key(
    tokens: list[str],
    global_key: str | None,
) -> tuple[list[str], str | None]:
    """Enforce key consistency across systems.

    If the first system detects a key, use it for all subsequent systems.
    If no key found and global_key is provided, insert it after the clef.
    """
    first_key_idx = None
    for i, tok in enumerate(tokens):
        if _is_key(tok):
            first_key_idx = i
            break

    if first_key_idx is None:
        if global_key is not None:
            insert_after = 0
            for j, tok in enumerate(tokens):
                if _is_clef(tok):
                    insert_after = j + 1
                    break
            tokens = list(tokens)
            tokens.insert(insert_after, global_key)
            return tokens, global_key
        return tokens, None

    local_key = tokens[first_key_idx]
    if global_key is not None and local_key != global_key:
        tokens = list(tokens)
        tokens[first_key_idx] = global_key
        return tokens, global_key

    return tokens, local_key


# ---------------------------------------------------------------------------
# Post-validation cleanup
# ---------------------------------------------------------------------------

def _clean_ties(tokens: list[str]) -> list[str]:
    """Remove nonsensical tie patterns.

    - tied:start immediately followed by tied:stop (no intervening note)
    - Dangling tied:start at the end with no corresponding tied:stop
    """
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        if tokens[i] == "tied:start":
            # Check if tied:stop follows before any note
            has_note_before_stop = False
            j = i + 1
            while j < n:
                if _is_pitch(tokens[j]):
                    has_note_before_stop = True
                    break
                if tokens[j] == "tied:stop":
                    break
                if tokens[j] == "measure":
                    break
                j += 1
            if j < n and tokens[j] == "tied:stop" and not has_note_before_stop:
                # Skip both tied:start and tied:stop
                i = j + 1
                continue
            out.append(tokens[i])
        else:
            out.append(tokens[i])
        i += 1

    # Remove dangling tied:start at the very end
    while out and out[-1] == "tied:start":
        out.pop()

    return out


def _remove_empty_measures(tokens: list[str]) -> list[str]:
    """Remove measures that contain no notes or rests."""
    measures: list[list[str]] = []
    current: list[str] = []

    for tok in tokens:
        if tok == "measure" and current:
            measures.append(current)
            current = []
        current.append(tok)
    if current:
        measures.append(current)

    out: list[str] = []
    for m in measures:
        has_content = any(
            _is_pitch(t) or t == "rest" for t in m
        )
        # Keep header measures (with clef/key/time) and content measures
        has_header = any(_is_clef(t) or _is_key(t) or t == "time" for t in m)
        if has_content or has_header:
            out.extend(m)

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fix_sequence(
    lmx_string: str,
    global_key: str | None = None,
    force_clef: bool = True,
) -> tuple[str, str | None]:
    """Parse, validate, and repair an LMX token string.

    Returns (fixed_string, detected_key).
    """
    if not lmx_string or not lmx_string.strip():
        return lmx_string, global_key

    tokens = lmx_string.split()
    tokens = _parse_and_repair(tokens, force_clef)
    tokens = _clean_ties(tokens)
    tokens, detected_key = _propagate_key(tokens, global_key)
    tokens = _remove_empty_measures(tokens)

    return " ".join(tokens), detected_key
