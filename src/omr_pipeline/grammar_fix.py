"""
Music grammar post-processor — apply music-theory constraints to CRNN output.

Fixes common CRNN errors on jazz lead sheets:
  1. Clef override:     Force treble clef (G2) for lead sheets.
  2. Key consistency:   Propagate key signature across systems.
  3. Time consistency:  Validate time signature against common jazz metres.
  4. Sequence cleanup:  Remove impossible token sequences.
  5. Octave sanity:     Clamp pitches to the expected lead-sheet range.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

_VALID_CLEFS = {"clef:G2"}
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

_LEAD_SHEET_OCTAVE_MIN = 3
_LEAD_SHEET_OCTAVE_MAX = 6


def fix_clefs(tokens: list[str], target_clef: str = _LEAD_SHEET_CLEF) -> list[str]:
    """Replace all clef tokens with *target_clef*.

    Jazz lead sheets are virtually always in treble clef.  The CRNN
    frequently predicts bass or C-clefs due to domain gap, which cascades
    into wrong octave expectations.
    """
    return [
        target_clef if tok.startswith("clef:") else tok
        for tok in tokens
    ]


def propagate_key(
    tokens: list[str],
    global_key: str | None = None,
) -> tuple[list[str], str | None]:
    """Enforce key-signature consistency across a piece.

    If *global_key* is provided (from a previous system), the first key token
    is checked against it; if it differs, the global key is trusted and the
    local key is replaced.

    Returns (fixed_tokens, detected_key).
    """
    first_key_idx = None
    for i, tok in enumerate(tokens):
        if tok.startswith("key:fifths:"):
            first_key_idx = i
            break

    if first_key_idx is None:
        if global_key is not None:
            out = list(tokens)
            insert_pos = 0
            for j, tok in enumerate(out):
                if tok.startswith("clef:"):
                    insert_pos = j + 1
                    break
            out.insert(insert_pos, global_key)
            return out, global_key
        return list(tokens), None

    local_key = tokens[first_key_idx]

    if global_key is not None and local_key != global_key:
        out = list(tokens)
        out[first_key_idx] = global_key
        return out, global_key

    return list(tokens), local_key


def validate_time_signature(tokens: list[str]) -> list[str]:
    """Remove time-signature tokens that don't form valid pairs."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok == "time":
            beats_tok = None
            beat_type_tok = None
            j = i + 1
            window_end = min(i + 4, len(tokens))
            consumed = [i]
            while j < window_end:
                if tokens[j].startswith("beats:"):
                    beats_tok = tokens[j]
                    consumed.append(j)
                elif tokens[j].startswith("beat-type:"):
                    beat_type_tok = tokens[j]
                    consumed.append(j)
                elif tokens[j] == "time":
                    break
                j += 1

            if beats_tok and beat_type_tok:
                pair = (beats_tok, beat_type_tok)
                if pair in _COMMON_TIME_SIGS:
                    out.append("time")
                    out.append(beats_tok)
                    out.append(beat_type_tok)
                else:
                    log.debug("Dropping unlikely time sig: %s", pair)
                i = max(consumed) + 1
            else:
                i += 1
        else:
            out.append(tok)
            i += 1

    return out


def fix_orphan_tokens(tokens: list[str]) -> list[str]:
    """Remove structurally invalid token sequences.

    - ``pitch:X`` without a following ``octave:Y`` within the next 3 tokens
    - consecutive ``measure`` tokens
    - orphan ``tied:stop`` without a preceding ``tied:start``
    - bare ``octave:N`` without a preceding ``pitch:X``
    """
    out: list[str] = []
    tie_open = False

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok == "measure" and out and out[-1] == "measure":
            i += 1
            continue

        if tok.startswith("pitch:"):
            has_octave = False
            for j in range(i + 1, min(i + 5, len(tokens))):
                if tokens[j].startswith("octave:"):
                    has_octave = True
                    break
                if tokens[j].startswith("pitch:") or tokens[j] == "measure":
                    break
            if not has_octave:
                log.debug("Dropping orphan pitch token: %s at pos %d", tok, i)
                i += 1
                continue

        if tok.startswith("octave:"):
            if not out or not out[-1].startswith("pitch:"):
                has_pitch_before = False
                for prev in out[-3:]:
                    if prev.startswith("pitch:"):
                        has_pitch_before = True
                        break
                if not has_pitch_before:
                    log.debug("Dropping orphan octave token: %s at pos %d", tok, i)
                    i += 1
                    continue

        if tok == "tied:start":
            tie_open = True
        elif tok == "tied:stop":
            if not tie_open:
                log.debug("Dropping orphan tied:stop at pos %d", i)
                i += 1
                continue
            tie_open = False

        out.append(tok)
        i += 1

    return out


def clamp_octaves(
    tokens: list[str],
    oct_min: int = _LEAD_SHEET_OCTAVE_MIN,
    oct_max: int = _LEAD_SHEET_OCTAVE_MAX,
) -> list[str]:
    """Clamp octave values to the expected lead-sheet range."""
    out: list[str] = []
    for tok in tokens:
        if tok.startswith("octave:"):
            try:
                val = int(tok.split(":")[1])
                clamped = max(oct_min, min(oct_max, val))
                if clamped != val:
                    log.debug("Clamped %s → octave:%d", tok, clamped)
                out.append(f"octave:{clamped}")
            except ValueError:
                out.append(tok)
        else:
            out.append(tok)
    return out


def fix_sequence(
    lmx_string: str,
    global_key: str | None = None,
    force_clef: bool = True,
) -> tuple[str, str | None]:
    """Apply all grammar fixes to an LMX token string.

    Parameters
    ----------
    lmx_string : str
        Space-separated LMX tokens from CRNN output.
    global_key : str | None
        Key signature from a previous system (for cross-system consistency).
    force_clef : bool
        If True, replace all clefs with treble.

    Returns
    -------
    (fixed_string, detected_key)
    """
    if not lmx_string or not lmx_string.strip():
        return lmx_string, global_key

    tokens = lmx_string.split()

    if force_clef:
        tokens = fix_clefs(tokens)

    tokens, detected_key = propagate_key(tokens, global_key)
    tokens = validate_time_signature(tokens)
    tokens = fix_orphan_tokens(tokens)
    tokens = clamp_octaves(tokens)

    return " ".join(tokens), detected_key
