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

_DURATIONS = frozenset(
    {
        "whole",
        "half",
        "quarter",
        "eighth",
        "16th",
        "32nd",
        "64th",
        "breve",
        "longa",
    }
)

# Duration → fraction of a whole note (used by barline regulariser)
_DUR_BEATS: dict[str, float] = {
    "longa": 4.0,
    "breve": 2.0,
    "whole": 1.0,
    "half": 0.5,
    "quarter": 0.25,
    "eighth": 0.125,
    "16th": 0.0625,
    "32nd": 0.03125,
    "64th": 0.015625,
}

_ACCIDENTALS = frozenset({"flat", "sharp", "natural"})

_LEAD_SHEET_CLEF = "clef:G2"

# Must stay in sync with ``_COMMON_TIME_SIGS`` in ``src/CRNN_CTC/dataset.py``
# (the training-side counterpart used for dataset filtering).
_COMMON_TIME_SIGS: frozenset[tuple[str, str]] = frozenset(
    {
        ("beats:4", "beat-type:4"),
        ("beats:3", "beat-type:4"),
        ("beats:2", "beat-type:4"),
        ("beats:2", "beat-type:2"),
        ("beats:6", "beat-type:8"),
        ("beats:6", "beat-type:4"),
        ("beats:5", "beat-type:4"),
        ("beats:12", "beat-type:8"),
    }
)

_OCTAVE_MIN = 3
# Cap raised to 7 to accommodate jazz altissimo (a high B5/C6 in alto-sax
# pitch is already an octave-7 sounding pitch; lead sheets occasionally
# notate ledgered C7/D7 melodies).  Octaves above 7 are notational outliers
# in The Real Book and almost always indicate model error.
_OCTAVE_MAX = 7


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
    last_had_duration = False  # tracks whether last note/rest has its duration

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
                # Insert synthetic clef — no input token consumed, so i stays.
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
                log.debug("Malformed octave token %r — clamping to %d", octave_tok, _OCTAVE_MIN)
                octave_tok = f"octave:{_OCTAVE_MIN}"

            # Emit the note
            out.append(tok)  # pitch:X
            out.append(octave_tok)  # octave:Y
            out.append(dur_tok)  # duration
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
            # Default to after the first measure token; advance to after clef
            # if one is present (normal case after _parse_and_repair).
            insert_after = 0
            for j, tok in enumerate(tokens):
                if tok == "measure":
                    insert_after = j + 1
                if _is_clef(tok):
                    insert_after = j + 1
                    break
            tokens = list(tokens)
            tokens.insert(insert_after, global_key)
            return tokens, global_key
        return tokens, None

    local_key = tokens[first_key_idx]
    # Local key takes priority — each Real Book staff prints its own key
    # signature, so a locally-detected key is more reliable than a global
    # carried forward from a previous (possibly wrong) detection.
    # The local key becomes the new global for staves that follow without one.
    return tokens, local_key


# ---------------------------------------------------------------------------
# Time-signature propagation
# ---------------------------------------------------------------------------


def _propagate_time(
    tokens: list[str],
    global_time: tuple[str, str, str] | None,
) -> tuple[list[str], tuple[str, str, str] | None]:
    """Carry time signature across systems, mirroring ``_propagate_key``.

    Parameters
    ----------
    tokens : list[str]
        Repaired LMX token list for one system.
    global_time : tuple | None
        ``("time", "beats:N", "beat-type:N")`` detected from a previous
        system, or *None* if not yet seen.

    Returns
    -------
    (tokens, detected_time)
        The possibly-modified token list and the time signature that should
        be forwarded to the next system.
    """
    # Find time signature in the header only (before the second 'measure').
    # Scanning the full sequence would mistake a mid-sequence time change for
    # the header time sig and skip the global-time injection.
    local_time_idx: int | None = None
    local_time: tuple[str, str, str] | None = None
    measure_count = 0
    for i, tok in enumerate(tokens):
        if tok == "measure":
            measure_count += 1
            if measure_count >= 2:
                break
        if tok == "time" and i + 2 < len(tokens):
            if _is_beats(tokens[i + 1]) and _is_beat_type(tokens[i + 2]):
                local_time = (tok, tokens[i + 1], tokens[i + 2])
                local_time_idx = i
                break

    # Found a local time signature
    if local_time is not None:
        # If a global is already established, override the local with the global
        # (Real Book pages keep the same time sig across all staves; the first
        # detected one is authoritative).  Mirrors _propagate_key behaviour.
        if global_time is not None and local_time != global_time:
            tokens = list(tokens)
            tokens[local_time_idx] = global_time[0]
            tokens[local_time_idx + 1] = global_time[1]
            tokens[local_time_idx + 2] = global_time[2]
            return tokens, global_time
        return tokens, local_time

    # No local time signature — inject the global one if available
    if global_time is None:
        return tokens, None

    # Insert after the last header token (clef or key) in the first measure
    insert_after = 0
    for j, tok in enumerate(tokens):
        if _is_clef(tok) or _is_key(tok):
            insert_after = j + 1
    tokens = list(tokens)
    for k, t in enumerate(global_time):
        tokens.insert(insert_after + k, t)

    return tokens, global_time


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
        has_content = any(_is_pitch(t) or t == "rest" for t in m)
        # Keep header measures (with clef/key/time) and content measures
        has_header = any(_is_clef(t) or _is_key(t) or t == "time" for t in m)
        if has_content or has_header:
            out.extend(m)

    return out


# ---------------------------------------------------------------------------
# Barline regularisation — re-insert / remove barlines by beat counting
# ---------------------------------------------------------------------------


def _element_duration(tokens: list[str], start: int) -> tuple[float, int]:
    """Return (duration_in_whole_notes, tokens_consumed) for a note/rest at *start*.

    Handles dotted durations: each dot adds half the previous dot's value.
    """
    i = start
    n = len(tokens)

    # skip pitch + octave / rest token to reach the duration
    if _is_pitch(tokens[i]):
        i += 1  # pitch:X
        if i < n and _is_octave(tokens[i]):
            i += 1  # octave:Y
    elif tokens[i] == "rest":
        i += 1
    else:
        return 0.0, 1

    dur = 0.0
    if i < n and tokens[i] in _DUR_BEATS:
        dur = _DUR_BEATS[tokens[i]]
        i += 1
    else:
        return 0.25, i - start  # fallback: quarter

    # dots
    dot_val = dur / 2
    while i < n and tokens[i] == "dot":
        dur += dot_val
        dot_val /= 2
        i += 1

    # accidental
    if i < n and tokens[i] in _ACCIDENTALS:
        i += 1

    # tie
    if i < n and tokens[i] in ("tied:start", "tied:stop"):
        i += 1

    # fermata
    if i < n and tokens[i] == "fermata":
        i += 1

    return dur, i - start


def _regularise_barlines(tokens: list[str]) -> list[str]:
    """Insert missing barlines based on accumulated beat duration.

    Requires a valid time signature in the header (``time beats:N beat-type:N``).
    If no time signature is found the tokens are returned unchanged.

    The algorithm **trusts** model-emitted barlines (resetting the beat
    accumulator when one is encountered) but **inserts** a ``measure`` token
    whenever the accumulated duration reaches a full measure without one.
    This handles pickup measures (anacrusis) naturally — the first measure
    may be shorter than the time signature indicates, and the model's first
    barline resets the grid.
    """
    # Extract time signature from the header
    measure_len: float | None = None

    for i, tok in enumerate(tokens):
        if tok == "time" and i + 2 < len(tokens):
            try:
                b = int(tokens[i + 1].split(":")[1])  # beats:N
                bt = int(tokens[i + 2].split(":")[1])  # beat-type:N
                measure_len = b / bt  # in whole notes
            except IndexError, ValueError:
                pass
            break

    if measure_len is None or measure_len <= 0:
        return tokens

    out: list[str] = []
    accum = 0.0
    eps = 1e-6
    i = 0
    n = len(tokens)

    while i < n:
        tok = tokens[i]

        # Model-emitted barline — trust it and reset accumulator
        if tok == "measure":
            out.append(tok)
            accum = 0.0
            i += 1
            continue

        if _is_pitch(tok) or tok == "rest":
            dur, consumed = _element_duration(tokens, i)

            # If the accumulator has reached a full measure, the model
            # missed a barline — insert one before this element.
            if accum > eps and accum + eps >= measure_len:
                out.append("measure")
                accum = 0.0

            out.extend(tokens[i : i + consumed])
            accum += dur
            i += consumed
            continue

        # Everything else: pass through
        out.append(tok)
        i += 1

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fix_sequence(
    lmx_string: str,
    global_key: str | None = None,
    global_time: tuple[str, str, str] | None = None,
    force_clef: bool = True,
) -> tuple[str, str | None, tuple[str, str, str] | None]:
    """Parse, validate, and repair an LMX token string.

    Returns ``(fixed_string, detected_key, detected_time)``.
    """
    if not lmx_string or not lmx_string.strip():
        return lmx_string, global_key, global_time

    tokens = lmx_string.split()
    tokens = _parse_and_repair(tokens, force_clef)
    tokens = _clean_ties(tokens)
    tokens, detected_key = _propagate_key(tokens, global_key)
    tokens, detected_time = _propagate_time(tokens, global_time)
    tokens = _remove_empty_measures(tokens)
    tokens = _regularise_barlines(tokens)

    return " ".join(tokens), detected_key, detected_time
