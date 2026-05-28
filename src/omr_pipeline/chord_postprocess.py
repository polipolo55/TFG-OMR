"""Jazz chord post-processor.

Takes the raw string returned by an OCR engine and extracts valid jazz chord
tokens from it.  The strategy:

1. Apply character-level OCR confusion repairs.
2. Split the text into candidate tokens (whitespace + root-letter boundaries).
3. For each token try to match the jazz chord grammar.
4. Reject tokens that look like noise.
5. Re-join surviving tokens with two spaces.

The grammar supported (all fields optional after the root):
  CHORD = ROOT ACC? QUALITY? EXTEN? (ALT)* SLASH?

  ROOT    = [A-G]
  ACC     = # | b
  QUALITY = - | m | min | maj | M | dim | aug | + | o | ø | sus
  EXTEN   = 2 | 4 | 5 | 6 | 7 | 9 | 11 | 13
  ALT     = (b|#) DIGIT+          e.g. b5  #9  b13
  SLASH   = / ROOT ACC?
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Character-level OCR confusion fixes
# ---------------------------------------------------------------------------

# Applied before tokenisation.  Order matters — longer first.
_CHAR_REPAIRS: list[tuple[re.Pattern, str]] = [
    # Leading '6' → 'G'  ("6-7" → "G-7", "6b7" → "Gb7")
    # Only at true word-start (not inside a chord like "C6" or "Cmaj6").
    (re.compile(r"(?<![A-Za-z0-9])6(?=[A-Ga-g#b\-+]|$)"), "G"),
    # Leading '8' → 'B'  ("87" → "B7", "89" → "B9")
    (re.compile(r"(?<![A-Za-z0-9])8(?=[A-Za-z7b#]|$)"), "B"),
    # 'mio' / 'mj' / 'mjj' / 'mii' / 'moj' → 'maj'  (cursive 'a' misread as o/i/j)
    (re.compile(r"m(?:io|j+|ii|oj)(?=\d|\b)"), "maj"),
    # 'majl' → 'maj7'  (l ↔ 1 ↔ 7 confusion, specific maj case first)
    (re.compile(r"(?<=maj)l\b"), "7"),
    # Any trailing 'l' after a chord character → '7'
    # Covers "G-l"→"G-7", "Gbl"→"Gb7", "Ebl"→"Eb7"
    (re.compile(r"(?<=[A-Za-z0-9#b\-+])l\b"), "7"),
    # 'susl' → 'sus4'
    (re.compile(r"susl\b"), "sus4"),
    # Trailing '0' after a chord letter → 'o'  (dim circle)
    (re.compile(r"(?<=[A-Ga-g#b])0\b"), "o"),
    # '|' → '1'
    (re.compile(r"\|"), "1"),
    # Remove stray bar-line / bracket characters
    (re.compile(r"[\[\]{}\\]"), ""),
]

# ---------------------------------------------------------------------------
# Jazz chord grammar regex
# ---------------------------------------------------------------------------

# A single jazz chord token.  Groups:
#   1 root letter, 2 accidental, 3 quality string, 4 extension digits, 5 alterations, 6 slash bass
#
# Longer quality alternatives before shorter ones (maj before m).
_CHORD_RE = re.compile(
    r"([A-G])"  # root
    r"([#b]?)"  # accidental
    r"(maj|min|m(?:in)?|M|dim|aug|sus|-|\+|o|ø|°)?"  # quality — longer first!
    r"(maj\d{1,2}|\d{1,2})?"  # extension: maj7/maj9 or plain digit(s)
    r"((?:[#b]\d+)*)"  # alterations: b5 #9 b13 …
    r"(/[A-G][#b]?)?",  # slash bass
    re.ASCII,
)

# Tokens that are clearly not chord-related even after repairs
_JUNK_RE = re.compile(
    r"^(?:"
    r"\d{2,}"  # multi-digit number (bar numbers, page numbers)
    r"|[^A-Ga-g]"  # doesn't start with a note letter (after stripping)
    r")$"
)

# Minimum fraction of "chord-valid" characters a token must have
_CHORD_CHARS = set("ABCDEFGabcdefg#bmajdimaugsusoMø°+−-0123456789/")


def _apply_char_repairs(text: str) -> str:
    for pat, repl in _CHAR_REPAIRS:
        text = pat.sub(repl, text)
    return text


def _looks_like_chord(token: str) -> bool:
    """Quick check: does this token start with a root note?"""
    if not token:
        return False
    if not re.match(r"^[A-Ga-g]", token):
        return False
    # Make sure it contains enough chord-valid characters
    valid = sum(1 for c in token if c in _CHORD_CHARS)
    return valid / len(token) >= 0.7


def _normalise_token(token: str) -> str:
    """Try to parse and normalise a single chord token.

    Returns the canonical chord string if it matches the grammar, or '' if
    the token looks like garbage.
    """
    if not token:
        return ""

    # Capitalise root
    token = token[0].upper() + token[1:]

    # Try strict grammar match from the beginning of the token
    m = _CHORD_RE.match(token)
    if not m or m.start() != 0:
        return ""

    matched_len = m.end()
    leftover = token[matched_len:]

    # Allow a tiny leftover (1 char OCR artefact), reject anything bigger
    if len(leftover) > 1:
        return ""

    root = m.group(1)
    acc = m.group(2) or ""
    qual = m.group(3) or ""
    ext = m.group(4) or ""
    alt = m.group(5) or ""
    slsh = m.group(6) or ""

    # Normalise quality synonyms
    qual_map: dict[str, str] = {
        "min": "m",
        "M": "maj",
        "o": "dim",
        "°": "dim",
        "ø": "m7b5",
    }
    qual = qual_map.get(qual, qual)

    # 'Gmaj' + ext='7' → 'Gmaj7';  'Gmaj' + ext='maj7' (duplicate) → just 'Gmaj7'
    if ext.startswith("maj") and qual == "maj":
        ext = ext[3:]  # strip redundant 'maj' prefix from extension

    # '1' is not a valid extension — EasyOCR regularly confuses '7' and '1'
    # in handwritten/stylised fonts.  Single '1' → '7'.
    if ext == "1":
        ext = "7"

    return root + acc + qual + ext + alt + slsh


def clean_chord_line(raw_ocr: str) -> str:
    """Convert raw OCR text into a clean jazz chord line.

    Returns a space-separated string of canonical chord tokens, or '' if
    nothing recognisable was found.
    """
    if not raw_ocr or not raw_ocr.strip():
        return ""

    text = _apply_char_repairs(raw_ocr)

    # Split by whitespace into candidate tokens
    tokens = text.split()

    cleaned: list[str] = []
    for tok in tokens:
        tok = tok.strip(".,;:!?()\"' ")
        if not tok:
            continue

        # Filter obvious non-chord tokens fast
        if _JUNK_RE.match(tok):
            continue

        if _looks_like_chord(tok):
            normed = _normalise_token(tok)
            # Skip consecutive duplicates (can arise from overlapping OCR segments)
            if normed and (not cleaned or cleaned[-1] != normed):
                cleaned.append(normed)
        # else: drop the token (it's noise)

    return "  ".join(cleaned)
