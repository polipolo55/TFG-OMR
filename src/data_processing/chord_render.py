"""
chord_render.py
===============
Synthetic Real Book-style chord rendering for the chord CRNN.

The chord CRNN reads chord symbols off Real Book scans.  Training data is
generated synthetically by:

1. Sampling a sequence of jazz chords with realistic frequency weights.
2. Rendering them as a horizontal strip with LilyPond + LilyJAZZ font.
3. Overriding LilyPond's chord-name function so the output uses Real Book
   conventions: ``-`` for minor (not ``m``), ``maj`` for major 7 (not ``M``),
   ``ø`` for half-diminished, ``dim`` for diminished, ``+`` for augmented.

Each chord has a fixed ``(lilypond_input, visual_label)`` pair so the image
content matches its label deterministically — no need to OCR-extract labels
from the rendering.

This module is the *single source of truth* for chord notation conventions
on the training side.  The chord CRNN's vocabulary, dataset, and inference
decoder all consume the visual labels produced here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Roots — LilyPond pitch name ↔ visual root letter, with sampling weights
# ---------------------------------------------------------------------------

ROOTS: list[tuple[str, str, int]] = [
    # (lily_input, visual, weight)
    ("c", "C", 10),
    ("cis", "C#", 3),
    ("des", "Db", 5),
    ("d", "D", 10),
    ("dis", "D#", 2),
    ("ees", "Eb", 7),
    ("e", "E", 10),
    ("f", "F", 10),
    ("fis", "F#", 5),
    ("ges", "Gb", 5),
    ("g", "G", 10),
    ("gis", "G#", 2),
    ("aes", "Ab", 7),
    ("a", "A", 10),
    ("ais", "A#", 1),
    ("bes", "Bb", 7),
    ("b", "B", 10),
]


# ---------------------------------------------------------------------------
# Qualities — every quality maps to a chord-exception entry in the LY source
# ---------------------------------------------------------------------------
# Each row: (lily_quality, visual_quality, weight, chord_notes_from_C, markup)
#
# `chord_notes_from_C` is the LilyPond pitch list (intervals from C) that
# defines the chord shape — used to build the chord-exception list.
# LilyPond matches exceptions root-agnostically: define once for C, applies
# to every root.
#
# `markup` is the LilyPond markup expression that renders the *quality*
# portion (everything after the root letter).  The root + accidental are
# emitted by the root-namer; the markup only handles the suffix.
#
# Visual labels NEVER contain the root.  Final label = root_visual + visual_quality.


@dataclass(frozen=True)
class QualitySpec:
    lily: str  # e.g. ":m7"
    visual: str  # e.g. "-7"
    weight: int
    notes: str  # e.g. "c es g bes"
    markup: str  # LilyPond markup, e.g. r'\markup { \concat { "-" \super "7" } }'


QUALITIES: list[QualitySpec] = [
    # Triads -----------------------------------------------------------------
    QualitySpec("", "", 30, "c e g", r"\markup { }"),
    QualitySpec(":m", "-", 12, "c es g", r'\markup { "-" }'),
    QualitySpec(":dim", "dim", 3, "c es ges", r'\markup { \super "dim" }'),
    QualitySpec(":aug", "+", 3, "c e gis", r'\markup { \super "+" }'),
    QualitySpec(":sus", "sus", 3, "c f g", r'\markup { \super "sus" }'),
    # 6ths -------------------------------------------------------------------
    QualitySpec(":6", "6", 4, "c e g a", r'\markup { \super "6" }'),
    QualitySpec(":m6", "-6", 2, "c es g a", r'\markup { \concat { "-" \super "6" } }'),
    # 7ths -------------------------------------------------------------------
    QualitySpec(":7", "7", 35, "c e g bes", r'\markup { \super "7" }'),
    QualitySpec(":m7", "-7", 35, "c es g bes", r'\markup { \concat { "-" \super "7" } }'),
    QualitySpec(":maj7", "maj7", 28, "c e g b", r'\markup { \concat { "maj" \super "7" } }'),
    QualitySpec(":m7.5-", "ø", 8, "c es ges bes", r'\markup { \super "ø" }'),
    QualitySpec(":dim7", "dim7", 4, "c es ges beses", r'\markup { \concat { "dim" \super "7" } }'),
    QualitySpec(":sus4.7", "7sus", 3, "c f g bes", r'\markup { \super "7sus" }'),
    # 9ths -------------------------------------------------------------------
    QualitySpec(":9", "9", 8, "c e g bes d'", r'\markup { \super "9" }'),
    QualitySpec(":m9", "-9", 3, "c es g bes d'", r'\markup { \concat { "-" \super "9" } }'),
    QualitySpec(":maj9", "maj9", 3, "c e g b d'", r'\markup { \concat { "maj" \super "9" } }'),
    # 7 + altered 9 ----------------------------------------------------------
    QualitySpec(":7.9-", "7b9", 6, "c e g bes des'", r'\markup { \concat { \super "7" \super "b9" } }'),
    QualitySpec(":7.9+", "7#9", 3, "c e g bes dis'", r'\markup { \concat { \super "7" \super "#9" } }'),
    # 11ths, 13ths -----------------------------------------------------------
    QualitySpec(":11", "11", 1, "c e g bes d' f'", r'\markup { \super "11" }'),
    QualitySpec(":m11", "-11", 1, "c es g bes d' f'", r'\markup { \concat { "-" \super "11" } }'),
    QualitySpec(":13", "13", 2, "c e g bes d' a'", r'\markup { \super "13" }'),
    QualitySpec(":maj13", "maj13", 1, "c e g b d' a'", r'\markup { \concat { "maj" \super "13" } }'),
]


# ---------------------------------------------------------------------------
# Half-diminished print variants
# ---------------------------------------------------------------------------
# Real Real Book pages overwhelmingly print half-diminished as "-7b5"
# (≈85 % of half-dim occurrences in the hand-labelled corpus), occasionally
# "m7b5", and only rarely the "ø" glyph.  All three carry the SAME canonical
# label "ø" (the `visual` of the half-dim QualitySpec); only the rendered
# glyphs differ, so the CRNN learns to map every printed form to one token.
# Weights mirror the observed corpus frequencies.

_HALFDIM_LILY = ":m7.5-"

HALFDIM_STYLES: list[tuple[str, str, int]] = [
    # (style, markup, weight)
    ("-7b5", r'\markup { \concat { "-" \super "7" \super "b5" } }', 85),
    ("m7b5", r'\markup { \concat { "m" \super "7" \super "b5" } }', 10),
    ("ø", r'\markup { \super "ø" }', 5),
]
_HALFDIM_MARKUP = {s: m for s, m, _ in HALFDIM_STYLES}


def choose_halfdim_style(rng: random.Random) -> str:
    """Pick a half-diminished print style with corpus-matched weights."""
    styles = [s for s, _, _ in HALFDIM_STYLES]
    weights = [w for _, _, w in HALFDIM_STYLES]
    return rng.choices(styles, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Build the LilyPond chord-exception list
# ---------------------------------------------------------------------------


def _build_exception_list(halfdim_markup: str | None = None) -> str:
    """Build the body of RealBookChordsList for the LY template.

    ``halfdim_markup`` overrides the markup used for the half-diminished
    chord shape so the same chord (and the same canonical ``ø`` label) can be
    drawn as ``-7b5``, ``m7b5``, or ``ø``.  ``None`` keeps the default ``ø``.
    """
    lines: list[str] = []
    for q in QUALITIES:
        if not q.notes:
            continue
        markup = q.markup
        if halfdim_markup is not None and q.lily == _HALFDIM_LILY:
            markup = halfdim_markup
        # ``<c e g>1-\markup { ... }`` — the `-` attaches the markup
        lines.append(f"  <{q.notes}>1-{markup}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LilyPond template
# ---------------------------------------------------------------------------
# Built once; uses the lilyjazz font for handwritten styling but supplies its
# own chord-name exceptions so the visual output matches Real Book conventions.

LY_TEMPLATE = r"""
\version "2.26.0"
{staff_size_directive}
\include "lilyjazz.ily"

%% Real Book chord-naming overrides
RealBookChordsList = {{
{chord_exceptions}
}}

RealBookChords = #(append (sequential-music-to-chord-exceptions RealBookChordsList #t) ignatzekExceptions)

\header {{ tagline = ##f }}
\paper {{
  indent = 0
  ragged-right = ##t
  paper-height = {paper_height}\mm
  paper-width = {paper_width}\mm
  top-margin = 3\mm
  bottom-margin = 3\mm
  left-margin = 4\mm
  right-margin = 4\mm
}}
\score {{
  \new ChordNames \with {{
    chordNameExceptions = #RealBookChords
    \override ChordName.font-name = #"lilyjazz-chord"
  }}
  \chordmode {{ {chord_body} }}
  \layout {{ }}
}}
"""


def make_ly_source(
    chord_body: str,
    *,
    paper_height: int = 30,
    paper_width: int = 280,
    staff_size_directive: str = "",
    halfdim_style: str = "ø",
) -> str:
    """Build a full LilyPond source for a chord strip.

    Parameters
    ----------
    chord_body
        LilyPond ``\\chordmode`` body, e.g. ``"c1 d:m7 g:7 c:maj7"``.
    paper_height, paper_width
        Page size in millimetres.  Width should scale with chord count so
        chords don't get squeezed.
    staff_size_directive
        Optional ``#(set-global-staff-size N)`` directive; varies the chord
        text size for augmentation.
    halfdim_style
        Which printed form to draw for half-diminished chords (``"-7b5"``,
        ``"m7b5"``, or ``"ø"``); see :func:`choose_halfdim_style`.  The label
        is unaffected — every form maps to the canonical ``ø`` token.
    """
    return LY_TEMPLATE.format(
        chord_exceptions=_build_exception_list(_HALFDIM_MARKUP[halfdim_style]),
        chord_body=chord_body,
        paper_height=paper_height,
        paper_width=paper_width,
        staff_size_directive=staff_size_directive,
    )


# ---------------------------------------------------------------------------
# Chord sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chord:
    """One sampled chord with its LilyPond input and visual label."""

    lily: str  # e.g. "cis:m7" or "f:maj7/c"
    label: str  # e.g. "C#-7" or "Fmaj7/C"


def _weighted_choice(rng: random.Random, items: list, weights: list[int]):
    """Pure-Python alternative to random.choices (works with frozen dataclasses)."""
    return rng.choices(items, weights=weights, k=1)[0]


def sample_chord(
    rng: random.Random,
    *,
    slash_bass_prob: float = 0.04,
) -> Chord:
    """Sample a single random jazz chord with realistic frequency weights."""
    # Root
    root_lily, root_vis, _ = _weighted_choice(rng, ROOTS, [w for _, _, w in ROOTS])
    # Quality
    qual = _weighted_choice(rng, QUALITIES, [q.weight for q in QUALITIES])

    lily = f"{root_lily}{qual.lily}"
    label = f"{root_vis}{qual.visual}"

    # Optional slash bass — appended to BOTH lily input and visual label
    if rng.random() < slash_bass_prob:
        bass_lily, bass_vis, _ = _weighted_choice(rng, ROOTS, [w for _, _, w in ROOTS])
        if bass_vis != root_vis:  # don't render C/C
            lily = f"{lily}/{bass_lily}"
            label = f"{label}/{bass_vis}"

    return Chord(lily=lily, label=label)


def sample_progression(
    rng: random.Random,
    n_chords: int,
    *,
    slash_bass_prob: float = 0.04,
) -> list[Chord]:
    """Sample a progression of ``n_chords`` independent chords."""
    return [sample_chord(rng, slash_bass_prob=slash_bass_prob) for _ in range(n_chords)]


# ---------------------------------------------------------------------------
# All visible characters in any rendered label — used to seed the vocabulary
# ---------------------------------------------------------------------------


def all_label_characters() -> set[str]:
    """Return the full character set that can appear in any chord label.

    Used to construct the chord CRNN's character vocabulary.  Includes the
    space character because multi-chord strips space-separate their chords.
    """
    chars: set[str] = {" "}
    for _, vis, _ in ROOTS:
        chars.update(vis)
    for q in QUALITIES:
        chars.update(q.visual)
    # Slash bass adds '/'
    chars.add("/")
    return chars
