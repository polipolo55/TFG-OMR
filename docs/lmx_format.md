# LMX Token Format

LMX (Lead-sheet Music eXchange) is the internal representation used for ground-truth labels and CRNN output. It is a flat token sequence — no nesting, no XML — designed to be compact and CTC-friendly.

## Grammar

```
SEQUENCE  :=  HEADER ELEMENT* (BARLINE ELEMENT*)*

HEADER    :=  "measure" [CLEF] [KEY] [TIME]
BARLINE   :=  "measure" [KEY] [TIME]

CLEF      :=  "clef:G2"
KEY       :=  "key:fifths:" <fifths>     # e.g. key:fifths:0, key:fifths:-3, key:fifths:2
TIME      :=  "time" "beats:" <num> "beat-type:" <den>   # three separate tokens

ELEMENT   :=  NOTE | REST

NOTE      :=  "pitch:" <A-G>
              "octave:" <0-8>
              DURATION
              DOT*
              [ACCIDENTAL]
              [TIE_START]

REST      :=  "rest" DURATION DOT*
            | "rest:measure"          # full-bar rest

DURATION  :=  "breve" | "whole" | "half" | "quarter"
            | "eighth" | "16th" | "32nd" | "64th" | "longa"

DOT       :=  "dot"

ACCIDENTAL :=  "flat" | "sharp" | "natural"

TIE       :=  "tied:start" … "tied:stop"  # wraps a note sequence
```

> **Important:** KEY, TIME, and ACCIDENTAL use a different format than MusicXML or older project versions.
> - Key is `key:fifths:-3`, not `key:-3`.
> - Time is three tokens: `time beats:4 beat-type:4`, not a single `time:4:4`.
> - Accidentals are bare words: `flat`, `sharp`, `natural` — no `acc:` prefix.

## Key Encoding

`key:fifths:N` where N is the number of sharps (positive) or flats (negative):

| N | Key (major) |
|---|-------------|
| -7 | Cb major |
| -6 | Gb major |
| -5 | Db major |
| -4 | Ab major |
| -3 | Eb major |
| -2 | Bb major |
| -1 | F major |
| 0 | C major |
| 1 | G major |
| 2 | D major |
| 3 | A major |
| 4 | E major |
| 5 | B major |
| 6 | F# major |
| 7 | C# major |

## Time Signature Encoding

Three separate tokens: `time` then `beats:N` then `beat-type:N`.

| Notation | Tokens |
|----------|--------|
| 4/4 | `time beats:4 beat-type:4` |
| 3/4 | `time beats:3 beat-type:4` |
| 2/4 | `time beats:2 beat-type:4` |
| 2/2 | `time beats:2 beat-type:2` |
| 6/8 | `time beats:6 beat-type:8` |
| 6/4 | `time beats:6 beat-type:4` |
| 5/4 | `time beats:5 beat-type:4` |
| 12/8 | `time beats:12 beat-type:8` |

Common time (C) and cut time (C/) are normalised to `4/4` and `2/2` respectively during conversion.

## Example Sequence

A two-bar phrase in A major, 3/4 time: dotted quarter A, followed by B half.

```
measure clef:G2 key:fifths:3 time beats:3 beat-type:4
pitch:A octave:4 quarter dot
pitch:B octave:4 half
measure
pitch:A octave:5 quarter
pitch:G octave:5 quarter sharp
pitch:F octave:5 quarter sharp
```

Fully flattened (as written in `.lmx` files — one line, space-separated):

```
measure clef:G2 key:fifths:3 time beats:3 beat-type:4 pitch:A octave:4 quarter dot pitch:B octave:4 half measure pitch:A octave:5 quarter pitch:G octave:5 quarter sharp pitch:F octave:5 quarter sharp
```

## Rules Enforced by Grammar Fixer

1. First token of every sequence is `measure` (header start)
2. `pitch:X` must be followed by `octave:Y` within the next two tokens
3. `octave:Y` must be followed by a duration token
4. Duration token comes before any `dot` or accidental tokens
5. `flat`, `sharp`, `natural` appear after the duration (and any dots) of the same note
6. `dot` appears only after a duration or another `dot`
7. `tied:stop` must be matched by a prior `tied:start`; both are emitted as part of a note
8. Octaves outside [3, 6] are clamped to that range by the grammar fixer
9. Clef is normalised to `clef:G2` (only treble supported in the jazz lead-sheet domain)
10. Time signatures not in the common jazz set are rejected; the previous system's time is propagated instead

## Accidental Display Rules

Accidentals are written in LMX only when they must be displayed on the staff:

- If the pitch's alteration conflicts with the current key signature → display `flat` or `sharp`
- If a pitch was altered earlier in the same measure and the current note is natural → display `natural`
- If the pitch matches the key signature → no accidental token
- Consecutive identical alterations on the same step are suppressed (e.g. two Bb's in a row: only the first gets `flat`)

Double accidentals (`flat-flat`, `double-sharp`) can appear in `semantic_to_lmx` output for
highly chromatic classical PrIMuS samples, but are dropped by `grammar_fix.py` since they never
occur in jazz lead sheets (and are filtered out at training time by `filter_non_leadsheet_clef`).

## Differences from MusicXML

| Aspect | LMX | MusicXML |
|--------|-----|---------|
| Format | flat token list | nested XML |
| Polyphony | no (monophonic only) | yes |
| Chord symbols | separate OCR output | `<harmony>` elements |
| Articulations | not encoded | full support |
| Dynamics | not encoded | full support |
| File size | tiny | large |
| CTC compatibility | direct | requires linearization |

## Vocabulary File Format

Plain text, one token per line, no special tokens (those are assigned fixed indices in code):

```
16th
32nd
64th
beat-type:2
beat-type:4
beat-type:8
beats:2
beats:3
beats:4
beats:5
beats:6
beats:12
breve
clef:G2
dot
eighth
flat
half
key:fifths:-1
key:fifths:-2
...
key:fifths:0
key:fifths:1
...
longa
measure
natural
octave:0
octave:1
...
octave:8
pitch:A
pitch:B
...
pitch:G
quarter
rest
rest:measure
sharp
tied:start
tied:stop
time
whole
```

Special indices (hardcoded, not in file):
- `0` → `<blank>` (CTC blank)
- `1` → `<pad>`
- `2` → `<unk>`
