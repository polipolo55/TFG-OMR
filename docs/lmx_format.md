# LMX Token Format

LMX (Lead-sheet Music eXchange) is the internal representation used for ground-truth labels and CRNN output. It is a flat token sequence — no nesting, no XML — designed to be compact and CTC-friendly.

## Grammar

```
SEQUENCE  :=  HEADER ELEMENT* (BARLINE ELEMENT*)*

HEADER    :=  "measure" [CLEF] [KEY] [TIME]
BARLINE   :=  "measure" [KEY] [TIME]

CLEF      :=  "clef:G2"
KEY       :=  "key:" <fifths>         # e.g. key:0, key:-3, key:2
TIME      :=  "time:" <num> ":" <den> # e.g. time:4:4, time:3:4

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

ACCIDENTAL :=  "acc:sharp" | "acc:flat" | "acc:nat"

TIE       :=  "tied:start" … "tied:stop"  # wraps a note sequence
```

## Key Encoding

`key:N` where N is the number of sharps (positive) or flats (negative):

| N | Key (major) |
|---|-------------|
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

## Example Sequence

A two-bar phrase: Cmaj7 melody in C major, 4/4.

```
measure clef:G2 key:0 time:4:4
pitch:C octave:5 quarter
pitch:E octave:5 quarter
pitch:G octave:5 quarter
pitch:B octave:5 quarter
measure
pitch:C octave:6 half
pitch:G octave:5 half
```

## Rules Enforced by Grammar Fixer

1. First token of every sequence is `measure` (header start)
2. `pitch:X` must be immediately followed by `octave:Y`
3. `octave:Y` must be followed by a duration token
4. `acc:*` appears only after `octave:Y` (part of the same note)
5. `dot` appears only after a duration
6. `tied:stop` must follow a complete note and be matched by a prior `tied:start`
7. Octaves outside [3, 6] are rejected
8. Clef must be `clef:G2` (only treble supported)
9. Time denominator must be one of: 1, 2, 4, 8, 16

## Accidental Display Rules

Accidentals are written in LMX only when they must be displayed on the staff:
- If the pitch's alteration conflicts with the current key signature → display the accidental
- If a pitch was altered in the same measure and is now natural → display `acc:nat`
- If the pitch matches the key signature → no accidental token

This mirrors engraving conventions and reduces label noise.

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
acc:flat
acc:nat
acc:sharp
breve
clef:G2
dot
eighth
half
key:-1
key:-2
...
measure
octave:0
octave:1
...
pitch:A
pitch:B
...
quarter
rest
rest:measure
tied:start
tied:stop
time:2:2
time:3:4
time:4:4
...
whole
```

Special indices (hardcoded, not in file):
- `0` → `<blank>` (CTC blank)
- `1` → `<pad>`
- `2` → `<unk>`
