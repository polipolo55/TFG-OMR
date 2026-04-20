# Data Pipeline

The data pipeline converts raw PrIMuS dataset packages into training-ready image/label pairs. It runs in four sequential stages.

## Source Dataset: PrIMuS

PrIMuS (Printed Images of Music Staves) provides monophonic music notation in three forms per sample:
- `.png` — typeset score image (Sibelius font)
- `.semantic` — pitch/duration/clef/key/time token sequences
- `.agnostic` — staff-relative position tokens

PrIMuS packages live at `data/raw/primus/package_aa/`, `package_ab/`, etc.

## Stage 1 — Render: PrIMuS → LilyJAZZ

**Script:** `src/data_processing/generate_realbook.py`
**CLI:** `python src/cli.py render`

The original PrIMuS PNGs use Sibelius font. This stage re-renders every sample with **LilyJAZZ** to match the Real Book aesthetic that the model will face at inference time.

**Per-sample steps:**
1. Parse `.semantic` file (pitch, duration, clef, key, time)
2. Normalize clefs: C1, C2, F3 → treble (G2), preserving absolute pitches via ledger lines
3. Generate LilyPond `.ly` source
4. Run `lilypond` subprocess at 200 DPI → PNG
5. Auto-crop white margins

**Output:** `data/processed/primus/clean/{sample_id}/{sample_id}.png`

**Shared lookup tables** (in `src/CRNN_CTC/lilypond_render.py`):
- `CLEF_LY` — PrIMuS clef IDs → LilyPond names
- `KEY_LY` — fifths count → `\key` command
- `DUR_LY` — duration tokens → LilyPond duration strings
- `CLEF_IDS_NORMALIZE_TO_G2` — set of clefs that map to treble

## Stage 2 — Convert: PrIMuS .semantic → LMX

**Script:** `src/data_processing/semantic_to_lmx.py`
**CLI:** `python src/cli.py convert`

Produces the ground-truth label files (`.lmx`) for CRNN training. Conversion is direct (no intermediate MusicXML), token-by-token.

**LMX grammar:**
```
SEQUENCE  := HEADER ELEMENT* (BARLINE ELEMENT*)*
HEADER    := "measure" [CLEF] [KEY] [TIME]
BARLINE   := "measure" [KEY] [TIME]
CLEF      := "clef:G2"
KEY       := "key:" <fifths>
TIME      := "time:" <num> ":" <denom>
ELEMENT   := NOTE | REST
NOTE      := "pitch:" <A-G> "octave:" <0-8> DURATION [DOT]* [ACCIDENTAL] [TIE]
REST      := "rest" DURATION [DOT]*
DURATION  := "whole"|"half"|"quarter"|"eighth"|"16th"|"32nd"|"64th"|"breve"|"longa"
ACCIDENTAL:= "acc:sharp"|"acc:flat"|"acc:nat"
TIE       := "tied:start" … "tied:stop"
```

**Key conversions:**
- PrIMuS pitch strings (e.g., `Bb5`) → LMX tokens (`pitch:B octave:5 acc:flat`)
- Accidental display computed from semantic pitch spelling + key signature context
- Natural signs emitted when canceling key-signature accidentals
- Duration aliases normalized to canonical LMX names

**Output:** `.lmx` files co-located with clean PNGs

## Stage 3 — Augment: Scan Simulation

**Script:** `src/data_processing/augment_scanned.py`
**CLI:** `python src/cli.py augment`

Bridges the synthetic-to-real domain gap by simulating real-world scanning artifacts on clean LilyJAZZ images.

**Augmentation pipeline (albumentations):**

| Transform | Purpose | Probability |
|-----------|---------|------------|
| ElasticTransform | page warping | 0.5 |
| GridDistortion | grid bending | 0.4 |
| OpticalDistortion | lens distortion | 0.35 |
| Affine (rotation, shear, scale, translate) | slight misalignment | 0.6 |
| GaussianBlur | focus blur | 0.5 |
| Sharpen | over-sharpened scanner | 0.4 |
| RandomToneCurve | exposure variation | 0.5 |
| GaussNoise | sensor noise | 0.5 |
| RandomBrightnessContrast | uneven illumination | 0.6 |
| ImageCompression | JPEG artifacts | 0.4 |

**Post-augmentation steps** (outside albumentations):
- `dilate_ink()` — morphological erosion simulating ink bleed
- `add_vignette()` — darken edges (scanner shadow)

**Output:** `data/processed/primus/scanned/{sample_id}/{sample_id}.png`
Labels (`.lmx`) are identical to clean — copied unchanged.

## Stage 4 — Vocabulary

**Script:** `src/CRNN_CTC/vocab.py`
**CLI:** `python src/cli.py vocab`

Scans all `.lmx` files and builds a sorted token list.

**Index layout:**
```
0  <blank>   (CTC blank)
1  <pad>     (sequence padding)
2  <unk>     (out-of-vocabulary)
3+ tokens    (alphabetically sorted)
```

**Robustness:** Full pitch (A–G) and octave (0–8) ranges are always included, even if absent in the corpus, so OOV pitch variants decode gracefully.

Output: a plain-text file, one token per line (excluding specials).

## Running the Full Pipeline

```bash
# All four stages in one command:
poetry run python src/cli.py pipeline \
  --raw-primus-dir data/raw/primus \
  --clean-dir data/processed/primus/clean \
  --scanned-dir data/processed/primus/scanned \
  --vocab-path data/vocab/primus_lmx.txt \
  --workers 8

# Or individually:
poetry run python src/cli.py render   --source data/raw/primus --output data/processed/primus/clean
poetry run python src/cli.py convert  --source data/processed/primus/clean --workers 8
poetry run python src/cli.py augment  --source data/processed/primus/clean --output data/processed/primus/scanned
poetry run python src/cli.py vocab    --data-dir data/processed/primus/clean --output data/vocab/primus_lmx.txt
```

## Data Filtering (applied at training time)

Filtering is not applied during pipeline stages but during dataset construction (`src/CRNN_CTC/dataset.py`). It removes samples that would hurt model generalization:

| Filter | Condition | Rationale |
|--------|-----------|-----------|
| `filter_rest_heavy` | >80% structural tokens and len>50 | orchestral tacet, absent in jazz |
| `filter_unwanted_clefs` | C1, C2, F3, C3, C4, F4, G1 | not used in jazz lead sheets |
| `filter_multi_staff` | original height > 180 px | removes polyphonic/grand-staff samples |
| `filter_non_leadsheet_clef` | not G2 (treble) | strict domain focus |
| `filter_unusual_time` | non-jazz meters | keeps 4/4, 3/4, 2/4, 2/2, 6/8, 5/4 |

## Dataset Directories

```
data/raw/primus/
  package_aa/    package_ab/    ...    (raw PrIMuS)

data/processed/primus/
  clean/
    {sample_id}/
      {sample_id}.png        LilyJAZZ render
      {sample_id}.lmx        LMX label
      {sample_id}.semantic   original (copied)
  scanned/
    {sample_id}/
      {sample_id}.png        distorted copy
      {sample_id}.lmx        same label (copied)
```

## Scripts

- `scripts/validate_lmx_pairs.py` — Sanity-check that every `.png` has a matching `.lmx`
- `scripts/audit_lmx_corpus.py` — Corpus statistics (token frequencies, sample counts per filter)
