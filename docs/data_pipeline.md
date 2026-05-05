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

**Per-sample render-time variations** (applied in `make_lily_source`):

| Variation | Probability | Effect |
|-----------|-------------|--------|
| `\numericTimeSignature` | 0.5 | Forces "4/4"/"2/2" numeric vs LilyPond's default C / cut-time glyph |
| `\accidentalStyle modern` | 0.5 | Adds cautionary accidentals (Real Book convention) vs LilyPond default |
| `\autoBeamOff` | 0.15 | Unbeamed eighth/sixteenth notes vs default beat-grouped beaming |
| `set-global-staff-size` | uniform over {17,18,19,20,21,22} | Varies note size and horizontal density |

> **Variation tuning audit (2026):** `\autoBeamOff` was reduced from p=0.5 to p=0.15.
> Real Book overwhelmingly beams its eighth notes; the previous setting gave half
> of training samples an unbeamed style that does not match the inference domain.

**Prerequisites — LilyJAZZ installation:**

LilyJAZZ is a third-party font/stylesheet not bundled with LilyPond. It must be installed manually. Run these commands once after installing or upgrading LilyPond:

```bash
LILY_VER=$(lilypond --version 2>&1 | grep -oP '\d+\.\d+\.\d+' | head -1)
LILY_SHARE="/usr/share/lilypond/$LILY_VER"
git clone --depth=1 https://github.com/OpenLilyPondFonts/lilyjazz.git /tmp/lilyjazz
sudo cp /tmp/lilyjazz/otf/*.otf "$LILY_SHARE/fonts/otf/"
sudo cp /tmp/lilyjazz/svg/*.svg "$LILY_SHARE/fonts/svg/"
sudo cp /tmp/lilyjazz/supplementary-files/**/*.otf "$LILY_SHARE/fonts/otf/"
sudo cp /tmp/lilyjazz/stylesheet/*.ily "$LILY_SHARE/ly/"
```

**LilyPond ≥ 2.25 note:** The LilyJAZZ stylesheet uses `set-global-fonts` which was removed in LilyPond 2.25. The installed `lilyjazz.ily` must use the new `property-defaults.fonts.*` API. The patched file at `/usr/share/lilypond/<version>/ly/lilyjazz.ily` in this project's environment already applies this fix.

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

**C major implicit key fix:** PrIMuS omits the `keySignature-` token for C major (no accidentals = default). The converter detects this and injects `key:fifths:0` before the time signature so every sample has an explicit key label. Without this, ~45% of training images (which visually show a blank key area) would have no `key:fifths:0` supervision and the model could never learn to predict it.

**Output:** `.lmx` files co-located with clean PNGs

## Stage 3 — Augment: Scan Simulation

**Script:** `src/data_processing/augment_scanned.py`
**CLI:** `python src/cli.py augment`

Bridges the synthetic-to-real domain gap by simulating real-world scanning artifacts on clean LilyJAZZ images.

**Augmentation pipeline (albumentations):**

| Transform | Purpose | Probability |
|-----------|---------|------------|
| ElasticTransform | page warping / staff curve | 0.82 |
| GridDistortion | grid bending | 0.72 |
| OpticalDistortion | lens distortion | 0.38 |
| Affine (rotation ±4°, shear ±1.2°, scale 0.97–1.03, translate ±1.5%) | scan misalignment | 0.85 |
| GaussianBlur (σ 0.2–1.1) | focus blur | 0.62 |
| Sharpen | over-sharpened scanner | 0.48 |
| RandomToneCurve | exposure variation | 0.72 |
| GaussNoise | sensor noise | 0.78 |
| RandomBrightnessContrast (brightness ±0.06, contrast 0.03–0.18) | uneven illumination | 0.70 |
| ImageCompression (quality 72–92) | JPEG artifacts | 0.35 |

**Post-augmentation steps** (outside albumentations, applied in order):
- `dilate_ink()` — morphological erosion simulating ink bleed; 0/1/2 iterations chosen stochastically
- `vary_staff_line_thickness()` — randomly thin or thicken staff lines (p=0.40)
- `remap_tones()` — linear remap from pure white/black to paper-like [28, 245] range
- `add_vignette()` — darken edges with a radial mask, jittered strength (p=0.55)
- `add_uneven_illumination()` — directional brightness gradient from one edge, simulating phone scans / book-spine shadow (p=0.45)
- `add_halftone_lines()` — faint horizontal scan-line banding, simulating photocopier artifacts (p=0.25)

> **Augmentation overlap audit (2026):** `add_edge_shadow` was removed because
> `add_uneven_illumination` covers the same domain (single-edge gradient) and
> stacking both with `add_vignette` produced over-dark corners. `add_vignette`
> was also changed from always-on to probabilistic, and `RandomBrightnessContrast`'s
> negative bias was tightened from −0.10 to −0.06 to prevent over-darkening when
> multiple lighting effects compose.

**Online augmentation (training only).** On top of the offline-augmented PNG,
`OMRDataset.__getitem__` applies a cheap per-sample jitter (brightness ±5%, contrast bias ±3%,
gaussian noise σ ≈ 0.005–0.015, ±2 px horizontal shift) with probability `Config.online_aug_prob`
(default 0.5). Without this, every epoch sees identical pixel grids and the model overfits the
exact augmentations baked into `scanned/`. Cost is ~50 µs per sample.

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
| `filter_multi_staff` | original height > 180 px | removes LilyPond two-line wraps |
| `filter_non_leadsheet_clef` | not G2 (treble) | lead-sheet domain — see `docs/overview.md` |
| `filter_unusual_time` | non-jazz meters | keeps 4/4, 3/4, 2/4, 2/2, 6/8, 6/4, 5/4, 12/8 |

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
