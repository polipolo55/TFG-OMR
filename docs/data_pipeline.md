# Data Pipeline

The data pipeline converts raw PrIMuS dataset packages into training-ready image/label pairs. It runs in five sequential stages (orchestrated by `cli.py pipeline` / `pipeline-train`).

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

**LMX grammar** (canonical definition lives in `docs/lmx_format.md`; this is the
shape produced by `semantic_to_lmx.py`):

```
SEQUENCE  := HEADER ELEMENT* (BARLINE ELEMENT*)*
HEADER    := "measure" [CLEF] [KEY] [TIME]
BARLINE   := "measure" [KEY] [TIME]
CLEF      := "clef:G2"
KEY       := "key:fifths:" <fifths>
TIME      := "time" "beats:" <num> "beat-type:" <den>     # three tokens
ELEMENT   := NOTE | REST
NOTE      := "pitch:" <A-G> "octave:" <0-8> DURATION DOT* [ACCIDENTAL] [TIE_STOP] [FERMATA]
REST      := "rest" DURATION DOT* [FERMATA]
DURATION  := "whole"|"half"|"quarter"|"eighth"|"16th"|"32nd"|"64th"|"breve"|"longa"
ACCIDENTAL:= "flat" | "sharp" | "natural"           # bare words, no acc: prefix
TIE       := "tied:start" … "tied:stop"
FERMATA   := "fermata"                              # attaches to previous note/rest
```

**Key conversions:**
- PrIMuS pitch strings (e.g., `Bb5`) → LMX tokens (`pitch:B octave:5 quarter flat`).
  The accidental token is **bare** (`flat`, not `acc:flat`) and appears **after**
  the duration, not as part of the pitch.
- Accidental display computed from semantic pitch spelling + key signature context
- Natural signs emitted when canceling key-signature accidentals
- Duration aliases normalized to canonical LMX names

**C major implicit key fix:** PrIMuS omits the `keySignature-` token for C major (no accidentals = default). The converter detects this and injects `key:fifths:0` before the time signature so every sample has an explicit key label. Without this, ~22.6% of training images (19,778 / 87,677 samples — those in C major, which visually show a blank key area) would have no `key:fifths:0` supervision and the model could never learn to predict it.

**Output:** `.lmx` files co-located with clean PNGs

On incremental runs, samples whose `.lmx` is newer than `.semantic` are skipped unless `--force-convert` or `--force-all` is set.

## Stage 3 — Header template generation

**Script:** `src/data_processing/generate_header_templates.py`
**CLI:** `poetry run python src/cli.py generate-header-templates`
**Output:** `data/header_templates/key_{N}_time_{beats}_{beat_type}.png` (120 files)

Prerenders 120 header-strip images (15 key signatures × 8 time signatures) using
LilyPond + LilyJAZZ at 200 DPI. These are used at inference time by
`header_injector.py` to prepend the correct clef+key+time glyphs to continuation
staff images before the CRNN. Run once; re-run with `--force` if LilyJAZZ
styling changes.

## Stage 4 — Augment: Scan Simulation

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

Labels (`.lmx`) are identical to clean — copied unchanged. Existing scanned PNGs are skipped unless `--force-augment` or `--force-all` (or when implied by `--force-render`).

## Stage 5 — Vocabulary

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
# All five stages in one command:
poetry run python src/cli.py pipeline \
  --raw-primus-dir data/raw/primus \
  --clean-dir data/processed/primus/clean \
  --scanned-dir data/processed/primus/scanned \
  --vocab-path data/vocab/primus_lmx.txt \
  --workers 8

# Full rebuild (re-render, re-convert, re-augment everything):
poetry run python src/cli.py pipeline --force-all --workers $(nproc) ...

# Or individually:
poetry run python src/cli.py render   --source data/raw/primus --output data/processed/primus/clean
poetry run python src/cli.py convert  --source data/processed/primus/clean --workers 8
poetry run python src/cli.py generate-header-templates
poetry run python src/cli.py augment  --source data/processed/primus/clean --output data/processed/primus/scanned
poetry run python src/cli.py vocab    --data-dir data/processed/primus/clean --output data/vocab/primus_lmx.txt
```

`pipeline-train` runs the same five stages, then `train`. See `docs/cli.md` for `--force-all` and other pipeline flags.

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

---

## Chord CRNN Data Pipeline

The chord OCR system uses its own independent data pipeline, separate from the PrIMuS OMR pipeline above.  It produces a character-level CRNN trained to read jazz chord symbols from Real Book scans.

### Chord Vocabulary

**File:** `data/vocab/chord.txt` — 26 character tokens (no blank/pad/unk — those are injected at indices 0/1/2 by `Vocabulary`).

| Category | Characters |
|----------|-----------|
| Root letters | `A B C D E F G` |
| Accidentals | `# b` |
| Separators | ` ` (space), `/` |
| Quality letters | `a d i j m s u` (spell `dim`, `maj`, `sus`) |
| Special quality | `- + ø` |
| Numbers | `1 3 6 7 9` |

Notable absences: `2`, `4`, `5`, `8`, `l`, `t`, `°`.
- `sus4` → `sus` after `4` is dropped; both align with the `sus` training label.
- `alt` cannot be represented; label `G7alt` as `G7`.
- `°` is not in vocab; always use `dim` / `dim7`.
- `5` is not in vocab; `-7b5` and `m7b5` are canonicalized to `ø` by `RealChordDataset._canon()` before the vocab filter, so half-dim labels survive. The synthetic renderer still *draws* all three printed forms (see below), so the model recognises whichever the page uses while predicting the single `ø` token.

### Stage 1 — Synthetic Data Generation

**Files:** `src/data_processing/chord_render.py`, `src/data_processing/generate_chord_crops.py`

`chord_render.py` defines the full set of chord quality specs (`QUALITIES`) and root
weights (`ROOTS`).  It is the **single source of truth** for chord notation conventions
on the training side — both `visual_quality` strings and LilyPond rendering live here.

| Quality | Visual label | Sampling weight |
|---------|-------------|----------------|
| Major triad | `` (empty) | 30 |
| Dominant 7 | `7` | 35 |
| Minor 7 | `-7` | 35 |
| Major 7 | `maj7` | 28 |
| Minor triad | `-` | 12 |
| Half-dim | `ø` | 8 |
| … | … | … |

**Half-diminished print variants.** Real Book pages spell half-diminished
inconsistently — predominantly `-7b5`, occasionally `m7b5`, and rarely the `ø`
glyph. `chord_render.py` (`HALFDIM_STYLES`, `choose_halfdim_style`) picks one of
the three glyph forms per strip with corpus-matched weights (≈85 / 10 / 5) while
keeping the label fixed at the canonical `ø`. The label vocabulary is therefore
unchanged; only the rendered image varies, so the CRNN learns to map every
printed half-dim form to one token.

`generate_chord_crops.py` generates strips of 1–5 sampled chords per image, rendered
with LilyJAZZ font (passing a per-strip `halfdim_style`), with Albumentations
augmentation + synthetic Real Book clutter
(`_add_realbook_clutter`: binder-hole shadow, staff-line bleed, slash repeat marks).

```bash
# Generate 12 000 train + 1 200 val strips:
cd src && poetry run python -m data_processing.generate_chord_crops \
    --output ../data/chord_synth --train 12000 --val 1200
```

**Output:**
```
data/chord_synth/
    train/          PNG strips
    val/            PNG strips
    train_labels.csv
    val_labels.csv
```

### Stage 2 — Train Chord CRNN from Scratch

**File:** `src/CRNN_CTC/chord_train.py`

Same CRNN-CTC architecture as the OMR model (ResNet18 backbone, 2-layer BiLSTM), but
with a ~29-token character vocabulary instead of the ~80-token LMX vocabulary (77 content tokens in `primus_lmx.txt` after a full rebuild).

```bash
PYTHONPATH=src poetry run python -m CRNN_CTC.chord_train \
    --data-dir data/chord_synth \
    --model-dir models/chord \
    --epochs 30
```

Checkpoint saved to `models/chord/run_TIMESTAMP/best_model.pt`.

### Stage 3 — Extract Real Strips and Pre-Label

**File:** `src/data_processing/extract_real_chord_strips.py`

Extracts chord-strip crops from real Real Book PDF pages using the same staff
detection pipeline as inference, then pre-labels them with the current chord CRNN
so the human reviewer can correct rather than type from scratch.

```bash
poetry run python src/data_processing/extract_real_chord_strips.py \
    --pdf data/real_book/full_realbook.pdf \
    --output data/chord_real \
    --page-step 10
```

Appends records to `data/chord_real/labels.jsonl`:
```json
{"filename": "page0030_staff2.png", "predicted": "G-7 C7 Fmaj7", "label": null, "status": "pending"}
```

**Resume support:** pages already present in `labels.jsonl` are skipped automatically.

### Stage 4 — Hand Labeling

Start the API server and open `http://localhost:8000/labeler` in a browser.

**Keyboard shortcuts:** `Enter` = save & next · `Esc` = skip · `Tab` = restore CRNN prediction · `Shift+Tab` = clear.

**Labeling conventions:**

| What you see | Label as |
|-------------|---------|
| `Fmaj7` | `Fmaj7` |
| `G-7` or `Gm7` | `G-7` |
| `B-7b5` or `Bm7b5` or `Bø` | any of the three — all auto-converted to `Bø` by `_canon` |
| `Cdim` or `C°` | `Cdim` (never `°`) |
| `G7alt` or `G7alt.` | `G7` (alt not in vocab) |
| `A7(11)` | `A711` (drop parens) |
| Multiple chords | space-separated: `Fmaj7 G-7 C7` |
| No chords visible | click **Skip** |

### Stage 5 — Fine-Tune on Real Strips

**File:** `src/CRNN_CTC/chord_finetune.py`

Fine-tunes the synthetic checkpoint on hand-labeled real strips mixed with synthetic
data to prevent catastrophic forgetting of rare chord types.

```bash
PYTHONPATH=src poetry run python -m CRNN_CTC.chord_finetune \
    --checkpoint models/chord/latest/best_model.pt \
    --real-strips-dir data/chord_real/strips \
    --real-labels data/chord_real/labels.jsonl \
    --synth-dir data/chord_synth \
    --model-dir models/chord \
    --epochs 20 --synth-weight 0.5 --lr 2e-4
```

**Label canonicalization** (`RealChordDataset._canon()`):
- `m7b5` / `-7b5` / `min7b5` → `ø`
- `Cm7` → `C-7` (minor written with `m`)
- `sus4` / `sus2` → `sus`
- Characters not in the chord vocabulary are dropped, and the dropped
  characters are logged once with counts (no longer silent)
- Whitespace collapsed

**Weighted sampling:** real strips weight 1.0, synthetic weight `--synth-weight` (default 0.4).
Epoch sample count ≈ `n_real + synth_weight × n_synth`.

Checkpoint saved to `models/chord/finetune_TIMESTAMP/best_model.pt`; `models/chord/latest` symlink updated.
