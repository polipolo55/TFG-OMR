# TFG-OMR — Project Overview

## What It Is

TFG-OMR is an Optical Music Recognition (OMR) system specialized for **monophonic jazz lead sheets** (The Real Book style). It takes a PDF or image of a lead sheet and outputs a structured JSON transcription of the musical content: notes, rhythms, key/time signatures, clefs, and chord symbols.

**Institution:** FIB/UPC (Facultat d'Informàtica de Barcelona, Universitat Politècnica de Catalunya)
**Thesis type:** Treball de Final de Grau (TFG)
**Author:** Pol Casanovas Puig
**Supervisor:** Manel Frigola Bourlon

## System Goals

1. Accept PDF or image input of a monophonic lead sheet.
2. Detect and isolate individual staff lines.
3. Recognize music notation (notes, rests, barlines, accidentals, ties) using a CRNN-CTC model.
4. Recognize chord symbols printed above the staff using OCR.
5. Validate and correct the recognized token sequence against the LMX grammar.
6. Return a structured JSON with the full transcription, including staff positions and page images.

## Domain Specification

This is the **contract of the system** — the precise definition of the
lead-sheet sub-domain TFG-OMR is built for.  Inputs that match this
specification are in scope; inputs outside it are not, and the system makes
no quality guarantees on them.

The domain is intentionally narrow.  The dataset filters (`filter_multi_staff`,
`filter_non_leadsheet_clef`, `filter_unusual_time` in `OMRDataset`) and the
inference-time grammar fixer (`src/omr_pipeline/grammar_fix.py`) are
*expressions* of this contract — not a list of paranoid edge cases.  Disabling
them does not produce a more general model; it only forces the CRNN to spend
representational capacity on visual patterns that never appear in The Real
Book.  See "Known Limitations" below for what falls *outside* the contract.

### Inputs

- **Document type.** PDF or raster image (PNG, JPEG, TIFF) of a monophonic
  jazz lead sheet — typically a single page from The Real Book or an
  equivalent fake-book.  Multi-page PDFs: only page 0 is processed today.
- **Print quality.** Photocopy / phone-photo / clean PDF export are all in
  scope.  The augmentation pipeline simulates realistic phone-scan noise,
  uneven illumination, ink bleed, JPEG-style compression, and elastic warp.
- **Resolution.** PDFs are rasterised at 300 DPI by default
  (`OMR_PDF_DPI`, clamped 72–600).  Training renders use 200/250/300 DPI
  with per-sample jitter, so the CRNN is robust to small DPI mismatches.

### Music notation

| Aspect | Contract |
|--------|----------|
| **Voicing** | Monophonic — one melody line per staff.  No polyphony, no piano grand-staff, no stem-down counterpoint. |
| **Clef** | Treble (`clef:G2`) only.  C-clefs, F-clefs, G1 are out of scope. |
| **Key signatures** | Any of `key:fifths:N` for `N ∈ [-6, +7]` (Gb major through C# major).  Cb major (`-7`) is the one nominal major key absent from PrIMuS and from the trained vocabulary — extremely rare in the Real Book, so this is not a practical limitation.  C major (`key:fifths:0`) is the most common class after the converter's implicit-key fix (~22 % of corpus). |
| **Time signatures** | One of `{4/4, 3/4, 2/4, 2/2, 6/8, 6/4, 5/4, 12/8}`.  4/4 is overwhelmingly the common case.  Exotic meters (5/8, 7/8, 7/4, 9/8, 11/8, …) are out of scope. |
| **Pitch range** | Octaves 3–7 inclusive (covers the standard treble range plus altissimo).  Octave 0–2 and 8 are accepted by the vocabulary but never produced by the model. |
| **Note values** | `whole`, `half`, `quarter`, `eighth`, `16th`, `32nd`, `64th`, plus dotted variants via the `dot` token, plus ties (`tied:start` / `tied:stop`). |
| **Rests** | `rest` (with a duration token).  `rest:measure` is defined in the grammar but never emitted by the converter — whole-measure rests are encoded as a regular `rest whole` pair. |
| **Accidentals** | `flat`, `sharp`, `natural` — display-only, the actual pitch is already encoded in the `pitch:X` / `octave:Y` pair.  Double accidentals (`double-sharp`, `flat-flat`) are out of scope. |
| **Barlines** | Single barline only.  Repeats, voltas, double barlines, segno/coda/D.C./D.S. navigation marks are out of scope. |
| **Tuplets** | Out of scope.  See "Known Limitations" below. |
| **Slurs / articulations** | Out of scope. |

### Chord symbols

- **Position.** Above the staff, in the strip detected by
  `staff_detect.py::_associate_chords`.
- **Format.** Real Book chord shorthand: `ROOT [acc] [quality] [extension] [alterations]* [/BASS]`.  Minor uses a hyphen (`G-7`), half-diminished `-7b5` (the dominant printed form; `m7b5` and `ø` also appear and are normalised to the same canonical token), major-7 `maj7`.  Examples: `Cmaj7`, `G-7b5`, `D7#9`, `Bb6/9`, `Am7/D`.
- **Root letters.** A through G, with optional flat (`b`) or sharp (`#`).
- **OCR backend.** A dedicated CRNN trained on synthetic Real Book-style chord
  strips (LilyJAZZ font).  Character-level CTC output; fine-tuned on hand-labeled
  real strips.  Checkpoint at `models/chord/latest/best_model.pt`, overridable
  via `OMR_CHORD_CHECKPOINT`.  Chord post-processing in
  `src/omr_pipeline/chord_postprocess.py` rejects non-chord strings (page
  numbers, tempo markings, lyrics).

### Outputs

Structured JSON with one entry per page, one segment per detected staff,
each segment carrying:

- `staff_bbox` — `[x, y, w, h]` in page-pixel coordinates.
- `lmx_tokens` — flat token sequence after the grammar fixer.
- `chords` — list of jazz chord strings.

See the "Output Format" section below for the full schema.

## High-Level Architecture

```
Input (PDF / image bytes)
        │
        ▼
  [Preprocessing]          binarize, deskew
        │
        ▼
  [Staff Detection]         morphology → 5-line clusters
        │
        ├──── [Music Recognition] ──── CRNN-CTC → LMX tokens
        │                                      │
        │                              [Grammar Fix]  LMX validator
        │
        └──── [Chord OCR] ──────────── Chord CRNN → chord tokens
                                               │
                                       [Chord Postprocess] grammar cleanup
        │
        ▼
  JSON Output               pages, segments, chords, staff positions
```

## Output Format

```json
{
  "pages": [
    {
      "image_data": "data:image/png;base64,...",
      "segments": [
        {
          "staff_bbox": [x, y, w, h],
          "lmx_tokens": ["clef:G2", "key:fifths:0", "time", "beats:4", "beat-type:4", "pitch:C", "octave:5", ...],
          "chords": ["Cmaj7", "Am7", "Dm7", "G7"]
        }
      ]
    }
  ]
}
```

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Deep learning | PyTorch 2.10+, CRNN-CTC |
| CNN backbone | ResNet18 (default) or VGG |
| Image processing | OpenCV, albumentations |
| Music rendering | LilyPond + LilyJAZZ font |
| PDF handling | PyMuPDF (fitz) |
| Chord OCR | Custom CRNN-CTC (ResNet18, LilyJAZZ-trained, fine-tuned on real strips) |
| Web API | FastAPI + Uvicorn |
| Data augmentation | albumentations |
| Package manager | Poetry |

## Repository Structure

```
src/
├── cli.py                      unified CLI (12 subcommands; see docs/cli.md)
├── style.py                    matplotlib theme
├── CRNN_CTC/                   model, training, vocab, evaluation
│   ├── model.py                CRNN-CTC architecture (ResNet18 / VGG backbone)
│   ├── vocab.py                Vocabulary: blank=0, pad=1, unk=2, tokens 3+
│   ├── dataset.py / config.py  PrIMuS OMR dataset + Config dataclass
│   ├── train.py / evaluate.py  OMR training loop and evaluation
│   ├── chord_dataset.py        Synthetic chord-strip dataset + augmentation
│   ├── chord_train.py          Chord CRNN training from scratch
│   └── chord_finetune.py       Fine-tune chord CRNN on real labeled strips
├── data_processing/            dataset generation and augmentation
│   ├── generate_realbook.py    PrIMuS → LilyJAZZ render
│   ├── semantic_to_lmx.py      PrIMuS .semantic → LMX labels
│   ├── generate_header_templates.py  prerender 120 header-strip templates
│   ├── augment_scanned.py      scan-simulation augmentation
│   ├── chord_render.py         Synthetic chord strip renderer (LilyJAZZ)
│   ├── generate_chord_crops.py Bulk chord strip generation for CRNN training
│   └── extract_real_chord_strips.py  Extract + pre-label strips from real PDFs
├── omr_pipeline/               full inference pipeline
│   ├── pipeline.py             run_pipeline() entry point
│   ├── preprocess.py           load / binarize / deskew
│   ├── staff_detect.py         morphology → 5-line clusters + chord strip crop
│   ├── inference.py            CRNN-CTC music recognition
│   ├── header_injector.py      prepend header templates to continuation staves
│   ├── chord_recognizer.py     CRNN chord OCR (recognize_chords_crnn)
│   ├── chord_postprocess.py    jazz-chord grammar filter
│   └── grammar_fix.py          LMX token sequence validator
└── api/                        FastAPI web server

data/
├── raw/primus/             PrIMuS dataset packages
├── processed/primus/
│   ├── clean/              rendered LilyJAZZ PNGs + .lmx labels
│   └── scanned/            augmented (distorted) copies
├── header_templates/       prerendered clef+key+time strips (120 files)
├── chord_synth/            synthetic chord-strip images + CSV labels
│   ├── train/  val/        split folders
│   └── train_labels.csv  val_labels.csv
├── chord_real/             real Real Book chord strips for fine-tuning
│   ├── strips/             cropped PNG images from real PDF pages
│   └── labels.jsonl        hand-corrected labels (status: pending/done/skip)
└── vocab/
    ├── primus_lmx.txt      OMR vocabulary (~77 content tokens; 80 incl. blank/pad/unk)
    └── chord.txt           Chord CRNN vocabulary (26 character tokens)

models/
├── latest/best_model.pt    OMR CRNN checkpoint (latest)
└── chord/
    ├── latest/             → symlink to latest chord run/finetune dir
    └── finetune_*/         fine-tune checkpoints with training_log.csv

static/
├── index.html              Lead-sheet upload UI
└── chord_labeler.html      Chord strip hand-labeling UI

scripts/                    standalone utilities
notebooks/                  Jupyter evaluation notebooks
latex_documents/gep/        GEP thesis-management deliverables
```

## Performance

Held-out test split (4 608 samples), greedy CTC decoding, latest checkpoint
(`models/latest/best_model.pt` = `run_20260612_101637`, best epoch 83):

> **Note on comparability.** Evaluation numbers from CRNN runs *before
> 2026-06-03* (e.g. 0.23 % SER / 94.1 % perfect from `run_20260601_134845`)
> used a header-stripped-twin split that leaked near-duplicate samples across
> train/test and are **not comparable** to the numbers below. They are also not
> reproducible — see `docs/experiments/2026-06-10-volume-collapse-findings.md`.

- **Aggregate SER:** 1.23 % (scanned), 1.17 % (clean) — token-level edit distance.
- **Melodic SER:** 0.14 % (scanned), 0.10 % (clean) — same metric with `measure`
  and tie tokens stripped, isolating actual pitch/duration/accidental errors.
- **Perfect transcriptions:** 72.7 % (scanned), 73.7 % (clean).
- **Error breakdown:** barline (`measure`) tokens ≈14.7 % per-category error and
  ties ≈31 % together account for ~87 % of all edits.
  Pitch/octave/duration error rates are each well below 0.1 %.
- **Beam search** (width 5) yields ≤ 1 edit/1000 SER improvement at ~6× decode
  cost; greedy is the recommended default.
- **Hardware target:** NVIDIA RTX 3060 (12 GB VRAM), batch size 16–24
- **Training time:** 90-epoch OneCycle schedule (~8 min/epoch on the 3060),
  early stopping at patience 12; best at epoch 83.

Reproduce: `poetry run python scripts/evaluate_full.py --checkpoint models/latest/best_model.pt --split test --both-splits`

## Known Limitations and Out-of-Scope Notation

The system is intentionally narrow: it targets monophonic Real-Book-style
lead sheets in C, and a number of music-notation features are not handled.
The list below is exhaustive as of 2026 and reflects deliberate
simplifications, not bugs.  None of these affect typical Real Book scans.

### Notation features not represented in the LMX vocabulary

| Feature | Status | Why |
|---------|--------|-----|
| **Tuplets (triplets, quintuplets, …)** | **Not in vocab.** Renderer can produce them, but `semantic_to_lmx.py` flattens tuplet members to plain notes and the grammar fixer has no tuplet productions. | Adding tuplets requires a new bracket-token grammar (open/close + ratio), retraining vocab, and PrIMuS source filtering — too large for the current scope. The Real Book uses tuplets sparingly; expect occasional rhythmic misreads on triplet-heavy sections. |
| **Slurs / phrasing arcs** | Not modelled. PrIMuS sources contain slur metadata that is dropped during semantic→LMX conversion. | Slurs span multiple notes and require a paired open/close token grammar.  Visually, phrasing rarely changes the symbolic transcription.  |
| **Articulations (staccato, accent, tenuto, marcato, …)** | Not modelled. | Real Book sources almost never print articulations. |
| **Repeat barlines, voltas, segno / coda / D.C. / D.S.** | Not modelled — barlines are unified into a single `measure` token, and navigation marks are dropped. | Real Books have these but they are out of scope; downstream (player) software must reconstruct form from the chord-symbol grid. |
| **Double accidentals (`flat-flat`, `double-sharp`)** | Generated by the converter but stripped by the grammar fixer because they are absent from the training vocabulary. | True jazz lead sheets virtually never use double accidentals. |
| **Time signatures outside `{4/4, 3/4, 2/4, 2/2, 6/8, 6/4, 5/4, 12/8}`** | Filtered out at dataset construction (`filter_unusual_time`) so the model never sees them.  Predictions outside this set are coerced to `4/4` by the grammar fixer. | These cover ~99 % of the Real Book.  Extending the set requires updating both the filter and `grammar_fix._COMMON_TIME_SIGS`. |
| **Polyphony, multiple voices, chords on the staff** | Not supported — the model is trained monophonic-only and `filter_multi_staff` excludes any sample whose source PNG is taller than the configured threshold. | Lead sheets are by definition monophonic on the staff (chords are printed as text symbols above). |
| **Clefs other than G2** | All non-G2 clefs are silently rewritten to G2 during rendering (lossy but intentional — see `CLAUDE.md`).  Predictions of F4/C-clefs at inference are coerced to G2. | Real Book uses treble clef exclusively. |

### Vocabulary "dead output classes"

The vocabulary file (`data/vocab/primus_lmx.txt`) is rebuilt empirically from
all `.lmx` files (`cli.py vocab`) and is typically **~77 content tokens**
(**80** including `<blank>`, `<pad>`, `<unk>`).  It can still list tokens that
rarely appear in **training** once dataset filters are on (e.g. `clef:C3`,
`clef:F4`, low octaves) because those labels exist in the corpus before
`filter_non_leadsheet_clef` removes the samples.  Those logits cost capacity
but are harmless at inference when the grammar fixer enforces the lead-sheet
contract.  See `docs/lmx_format.md` for the canonical token list shape.

### Augmentation effects not implemented

The scan-simulation augmentation models distortion, blur, ink-bleed,
compression, lighting gradients, and halftone banding, but does **not** model:

- **Show-through / paper bleed-through** from the back side of a page.
  This is visually noticeable on cheap photocopies but rare in the digital
  Real Books typically used as input.
- **Realistic page texture / fibre noise.** The remap-tones step uses a
  uniform paper colour; no per-pixel paper grain is added.
- **Coffee-stain / annotation noise.** Hand-written marks above or below
  the staff are not synthesized; the model has not been trained to ignore them.

These would each be useful for very rough phone-photographed scans but are
not justified for the typical PDF input the system is designed for.

### Inference-time caveats

- **PDFs with text overlays** (e.g. exported from Sibelius with footers)
  may produce spurious staves; staff detection only filters by line count.
- **Very wide pages** (>2048 px after preprocessing) are clamped at the
  configured `max_image_width`.  Re-render the PDF at lower DPI if a single
  staff does not fit.
- **Multi-page PDFs:** only page 0 is processed by `run_pipeline()` and the
  API endpoint.  To transcribe additional pages today, call
  `omr_pipeline.preprocess.load_pdf_page(..., page=N)` and the staff /
  inference stages directly from a script — there is no batch multi-page CLI
  yet.

If you need any of the above, fork the project — none of these are simple
add-ons, all require coordinated changes across vocabulary, dataset
filtering, the grammar fixer, and (in some cases) the model itself.
