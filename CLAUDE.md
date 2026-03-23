# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TFG-OMR is an Optical Music Recognition (OMR) system for monophonic jazz lead sheets (The Real Book). It extracts musical information from score images using computer vision and deep learning.

**Core Model**: CRNN-CTC (ResNet18 CNN → BiLSTM → CTC loss)
**Output Format**: LMX (Linear MusicXML), vocabulary in `src/CRNN_CTC/vocabulary.txt`
**Scope**: Monophonic only (single melody line, no chords, no polyphony)

## Environment Setup

- **Python**: ~3.14
- **Package Manager**: Poetry
- **Install**: `poetry install`

## Common Commands

All commands are run through the unified CLI at `src/cli.py`:

```bash
# Data pipeline (render → convert → augment → vocab)
poetry run python src/cli.py pipeline

# Full pipeline + training
poetry run python src/cli.py pipeline-train --epochs 50

# Individual steps
poetry run python src/cli.py render --source data/raw/primus --output data/processed/primus/clean
poetry run python src/cli.py convert --source data/processed/primus/clean
poetry run python src/cli.py augment --source data/processed/primus/clean --output data/processed/primus/scanned
poetry run python src/cli.py vocab --data-dir data/processed/primus/clean

# Training
poetry run python src/cli.py train --epochs 50 --batch-size 16 --lr 1e-3
poetry run python src/cli.py train --resume  # auto-resume from latest checkpoint

# Evaluation
poetry run python src/cli.py evaluate --checkpoint models/latest/best_model.pt --split test --beam-width 10

# API server
poetry run python src/cli.py api --host 0.0.0.0 --port 8000
```

## Architecture

### CRNN-CTC Model (`src/CRNN_CTC/`)

```
Input image (B, 1, H=128, W)
    ↓
CNN backbone (ResNet18, modified for grayscale, collapses height → 1)
    ↓  (B, C, 1, W')
BiLSTM (2 layers, hidden=256 per direction)
    ↓  (B, W', 512)
Linear → log_softmax  (vocab_size = 98 + blank + pad)
    ↓
CTCLoss (blank index = 0)
```

**Key Files**:
- `model.py` — `CRNN` class with swappable CNN (`backbone: "resnet18"` or `"vgg"`)
- `config.py` — Single `Config` dataclass for all hyperparameters; serialized into every checkpoint
- `dataset.py` — `OMRDataset` loads `{id}.png` + `{id}.lmx` pairs; applies multi-stage filtering
- `vocab.py` — `Vocabulary`: blank at index 0, pad at index 1, music tokens from index 2
- `train.py` — AdamW + OneCycleLR + AMP + early stopping; saves to `models/run_<timestamp>/`
- `evaluate.py` — Greedy CTC decode, SER (Symbol Error Rate = edit distance / ground-truth length)

### OMR Pipeline (`src/omr_pipeline/`)

End-to-end inference pipeline for uploaded PDFs/images:

```
Load → Preprocess (deskew + binarize) → Detect staves → Recognize music (CRNN) + chords (OCR) → Grammar correction → JSON output
```

**Key Files**:
- `pipeline.py` — Main entry point `run_pipeline()`
- `preprocess.py` — `PageImage`, `load_image()`, `load_pdf_page()`, `preprocess_page()`
- `staff_detect.py` — `System`, `detect_systems()` — morphological staff-line finder
- `inference.py` — `recognize_music()`, `normalize_staff_crop()`
- `ocr_chords.py` — `recognize_chords()` using EasyOCR
- `grammar_fix.py` — `fix_sequence()` applies music-theory corrections

### Data Processing (`src/data_processing/`)

- `generate_realbook.py` — LilyPond + LilyJAZZ rendering of PrIMuS → PNG + LMX
- `semantic_to_lmx.py` — Converts PrIMuS `.semantic` to monophonic LMX via music21
- `augment_scanned.py` — Scan-simulation augmentations (noise, rotation, distortion)

## Key Conventions

### Python Styling

Every notebook and script must start with:

```python
import sys; sys.path.insert(0, "../src")  # adjust relative depth
import style; style.apply()
```

Use `style.C["<role>"]` for all explicit colors — never hardcode hex literals. Available roles: `primary`, `secondary`, `tertiary`, `highlight`, `primary_light`, `secondary_light`, `tertiary_light`, `highlight_light`, `neutral_dark`, `neutral_mid`, `neutral_light`.

### Data Filtering

Three filters are applied before training (all enabled by default in `Config`):
- **`filter_rest_heavy`** — Drops samples where >80% of tokens are `rest/rest:measure/measure` AND total length >50 (orchestral tacet passages irrelevant to jazz)
- **`filter_unwanted_clefs`** — Drops samples with `clef:C1`, `clef:C2`, or `clef:F3` (not used in jazz, cause pitch-cascade substitution errors)
- **`filter_multi_staff`** — Drops images whose original height >180 px (LilyPond multi-staff wraps; single-staff images are 84–152 px)

### LMX Annotation Format

`.lmx` files are space-separated token sequences. Example: `clef:G2 key:fifths:2 time:4/4 measure E4 quarter F#4 quarter G4 half measure`. Token index 0 = CTC blank, index 1 = pad; music tokens start at index 2.

### Model Checkpoints

Checkpoints are saved to `models/run_<timestamp>/best_model.pt` and symlinked as `models/latest/best_model.pt`. Each checkpoint embeds the `Config` dict for full reproducibility.

### LaTeX (Thesis Document)

All `.tex` files under `docs/` must include `\usepackage{tfg}` (points to `docs/main/tfg.sty`). Use macros: `\code{}`, `\term{}` (first use of a term), `\important{}`. Use `\tfgheadrule` for table top rules. Bibliography: `biblatex` with `style=ieee, sorting=none`. Citation keys follow Better BibLaTeX camelCase: e.g., `\cite{dalitzComparativeStudyStaff2008}` — never `{author2024}` style.

LaTeX color names mirror the Python palette: `tfgPrimary`, `tfgSecondary`, `tfgTertiary`, `tfgHighlight`, `tfgNeutralDark/Mid/Light`.

## Project Structure

```
data/
  raw/primus/           # PrIMuS dataset source
  processed/primus/
    clean/              # Rendered PNGs + LMX annotations
    scanned/            # Augmented scan-simulation images
  vocab/
    primus_lmx.txt      # Built vocabulary file

models/
  run_<timestamp>/      # Training run artifacts
  latest/               # Symlink to most recent run

src/
  cli.py                # Unified CLI entry point
  style.py              # Central styling module (matplotlib + colors)
  CRNN_CTC/             # Model training code
  omr_pipeline/         # Inference pipeline
  data_processing/      # Dataset generation
  api/                  # FastAPI server

notebooks/              # Jupyter notebooks for evaluation
```

## Development Notes

- **No automated tests or linters** are configured
- Run notebooks via `poetry run jupyter lab`
- **Memory target**: RTX 3060 (12 GB); use ResNet18/MobileNet backbones and `torch.amp` mixed precision
- **Primary evaluation metric**: SER (Symbol Error Rate), not accuracy
- **No Verovio** — use LilyPond or MuseScore for score rendering
- Set `OMR_DEBUG_DIR` to save intermediate crops during pipeline execution for inspection
