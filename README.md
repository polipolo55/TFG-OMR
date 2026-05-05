# TFG-OMR

Optical Music Recognition (OMR) for monophonic jazz lead sheets (The Real Book style).
CRNN-CTC model trained on LilyJAZZ-rendered PrIMuS data.

Bachelor's thesis (TFG) at FIB/UPC by Pol Casanovas Puig.

---

## System Requirements

Before installing Python dependencies, install these system packages:

### LilyPond

LilyPond is used to render `.semantic` notation into PNG images.

```bash
# Fedora / RHEL
sudo dnf install lilypond

# Debian / Ubuntu
sudo apt install lilypond

# macOS (Homebrew)
brew install lilypond
```

Verify: `lilypond --version` (tested with 2.26.0)

### LilyJAZZ Font

LilyJAZZ is a separate third-party font that gives renders the hand-written Real Book look.
It is **not** bundled with LilyPond and must be installed manually after every LilyPond upgrade.

```bash
# 1. Clone the LilyJAZZ repository
git clone --depth=1 https://github.com/OpenLilyPondFonts/lilyjazz.git /tmp/lilyjazz

# 2. Detect your LilyPond version
LILY_VER=$(lilypond --version 2>&1 | grep -oP '\d+\.\d+\.\d+' | head -1)
LILY_SHARE="/usr/share/lilypond/$LILY_VER"

# 3. Copy font files into LilyPond's directories
sudo cp /tmp/lilyjazz/otf/*.otf                            "$LILY_SHARE/fonts/otf/"
sudo cp /tmp/lilyjazz/svg/*.svg                            "$LILY_SHARE/fonts/svg/"
sudo cp /tmp/lilyjazz/supplementary-files/**/*.otf         "$LILY_SHARE/fonts/otf/"
sudo cp /tmp/lilyjazz/stylesheet/*.ily                     "$LILY_SHARE/ly/"
```

> **LilyPond ≥ 2.25 compatibility:** The official LilyJAZZ stylesheet uses `set-global-fonts`
> which was removed in LilyPond 2.25. After copying the stylesheet, patch
> `$LILY_SHARE/ly/lilyjazz.ily` to use the new `property-defaults.fonts.*` API:
>
> ```lilypond
> \version "2.26.0"
>
> \paper {
>   property-defaults.fonts.music = "lilyjazz"
>   property-defaults.fonts.brace = "lilyjazz"
>   property-defaults.fonts.serif = "lilyjazz-text"
>   property-defaults.fonts.sans  = "lilyjazz-chord"
> }
>
> \layout {
>   \override Score.Hairpin.thickness = #2
>   \override Score.Stem.thickness = #2
>   \override Score.TupletBracket.thickness = #2
>   \override Score.VoltaBracket.thickness = #2
>   \override Score.SystemStartBar.thickness = #4
>   \override StaffGroup.SystemStartBracket.padding = #0.25
>   \override ChoirStaff.SystemStartBracket.padding = #0.25
>   \override Staff.Tie.line-thickness = #2
>   \override Staff.Slur.thickness = #3
>   \override Staff.PhrasingSlur.thickness = #3
>   \override Staff.BarLine.hair-thickness = #4
>   \override Staff.BarLine.thick-thickness = #8
>   \override Staff.MultiMeasureRest.hair-thickness = #3
>   \override Staff.MultiMeasureRestNumber.font-size = #2
>   \override LyricHyphen.thickness = #3
>   \override LyricExtender.thickness = #3
>   \override PianoPedalBracket.thickness = #2
> }
> ```
>
> Test with: `poetry run python -c "from src.CRNN_CTC.lilypond_render import render_tokens; print(render_tokens('measure clef:G2 key:fifths:0 time beats:4 beat-type:4 pitch:C octave:5 quarter'.split()))"`
> Expected output: `Render OK — image shape: (...)` or similar non-None result.

---

## Python Setup

Requires Python `~3.14`. Install dependencies with Poetry:

```bash
poetry install
```

---

## Quick Start

All commands run through the unified CLI via `poetry run`:

```bash
# 1. Run the full data pipeline (render + convert + augment + vocab)
poetry run python src/cli.py pipeline \
  --raw-primus-dir data/raw/primus \
  --clean-dir data/processed/primus/clean \
  --scanned-dir data/processed/primus/scanned \
  --vocab-path data/vocab/primus_lmx.txt \
  --workers 8

# 2. Train the model
poetry run python src/cli.py train \
  --epochs 60 --batch-size 16

# 3. Evaluate
poetry run python src/cli.py evaluate \
  --checkpoint models/latest/best_model.pt \
  --split test --beam-width 10

# 4. Or: run everything in one shot
poetry run python src/cli.py pipeline-train \
  --raw-primus-dir data/raw/primus \
  --workers 8 --epochs 60 --batch-size 16
```

---

## Project Structure

```
src/
├── cli.py                  Unified CLI (render / convert / augment / vocab / train / evaluate / api / pipeline)
├── CRNN_CTC/               Model, training, vocab, evaluation
│   ├── lilypond_render.py  Single source of truth for LMX→LilyPond lookup tables + render pipeline
│   ├── config.py           Config dataclass (serialized in every checkpoint)
│   ├── dataset.py          OMRDataset + sample filters
│   ├── vocab.py            Vocabulary (blank=0, pad=1, unk=2, tokens=3+)
│   ├── model.py            CRNN-CTC model (ResNet18 or VGG backbone)
│   └── train.py / evaluate.py
├── data_processing/
│   ├── generate_realbook.py    PrIMuS .semantic → LilyJAZZ PNG
│   ├── semantic_to_lmx.py      PrIMuS .semantic → LMX tokens
│   └── augment_scanned.py      Clean PNG → scan-simulated PNG
└── omr_pipeline/           Full inference pipeline (staff detect + CRNN + chord OCR)

data/
├── raw/primus/             PrIMuS dataset packages (package_aa/, package_ab/, …)
└── processed/primus/
    ├── clean/              LilyJAZZ PNGs + .lmx labels
    └── scanned/            Scan-augmented copies

models/latest/              Best checkpoint + config
notebooks/                  Jupyter evaluation notebooks
docs/                       Architecture and format documentation
```

---

## Documentation

| Topic | File |
|-------|------|
| Architecture overview | `docs/overview.md` |
| Data pipeline stages | `docs/data_pipeline.md` |
| LMX token format | `docs/lmx_format.md` |
| Model and training | `docs/model.md`, `docs/training.md` |
| CLI reference | `docs/cli.md` |
| Config fields | `docs/configuration.md` |
| Inference pipeline | `docs/inference_pipeline.md` |
| API endpoints | `docs/api.md` |

---

## Performance

- **Synthetic test SER:** ~1.17% (symbol error rate)
- **Hardware target:** NVIDIA RTX 3060 (12 GB VRAM), batch size 16
- **Training time:** ~60 epochs, early stopping at patience 12
