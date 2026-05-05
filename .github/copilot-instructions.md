# TFG-OMR — GitHub Copilot Instructions

End-to-end Optical Music Recognition (OMR) system for monophonic jazz lead sheets (The Real Book). Core model: CRNN-CTC (ResNet18 CNN → BiLSTM → CTC loss). Symbolic output encoding: LMX (Linear MusicXML), vocabulary in `src/CRNN_CTC/vocabulary.txt`.

## Environment & Commands

**Python 3.14, managed with Poetry.**

```bash
# Install
poetry install

# Full pipeline via unified CLI
poetry run python src/cli.py render   --source data/primus/package_aa --output data/realbook_primus_aa
poetry run python src/cli.py convert  --source data/realbook_primus_aa
poetry run python src/cli.py augment  --source data/realbook_primus_aa --output data/realbook_primus_aa_scanned
poetry run python src/cli.py vocab    --data-dir data/realbook_primus_aa
poetry run python src/cli.py train    --epochs 50 --batch-size 16
poetry run python src/cli.py evaluate --checkpoint models/latest/best_model.pt --split test

# Include multiple datasets
poetry run python src/cli.py train \
  --data-dir data/realbook_primus_aa \
  --scanned-dir data/realbook_primus_aa_scanned \
  --extra-data-dir data/realbook_primus_ab \
  --extra-scanned-dir data/realbook_primus_ab_scanned
```

There are no automated tests or linters configured. Run notebooks via `poetry run jupyter lab`.

## Architecture

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

- `src/CRNN_CTC/model.py` — `CRNN` class with swappable CNN (`backbone: "resnet18"` or `"vgg"`)
- `src/CRNN_CTC/config.py` — single `Config` dataclass for all hyperparameters; serialised into every checkpoint
- `src/CRNN_CTC/dataset.py` — `OMRDataset` loads `{id}.png` + `{id}.lmx` pairs; applies multi-stage filtering (rest-heavy, unwanted clefs, multi-staff by height)
- `src/CRNN_CTC/vocab.py` — `Vocabulary`: blank at index 0, pad at index 1, music tokens from index 2
- `src/CRNN_CTC/train.py` — AdamW + OneCycleLR + AMP + early stopping; saves `best_model.pt` per run under `models/run_<timestamp>/`
- `src/CRNN_CTC/evaluate.py` — greedy CTC decode, SER (Symbol Error Rate = edit distance / ground-truth length)
- `src/data_processing/generate_realbook.py` — LilyPond + LilyJAZZ rendering of PrIMuS → PNG + LMX
- `src/data_processing/semantic_to_lmx.py` — converts PrIMuS `.semantic` to monophonic LMX via music21; skips `multirest-N` tokens

## Key Conventions

### Python styling (every notebook and script)
Every notebook and standalone script **must** start with:
```python
import sys; sys.path.insert(0, "../src")  # adjust relative depth
import style; style.apply()
```
Use `style.C["<role>"]` for all explicit colours — never hardcode hex literals. Available roles: `primary`, `secondary`, `tertiary`, `highlight`, `primary_light`, `secondary_light`, `tertiary_light`, `highlight_light`, `neutral_dark`, `neutral_mid`, `neutral_light`.

### Data filtering in `OMRDataset`
Three filters are applied before training (all enabled by default in `Config`).
They collectively realise the **lead-sheet domain spec** — see
`docs/overview.md` → "Domain Specification" for the full rationale:
- **`filter_multi_staff`** — drops images whose original height >180 px (LilyPond multi-staff wraps; single-staff images are 84–152 px).
- **`filter_non_leadsheet_clef`** — drops any sample whose clef is not `clef:G2` (treble).  Real Book is treble-only.
- **`filter_unusual_time`** — drops time signatures outside `{4/4, 3/4, 2/4, 2/2, 6/8, 6/4, 5/4, 12/8}` — the jazz common-time set.

### LMX annotation format
`.lmx` files are space-separated token sequences. Example: `clef:G2 key:fifths:2 time:4/4 measure E4 quarter F#4 quarter G4 half measure`. Token index 0 = CTC blank, index 1 = pad; music tokens start at index 2.

### Model checkpoints
Checkpoints are saved to `models/run_<timestamp>/best_model.pt` and symlinked as `models/latest/best_model.pt`. Each checkpoint embeds the `Config` dict for full reproducibility.

### LaTeX (thesis document)
All `.tex` files under `docs/` must include `\usepackage{tfg}` (points to `docs/main/tfg.sty`). Use macros: `\code{}`, `\term{}` (first use of a term), `\important{}`. Use `\tfgheadrule` for table top rules. Bibliography: `biblatex` with `style=ieee, sorting=none`. Citation keys follow Better BibLaTeX camelCase: e.g., `\cite{dalitzComparativeStudyStaff2008}` — never `{author2024}` style.

LaTeX colour names mirror the Python palette: `tfgPrimary`, `tfgSecondary`, `tfgTertiary`, `tfgHighlight`, `tfgNeutralDark/Mid/Light`.

### Scope constraints
- **Monophonic only** — single melody line, no chords, no polyphony.
- **No Verovio** — use LilyPond or MuseScore for score rendering.
- **Memory target** — RTX 3060 (12 GB); use ResNet18/MobileNet backbones and `torch.amp` mixed precision.
- Primary evaluation metric is **SER** (Symbol Error Rate), not accuracy.
