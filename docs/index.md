# TFG-OMR Documentation

Optical Music Recognition system for monophonic jazz lead sheets (The Real Book).
Bachelor's thesis (TFG) at FIB/UPC by Pol Casanovas Puig.

## Documents

| File | Contents |
|------|---------|
| [overview.md](overview.md) | Project goals, architecture diagram, tech stack, output format |
| [data_pipeline.md](data_pipeline.md) | Data generation: PrIMuS → LilyJAZZ → LMX → twins → augmentation → vocab |
| [model.md](model.md) | CRNN-CTC architecture: CNN backbones, BiLSTM, CTC, vocabulary |
| [training.md](training.md) | Training loop, dataset splits, hyperparameters, checkpointing, fine-tuning |
| [inference_pipeline.md](inference_pipeline.md) | Full OMR pipeline: preprocess → staff detect → CRNN → chord OCR → grammar fix |
| [lmx_format.md](lmx_format.md) | LMX token grammar, key encoding, accidental rules, vocabulary file format |
| [cli.md](cli.md) | All CLI subcommands with flags and usage examples |
| [api.md](api.md) | FastAPI web server endpoints and response format |
| [configuration.md](configuration.md) | Complete Config dataclass field reference |
| [notebooks.md](notebooks.md) | Jupyter notebook descriptions and usage |

## Quick Start

```bash
# Install dependencies
poetry install

# Full pipeline: data generation + training (full rebuild)
poetry run python src/cli.py pipeline-train \
  --raw-primus-dir data/raw/primus \
  --clean-dir data/processed/primus/clean \
  --scanned-dir data/processed/primus/scanned \
  --vocab-path data/vocab/primus_lmx.txt \
  --model-dir models/run1 \
  --force-all \
  --workers $(nproc) \
  --epochs 60 --batch-size 24 --num-workers 12

# Evaluate
poetry run python src/cli.py evaluate \
  --checkpoint models/run1/best_model.pt \
  --data-dir data/processed/primus/clean \
  --vocab-path data/vocab/primus_lmx.txt \
  --split test --beam-width 10

# Serve web UI
poetry run python src/cli.py api --port 8000
```

## Source Map

```
src/
├── cli.py                       Unified CLI (13 subcommands; see cli.md)
├── style.py                     Matplotlib theme for figures
│
├── CRNN_CTC/
│   ├── config.py                Config dataclass (all hyperparameters)
│   ├── vocab.py                 Vocabulary: encode/decode/build/load/save
│   ├── model.py                 CRNN: CNNBackbone, ResNetBackbone, BiLSTM
│   ├── dataset.py               OMRDataset, make_splits, collate_fn
│   ├── train.py                 Training loop (AdamW, OneCycleLR, AMP, CTC)
│   ├── evaluate.py              SER computation, greedy/beam decode
│   └── lilypond_render.py       Shared LilyPond lookup tables & subprocess runner
│
├── data_processing/
│   ├── generate_realbook.py     PrIMuS .semantic → LilyJAZZ PNG
│   ├── semantic_to_lmx.py       PrIMuS .semantic → .lmx tokens
│   ├── generate_header_templates.py  prerender 120 header-strip templates
│   └── augment_scanned.py       Scan simulation augmentation
│
├── omr_pipeline/
│   ├── pipeline.py              Orchestrates full inference flow
│   ├── preprocess.py            Binarize, deskew (CLAHE + Otsu + projection)
│   ├── staff_detect.py          Morphological staff line detection
│   ├── inference.py             CRNN batch inference + CTC decode
│   ├── header_injector.py       Prepend header templates to continuation staves
│   ├── chord_recognizer.py      Chord symbol OCR (CRNN-CTC, character-level)
│   ├── grammar_fix.py           Stateful LMX token validator
│   └── chord_postprocess.py     Jazz chord grammar cleanup
│
└── api/
    └── main.py                  FastAPI: GET /, POST /api/omr/lead-sheet, /labeler endpoints
```

## Key Design Decisions

1. **LMX flat token format** — simpler than MusicXML for CTC, easy to validate
2. **CRNN-CTC** — alignment-free, handles variable-width staff images
3. **ResNet18 default** — better gradient flow and convergence than VGG
4. **Domain filtering** — remove orchestral patterns foreign to jazz lead sheets
5. **Scan simulation augmentation** — bridges synthetic-to-real domain gap
6. **Header-less twin samples** — first-class continuation-staff renders (clef+time hidden, key kept) teach the model to read lines 2+ without a header
7. **Single lookup table file** (`lilypond_render.py`) — PNG render and LMX label always in sync
8. **Unified CLI** — all stages accessible via one entry point with full config control
