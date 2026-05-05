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

## Domain Scope

- **Monophonic only** (single melodic line per staff, no polyphony)
- **Jazz lead sheets** (treble clef, G2 / treble clef only)
- **Common jazz meters:** 4/4, 3/4, 2/4, 2/2, 6/8, 5/4
- **Real Book aesthetics:** LilyJAZZ-rendered synthetic training data

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
        └──── [Chord OCR] ──────────── EasyOCR → chord tokens
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
| Chord OCR | EasyOCR |
| Web API | FastAPI + Uvicorn |
| Data augmentation | albumentations |
| Package manager | Poetry |

## Repository Structure

```
src/
├── cli.py                  unified CLI (9 subcommands)
├── style.py                matplotlib theme
├── CRNN_CTC/               model, training, vocab, evaluation
├── data_processing/        dataset generation and augmentation
├── omr_pipeline/           full inference pipeline
└── api/                    FastAPI web server

data/
├── raw/primus/             PrIMuS dataset packages
└── processed/primus/
    ├── clean/              rendered LilyJAZZ PNGs + .lmx labels
    └── scanned/            augmented (distorted) copies

models/latest/best_model.pt best checkpoint
scripts/                    standalone utilities
notebooks/                  Jupyter evaluation notebooks
latex_documents/gep/        GEP thesis-management deliverables
```

## Performance

- **Synthetic test SER:** ~1.17% (symbol error rate, token-level edit distance)
- **Hardware target:** NVIDIA RTX 3060 (12 GB VRAM), batch size 16
- **Training time:** ~60 epochs, early stopping at patience 12
