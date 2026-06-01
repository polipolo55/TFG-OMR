# TFG-OMR Pipeline Overview

This document provides a high-level overview of the end-to-end Optical Music Recognition (OMR) system: **data preparation** (PrIMuS → training pairs), **model training**, and **inference** on lead-sheet images.

---

## 1. Data processing (offline)

The training-data pipeline is implemented in `src/cli.py` as five stages, run in order by `pipeline` or `pipeline-train`:

| Stage | CLI / script | Purpose |
|-------|----------------|---------|
| 1. Render | `render` → `generate_realbook.py` | PrIMuS `.semantic` → LilyJAZZ PNG (+ inline `.lmx`) |
| 2. Convert | `convert` → `semantic_to_lmx.py` | Re-sync `.semantic` → `.lmx` (skips up-to-date unless forced) |
| 3. Twins | `generate_headerless_twins.py` | `__nh` continuation-staff samples (clef/time hidden) |
| 4. Augment | `augment` → `augment_scanned.py` | Scan simulation on clean + twin PNGs |
| 5. Vocab | `vocab` | Union of all `.lmx` tokens → `data/vocab/primus_lmx.txt` |

Use `--force-all` on `pipeline` / `pipeline-train` to re-render, re-convert, re-twin, and re-augment every sample.

Domain filters (treble clef, common time signatures, single-staff height) are applied later in `OMRDataset` at training time — see `docs/data_pipeline.md`.

### A. Rendering (`render`)

- **Input:** PrIMuS `.semantic` packages under `data/raw/primus/`.
- **Output:** `data/processed/primus/clean/{id}/{id}.png` and labels.

### B. Conversion (`convert`)

- Direct token mapping from PrIMuS semantic format to monophonic LMX (not MusicXML).
- Ground truth for CTC training.

### C. Header-less twins

- Fraction of treble samples (default 35%) get a sibling `{id}__nh/` directory.
- Image omits clef/time glyphs; label drops matching header tokens.

### D. Augmentation (`augment`)

- Albumentations + post-processing to simulate scans.
- **Output:** `data/processed/primus/scanned/` mirroring clean sample ids (including twins).

### E. Vocabulary (`vocab`)

- Scans all `.lmx` files; builds sorted token list (~77 content tokens on a full rebuild, 80 with specials).
- Special indices 0–2 (`blank`, `pad`, `unk`) are injected in code only.

---

## 2. Model architecture

Convolutional Recurrent Neural Network (CRNN) with Connectionist Temporal Classification (CTC) for monophonic melody recognition. See `docs/model.md`.

---

## 3. Training

`cli.py train` or `pipeline-train` (pipeline + train). Checkpoints store full `Config` and vocabulary tokens.

---

## 4. Inference (five stages)

PDF/image → preprocess → staff detect → music CRNN + chord CRNN → grammar fix → JSON. See `docs/inference_pipeline.md`.

---

For authoritative CLI flags and examples, see `docs/cli.md`.
