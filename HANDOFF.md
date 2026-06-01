# Handoff: Chapter 5 (Implementation) — TFG-OMR

## Goal
Complete Chapter 5 (`latex_documents/main/chapters/05_implementation.tex`) of the Bachelor's thesis memoir for the TFG-OMR Optical Music Recognition project at FIB/UPC.

---

## Current Progress

### Completed this session

**Section 5.1 — Technology Stack:**
- All 8 `\todo{cite ...}` items resolved. New bibtex entries appended to `references.bib` under the comment `% ---- Implementation technology stack ----`:
  - `paszkeImperativeStyleHighPerformance2019` — PyTorch NeurIPS 2019
  - `torchvision2016` — torchvision @software
  - `harrisArrayProgramming2020` — NumPy Nature 2020
  - `bradskiOpenCVLibrary2000` — OpenCV Dr. Dobb's 2000
  - `nienhuysLilyPondSystem2003` — LilyPond CIM 2003
  - `hammerleOpenLilyPondFontsLilyJAZZ` — LilyJAZZ GitHub 2013
  - `jaidedaiEasyOCR` — EasyOCR GitHub 2020
  - `ramirezFastAPI` — FastAPI GitHub 2018
  - `artifexPyMuPDF` — PyMuPDF GitHub (no year; no formal paper)

**Other sections (from earlier in session):**
- Epoch wall-clock filled in: ~8 min/epoch (mean of epochs 2–49; best epoch = 37)
- Model size filled in: ~165 MB
- Batch-size VRAM todo removed (value ~8.5 GB already stated in prose)
- ResNet18 parameter/FLOPs todo removed (approx values already in prose)
- `persistent_workers` todo replaced with clean prose
- Chord network hyperparameter section completed: two-phase description + `tab:chord-hyperparams` table with values from `chord_train.py` (`rnn_hidden=192`, `batch=32`, `lr=1e-3`/`2e-4`, `epochs=30`/`20`, `patience=8`, `synth_weight=0.4`)
- Random-seed strategy paragraph written: documents `seed_everything(42)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, Generator-seeded splits, augmentation `--seed` flag, CUDA non-determinism caveat
- LilyPond version filled in: **2.26.0** (Guile 3.0)

---

## Remaining Todos in Chapter 5

All remaining `\todo{}` items require either hardware measurement or a new figure asset. None can be resolved from code reading alone.

| Line | Todo | What's needed |
|------|------|---------------|
| ~277 | Figure: augmentation gallery (6 panels) | Run `scripts/generate_augment_samples.py --n-samples 1 --show-steps`; save as `figures/fig_augment_samples.png` and insert with `\includegraphics` |
| ~880 | Median wall-clock latency per page on RTX 3060 | Run `run_pipeline()` on a representative Real Book page, time it, break down by stage |
| ~1054 | Total wall-clock for `run_pipeline()` | Same measurement |
| ~1068 | Per-strip CRNN forward + greedy decode latency at W≈1500 | Time the inference stage separately |

Quick verification that no citation todos remain:
```bash
grep -n "\\\\todo{cite" latex_documents/main/chapters/05_implementation.tex
# should return nothing
```

---

## What Worked

- Reading `models/latest/training_log.csv` directly to extract epoch timing (avg ~494 s/epoch) and best epoch (37)
- Using `ls -lh models/latest/best_model.pt` for model size (165 MB)
- Reading `src/CRNN_CTC/chord_train.py` for all chord hyperparameters (the dataclass `ChordConfig` has every default)
- Reading `src/CRNN_CTC/training_utils.py` for the seed policy (`seed_everything` function)
- Web search + CITATION.cff fetches for library citations

---

## Project Context

- **Skill to invoke first**: `writing-tfg-fib` (loads chapter scope rules and academic voice guidelines)
- **Humanizer**: always invoke `humanizer` skill before writing new prose — no em dashes, varied sentence length, no AI vocabulary
- **Chapter 5 scope** (from skill): *how* the Ch. 4 decisions were realised — concrete libraries, versions, hyperparameters, magic numbers. Do NOT re-justify architecture (that belongs in Ch. 4).
- **Key docs**: `docs/overview.md` (read first), `docs/data_pipeline.md`, `docs/training.md`
- **Do not build the PDF** — user compiles it themselves (see memory)
- **Always `poetry run python`**, never bare `python`

---

## File Locations

| File | Purpose |
|------|---------|
| `latex_documents/main/chapters/05_implementation.tex` | Chapter 5 source |
| `latex_documents/main/references.bib` | Bibliography (new entries at bottom) |
| `models/latest/training_log.csv` | Per-epoch metrics (loss, SER, elapsed_s) |
| `models/latest/best_model.pt` | OMR checkpoint (165 MB, epoch 37) |
| `models/chord/finetune_20260601_114246/best_model.pt` | Chord checkpoint (51 MB) |
| `src/CRNN_CTC/chord_train.py` | Chord hyperparameter defaults (ChordConfig) |
| `src/CRNN_CTC/training_utils.py` | seed_everything() implementation |
