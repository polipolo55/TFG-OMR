# TFG-OMR

Optical Music Recognition (OMR) project for sheet music analysis using computer vision and deep learning.

## Overview

This repository contains experiments and development work for a Final Degree Project focused on extracting musical information from score images.

Current work is organized around:
- Baseline approaches in `src/simple_baseline/`
- CRNN + CTC approaches in `src/CRNN-CTC/`
- Supporting documentation in `docs/`

## Quick start

Requirements:
- Python `~3.14`

Install dependencies with Poetry:

```bash
poetry install
```

Run commands via the unified CLI inside the virtual environment:

```bash
poetry run python src/cli.py <command> [options]
```

## CLI Usage Guide

The `src/cli.py` script is the main entry point for the entire pipeline. Below are the key commands and typical workflows. Run any command with `--help` for full options.

### 1. Data Processing
Process raw PrIMuS datasets into usable formats (images + annotations).

```bash
# Render raw semantic data to images and LMX annotations
poetry run python src/cli.py render --source data/primus/package_aa --output data/realbook_primus_aa

# Alternatively, convert only the annotations without rendering images
poetry run python src/cli.py convert --source data/realbook_primus_aa 

# Create scanned image augmentations
poetry run python src/cli.py augment --source data/realbook_primus_aa --output data/realbook_primus_aa_scanned
```

### 2. Training
Train the CRNN-CTC model on the processed dataset.

```bash
# Build the vocabulary first
poetry run python src/cli.py vocab --data-dir data/realbook_primus_aa

# Train the model (uses ResNet18 by default)
poetry run python src/cli.py train \
  --data-dir data/realbook_primus_aa \
  --scanned-dir data/realbook_primus_aa_scanned \
  --epochs 50 \
  --batch-size 16
```
*Note: To include additional datasets (like `package_ab`), simply use `--extra-data-dir` and `--extra-scanned-dir` flags.*

### 3. Evaluation
Evaluate a trained checkpoint to calculate the Symbol Error Rate (SER).

```bash
# Evaluate on the test set using beam-search decoding
poetry run python src/cli.py evaluate \
  --checkpoint models/best_model.pt \
  --split test \
  --beam-width 10
```
or the premade evaluation notebook found in `notebooks/02_evaluate_model.ipynb`.
## Project structure

- `data/`: datasets and data notes
- `docs/`: report drafts and project documentation
- `models/`: saved model artifacts
- `notebooks/`: exploratory notebooks
- `src/`: implementation code
