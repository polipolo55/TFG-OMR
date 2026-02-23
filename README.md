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

Run commands inside the virtual environment:

```bash
poetry run python -m <module>
```

## Project structure

- `data/`: datasets and data notes
- `docs/`: report drafts and project documentation
- `models/`: saved model artifacts
- `notebooks/`: exploratory notebooks
- `src/`: implementation code
