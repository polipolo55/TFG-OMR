# TFG-OMR Pipeline Overview

This document provides a high-level overview of the end-to-end Optical Music Recognition (OMR) pipeline implemented in this project. The system processes raw symbolic music annotations into rendered images and sequence labels, trains a neural network to recognize these sequences, and evaluates the model's accuracy.

---

## 1. Data Processing

The pipeline begins with the PrIMuS dataset, extracting annotations, rendering images, and preparing data for sequence modeling. These steps correspond to the `render`, `convert`, `augment`, and `vocab` subcommands in `src/cli.py`.

### A. Rendering (`render`)
The `render` step takes the PrIMuS `.semantic` encoding and generates clean, stylized sheet music images. 
- **Input:** PrIMuS semantic data.
- **Output:** PNG images styled using the LilyJAZZ font to simulate handwritten/jazz notation aesthetics.
- **Optional:** In parallel, it can also produce corresponding `.lmx` (LilyPond Music XML) annotations.

### B. Conversion (`convert`)
The `convert` step translates the raw PrIMuS representations into the target sequence format.
- It converts `.semantic` annotations into monophonic LMX sequences using `music21`.
- The token representations serve as the ground truth sequence for the CTC loss during training.

### C. Augmentation (`augment`)
To improve model robustness on real-world data, the `augment` step distorts the clean rendered images.
- Applies transformations to simulate physical scanning processes (e.g., blurring, noise, rotations, morphological operations).
- **Output:** A secondary dataset of "scanned" image variants.

### D. Vocabulary Building (`vocab`)
Before training, the system defines the model's token universe.
- The `vocab` step scans the generated `.lmx` sequence files.
- Builds a sorted vocabulary of all unique tokens present in the dataset, adding necessary special tokens (e.g., a CTC blank token).
- Saves this vocabulary to a file used by the model for classification and decoding.

---

## 2. Model Architecture

The project employs a Convolutional Recurrent Neural Network (CRNN) optimized with Connectionist Temporal Classification (CTC) loss, tailored for monophonic music recognition.

### A. CNN Feature Extractor
A Convolutional Neural Network (CNN) acts as the vision backbone.
- Typically a VGG-style block or ResNet variant.
- It takes a grayscale input image of fixed height (e.g., `H=128`) and arbitrary width.
- Collapses the height dimension to 1 (e.g., via `AdaptiveAvgPool2d`) while retaining a downsampled width dimension `W'`. This produces a sequence of visual feature vectors along the horizontal axis.

### B. RNN Sequence Modeling
The visual feature sequence is passed to a Recurrent Neural Network (RNN).
- Employs stacked Bidirectional LSTMs (e.g., 2 layers).
- The BiLSTM contextualizes the visual features based on the entire sequence, capturing dependencies between musical symbols, notes, and staff lines.

### C. Fully Connected Output & CTC Loss
The RNN output at each horizontal timestep is projected to the vocabulary size.
- **CTC Alignment:** Since the raw horizontal timesteps do not have a 1-to-1 alignment with the sequence labels, Connectionist Temporal Classification (CTC) loss is used during training.
- The CRNN predicts the probability of every vocabulary token (plus the blank token) for every timestep.

---

## 3. Training & Evaluation

The final stages involve training the CRNN on the prepared dataset and evaluating its performance.

### A. Training (`train`)
The `train` subcommand orchestrates the learning loop.
- Loads image-sequence pairs (using clean or `augmented` datasets).
- Can apply on-the-fly filtering (e.g., removing rest-heavy or multi-staff samples).
- Uses AdamW optimization with Automatic Mixed Precision (AMP) for faster and memory-efficient training.
- Periodically validates on a hold-out set, saving the best checkpoint based on validation loss/accuracy.

### B. Evaluation (`evaluate`)
The `evaluate` subcommand measures the model's predictive accuracy.
- Loads a trained `.pt` checkpoint.
- Runs inference on an evaluation split (train/val/test).
- Evaluates the predicted sequences against the ground truth using **Symbol Error Rate (SER)**, summarizing how well the model acts as an OMR transcriber.

---

## Summary CLI Usage

`cli.py` is the unified entry point. You can run `poetry run python src/cli.py <command> --help` to see specific flags.

```bash
# Data Preparation
poetry run python src/cli.py render --source data/primus/... --output data/rendered
poetry run python src/cli.py convert --source data/rendered
poetry run python src/cli.py vocab --data-dir data/rendered

# Train & Eval
poetry run python src/cli.py train --epochs 50 --batch-size 16 --lr 1e-3
poetry run python src/cli.py evaluate --checkpoint models/best_model.pt --split test
```
