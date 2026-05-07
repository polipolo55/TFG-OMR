# Training

**Location:** `src/CRNN_CTC/train.py`, `src/CRNN_CTC/dataset.py`

## Running Training

```bash
# Full pipeline then train:
poetry run python src/cli.py pipeline-train [options]

# Train only (data already prepared):
poetry run python src/cli.py train \
  --data-dir data/processed/primus/clean \
  --scanned-dir data/processed/primus/scanned \
  --vocab-path data/vocab/primus_lmx.txt \
  --model-dir models/my_run \
  --epochs 60 --batch-size 16 --lr 1e-3
```

## Dataset Construction

**File:** `src/CRNN_CTC/dataset.py`

### Splits

`make_splits()` partitions samples deterministically (seeded) into train/val/test.

| Split | Fraction | Use |
|-------|----------|-----|
| Train | 1 - val_frac - test_frac | gradient updates |
| Val | 0.10 | early stopping, checkpoint selection |
| Test | 0.10 | final SER report |

Both clean (LilyJAZZ) and scanned (augmented) images are included when `use_scanned=True`.

### Sample Filtering

Applied before splits. Filters are flags on `Config`:

| Flag | What gets removed |
|------|------------------|
| `filter_multi_staff` | images taller than 180 px (LilyPond two-line wraps) |
| `filter_non_leadsheet_clef` | any clef that is not G2 (treble) |
| `filter_unusual_time` | time signatures not in jazz common set |

These three flags collectively realise the lead-sheet domain spec — see
`docs/overview.md` → "Domain Specification" for the full rationale.

### Rare Token Oversampling

Samples whose `.lmx` contains any token in `Config.rare_lmx_tokens` are
duplicated `rare_lmx_oversample` times (default: 2×) in the training index
only.  The default set up-weights:

- `tied:start` / `tied:stop` — visually subtle, under-predicted on real scans.
- `key:fifths:0` (C major) — under-represented in PrIMuS (~0.05 % of corpus)
  but extremely common in the Real Book.

### Image Preprocessing (at load time)

1. Load PNG as grayscale uint8
2. Resize to height `img_height=128` px; scale width proportionally
3. Convert to float32 in [0.0, 1.0]
4. Per-image zero-mean, unit-variance normalization

### Header Stripping Augmentation

With probability `strip_header_prob=0.4` during training, the clef, key, and time-signature tokens are removed from the label AND the corresponding pixels are cropped from the image. This teaches the model to handle continuation lines (lines 2+ in a Real Book page, which typically omit clef/key/time).

### Collation

Images in a batch are right-padded (with zeros after normalization) to the maximum width in that batch. CTC labels are packed into a 1-D tensor with a separate `label_lens` tensor.

Batch dict keys: `images`, `labels`, `label_lens`, `image_widths`

## Training Loop

**Optimizer:** AdamW (`lr=1e-3` peak, `weight_decay=1e-4`)

**Scheduler:** OneCycleLR (single-cycle cosine annealing)
- Warm-up fraction: 8% of total steps
- Peak at `lr`, cosine decay to `lr/1e4`

**Loss:** `nn.CTCLoss(blank=0, zero_infinity=True)`

**Mixed Precision:** `torch.amp` with `GradScaler` — reduces VRAM usage, accelerates training on RTX hardware.

**Gradient clipping:** max norm = 5.0 (prevents exploding gradients in LSTM)

**Per epoch:**
1. Train loop: forward → CTC loss → backward → clip → optimizer step → scheduler step
2. Val loop: SER on full validation set (greedy decode)
3. Checkpoint: save `best_model.pt` if val SER improved; save `latest.pt` always
4. Log: `{model_dir}/training_log.csv` (epoch, train_loss, val_loss, val_ser)
5. Early stopping: if val SER does not improve for `early_stopping_patience=12` epochs, stop

## Hyperparameter Reference

| Parameter | Default | Notes |
|-----------|---------|-------|
| `epochs` | 60 | maximum epochs |
| `batch_size` | 16 | per-GPU |
| `lr` | 1e-3 | OneCycleLR peak |
| `weight_decay` | 1e-4 | AdamW |
| `warmup_frac` | 0.08 | fraction of steps for warm-up |
| `early_stopping_patience` | 12 | epochs without improvement |
| `max_grad_norm` | 5.0 | gradient clip |
| `num_workers` | 10 | DataLoader workers |
| `val_frac` | 0.10 | validation split fraction |
| `test_frac` | 0.10 | test split fraction |
| `use_scanned` | True | include augmented images |
| `strip_header_prob` | 0.4 | header-stripping augmentation |
| `rare_lmx_oversample` | 2 | oversampling factor for ties |

## Checkpoint Files

| File | Contents |
|------|---------|
| `best_model.pt` | state dict with lowest val SER |
| `latest.pt` | most recent epoch (resume support) |
| `training_log.csv` | per-epoch metrics |
| `config.json` | full Config serialized |

Checkpoint format (PyTorch dict):
```python
{
  "model_state_dict": ...,
  "optimizer_state_dict": ...,
  "epoch": int,
  "val_ser": float,
  "config": dict,
}
```

## Resuming Training

The CLI auto-detects `latest.pt` in `--model-dir` and resumes from that epoch, restoring optimizer and scheduler state.

```bash
poetry run python src/cli.py train \
  --model-dir models/my_run \
  --epochs 80   # extend from where it stopped
```

## Fine-tuning

Use `--finetune-data-dir` (repeatable) to inject additional labeled samples into the training split only (not val/test). This is designed for fine-tuning on real scanned Real Book pages.

```bash
poetry run python src/cli.py train \
  --data-dir data/processed/primus/clean \
  --finetune-data-dir data/realbook_scans/annotated \
  --checkpoint models/latest/best_model.pt
```

## Evaluation

```bash
# Test set SER:
poetry run python src/cli.py evaluate \
  --checkpoint models/latest/best_model.pt \
  --split test \
  --beam-width 10

# Compare beam widths:
poetry run python src/cli.py evaluate-ab \
  --checkpoint models/latest/best_model.pt \
  --beams 1,5,10
```

**Metric:** Symbol Error Rate (SER) — token-level edit distance (Levenshtein) normalized by ground-truth length.

```
SER = total_edit_distance / total_ground_truth_length
```

A complementary **melodic SER** (`evaluate --melodic` / `melodic_ser` in
`evaluate.py`) is computed on the same edit distance after stripping
`measure`, `tied:start`, `tied:stop`.  On the current PrIMuS test split
this metric is roughly 7× lower than aggregate SER because barlines and
ties dominate the error budget; it is the better headline number for
"how often does the model get the actual notes right".
