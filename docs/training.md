# Training

**Location:** `src/CRNN_CTC/train.py`, `src/CRNN_CTC/dataset.py`

## Running Training

```bash
# Full pipeline (render → convert → twins → augment → vocab) then train:
poetry run python src/cli.py pipeline-train [options]

# Full rebuild of all prepared data:
poetry run python src/cli.py pipeline-train --force-all [options]

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

### Rare Token Oversampling (disabled by default)

Samples whose `.lmx` contains any token in `Config.rare_lmx_tokens` can be
duplicated `rare_lmx_oversample` times in the training index only. The intended
target is `tied:start` / `tied:stop` (visually subtle, under-predicted on real
scans).

**This is disabled by default (`rare_lmx_oversample=1`) as of the 2026-06-12
retrain.** That run used 2× oversampling and the tie category still showed
**31 % error** on the test split — duplicating whole staff images does not teach
the model to disambiguate faint tie arcs on degraded scans, so the ~10 % extra
train time bought no tie accuracy. The machinery and `--rare-lmx-oversample`
flag are kept for ablations. See
`docs/experiments/2026-06-10-volume-collapse-findings.md`.

(`key:fifths:0` was previously in the target set; it was removed once the
converter bug under-representing C major was fixed — see
`docs/configuration.md`.)

### Scan-Variant Sampling (train-only)

When `Config.scanned_variant_dirs` is set (CLI `--scanned-variant-dirs`), each
training `__getitem__` picks uniformly among the base scanned image plus that
sample's extra variants (`{sid}_augNN` dirs produced by `augment_scanned.py
--copies N`). Val/test always use the deterministic base image. Because
variants are keyed by sample id and the split is over ids, no variant can cross
the train/val/test boundary — this is the **leakage-free** replacement for the
header-stripped twins that were deleted on 2026-06-03 (those twins had been
added to the *split pool*, leaking near-duplicates across train/test). See
`docs/experiments/2026-06-10-volume-collapse-findings.md`.

> **Note:** before 2026-06-10, `_AugSubset` mutated a shared dataset attribute,
> so online jitter leaked into in-loop validation. Fixed in `e786a40`; offline
> `scripts/evaluate_full.py` was never affected.

### Image Preprocessing (at load time)

1. Load PNG as grayscale uint8
2. Resize to height `img_height=128` px; scale width proportionally
3. Convert to float32 in [0.0, 1.0]
4. Per-image zero-mean, unit-variance normalization

### Continuation staves

Real Book continuation staves (all staves after the first on a page) carry no
header. The model is trained only on full-header PrIMuS staves — this is by
design. At inference, `header_injector.py` prepends a prerendered header to
each continuation staff before the CRNN (see `docs/inference_pipeline.md`), so
the model always receives its training distribution.

`strip_header_prob` appears in `Config` but is **DEPRECATED/inert** — it is
kept only so existing checkpoints deserialise without error.

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
| `strip_header_prob` | 0.0 | DEPRECATED/inert — continuation staves handled at inference via virtual header injection |
| `rare_lmx_oversample` | 1 | oversampling factor for ties (disabled by default; see above) |

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

Warm-start fine-tuning adapts a trained model to real data. It combines two flags:

- `--init-weights <ckpt>` loads **only** the model weights from a checkpoint, then
  starts a **fresh** run (new run dir, fresh optimiser, fresh OneCycle at `--lr`).
  This is the correct way to fine-tune — it does **not** touch the baseline run, and
  it lets you pick a low LR. Mutually exclusive with `--resume`.
- `--finetune-data-dir <dir>` (repeatable) appends labeled real samples to the
  **train split only** (never val/test — see `dataset.py::make_splits`), so the
  synthetic→real domain-gap measurement on val/test stays honest.

```bash
poetry run python src/cli.py train \
  --data-dir data/processed/primus/clean \
  --scanned-dir data/processed/primus/scanned \
  --vocab-path data/vocab/primus_lmx.txt \
  --init-weights models/latest/best_model.pt \
  --finetune-data-dir data/finetune/realbook/clean \
  --model-dir models/finetune_realbook \
  --epochs 12 --lr 1e-4
```

Use a **low `--lr`** (≈1e-4) and **few `--epochs`** (≈10–15): a small real set
overfits fast. After fine-tuning, re-calibrate the staff-reject gate
(`cli.py calibrate-reject …`, CLAUDE.md hard constraint #7).

> **`--init-weights` vs `--resume`.** `--resume` *continues an interrupted run*:
> it restores optimiser + scheduler + epoch counter and keeps training the **same**
> run in place (used to recover from a crash). It is **not** a fine-tuning tool —
> resuming a finished run replays the decayed tail of its LR schedule and would
> overwrite the original checkpoints. For fine-tuning always use `--init-weights`.

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
