# CLI Reference

**Entry point:** `src/cli.py`
**Run with:** `poetry run python src/cli.py <subcommand> [options]`

All subcommands share a top-level `-v` / `--verbose` flag (enables DEBUG logging).

---

## render

Re-render PrIMuS `.semantic` samples as LilyJAZZ PNGs with inline `.lmx` labels.

```
poetry run python src/cli.py render \
  [--source data/raw/primus]          # dir containing package_aa/, package_ab/, …
  [--output data/processed/primus/clean]
  [--dpi 200]                         # LilyPond render resolution
  [--workers N]                       # parallel processes (default: cpu_count-2)
  [--limit N]                         # stop after N samples (smoke test)
  [--force]                           # re-render even if PNG already exists
  [--no-lmx]                          # skip inline LMX generation
```

---

## convert

Convert `.semantic` files to `.lmx` (direct token remapping, without re-rendering images).

```
poetry run python src/cli.py convert \
  [--source data/processed/primus/clean]
  [--workers N]
  [--limit N]
  [--verbose]
```

---

## augment

Apply scan-simulation augmentation to clean PNGs, writing distorted copies.

```
poetry run python src/cli.py augment \
  [--source data/processed/primus/clean]
  [--output data/processed/primus/scanned]
  [--copies 1]     # augmented copies per original (default: 1)
  [--seed 42]
  [--workers N]
  [--limit N]
```

---

## vocab

Build a vocabulary file from all `.lmx` files under `--data-dir`.

```
poetry run python src/cli.py vocab \
  [--data-dir data/processed/primus/clean]
  [--output data/vocab/primus_lmx.txt]
  [--extra-data-dir <dir>]     # repeatable: add more .lmx sources
  [--workers N]
```

---

## train

Train the CRNN-CTC model.

```
poetry run python src/cli.py train \
  [--data-dir data/processed/primus/clean]
  [--scanned-dir data/processed/primus/scanned]
  [--vocab-path data/vocab/primus_lmx.txt]
  [--model-dir models/]
  [--extra-data-dir <dir>]          # repeatable
  [--extra-scanned-dir <dir>]       # repeatable
  [--finetune-data-dir <dir>]       # repeatable: injected into train split only
  [--finetune-scanned-dir <dir>]    # repeatable
  [--resume [CHECKPOINT]]           # omit path to auto-detect latest checkpoint
  [--backbone resnet18|vgg]
  [--epochs 60]
  [--batch-size 16]
  [--lr 1e-3]
  [--weight-decay 1e-4]
  [--warmup-frac 0.08]
  [--rnn-hidden 256]
  [--rnn-layers 2]
  [--dropout 0.3]
  [--early-stopping-patience 12]
  [--num-workers 10]
  [--img-height 128]
  [--use-scanned | --no-use-scanned]
  [--val-frac 0.10]
  [--test-frac 0.10]
  [--seed 42]
  [--no-filter-rest-heavy]
  [--no-filter-unwanted-clefs]
  [--no-filter-multi-staff]
  [--max-source-height 180]
  [--rare-lmx-oversample 2]
  [--rare-lmx-tokens tied:start,tied:stop]
  [--strip-header-prob 0.4]    # 0 disables training-time header stripping
  [--online-aug-prob 0.5]      # 0 disables online jitter
```

---

## evaluate

Evaluate a checkpoint and report SER on val or test split.

```
poetry run python src/cli.py evaluate \
  --checkpoint <model.pt> \
  [--data-dir data/processed/primus/clean]
  [--vocab-path data/vocab/primus_lmx.txt]
  [--split val|test]           # default: test
  [--beam-width 1]             # 1 = greedy; >1 = beam search
  [--per-sample]               # log per-sample SER (worst first)
  [--melodic]                  # also report melodic SER (structural tokens stripped)
  [--num-workers 10]
```

The `--melodic` flag computes a second SER after stripping `measure`,
`tied:start`, and `tied:stop` from both reference and prediction.  On
this corpus the bulk of edits are structural (≈ 87 % barlines + ties),
so the melodic figure is a much better measure of the model's actual
musical recognition ability.

---

## evaluate-ab

Compare SER across multiple beam widths on the same split.

```
poetry run python src/cli.py evaluate-ab \
  --checkpoint <model.pt> \
  [--data-dir data/processed/primus/clean]
  [--vocab-path data/vocab/primus_lmx.txt]
  [--beams 1,5,10]             # comma-separated beam widths
  [--split test]
```

Outputs a table: beam width → SER.

---

## api

Start the FastAPI web server.

```
poetry run python src/cli.py api \
  [--host 0.0.0.0] \
  [--port 8000]
```

Visit `http://localhost:8000` for the upload UI. POST to `/api/omr/lead-sheet` for JSON.

---

## pipeline

Run all data preparation stages in sequence: render → convert → augment → vocab.

```
poetry run python src/cli.py pipeline \
  [--raw-primus-dir data/raw/primus]
  [--clean-dir data/processed/primus/clean]
  [--scanned-dir data/processed/primus/scanned]
  [--vocab-path data/vocab/primus_lmx.txt]
  [--render-dpi 200]
  [--force-render]             # re-render even if PNG exists
  [--augment-copies 1]
  [--augment-seed 42]
  [--extra-vocab-data-dir <dir>]    # repeatable: extra .lmx dirs for vocab
  [--workers N]
  [--limit N]
```

---

## pipeline-train

Run the full data pipeline and then immediately start training.

```
poetry run python src/cli.py pipeline-train \
  [all pipeline flags] \
  [all train flags]
```

---

## Common Patterns

### First-time full run
```bash
poetry install
poetry run python src/cli.py pipeline-train \
  --raw-primus-dir data/raw/primus \
  --clean-dir data/processed/primus/clean \
  --scanned-dir data/processed/primus/scanned \
  --vocab-path data/vocab/primus_lmx.txt \
  --model-dir models/run1 \
  --epochs 60 --batch-size 16 --workers 8
```

### Resume interrupted training
```bash
poetry run python src/cli.py train \
  --data-dir data/processed/primus/clean \
  --scanned-dir data/processed/primus/scanned \
  --vocab-path data/vocab/primus_lmx.txt \
  --model-dir models/run1 \
  --epochs 80 \
  --resume
# --resume with no path auto-detects models/run1/latest_checkpoint.pt
```

### Quick evaluation
```bash
poetry run python src/cli.py evaluate \
  --checkpoint models/latest/best_model.pt \
  --data-dir data/processed/primus/clean \
  --vocab-path data/vocab/primus_lmx.txt \
  --split test --beam-width 10
```

### Smoke-test render pipeline (10 samples)
```bash
poetry run python src/cli.py pipeline \
  --raw-primus-dir data/raw/primus \
  --clean-dir /tmp/test_clean \
  --scanned-dir /tmp/test_scanned \
  --vocab-path /tmp/test_vocab.txt \
  --limit 10
```

### Serve the web UI
```bash
poetry run python src/cli.py api --port 8000
```


---

### Harvest staff strips for reject calibration

```bash
poetry run python src/cli.py harvest-reject-fixtures \
  --pdfs 'data/real_book/*.pdf' --pages 5 \
  --out data/staff_reject/_harvest
```

Runs preprocessing + staff detection on each PDF page and saves every detected
music strip as a PNG into `--out`. Manually sort the harvested files into
`data/staff_reject/music/` (real music staves) and
`data/staff_reject/non_music/` (titles, footers, page numbers).

### Calibrate reject thresholds

```bash
poetry run python src/cli.py calibrate-reject \
  --fixtures data/staff_reject \
  --out models/staff_reject/thresholds.json
```

Sweeps each gate's signal (Youden's J) on the labelled fixtures, prints a
confusion matrix and per-strip diagnostics for misclassified samples, then
writes the chosen thresholds to `--out`. Point
`$OMR_REJECT_THRESHOLDS=<path>` at the JSON to make the pipeline use them.

Re-run after every CRNN re-train — the CTC log-prob distribution will shift.
