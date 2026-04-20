# CLI Reference

**Entry point:** `src/cli.py`
**Run with:** `poetry run python src/cli.py <subcommand> [options]`

All subcommands share a `--log-level` flag (DEBUG/INFO/WARNING, default INFO).

---

## render

Re-render PrIMuS `.semantic` samples as LilyJAZZ PNGs with `.lmx` labels.

```
python src/cli.py render \
  --source <primus-root>       # dir containing package_aa/, package_ab/, ...
  --output <clean-dir>         # output root for rendered samples
  [--dpi 200]                  # LilyPond render DPI
  [--workers N]                # parallel processes (default: cpu_count-2)
  [--limit N]                  # stop after N samples (debug)
```

---

## convert

Convert `.semantic` files already in `--source` to `.lmx` (direct token remapping).

```
python src/cli.py convert \
  --source <clean-dir>
  [--workers N]
```

---

## augment

Apply scan-simulation augmentation to clean PNGs, producing scanned copies.

```
python src/cli.py augment \
  --source <clean-dir> \
  --output <scanned-dir> \
  [--copies 1]                 # augmented copies per original
  [--seed 42]                  # reproducibility seed
  [--workers N]
```

---

## vocab

Build a vocabulary file from all `.lmx` files under `--data-dir`.

```
python src/cli.py vocab \
  --data-dir <clean-dir> \
  --output <vocab-path>        # e.g. data/vocab/primus_lmx.txt
  [--extra-data-dir <dir>]     # repeatable: add more .lmx sources
  [--workers N]
```

---

## train

Train the CRNN-CTC model.

```
python src/cli.py train \
  --data-dir <clean-dir> \
  --scanned-dir <scanned-dir> \
  --vocab-path <vocab-path> \
  --model-dir <output-dir> \
  [--extra-data-dir <dir>]          # repeatable
  [--extra-scanned-dir <dir>]       # repeatable
  [--finetune-data-dir <dir>]       # repeatable: injected into train only
  [--finetune-scanned-dir <dir>]    # repeatable
  [--checkpoint <path>]             # resume from checkpoint
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
  [--workers 10]
  [--no-scanned]                    # disable scanned data
  [--no-filter-rest-heavy]          # disable rest-heavy filter
  [--no-filter-clefs]               # disable clef filter
  [--no-filter-multi-staff]         # disable multi-staff filter
  [--no-filter-non-leadsheet]       # disable non-leadsheet clef filter
  [--no-filter-unusual-time]        # disable unusual time filter
```

---

## evaluate

Evaluate a checkpoint and report SER on val or test split.

```
python src/cli.py evaluate \
  --checkpoint <model.pt> \
  --data-dir <clean-dir> \
  --scanned-dir <scanned-dir> \
  --vocab-path <vocab-path> \
  [--split val|test]           # default: test
  [--beam-width 1]             # 1 = greedy; >1 = beam search
  [--worst-n 20]               # print N worst samples
  [--workers N]
```

---

## evaluate-ab

Compare SER across multiple beam widths.

```
python src/cli.py evaluate-ab \
  --checkpoint <model.pt> \
  --data-dir <clean-dir> \
  --vocab-path <vocab-path> \
  [--beams 1,5,10]             # comma-separated beam widths
  [--split test]
```

Outputs a table: beam width → SER.

---

## api

Start the FastAPI web server.

```
python src/cli.py api \
  [--host 0.0.0.0] \
  [--port 8000] \
  [--checkpoint <model.pt>]    # override model path
  [--vocab-path <vocab-path>]
```

Visit `http://localhost:8000` for the upload UI. POST to `/api/omr/lead-sheet` for JSON.

---

## pipeline

Run all data preparation stages (render → convert → augment → vocab).

```
python src/cli.py pipeline \
  --raw-primus-dir <primus-root> \
  --clean-dir <clean-dir> \
  --scanned-dir <scanned-dir> \
  --vocab-path <vocab-path> \
  [--dpi 200] \
  [--copies 1] \
  [--seed 42] \
  [--workers N] \
  [--limit N]
```

---

## pipeline-train

Run full pipeline then immediately train.

```
python src/cli.py pipeline-train \
  --raw-primus-dir <primus-root> \
  --clean-dir <clean-dir> \
  --scanned-dir <scanned-dir> \
  --vocab-path <vocab-path> \
  --model-dir <output-dir> \
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
  --epochs 80
# auto-detects models/run1/latest.pt and resumes
```

### Quick evaluation
```bash
poetry run python src/cli.py evaluate \
  --checkpoint models/latest/best_model.pt \
  --data-dir data/processed/primus/clean \
  --vocab-path data/vocab/primus_lmx.txt \
  --split test --beam-width 10
```

### Serve the web UI
```bash
poetry run python src/cli.py api --port 8000
```
