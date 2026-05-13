# Configuration Reference

**File:** `src/CRNN_CTC/config.py`

All training and data settings live in a single `Config` dataclass. It is populated from CLI flags by `_build_config_from_args()` in `src/cli.py` and serialized to `{model_dir}/config.json` at training time.

## Paths

| Field | Default | Description |
|-------|---------|-------------|
| `data_dir` | — | Root of clean rendered samples (`{id}.png` + `{id}.lmx`) |
| `scanned_dir` | — | Root of augmented scanned copies |
| `extra_data_dirs` | `[]` | Additional clean data roots (repeatable) |
| `extra_scanned_dirs` | `[]` | Additional scanned data roots (repeatable) |
| `finetune_data_dirs` | `[]` | Fine-tune data injected into train split only |
| `finetune_scanned_dirs` | `[]` | Fine-tune scanned data (train split only) |
| `model_dir` | `"models"` | Where checkpoints and logs are saved (training writes a `run_<timestamp>/` subdir and a `latest` symlink) |
| `vocab_path` | — | Path to vocabulary file (one token per line) |

## Data & Image

| Field | Default | Description |
|-------|---------|-------------|
| `img_height` | 128 | Fixed image height after resize (px) |
| `max_image_width` | 2048 | Maximum width (wider images are clamped) |
| `val_frac` | 0.10 | Fraction of data reserved for validation |
| `test_frac` | 0.10 | Fraction of data reserved for testing |
| `use_scanned` | True | Include augmented images in training |

## Domain Filters

These three flags realise the **lead-sheet domain spec** at dataset-construction
time.  See `docs/overview.md` → "Domain Specification" for the full rationale.
Disabling them on a PrIMuS-trained run does not produce a more general model;
it only spends model capacity on patterns the system will never encounter at
inference.

| Field | Default | Description |
|-------|---------|-------------|
| `filter_multi_staff` | True | Drop samples with original PNG height >`max_source_height` (180 px).  Removes LilyPond renders that wrapped onto two staves. |
| `filter_non_leadsheet_clef` | True | Keep only `clef:G2` (treble).  Drops alto/tenor/bass/French-violin clefs from orchestral PrIMuS sources. |
| `filter_unusual_time` | True | Keep only the eight jazz common-time signatures listed below. |

**Allowed time signatures** (when `filter_unusual_time=True`):
`4/4`, `3/4`, `2/4`, `2/2`, `6/8`, `6/4`, `5/4`, `12/8`

These must stay in sync with `_COMMON_TIME_SIGS` in both `src/CRNN_CTC/dataset.py` and `src/omr_pipeline/grammar_fix.py`. If you add or remove a time signature from one, update the other immediately (Hard Constraint #5 in `CLAUDE.md`).

## Training Augmentation

| Field | Default | Description |
|-------|---------|-------------|
| `strip_header_prob` | 0.4 | Probability of removing clef+key+time from image and label (training only) |
| `online_aug_prob` | 0.5 | Probability of light per-sample jitter (brightness, noise, ±2 px shift) on top of the offline-augmented PNG (training only) |
| `rare_lmx_oversample` | 2 | Oversampling factor for samples containing rare tokens |
| `rare_lmx_tokens` | `("tied:start", "tied:stop")` | Tokens that trigger oversampling.  Ties are visually subtle on degraded scans and chronically under-predicted.  `key:fifths:0` was previously included here because PrIMuS only had 8 explicit C-major labels; the root cause was a converter bug (missing default key injection) that is now fixed — C major is ~22.6 % of the corpus after the fix (19,778 / 87,677 samples) and needs no oversampling. |

## Model Architecture

| Field | Default | Description |
|-------|---------|-------------|
| `backbone` | `"resnet18"` | CNN backbone: `"resnet18"` or `"vgg"` |
| `cnn_out_channels` | 256 | VGG output channels (ResNet18 uses fixed 512) |
| `cnn_dropout` | 0.25 | Dropout in VGG blocks |
| `rnn_hidden` | 256 | BiLSTM hidden units per direction |
| `rnn_layers` | 2 | BiLSTM depth |
| `dropout` | 0.3 | Dropout between LSTM layers |

## Training Hyperparameters

| Field | Default | Description |
|-------|---------|-------------|
| `epochs` | 60 | Maximum training epochs |
| `batch_size` | 16 | Samples per GPU batch |
| `lr` | 1e-3 | OneCycleLR peak learning rate |
| `weight_decay` | 1e-4 | AdamW weight decay |
| `warmup_frac` | 0.08 | Fraction of total steps used for warm-up |
| `early_stopping_patience` | 12 | Epochs without val SER improvement before stopping |
| `max_grad_norm` | 5.0 | Gradient clipping max norm |
| `num_workers` | 10 | DataLoader worker processes |

## Environment Variables (Inference Only)

These override inference behavior without touching the Config dataclass:

| Variable | Effect |
|----------|--------|
| `OMR_ENABLE_TILING=1` | Enable legacy tiling mode in inference |
| `OMR_BEAM_WIDTH=N` | Override CTC beam width (default: 1 = greedy) |
| `OMR_CHORD_CHECKPOINT=<path>` | Path to chord CRNN checkpoint (default: `models/chord/latest/best_model.pt`) |

## Example: Custom Training Config

```python
from src.CRNN_CTC.config import Config

cfg = Config(
    data_dir="data/processed/primus/clean",
    scanned_dir="data/processed/primus/scanned",
    vocab_path="data/vocab/primus_lmx.txt",
    model_dir="models/experiment_A",
    backbone="resnet18",
    epochs=80,
    batch_size=32,
    lr=5e-4,
    rnn_hidden=512,
    filter_non_leadsheet_clef=True,
)
```

## Config Serialization

At training start, `config.json` is written to `model_dir`. When resuming or evaluating, the checkpoint contains the config dict so inference always uses the exact same settings as training.
