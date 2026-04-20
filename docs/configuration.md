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
| `model_dir` | `"models/latest"` | Where checkpoints and logs are saved |
| `vocab_path` | — | Path to vocabulary file (one token per line) |

## Data & Image

| Field | Default | Description |
|-------|---------|-------------|
| `img_height` | 128 | Fixed image height after resize (px) |
| `max_image_width` | 2048 | Maximum width (wider images are clamped) |
| `val_frac` | 0.10 | Fraction of data reserved for validation |
| `test_frac` | 0.10 | Fraction of data reserved for testing |
| `use_scanned` | True | Include augmented images in training |

## Sample Filtering

| Field | Default | Description |
|-------|---------|-------------|
| `filter_rest_heavy` | True | Drop samples where >80% of tokens are structural (rest, measure) and length >50 |
| `filter_unwanted_clefs` | True | Drop samples with non-jazz clefs (C1, C2, F3, etc.) |
| `filter_multi_staff` | True | Drop samples with original height >180 px |
| `filter_non_leadsheet_clef` | True | Keep only G2 (treble) clef |
| `filter_unusual_time` | True | Keep only jazz common time signatures |

**Allowed time signatures** (when `filter_unusual_time=True`):
`4/4`, `3/4`, `2/4`, `2/2`, `6/8`, `5/4`, `7/8`, `12/8`, `3/8`

## Training Augmentation

| Field | Default | Description |
|-------|---------|-------------|
| `strip_header_prob` | 0.4 | Probability of removing clef+key+time from image and label |
| `rare_lmx_oversample` | 2 | Oversampling factor for samples containing rare tokens |
| `rare_lmx_tokens` | `("tied:start", "tied:stop")` | Tokens that trigger oversampling |

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
| `OMR_CHORD_BACKEND=contour\|easyocr\|vlm` | Chord OCR backend |

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
