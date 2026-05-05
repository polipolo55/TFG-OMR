# CRNN-CTC Model

**Location:** `src/CRNN_CTC/model.py`, `src/CRNN_CTC/config.py`

## Architecture

The model is a Convolutional Recurrent Neural Network (CRNN) trained with Connectionist Temporal Classification (CTC) loss. It maps a variable-width staff image to a variable-length sequence of LMX tokens without explicit alignment.

```
Input: (B, 1, H=128, W)
        │
        ▼
  [CNN Backbone]        feature extraction
  ResNet18 (default)
  or VGG
        │
        ▼  (B, C, 1, W/4)
  [Squeeze height dim]  → (W/4, B, C)   [time-major]
        │
        ▼
  [BiLSTM × 2]          256 hidden units per direction
        │
        ▼  (W/4, B, 512)
  [Linear]              → vocab_size + 1 (CTC blank)
        │
        ▼
  log_softmax → CTCLoss
```

**Time steps T ≈ W/4** (width compressed 4× by the CNN). Each time step predicts a probability distribution over the vocabulary.

## CNN Backbones

### ResNet18 (default, `backbone="resnet18"`)

Modified ResNet18 with asymmetric strides to aggressively reduce height while preserving width.

| Layer | Stride | Output |
|-------|--------|--------|
| conv1 | (2, 1) | H/2, W |
| maxpool | (2, 2) | H/4, W/2 |
| layer1 | (1, 1) | H/4, W/2 |
| layer2 | (2, 2) | H/8, W/4 |
| layer3 | (2, 1) | H/16, W/4 |
| layer4 | (2, 1) | H/32, W/4 |
| AdaptiveAvgPool2d | → (1, W/4) | 1, W/4 |

Output channels: 512. ResNet18 is preferred for its residual connections (better gradient flow) and faster convergence vs. VGG.

### VGG (`backbone="vgg"`)

5-block VGG-style architecture with progressive max-pooling.

| Block | Pooling | Output |
|-------|---------|--------|
| Block 1 | (2, 2) | H/2, W/2 |
| Block 2 | (2, 2) | H/4, W/4 |
| Block 3 | (2, 1) | H/8, W/4 |
| Block 4 | (2, 1) | H/16, W/4 |
| Block 5 | (2, 1) | H/32, W/4 |

Output channels: configurable (`cnn_out_channels`, default 256). Dropout after each block (`cnn_dropout=0.25`).

## BiLSTM Sequence Modeler

- 2 stacked bidirectional LSTM layers
- Hidden size: 256 per direction (512 total)
- Dropout between layers: 0.3
- Input: flattened CNN features per time step
- Output: (T, B, 512) → projected to (T, B, vocab_size)

## CTC Loss

```python
nn.CTCLoss(blank=0, zero_infinity=True)
```

- `blank=0` is the CTC blank token (always index 0 in vocab)
- `zero_infinity=True` — silently ignores samples whose label lengths exceed input lengths (prevents NaN loss)
- Input to loss: `(log_probs, targets, input_lengths, target_lengths)`
- `input_lengths` derived from image widths (W/4 per sample)

## Greedy Decoding

During inference (and fast evaluation):
1. Argmax over vocab dimension at each time step → frame-level prediction
2. Collapse consecutive duplicate indices
3. Remove blank (index 0)
4. Map indices → LMX tokens via `Vocab.decode()`

## Beam Search Decoding

Greedy is the default everywhere (CLI `evaluate`, inference, API).  Beam
search is opt-in via `--beam-width N` (CLI) or `OMR_BEAM_WIDTH=N` (inference
env var).  Empirically, beam search yields little or no SER improvement on
monophonic Real Book staves while costing ~N× the decode time, so keep the
default unless you have evidence otherwise.

## Vocabulary

**File:** `src/CRNN_CTC/vocab.py`

```
Index 0  →  <blank>   (CTC blank, never in output)
Index 1  →  <pad>     (padding, skipped in decode)
Index 2  →  <unk>     (OOV token)
Index 3+ →  LMX tokens (alphabetically sorted)
```

Example tokens (categories — see `docs/lmx_format.md` for the full grammar):
```
flat  natural  sharp                       # display-only accidentals
clef:G2  clef:F4  clef:C3  ...              # only G2 reaches inference (others normalised)
key:fifths:-7 ... key:fifths:0 ... key:fifths:7
time  beats:2  beats:3  beats:4  beats:6  beats:12
       beat-type:2  beat-type:4  beat-type:8
pitch:A  pitch:B  pitch:C  pitch:D  pitch:E  pitch:F  pitch:G
octave:0  octave:1 ... octave:8
measure  rest  rest:measure
tied:start  tied:stop
whole  half  quarter  eighth  16th  32nd  64th  dot
```

> **Note — dead output classes.** Some vocabulary entries (e.g. clefs other
> than G2, time signatures outside the allowed set, octave 0–2/8, etc.) are
> kept in the vocab so that legacy checkpoints continue to load, but they are
> never produced by the data pipeline once filtering is on.  They are
> effectively dead output classes; they cost a few unused logits per time
> step and otherwise do no harm.  The grammar fixer rejects/normalises them
> if the model ever predicts one.

## Key Configuration Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `backbone` | `"resnet18"` | CNN architecture |
| `rnn_hidden` | 256 | BiLSTM hidden units per direction |
| `rnn_layers` | 2 | BiLSTM depth |
| `dropout` | 0.3 | LSTM inter-layer dropout |
| `cnn_dropout` | 0.25 | CNN block dropout (VGG only) |
| `cnn_out_channels` | 256 | VGG output channels |
| `img_height` | 128 | Fixed input height (px) |
| `max_image_width` | 2048 | Max width before clamping |

## Model Files

| File | Purpose |
|------|---------|
| `src/CRNN_CTC/model.py` | CRNN, CNNBackbone, ResNetBackbone, build_backbone |
| `src/CRNN_CTC/config.py` | Config dataclass (all hyperparameters) |
| `src/CRNN_CTC/vocab.py` | Vocabulary (encode/decode, build, load/save) |
| `models/latest/best_model.pt` | Checkpoint: model weights + config + vocab path |
