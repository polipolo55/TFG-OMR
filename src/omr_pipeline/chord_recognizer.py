"""CRNN-based chord-strip recognition.

The chord CRNN is trained on synthetic Real Book-style chord strips
(``data_processing.generate_chord_crops``) using a character-level CTC
target.  At inference we:

1. Load the chord-model checkpoint + character vocabulary (cached per process).
2. Resize each chord strip to the trained height (64 px), preserving aspect.
3. Right-pad the batch to common width.
4. Forward pass + greedy CTC decode → character sequence per strip.
5. Join characters and return one string per strip.

Output strings use Real Book conventions: ``-`` for minor, ``maj`` for
major 7, ``ø`` for half-diminished, ``dim`` for diminished, ``+`` for
augmented, ``/`` for slash bass.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.amp import autocast

from ._bootstrap import ensure_src_path

ensure_src_path()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-process model cache
# ---------------------------------------------------------------------------

_chord_model_cache: dict = {}


def _resolve_chord_checkpoint() -> Path | None:
    """Locate the chord CRNN checkpoint.

    Order of precedence:
    1. ``OMR_CHORD_CHECKPOINT`` env var.
    2. ``<project_root>/models/chord/latest/best_model.pt`` (training default).

    Project root is resolved relative to this file (src/omr_pipeline/...),
    so the lookup works regardless of the caller's CWD.
    """
    env = os.environ.get("OMR_CHORD_CHECKPOINT", "").strip()
    if env:
        p = Path(env)
        return p if p.exists() else None
    project_root = Path(__file__).resolve().parent.parent.parent
    default = project_root / "models" / "chord" / "latest" / "best_model.pt"
    return default if default.exists() else None


def _load_chord_model(checkpoint_path: Path) -> dict:
    key = str(checkpoint_path.resolve())
    if key in _chord_model_cache:
        return _chord_model_cache[key]

    from CRNN_CTC.model import CRNN
    from CRNN_CTC.vocab import Vocabulary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    # Vocab — checkpoint stores the content tokens (no blank/pad/unk)
    if "vocab_tokens" in ckpt:
        vocab = Vocabulary(list(ckpt["vocab_tokens"]))
    else:
        # Fall back to vocab_path in config
        vocab_path = Path(cfg.get("vocab_path", "data/vocab/chord.txt"))
        vocab = Vocabulary.from_file(vocab_path)

    model = CRNN(
        vocab_size=len(vocab),
        cnn_out_channels=cfg.get("cnn_out_channels", 256),
        rnn_hidden=cfg.get("rnn_hidden", 192),
        rnn_layers=cfg.get("rnn_layers", 2),
        dropout=0.0,
        cnn_dropout=0.0,
        backbone=cfg.get("backbone", "resnet18"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    entry = {
        "model": model,
        "vocab": vocab,
        "img_height": cfg.get("img_height", 64),
        "max_image_width": cfg.get("max_image_width", 2048),
        "device": device,
    }
    _chord_model_cache[key] = entry
    log.info(
        "Chord CRNN loaded from %s (epoch %d, val_CER=%.4f)",
        checkpoint_path,
        ckpt.get("epoch", -1),
        ckpt.get("val_cer", -1.0),
    )
    return entry


# ---------------------------------------------------------------------------
# Preprocessing — matches ChordDataset exactly (no augmentation at inference)
# ---------------------------------------------------------------------------


def _trim_binder_hole(img: np.ndarray) -> np.ndarray:
    """Trim left-edge columns dominated by binder-hole ink.

    Real Book pages have a dark binder-hole shadow in the first ~30-60 px of
    every chord strip.  The CRNN reads it as a phantom "B" chord.  We trim
    leading columns whose ink density exceeds 50 % (binder holes are nearly
    solid dark) up to at most 10 % of the strip width.
    """
    h, w = img.shape[:2]
    if h < 8 or w < 50:
        return img
    # Binarize using Otsu so we can measure ink density column-wise.
    thr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    col_density = thr.mean(axis=0) / 255.0  # fraction of inky pixels per column
    max_trim = min(w // 10, 80)
    trim = 0
    for x in range(max_trim):
        if col_density[x] >= 0.5:
            trim = x + 1
        else:
            # Found a sparse column — keep trimming only while density stays high
            if trim > 0 and x - trim > 3:
                break
    if trim > 0:
        return img[:, trim:]
    return img


def _preprocess_strip(img: np.ndarray, img_height: int, max_width: int) -> tuple[Tensor, int]:
    """Grayscale uint8 → trim binder-hole → resize → /255 → per-image norm → (1, H, W) tensor."""
    if img.ndim == 3:
        img = img[:, :, 0]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    img = _trim_binder_hole(img)

    h, w = img.shape
    if h == 0 or w == 0:
        return torch.zeros(1, img_height, 1), 1

    new_w = max(1, round(w * img_height / h))
    if max_width > 0 and new_w > max_width:
        new_w = max_width

    resized = cv2.resize(img, (new_w, img_height), interpolation=cv2.INTER_AREA)
    f = resized.astype(np.float32) / 255.0
    f = (f - f.mean()) / (f.std() + 1e-6)
    return torch.from_numpy(f).unsqueeze(0), new_w


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@torch.inference_mode()
def recognize_chords_crnn(
    strip_images: list[np.ndarray],
    checkpoint_path: Path | None = None,
) -> list[str]:
    """Run the chord CRNN on a list of chord-strip crops.

    Returns one string per crop (Real Book chord convention, space-separated
    when multiple chords on the same strip).  Empty / unreadable strips yield ``""``.

    If no checkpoint is available the function returns ``[""] * len(strips)``
    so the rest of the pipeline can proceed without chord text.
    """
    if not strip_images:
        return []

    if checkpoint_path is None:
        checkpoint_path = _resolve_chord_checkpoint()
    if checkpoint_path is None or not checkpoint_path.exists():
        log.warning(
            "Chord CRNN checkpoint not found — skipping chord recognition. "
            "Set OMR_CHORD_CHECKPOINT or place a model at models/chord/latest/best_model.pt"
        )
        return [""] * len(strip_images)

    entry = _load_chord_model(checkpoint_path)
    model = entry["model"]
    vocab = entry["vocab"]
    img_height = entry["img_height"]
    max_width = entry["max_image_width"]
    device = entry["device"]
    use_amp = device.type == "cuda"

    from CRNN_CTC.evaluate import greedy_decode

    # Preprocess all strips
    tensors: list[Tensor] = []
    widths: list[int] = []
    valid_mask: list[bool] = []
    for img in strip_images:
        if img is None or not hasattr(img, "size") or img.size == 0:
            tensors.append(torch.zeros(1, img_height, 1))
            widths.append(1)
            valid_mask.append(False)
            continue
        try:
            t, w = _preprocess_strip(img, img_height, max_width)
            tensors.append(t)
            widths.append(w)
            valid_mask.append(True)
        except Exception as exc:
            log.warning("Chord strip preprocess failed: %s", exc)
            tensors.append(torch.zeros(1, img_height, 1))
            widths.append(1)
            valid_mask.append(False)

    # Right-pad to common width
    max_w = max(t.shape[-1] for t in tensors)
    padded: list[Tensor] = []
    for t in tensors:
        if t.shape[-1] < max_w:
            pad = torch.zeros(1, img_height, max_w - t.shape[-1])
            t = torch.cat([t, pad], dim=-1)
        padded.append(t.unsqueeze(0))

    batch = torch.cat(padded, dim=0).to(device)
    width_t = torch.tensor(widths, dtype=torch.long, device=device)

    with autocast("cuda", enabled=use_amp):
        log_probs, out_lens = model(batch, width_t)
    token_lists = greedy_decode(log_probs, out_lens, vocab)

    # Drop bare roots (e.g. "B", "Eb") — usually clutter false-positives.
    import re as _re

    from .chord_postprocess import clean_chord_line

    _SINGLE_ROOT_RE = _re.compile(r"^[A-G][#b]?$")

    results: list[str] = []
    for chars, ok in zip(token_lists, valid_mask):
        if not ok:
            results.append("")
            continue
        # The CRNN occasionally emits <unk> on noisy regions (clutter the
        # synthetic data did not contain); treat it as a token separator so
        # the grammar filter can split real chords away from noise.
        raw = "".join(" " if c == "<unk>" else c for c in chars).strip()
        cleaned = clean_chord_line(raw)
        if cleaned:
            keep = [t for t in cleaned.split() if not _SINGLE_ROOT_RE.match(t)]
            results.append("  ".join(keep))
        else:
            results.append("")
    return results
