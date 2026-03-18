"""
CRNN inference — load model once, prepare strips matching training, run recognition.

Strip preprocessing exactly matches dataset.py:
  1. Grayscale uint8 → resize height to img_height, proportional width
  2. float32 / 255.0
  3. Per-image (img - mean) / (std + 1e-6)
  4. Pad batch to max width with zeros

Domain-gap mitigation — width:
  Training images are ~877 px wide at 128 px height (median).  Sliced Real Book
  strips upscale to ~2150 px after the same height-resize — outside the training
  distribution.  Wide strips are tiled into ~850 px chunks so each tile is
  in-distribution before being fed to the CRNN.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain-gap constants
# ---------------------------------------------------------------------------

# Training median width at 128px height = 877px; p95 = 1219px.
# Test strips resize to ~2150px — outside the training distribution.
# We tile wide strips into chunks of this size so each tile matches training.
_TILE_W_AT_128 = 850  # target tile width in resized (128px-height) coordinates

# ---------------------------------------------------------------------------
# Cached model state (loaded once per process)
# ---------------------------------------------------------------------------

_model_cache: dict = {}


def _load_model(checkpoint_path: Path):
    """Load and cache the CRNN model + vocab from a checkpoint."""
    key = str(checkpoint_path.resolve())
    if key in _model_cache:
        return _model_cache[key]

    from CRNN_CTC.model import CRNN
    from CRNN_CTC.vocab import Vocabulary
    from CRNN_CTC.config import Config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", Config())

    # Resolve vocab path.  The checkpoint stores a path that was valid when
    # training ran; resolve it relative to several anchors to handle CWD
    # differences and repo restructures.
    _repo_root = checkpoint_path.resolve().parent
    while _repo_root.parent != _repo_root and not (_repo_root / "pyproject.toml").exists():
        _repo_root = _repo_root.parent
    _vocab_candidates = [
        Path(cfg.vocab_path),                                      # as-saved (works if CWD == project root)
        _repo_root / cfg.vocab_path,                               # relative to repo root
        _repo_root / "data" / "vocab" / "primus_lmx.txt",         # canonical location
        checkpoint_path.parent.parent / "data" / "vocab" / "primus_lmx.txt",  # relative to run dir
    ]
    vocab_path = next((p for p in _vocab_candidates if p.exists()), None)
    if vocab_path is None:
        raise FileNotFoundError(
            f"Cannot find vocab file; tried: {[str(p) for p in _vocab_candidates]}"
        )
    vocab = Vocabulary.from_file(vocab_path)

    model = CRNN(
        vocab_size=ckpt.get("vocab_size", len(vocab)),
        cnn_out_channels=cfg.cnn_out_channels,
        rnn_hidden=cfg.rnn_hidden,
        rnn_layers=cfg.rnn_layers,
        dropout=0.0,
        backbone=cfg.backbone,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    entry = {"model": model, "vocab": vocab, "cfg": cfg, "device": device}
    _model_cache[key] = entry
    log.info("CRNN loaded from %s (epoch %d, val_SER=%.4f)",
             checkpoint_path, ckpt.get("epoch", -1), ckpt.get("val_ser", -1))
    return entry


# ---------------------------------------------------------------------------
# Strip preprocessing (must match dataset._load_image + __getitem__)
# ---------------------------------------------------------------------------

def _tile_strip(img: np.ndarray, img_height: int) -> list[np.ndarray]:
    """Split a wide strip into horizontal tiles whose resized width matches
    the training distribution (~850px at 128px height).

    Tiles are taken in the *original* image coordinates so that the
    proportional resize in _prepare_strip happens per-tile.
    """
    h, w = img.shape[:2]
    if h == 0:
        return [img]
    # Compute what width each tile should be in the ORIGINAL image
    tile_w_orig = max(1, round(_TILE_W_AT_128 * h / img_height))

    # If the strip is already narrow enough, no tiling needed
    if w <= int(tile_w_orig * 1.3):
        return [img]

    tiles = []
    x = 0
    while x < w:
        x1 = min(x + tile_w_orig, w)
        tiles.append(img[:, x:x1])
        x = x1
    return tiles


def _prepare_strip(img: np.ndarray, img_height: int, max_width: int) -> tuple[Tensor, int]:
    """Preprocess one grayscale strip exactly like the training dataset."""
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if len(img.shape) == 3:
        img = img[:, :, 0]
    h, w = img.shape
    if h == 0 or w == 0:
        raise ValueError(f"Empty strip: {img.shape}")

    new_w = max(1, round(w * img_height / h))
    if max_width > 0 and new_w > max_width:
        new_w = max_width
    resized = cv2.resize(img, (new_w, img_height), interpolation=cv2.INTER_AREA)
    f = resized.astype(np.float32) / 255.0
    f = (f - f.mean()) / (f.std() + 1e-6)
    return torch.from_numpy(f).unsqueeze(0), new_w  # (1, H, W), width


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recognize_music(
    strip_images: list[np.ndarray],
    checkpoint_path: Path | None = None,
) -> list[str]:
    """Run CRNN on staff strip images. Returns one LMX string per strip."""
    if not strip_images:
        return []

    if checkpoint_path is None:
        checkpoint_path = Path("models/latest/best_model.pt")
    if not checkpoint_path.exists():
        return [f"[no checkpoint: {checkpoint_path}]"] * len(strip_images)

    try:
        entry = _load_model(checkpoint_path)
    except Exception as e:
        log.error("Model load failed: %s", e)
        return [f"[model error: {e}]"] * len(strip_images)

    model = entry["model"]
    vocab = entry["vocab"]
    cfg = entry["cfg"]
    device = entry["device"]
    img_height = getattr(cfg, "img_height", 128)
    max_width = getattr(cfg, "max_image_width", 2048)

    from CRNN_CTC.evaluate import greedy_decode

    def _decode_batch(tiles: list[np.ndarray]) -> list[list[str]]:
        """Run CRNN on a list of image tiles; return one token list per tile."""
        tensors: list[Tensor] = []
        tile_widths: list[int] = []
        for tile in tiles:
            try:
                t, tw = _prepare_strip(tile, img_height, max_width)
                tensors.append(t)
                tile_widths.append(tw)
            except Exception as e:
                log.warning("Tile prep failed: %s", e)
                tensors.append(torch.zeros(1, img_height, 1))
                tile_widths.append(1)

        max_w = max(t.shape[-1] for t in tensors)
        padded = []
        for t in tensors:
            if t.shape[-1] < max_w:
                pad = torch.zeros(1, img_height, max_w - t.shape[-1])
                t = torch.cat([t, pad], dim=-1)
            padded.append(t.unsqueeze(0))

        batch = torch.cat(padded, dim=0).to(device)
        width_t = torch.tensor(tile_widths, dtype=torch.long, device=device)
        with torch.inference_mode():
            log_probs, out_lens = model(batch, width_t)
            return greedy_decode(log_probs, out_lens, vocab)

    results: list[str] = []
    for img in strip_images:
        if img is None or img.size == 0:
            results.append("")
            continue

        # Tile wide strips so each chunk is in the training width distribution
        tiles = _tile_strip(img, img_height)
        all_token_lists = _decode_batch(tiles)

        # Merge tile outputs.  Simple concatenation — the first token of each
        # tile after the first is usually a mid-sequence note, so we just join.
        # Drop leading 'measure' tokens from tiles 2+ to avoid duplicating
        # the barline that marks the tile boundary.
        merged: list[str] = list(all_token_lists[0]) if all_token_lists else []
        for tl in all_token_lists[1:]:
            token_list = list(tl)
            # Remove duplicate leading 'measure' token when the previous tile
            # already ended with one
            if merged and merged[-1] == "measure" and token_list and token_list[0] == "measure":
                token_list = token_list[1:]
            merged.extend(token_list)

        results.append(" ".join(merged))

    return results
