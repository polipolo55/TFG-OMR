"""
CRNN inference — load model, normalise crops to match training, recognise.

Strip preprocessing matches dataset.py exactly:
  1. Grayscale uint8 → resize height to img_height, proportional width
  2. float32 / 255
  3. Per-image zero-mean unit-variance normalisation
  4. Pad batch to max width with zeros

Staff-aware normalization centres the 5-line staff vertically in the crop
so it occupies the same region as in the clean training images.  Overlapping
tiles (50 %) with centre-crop merging avoid boundary-cut artefacts on wide
strips.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.amp import autocast

from .staff_detect import local_primary_staff_lines

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

log = logging.getLogger(__name__)

_TILE_W_AT_128 = 850
_TILE_OVERLAP = 0.50

# ---------------------------------------------------------------------------
# Model cache (loaded once per process)
# ---------------------------------------------------------------------------

_model_cache: dict = {}


def _load_model(checkpoint_path: Path):
    key = str(checkpoint_path.resolve())
    if key in _model_cache:
        return _model_cache[key]

    from CRNN_CTC.model import CRNN
    from CRNN_CTC.vocab import Vocabulary
    from CRNN_CTC.config import Config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", Config())

    repo_root = checkpoint_path.resolve().parent
    while repo_root.parent != repo_root and not (repo_root / "pyproject.toml").exists():
        repo_root = repo_root.parent

    vocab_candidates = [
        Path(cfg.vocab_path),
        repo_root / cfg.vocab_path,
        repo_root / "data" / "vocab" / "primus_lmx.txt",
        checkpoint_path.parent.parent / "data" / "vocab" / "primus_lmx.txt",
    ]
    vocab_path = next((p for p in vocab_candidates if p.exists()), None)
    if vocab_path is None:
        raise FileNotFoundError(
            f"Cannot find vocab file; tried: {[str(p) for p in vocab_candidates]}"
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
    log.info(
        "CRNN loaded from %s (epoch %d, val_SER=%.4f)",
        checkpoint_path, ckpt.get("epoch", -1), ckpt.get("val_ser", -1),
    )
    return entry


# ---------------------------------------------------------------------------
# Staff-aware normalization
# ---------------------------------------------------------------------------

def normalize_staff_crop(
    grayscale: np.ndarray,
    staff_line_ys: list[int] | None,
    bbox_y0: int,
) -> np.ndarray:
    """Re-crop so the staff is centred vertically with a fixed margin.

    Training images have a consistent vertical staff placement (~60 % of image
    height).  Real-page crops have variable whitespace above/below.  This
    function trims / pads the crop to match the training distribution.

    Parameters
    ----------
    grayscale : 2-D uint8 array — the music region crop.
    staff_line_ys : absolute page y-coordinates of the 5 staff lines, or None.
    bbox_y0 : absolute y-offset of *grayscale* within the full page.
    """
    if grayscale.size == 0 or staff_line_ys is None or len(staff_line_ys) < 5:
        return grayscale

    local = [y - bbox_y0 for y in staff_line_ys]
    top, bot = local[0], local[-1]
    span = bot - top
    if span <= 0:
        return grayscale

    # Target: staff occupies ~62 % of the crop height → multiplier ≈ 1.6
    desired_h = max(int(span * 1.6), span + 20)
    mid = (top + bot) / 2.0
    y0 = int(mid - desired_h / 2.0)
    y1 = y0 + desired_h

    h = grayscale.shape[0]
    pad_top = max(0, -y0)
    pad_bot = max(0, y1 - h)
    y0 = max(0, y0)
    y1 = min(h, y1)

    crop = grayscale[y0:y1, :]
    if pad_top or pad_bot:
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bot, 0, 0,
                                  cv2.BORDER_CONSTANT, value=255)
    return crop


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def _tile_strip(img: np.ndarray, img_height: int) -> list[tuple[np.ndarray, float, float]]:
    """Split a wide strip into overlapping tiles.

    Returns (tile, keep_start_frac, keep_end_frac) per tile.  Interior tiles
    keep only the central 50 %; edge tiles keep more to avoid losing content.

    Merging by slicing token lists is only a heuristic (CTC time ≠ token index).
    Set ``OMR_DISABLE_TILING=1`` to run one forward pass (may squash wide staves
    to ``max_image_width``, matching training clamp behaviour).
    """
    h, w = img.shape[:2]
    if h == 0:
        return [(img, 0.0, 1.0)]

    if os.environ.get("OMR_DISABLE_TILING", "").lower() in ("1", "true", "yes"):
        return [(img, 0.0, 1.0)]

    tile_w = max(1, round(_TILE_W_AT_128 * h / img_height))
    if w <= int(tile_w * 1.3):
        return [(img, 0.0, 1.0)]

    stride = max(1, int(tile_w * (1.0 - _TILE_OVERLAP)))
    positions: list[tuple[int, int]] = []
    x = 0
    while x < w:
        x1 = min(x + tile_w, w)
        positions.append((x, x1))
        if x1 >= w:
            break
        x += stride

    n = len(positions)
    tiles: list[tuple[np.ndarray, float, float]] = []
    for i, (x0, x1) in enumerate(positions):
        tile = img[:, x0:x1]
        if n == 1:
            tiles.append((tile, 0.0, 1.0))
        elif i == 0:
            tiles.append((tile, 0.0, 0.75))
        elif i == n - 1:
            tiles.append((tile, 0.25, 1.0))
        else:
            tiles.append((tile, 0.25, 0.75))
    return tiles


# ---------------------------------------------------------------------------
# Per-tile preprocessing (must match dataset.py)
# ---------------------------------------------------------------------------

def _prepare_tile(img: np.ndarray, img_height: int, max_width: int) -> tuple[Tensor, int]:
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.ndim == 3:
        img = img[:, :, 0]

    h, w = img.shape
    if h == 0 or w == 0:
        raise ValueError(f"Empty tile: {img.shape}")

    new_w = max(1, round(w * img_height / h))
    if 0 < max_width < new_w:
        new_w = max_width
    resized = cv2.resize(img, (new_w, img_height), interpolation=cv2.INTER_AREA)

    f = resized.astype(np.float32) / 255.0
    f = (f - f.mean()) / (f.std() + 1e-6)
    return torch.from_numpy(f).unsqueeze(0), new_w


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Greedy (1) matches notebook / default ``cli.py evaluate``; raise via OMR_BEAM_WIDTH.
_BEAM_WIDTH = int(os.environ.get("OMR_BEAM_WIDTH", "1"))


def recognize_music(
    strip_images: list[np.ndarray],
    checkpoint_path: Path | None = None,
    staff_line_positions: list[list[int] | None] | None = None,
    music_bbox_y0s: list[int] | None = None,
    music_binaries: list[np.ndarray | None] | None = None,
    beam_width: int | None = None,
) -> list[str]:
    """Run the CRNN on a list of music-staff crops.

    Returns one LMX token string per strip.  Decoding defaults to greedy CTC
    (same as ``evaluate`` / the phase-2 notebook); set ``OMR_BEAM_WIDTH`` > 1
    for beam search.
    """
    if not strip_images:
        return []

    if checkpoint_path is None:
        checkpoint_path = Path("models/latest/best_model.pt")
    if not checkpoint_path.exists():
        return [f"[no checkpoint: {checkpoint_path}]"] * len(strip_images)

    entry = _load_model(checkpoint_path)
    model = entry["model"]
    vocab = entry["vocab"]
    cfg = entry["cfg"]
    device = entry["device"]
    use_amp = device.type == "cuda"
    img_height = getattr(cfg, "img_height", 128)
    max_width = getattr(cfg, "max_image_width", 2048)

    if beam_width is None:
        beam_width = _BEAM_WIDTH

    if beam_width > 1:
        from CRNN_CTC.evaluate import beam_search_decode
        _decode_fn = lambda lp, ol, v: beam_search_decode(lp, ol, v, beam_width)
        log.info("Using beam search (width=%d)", beam_width)
    else:
        from CRNN_CTC.evaluate import greedy_decode
        _decode_fn = greedy_decode

    # ----- batch-decode helper -----
    def _decode(tiles: list[np.ndarray]) -> list[list[str]]:
        tensors: list[Tensor] = []
        widths: list[int] = []
        for t in tiles:
            try:
                tensor, tw = _prepare_tile(t, img_height, max_width)
            except Exception as exc:
                log.warning("Tile prep failed: %s", exc)
                tensor = torch.zeros(1, img_height, 1)
                tw = 1
            tensors.append(tensor)
            widths.append(tw)

        max_w = max(t.shape[-1] for t in tensors)
        padded = []
        for t in tensors:
            if t.shape[-1] < max_w:
                t = torch.cat([t, torch.zeros(1, img_height, max_w - t.shape[-1])], dim=-1)
            padded.append(t.unsqueeze(0))

        batch = torch.cat(padded, dim=0).to(device)
        width_t = torch.tensor(widths, dtype=torch.long, device=device)
        with torch.inference_mode():
            with autocast("cuda", enabled=use_amp):
                log_probs, out_lens = model(batch, width_t)
            return _decode_fn(log_probs, out_lens, vocab)

    # ----- fill defaults -----
    if staff_line_positions is None:
        staff_line_positions = [None] * len(strip_images)
    if music_bbox_y0s is None:
        music_bbox_y0s = [0] * len(strip_images)
    if music_binaries is None:
        music_binaries = [None] * len(strip_images)

    results: list[str] = []
    for img, staff_ys, y0, mbin in zip(
        strip_images, staff_line_positions, music_bbox_y0s, music_binaries,
    ):
        if img is None or img.size == 0:
            results.append("")
            continue

        local_lines: list[int] | None = None
        if mbin is not None and mbin.size > 0:
            local_lines = local_primary_staff_lines(mbin)
        if local_lines is not None:
            normed = normalize_staff_crop(img, local_lines, 0)
        else:
            normed = normalize_staff_crop(img, staff_ys, y0)
        tile_info = _tile_strip(normed, img_height)

        if len(tile_info) == 1:
            token_lists = _decode([tile_info[0][0]])
            merged = list(token_lists[0]) if token_lists else []
        else:
            tile_imgs = [t for t, _, _ in tile_info]
            fracs = [(ks, ke) for _, ks, ke in tile_info]
            token_lists = _decode(tile_imgs)

            merged: list[str] = []
            for tokens, (ks, ke) in zip(token_lists, fracs):
                n = len(tokens)
                if n == 0:
                    continue
                si = int(n * ks)
                ei = max(si + 1, int(n * ke))
                kept = tokens[si:ei]
                if merged and kept and merged[-1] == "measure" and kept[0] == "measure":
                    kept = kept[1:]
                merged.extend(kept)

        results.append(" ".join(merged))
    return results
