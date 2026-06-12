"""CRNN inference — preprocessing matches ``dataset.py`` training exactly."""

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
# Model cache (loaded once per process per checkpoint)
# ---------------------------------------------------------------------------

_model_cache: dict = {}


def _load_model(checkpoint_path: Path) -> dict:
    key = str(checkpoint_path.resolve())
    if key in _model_cache:
        return _model_cache[key]

    from CRNN_CTC.config import Config, ensure_config_defaults
    from CRNN_CTC.model import CRNN
    from CRNN_CTC.vocab import Vocabulary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ensure_config_defaults(ckpt.get("config", Config()))

    # Resolve project root by walking up from the checkpoint until pyproject.toml is found
    repo_root = checkpoint_path.resolve().parent
    while repo_root.parent != repo_root and not (repo_root / "pyproject.toml").exists():
        repo_root = repo_root.parent

    # Prefer the token list embedded in the checkpoint (authoritative; immune to
    # the vocab file being re-sorted at the same length). Fall back to the file.
    if ckpt.get("vocab_tokens") is not None:
        vocab = Vocabulary(list(ckpt["vocab_tokens"]))
    else:
        candidates = [
            Path(cfg.vocab_path),
            repo_root / cfg.vocab_path,
            repo_root / "data" / "vocab" / "primus_lmx.txt",
            checkpoint_path.parent.parent / "data" / "vocab" / "primus_lmx.txt",
        ]
        vocab_path = next((p for p in candidates if p.exists()), None)
        if vocab_path is None:
            raise FileNotFoundError(f"Vocab file not found; tried: {[str(p) for p in candidates]}")

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
        checkpoint_path,
        ckpt.get("epoch", -1),
        ckpt.get("val_ser", -1),
    )
    return entry


# ---------------------------------------------------------------------------
# Preprocessing — exact match to dataset.py
# ---------------------------------------------------------------------------


def _preprocess_strip(img: np.ndarray, img_height: int, max_width: int) -> tuple[Tensor, int]:
    """Grayscale uint8 → resize → /255 → per-image norm → (1, H, W) tensor."""
    if img.ndim == 3:
        img = img[:, :, 0]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

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
def recognize_music(
    strip_images: list[np.ndarray],
    checkpoint_path: Path | None = None,
    beam_width: int | None = None,
) -> tuple[list[str], list[Tensor], list[int]]:
    """Run CRNN on a list of music-staff crops.

    Returns three parallel lists, one entry per crop:

    * ``token_strings``: space-joined LMX token string (``""`` if the strip was
      empty or preprocess failed).
    * ``per_strip_log_probs``: a CPU tensor of shape ``(T_i, C)`` carrying the
      log-probabilities for the first ``out_len_i`` time-steps of strip ``i``.
      Used by the post-CRNN reject gate in ``staff_reject``.
    * ``per_strip_out_lens``: the effective time-step count ``out_len_i``.
    """
    if not strip_images:
        return [], [], []

    if checkpoint_path is None:
        env_ckpt = os.environ.get("OMR_CHECKPOINT", "").strip()
        checkpoint_path = Path(env_ckpt) if env_ckpt else Path("models/latest/best_model.pt")
    if not checkpoint_path.exists():
        log.error("Checkpoint not found: %s", checkpoint_path)
        n = len(strip_images)
        return ([""] * n, [torch.zeros(0, 1) for _ in range(n)], [0] * n)

    entry = _load_model(checkpoint_path)
    model = entry["model"]
    vocab = entry["vocab"]
    cfg = entry["cfg"]
    device = entry["device"]
    img_height = getattr(cfg, "img_height", 128)
    max_width = getattr(cfg, "max_image_width", 2048)
    use_amp = device.type == "cuda"

    if beam_width is None:
        beam_width = max(1, int(os.environ.get("OMR_BEAM_WIDTH", "1")))

    if beam_width > 1:
        from CRNN_CTC.evaluate import beam_search_decode

        def decode_fn(lp, ol, v):
            return beam_search_decode(lp, ol, v, beam_width)

        log.info("Using beam search (width=%d)", beam_width)
    else:
        from CRNN_CTC.evaluate import greedy_decode

        decode_fn = greedy_decode

    # Preprocess all strips into per-image tensors + widths
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
            log.warning("Strip preprocess failed: %s", exc)
            tensors.append(torch.zeros(1, img_height, 1))
            widths.append(1)
            valid_mask.append(False)

    # Right-pad batch to common width with zeros (matches dataset.collate_fn)
    max_w = max(t.shape[-1] for t in tensors)
    padded: list[Tensor] = []
    for t in tensors:
        if t.shape[-1] < max_w:
            pad = torch.zeros(1, img_height, max_w - t.shape[-1])
            t = torch.cat([t, pad], dim=-1)
        padded.append(t.unsqueeze(0))  # (1, 1, H, W)

    batch = torch.cat(padded, dim=0).to(device)
    width_t = torch.tensor(widths, dtype=torch.long, device=device)

    with autocast("cuda", enabled=use_amp):
        log_probs, out_lens = model(batch, width_t)
    token_lists = decode_fn(log_probs, out_lens, vocab)

    # CRNN returns log_probs shaped (T, B, C). Move to CPU and slice per-strip.
    lp_cpu = log_probs.detach().to("cpu")  # (T, B, C)
    ol_cpu = out_lens.detach().to("cpu").tolist()  # length B

    results: list[str] = []
    per_strip_lp: list[Tensor] = []
    per_strip_ol: list[int] = []
    for i, (tl, ok) in enumerate(zip(token_lists, valid_mask)):
        results.append(" ".join(tl) if ok else "")
        ol = int(ol_cpu[i]) if i < len(ol_cpu) else 0
        if ok and ol > 0:
            per_strip_lp.append(lp_cpu[:ol, i, :].clone())
        else:
            per_strip_lp.append(torch.zeros(0, lp_cpu.shape[-1]))
        per_strip_ol.append(ol if ok else 0)
    return results, per_strip_lp, per_strip_ol
