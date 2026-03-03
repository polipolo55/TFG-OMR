"""
evaluate.py
===========
Evaluation utilities for the CRNN-CTC OMR pipeline.

Provides
--------
* **Greedy CTC decoding** — collapse repeated tokens, strip blanks.
* **Symbol Error Rate (SER)** — edit distance at the token level, analogous
  to Character Error Rate in OCR.
* **Full evaluation loop** — load a checkpoint, run inference on a test split,
  report aggregate and per-sample SER.

Usage::

    poetry run python src/cli.py evaluate --checkpoint models/best_model.pt
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import Tensor
from torch.amp import autocast
from torch.utils.data import DataLoader

from .config import Config
from .dataset import OMRDataset, collate_fn, make_splits
from .model import CRNN
from .vocab import Vocabulary

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CTC greedy decoder
# ---------------------------------------------------------------------------

def greedy_decode(
    log_probs: Tensor,
    output_lengths: Tensor,
    vocab: Vocabulary,
) -> list[list[str]]:
    """Greedy-decode a batch of CTC log-probability matrices.

    Parameters
    ----------
    log_probs : Tensor
        (T, B, vocab_size) — log-softmax output from the model.
    output_lengths : Tensor
        (B,) — valid time-steps per sample.
    vocab : Vocabulary
        Mapping to convert indices back to token strings.

    Returns
    -------
    list[list[str]]
        Decoded token sequences, one per sample in the batch.
    """
    # Argmax along vocab dim → (T, B)
    best = log_probs.argmax(dim=2).T  # → (B, T)

    decoded: list[list[str]] = []
    for i, length in enumerate(output_lengths):
        raw = best[i, : length.item()].tolist()
        # Collapse consecutive duplicates, then remove blanks
        collapsed: list[int] = []
        prev = -1
        for idx in raw:
            if idx != prev:
                collapsed.append(idx)
            prev = idx
        # Remove blank symbol
        collapsed = [idx for idx in collapsed if idx != vocab.blank_idx]
        tokens = vocab.decode(collapsed)
        decoded.append(tokens)

    return decoded


def beam_search_decode(
    log_probs: Tensor,
    output_lengths: Tensor,
    vocab: Vocabulary,
    beam_width: int = 10,
) -> list[list[str]]:
    """Prefix beam search CTC decoding.

    Parameters
    ----------
    log_probs : Tensor
        (T, B, vocab_size) — log-softmax output from the model.
    output_lengths : Tensor
        (B,) — valid time-steps per sample.
    vocab : Vocabulary
        Mapping to convert indices back to token strings.
    beam_width : int
        Number of beams to keep at each time-step.

    Returns
    -------
    list[list[str]]
        Decoded token sequences, one per sample in the batch.
    """
    B = log_probs.shape[1]
    blank = vocab.blank_idx
    decoded: list[list[str]] = []

    for i in range(B):
        T_i = output_lengths[i].item()
        lp = log_probs[:T_i, i, :]  # (T_i, V)

        # Each beam: (prefix_tuple, (log_prob_blank_end, log_prob_non_blank_end))
        NEG_INF = float("-inf")
        beams: dict[tuple[int, ...], list[float]] = {
            (): [0.0, NEG_INF],  # empty prefix: blank-end prob=1, non-blank=0
        }

        for t in range(T_i):
            new_beams: dict[tuple[int, ...], list[float]] = {}
            scores_t = lp[t].tolist()  # vocab_size floats (log-probs)

            # Prune to top beam_width prefixes by total log-prob
            scored = [
                (prefix, _log_add(pb, pnb))
                for prefix, (pb, pnb) in beams.items()
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            scored = scored[:beam_width]

            for prefix, _total in scored:
                pb, pnb = beams[prefix]

                # --- extend with blank ---
                new_pb = _log_add(pb + scores_t[blank], pnb + scores_t[blank])
                if prefix not in new_beams:
                    new_beams[prefix] = [NEG_INF, NEG_INF]
                new_beams[prefix][0] = _log_add(new_beams[prefix][0], new_pb)

                # --- extend with non-blank tokens ---
                for c in range(len(scores_t)):
                    if c == blank:
                        continue
                    sc = scores_t[c]
                    # If last char of prefix == c, only blank-ended paths can extend
                    if prefix and prefix[-1] == c:
                        new_pnb = pb + sc  # repeat only from blank-ended
                        # Also continue the prefix without extension
                        if prefix not in new_beams:
                            new_beams[prefix] = [NEG_INF, NEG_INF]
                        new_beams[prefix][1] = _log_add(
                            new_beams[prefix][1], pnb + sc
                        )
                    else:
                        new_pnb = _log_add(pb + sc, pnb + sc)

                    ext = prefix + (c,)
                    if ext not in new_beams:
                        new_beams[ext] = [NEG_INF, NEG_INF]
                    new_beams[ext][1] = _log_add(new_beams[ext][1], new_pnb)

            beams = new_beams

        # Select best beam
        best_prefix = max(
            beams, key=lambda p: _log_add(beams[p][0], beams[p][1])
        )
        tokens = vocab.decode(list(best_prefix))
        decoded.append(tokens)

    return decoded


def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a > b:
        return a + _log_stable(b - a)
    return b + _log_stable(a - b)


def _log_stable(x: float) -> float:
    """log(1 + exp(x)) for small x."""
    import math
    if x < -50:
        return 0.0
    return math.log1p(math.exp(x))


# ---------------------------------------------------------------------------
# Edit distance (Levenshtein)
# ---------------------------------------------------------------------------

def _edit_distance(hyp: list[str], ref: list[str]) -> int:
    """Compute Levenshtein edit distance between two token sequences."""
    n, m = len(hyp), len(ref)
    # Optimise memory: only keep two rows
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if hyp[i - 1] == ref[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost, # substitution
            )
        prev, curr = curr, prev
    return prev[m]


def symbol_error_rate(hyp: list[str], ref: list[str]) -> float:
    """SER = edit_distance(hyp, ref) / len(ref).

    Returns 0.0 when both sequences are empty; returns float('inf') when
    ref is empty but hyp is not (pure insertions).
    """
    if not ref:
        return 0.0 if not hyp else float("inf")
    return _edit_distance(hyp, ref) / len(ref)


def compute_ser_batch(
    predictions: list[list[str]],
    references: list[list[str]],
) -> tuple[int, int]:
    """Accumulate edit distance and reference length across a batch.

    Returns ``(total_edits, total_ref_length)`` so the caller can aggregate
    across batches: ``SER = sum(edits) / sum(ref_lengths)``.
    """
    total_edit = 0
    total_len = 0
    for hyp, ref in zip(predictions, references):
        total_edit += _edit_distance(hyp, ref)
        total_len += len(ref)
    return total_edit, total_len


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate(
    cfg: Config,
    checkpoint_path: Path | str,
    *,
    split: str = "test",
    per_sample: bool = False,
    beam_width: int = 1,
) -> float:
    """Load a checkpoint, run inference on the requested split, report SER.

    Parameters
    ----------
    cfg : Config
        Pipeline configuration.
    checkpoint_path : Path | str
        Path to a ``torch.save``-d checkpoint dict.
    split : str
        Which split to evaluate: ``"test"`` (default), ``"val"``, or ``"train"``.
    per_sample : bool
        If True, log SER for each individual sample.

    Returns
    -------
    float
        Aggregate SER on the requested split.
    """
    decode_fn = (
        lambda lp, ol, v: beam_search_decode(lp, ol, v, beam_width)
        if beam_width > 1
        else greedy_decode(lp, ol, v)
    )
    if beam_width > 1:
        log.info("Using beam search decoding (beam_width=%d)", beam_width)
    checkpoint_path = Path(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # ── Vocabulary ─────────────────────────────────────────────────────────
    vocab = Vocabulary.from_file(cfg.vocab_path)

    # ── Load model from checkpoint ─────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CRNN(
        vocab_size=ckpt.get("vocab_size", len(vocab)),
        cnn_out_channels=cfg.cnn_out_channels,
        rnn_hidden=cfg.rnn_hidden,
        rnn_layers=cfg.rnn_layers,
        dropout=0.0,  # no dropout at inference
        backbone=cfg.backbone,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info("Loaded checkpoint: %s (epoch %d, val_SER=%.4f)",
             checkpoint_path, ckpt.get("epoch", -1), ckpt.get("val_ser", -1))

    # ── Data ───────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds = make_splits(
        data_dir=cfg.data_dir,
        vocab=vocab,
        img_height=cfg.img_height,
        max_image_width=cfg.max_image_width,
        scanned_dir=cfg.scanned_dir if cfg.use_scanned else None,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        seed=cfg.seed,
        filter_rest_heavy=cfg.filter_rest_heavy,
        filter_unwanted_clefs=cfg.filter_unwanted_clefs,
        filter_multi_staff=cfg.filter_multi_staff,
        max_source_height=cfg.max_source_height,
        extra_data_dirs=cfg.extra_data_dirs or None,
        extra_scanned_dirs=(
            cfg.extra_scanned_dirs if cfg.use_scanned else None
        ) or None,
    )
    ds_map = {"train": train_ds, "val": val_ds, "test": test_ds}
    ds = ds_map[split]
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    log.info("Evaluating on '%s' split (%d samples)", split, len(ds))

    # ── Inference ──────────────────────────────────────────────────────────
    total_edit = 0
    total_len = 0
    sample_results: list[tuple[str, float, list[str], list[str]]] = []

    for batch in loader:
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        label_lens = batch["label_lens"].to(device)
        image_widths = batch["image_widths"].to(device)
        sids = batch["sample_ids"]

        with autocast("cuda", enabled=use_amp):
            log_probs, output_lens = model(images, image_widths)

        preds = decode_fn(log_probs, output_lens, vocab)

        # Reconstruct per-sample ground truth
        offset = 0
        for i, length in enumerate(label_lens):
            l = length.item()
            ref = vocab.decode(labels[offset : offset + l].tolist())
            offset += l

            ed = _edit_distance(preds[i], ref)
            ser = ed / max(len(ref), 1)
            total_edit += ed
            total_len += len(ref)

            if per_sample:
                sample_results.append((sids[i], ser, preds[i], ref))

    aggregate_ser = total_edit / max(total_len, 1)

    # ── Report ─────────────────────────────────────────────────────────────
    if per_sample:
        sample_results.sort(key=lambda x: x[1], reverse=True)  # worst first
        log.info("Per-sample SER (worst → best):")
        for sid, ser, pred, ref in sample_results[:20]:  # top 20
            log.info("  %s  SER=%.4f  pred_len=%d  ref_len=%d",
                     sid, ser, len(pred), len(ref))

    log.info("Aggregate SER on '%s': %.4f  (%d edits / %d symbols)",
             split, aggregate_ser, total_edit, total_len)
    return aggregate_ser
