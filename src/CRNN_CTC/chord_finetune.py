"""
chord_finetune.py
=================
Fine-tune the synthetic-pretrained chord CRNN on hand-labeled real Real Book
chord strips.

The starting point is ``models/chord/latest/best_model.pt`` (the synthetic
checkpoint).  We continue training with:

* the hand-labeled real strips from ``data/chord_real/labels.jsonl``
  (``status == "done"`` entries only — skipped strips are excluded);
* a mix of synthetic samples weighted at ``--synth-weight`` (default 0.5)
  so the model doesn't catastrophically forget rare chord types that may
  not be in the labeled real set;
* a lower learning rate, fewer epochs, the augmentation pipeline kept on
  (real strips still benefit from jitter / clutter).

Usage::

    poetry run python -m CRNN_CTC.chord_finetune \
        --epochs 20 --synth-weight 0.5 --lr 2e-4
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from .chord_dataset import (
    ChordDataset,
    _apply_augmentation,
    _load_chord_image,
)
from .dataset import collate_fn
from .model import CRNN
from .training_utils import (
    seed_everything,
    train_one_epoch,
    update_latest_symlink,
    validate_ctc_epoch,
)
from .vocab import Vocabulary

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Real-data dataset (reads labels.jsonl)
# ---------------------------------------------------------------------------


class RealChordDataset(Dataset):
    """Hand-labeled real Real Book chord strips."""

    def __init__(
        self,
        strips_dir: Path,
        labels_jsonl: Path,
        vocab: Vocabulary,
        *,
        img_height: int = 64,
        max_image_width: int = 2048,
        augment: bool = True,
    ) -> None:
        self.strips_dir = Path(strips_dir)
        self.vocab = vocab
        self.img_height = img_height
        self.max_image_width = max_image_width
        self.augment = augment

        # Canonicalise hand-typed labels so the model sees a clean,
        # consistent character sequence.  The labeling UI lets users write
        # things like double spaces, '%' (repeat sign), 'm7b5' notation,
        # which we silently normalise here so they match the synthetic
        # training distribution.
        import re as _re

        vocab_chars = set(vocab._tok2idx.keys())

        # Half-diminished synonyms typed without ø: "m7b5", "-7b5", "min7b5"
        _HALFDIM_RE = _re.compile(r"(?:m7b5|-7b5|min7b5)", _re.IGNORECASE)
        # Minor with 'm' instead of '-': "Cm7" → "C-7", "Cm" → "C-"
        # Python regex needs fixed-width lookbehind, so use two alternations.
        _M_TO_DASH_RE = _re.compile(r"(?:(?<=[A-G])|(?<=[A-G][#b]))m(?=\d|\b)")

        def _canon(label: str) -> str:
            # 1. Normalise half-dim notation to ø
            label = _HALFDIM_RE.sub("ø", label)
            # 2. "Cm7" → "C-7" (some users may write minor with m)
            label = _M_TO_DASH_RE.sub("-", label)
            # 3. Drop any character not in the chord vocabulary (%, |, parens, …)
            label = "".join(c for c in label if c in vocab_chars)
            # 4. Collapse runs of whitespace into a single space, trim
            return " ".join(label.split())

        self._samples: list[tuple[str, str]] = []
        n_dropped = 0
        with open(labels_jsonl, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("status") != "done":
                    continue
                label = rec.get("label")
                if label is None:
                    continue
                label = _canon(label)
                if not label:
                    n_dropped += 1
                    continue
                if not (self.strips_dir / rec["filename"]).exists():
                    continue
                self._samples.append((rec["filename"], label))
        if n_dropped:
            log.warning("Dropped %d real samples with empty/non-vocab labels", n_dropped)

        if not self._samples:
            raise RuntimeError(f"No labeled real samples in {labels_jsonl} (status=done with non-null label)")
        log.info(
            "RealChordDataset: %d labeled strips from %s (augment=%s)",
            len(self._samples),
            labels_jsonl,
            augment,
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        filename, label = self._samples[idx]
        img = _load_chord_image(
            self.strips_dir / filename,
            self.img_height,
            self.max_image_width,
        )
        if self.augment:
            img_u8 = (img * 255).astype(np.uint8)
            # Real strips already have natural clutter — only apply
            # Albumentations geometric/photometric jitter, not synthetic clutter.
            img_u8 = _apply_augmentation(img_u8)
            img = img_u8.astype(np.float32) / 255.0

        img = (img - img.mean()) / (img.std() + 1e-6)
        img_t = torch.from_numpy(img).unsqueeze(0)
        label_indices = self.vocab.encode(list(label))
        return {
            "sample_id": filename,
            "image": img_t,
            "label": torch.tensor(label_indices, dtype=torch.long),
            "tokens": list(label),
        }


# ---------------------------------------------------------------------------
# Fine-tune driver
# ---------------------------------------------------------------------------


def _build_weighted_sampler(
    n_real: int,
    n_synth: int,
    synth_weight: float,
) -> WeightedRandomSampler:
    """Each real sample gets weight 1; each synth gets weight ``synth_weight``."""
    weights = [1.0] * n_real + [synth_weight] * n_synth
    # Sample ~ (n_real + synth_weight·n_synth) per epoch
    num_samples = int(n_real + synth_weight * n_synth)
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)


def finetune(args) -> Path:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    log.info("Device: %s | AMP: %s", device, use_amp)

    # ── Resolve checkpoint + vocab ─────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    if "vocab_tokens" in ckpt:
        vocab = Vocabulary(list(ckpt["vocab_tokens"]))
    else:
        vocab = Vocabulary.from_file(Path(args.vocab_path))
    log.info("Vocabulary: %d tokens", len(vocab))

    img_height = cfg.get("img_height", 64)
    max_image_width = cfg.get("max_image_width", 2048)

    # ── Datasets ───────────────────────────────────────────────────────────
    real_ds = RealChordDataset(
        strips_dir=Path(args.real_strips_dir),
        labels_jsonl=Path(args.real_labels),
        vocab=vocab,
        img_height=img_height,
        max_image_width=max_image_width,
        augment=True,
    )

    # Split real into train/val (small val so we don't waste real data)
    n_real = len(real_ds)
    n_val = max(10, n_real // 10)
    perm = torch.randperm(n_real, generator=torch.Generator().manual_seed(args.seed)).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    from torch.utils.data import Subset

    real_train = Subset(real_ds, train_idx)
    real_val = Subset(real_ds, val_idx)
    log.info("Real split: %d train / %d val", len(real_train), len(real_val))

    synth_train = ChordDataset(
        image_dir=Path(args.synth_dir) / "train",
        labels_csv=Path(args.synth_dir) / "train_labels.csv",
        vocab=vocab,
        img_height=img_height,
        max_image_width=max_image_width,
        augment=True,
    )
    log.info("Synthetic train pool: %d samples", len(synth_train))

    # Concat real_train + synth_train, sample with weights so real is upweighted
    combined = ConcatDataset([real_train, synth_train])
    sampler = _build_weighted_sampler(
        n_real=len(real_train),
        n_synth=len(synth_train),
        synth_weight=args.synth_weight,
    )
    log.info(
        "Combined sampler: %d epoch samples (real wt=1.0, synth wt=%.2f)",
        sampler.num_samples,
        args.synth_weight,
    )

    train_loader = DataLoader(
        combined,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=use_amp,
        drop_last=True,
    )
    val_loader = DataLoader(
        real_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=use_amp,
    )

    # ── Model — load synthetic checkpoint ──────────────────────────────────
    model = CRNN(
        vocab_size=len(vocab),
        cnn_out_channels=cfg.get("cnn_out_channels", 256),
        rnn_hidden=cfg.get("rnn_hidden", 192),
        rnn_layers=cfg.get("rnn_layers", 2),
        dropout=cfg.get("dropout", 0.2),
        cnn_dropout=cfg.get("cnn_dropout", 0.15),
        backbone=cfg.get("backbone", "resnet18"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    log.info("Loaded weights from %s (val_CER=%.4f)", ckpt_path, ckpt.get("val_cer", -1.0))

    criterion = nn.CTCLoss(blank=vocab.blank_idx, zero_infinity=True)
    optimiser = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    scheduler = OneCycleLR(
        optimiser,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=args.warmup_frac,
        anneal_strategy="cos",
    )
    scaler = GradScaler("cuda", enabled=use_amp)

    # ── Output dir ─────────────────────────────────────────────────────────
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.model_dir) / f"finetune_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output dir: %s", out_dir)

    log_csv = out_dir / "training_log.csv"
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_cer", "elapsed_s"])

    best_cer = float("inf")
    best_path = out_dir / "best_model.pt"
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimiser,
            scheduler,
            scaler,
            device,
            use_amp,
            max_grad_norm=args.max_grad_norm,
        )
        val_loss, val_cer = validate_ctc_epoch(
            model,
            val_loader,
            criterion,
            vocab,
            device,
            use_amp,
        )
        dt = time.time() - t0

        log.info(
            "Epoch %d/%d  train=%.4f  val=%.4f  CER=%.4f  (%.1fs)",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            val_cer,
            dt,
        )
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_cer, dt])

        if val_cer < best_cer:
            best_cer = val_cer
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "vocab_tokens": vocab._idx2tok[3:],
                    "val_cer": val_cer,
                    "val_loss": val_loss,
                    "finetuned_from": str(ckpt_path),
                },
                best_path,
            )
            log.info("  ↳ new best CER: %.4f  saved → %s", best_cer, best_path)
        else:
            no_improve += 1
            if args.early_stopping_patience > 0 and no_improve >= args.early_stopping_patience:
                log.info("Early stopping: no val-CER improvement for %d epochs", no_improve)
                break

    update_latest_symlink(Path(args.model_dir), out_dir)

    log.info("Fine-tune complete.  Best CER: %.4f  →  %s", best_cer, best_path)
    return best_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--checkpoint",
        default="models/chord/latest/best_model.pt",
        help="Synthetic-pretrained checkpoint to start from.",
    )
    p.add_argument("--real-strips-dir", default="data/chord_real/strips")
    p.add_argument("--real-labels", default="data/chord_real/labels.jsonl")
    p.add_argument("--synth-dir", default="data/chord_synth")
    p.add_argument("--vocab-path", default="data/vocab/chord.txt")
    p.add_argument("--model-dir", default="models/chord")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4, help="Lower LR than from-scratch (default: 2e-4).")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=5.0)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--synth-weight", type=float, default=0.5, help="Per-sample weight for synthetic data (real=1.0).")
    p.add_argument("--early-stopping-patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    finetune(args)


if __name__ == "__main__":
    main()
