"""
train.py
========
Training loop for the CRNN-CTC monophonic OMR model.

Features
--------
* CTC loss with ``torch.nn.CTCLoss(blank=0, zero_infinity=True)``.
* Mixed-precision training via ``torch.amp`` (CUDA AMP).
* Deterministic seeding for reproducibility.
* Cosine-annealing LR scheduler with warm-up.
* Periodic validation with SER reporting.
* Best-model checkpointing (lowest validation SER).
* CSV-based training log (epoch, train_loss, val_loss, val_ser).

Usage — called from ``cli.py``::

    poetry run python src/cli.py train --epochs 40 --batch-size 16

Or programmatically::

    from CRNN_CTC.train import train
    from CRNN_CTC.config import Config
    train(Config())
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from .config import Config
from .dataset import OMRDataset, collate_fn, make_splits
from .evaluate import compute_ser_batch, greedy_decode
from .model import CRNN
from .vocab import Vocabulary

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------

def _train_one_epoch(
    model: CRNN,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    optimiser: torch.optim.Optimizer,
    scheduler: OneCycleLR | None,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
) -> float:
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    bar = tqdm(loader, desc="train", leave=False, dynamic_ncols=True)
    for batch in bar:
        images: Tensor = batch["images"].to(device)
        labels: Tensor = batch["labels"].to(device)
        label_lens: Tensor = batch["label_lens"].to(device)
        image_widths: Tensor = batch["image_widths"].to(device)

        optimiser.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_amp):
            log_probs, output_lens = model(images, image_widths)
            # CTCLoss expects (T, B, C), targets flat, input_lengths, target_lengths
            loss = criterion(log_probs, labels, output_lens, label_lens)

        scaler.scale(loss).backward()
        # Gradient clipping — essential for CTC stability
        scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        # Track scale before step: if it drops, the step was skipped (inf/NaN
        # in gradients during AMP warm-up). Only advance the LR scheduler when
        # the optimizer actually performed an update.
        scale_before = scaler.get_scale()
        scaler.step(optimiser)
        scaler.update()

        if scheduler is not None and scaler.get_scale() >= scale_before:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1
        bar.set_postfix(loss=f"{total_loss / n_batches:.4f}",
                        lr=f"{optimiser.param_groups[0]['lr']:.2e}")

    return total_loss / max(n_batches, 1)


@torch.inference_mode()
def _validate(
    model: CRNN,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    vocab: Vocabulary,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    """Run validation. Returns (avg_loss, SER)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    total_edit = 0
    total_len = 0

    for batch in loader:
        images: Tensor = batch["images"].to(device)
        labels: Tensor = batch["labels"].to(device)
        label_lens: Tensor = batch["label_lens"].to(device)
        image_widths: Tensor = batch["image_widths"].to(device)

        with autocast("cuda", enabled=use_amp):
            log_probs, output_lens = model(images, image_widths)
            loss = criterion(log_probs, labels, output_lens, label_lens)

        total_loss += loss.item()
        n_batches += 1

        # Greedy decode + SER
        preds = greedy_decode(log_probs, output_lens, vocab)
        # Reconstruct per-sample ground truth from flat labels
        gt_tokens: list[list[str]] = []
        offset = 0
        for length in label_lens:
            l = length.item()
            gt_tokens.append(vocab.decode(labels[offset : offset + l].tolist()))
            offset += l

        edit, ref_len = compute_ser_batch(preds, gt_tokens)
        total_edit += edit
        total_len += ref_len

    avg_loss = total_loss / max(n_batches, 1)
    ser = total_edit / max(total_len, 1)
    return avg_loss, ser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train(cfg: Config) -> Path:
    """Full training run. Returns path to the best checkpoint.

    Parameters
    ----------
    cfg : Config
        Centralised configuration (paths, hyperparameters, etc.).
    """
    _seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    log.info("Device: %s | AMP: %s", device, use_amp)

    # ── Vocabulary ─────────────────────────────────────────────────────────
    vocab = Vocabulary.from_file(cfg.vocab_path)
    log.info("Vocabulary: %d tokens (incl. blank + pad)", len(vocab))

    # ── Data ───────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds = make_splits(
        data_dir=cfg.data_dir,
        vocab=vocab,
        img_height=cfg.img_height,
        scanned_dir=cfg.scanned_dir if cfg.use_scanned else None,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        seed=cfg.seed,
        filter_rest_heavy=cfg.filter_rest_heavy,
        filter_unwanted_clefs=cfg.filter_unwanted_clefs,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = CRNN(
        vocab_size=len(vocab),
        cnn_out_channels=cfg.cnn_out_channels,
        rnn_hidden=cfg.rnn_hidden,
        rnn_layers=cfg.rnn_layers,
        dropout=cfg.dropout,
        cnn_dropout=cfg.cnn_dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %s", f"{n_params:,}")

    # ── Optimiser & scheduler ──────────────────────────────────────────────
    criterion = nn.CTCLoss(blank=vocab.blank_idx, zero_infinity=True)
    optimiser = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = cfg.epochs * len(train_loader)
    scheduler = OneCycleLR(
        optimiser,
        max_lr=cfg.lr,
        total_steps=total_steps,
        pct_start=cfg.warmup_frac,
        anneal_strategy="cos",
    )

    scaler = GradScaler("cuda", enabled=use_amp)

    # ── Training log ───────────────────────────────────────────────────────
    log_path = cfg.model_dir / "training_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_ser", "lr", "elapsed_s"])

    # ── Training loop ──────────────────────────────────────────────────────
    best_ser = float("inf")
    best_ckpt = cfg.model_dir / "best_model.pt"
    patience_counter = 0
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        t_epoch = time.time()

        train_loss = _train_one_epoch(
            model, train_loader, criterion, optimiser, scheduler, scaler, device, use_amp,
        )
        val_loss, val_ser = _validate(
            model, val_loader, criterion, vocab, device, use_amp,
        )

        elapsed = time.time() - t0
        current_lr = optimiser.param_groups[0]["lr"]

        log.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_SER=%.4f  lr=%.2e  [%.0fs]",
            epoch, cfg.epochs, train_loss, val_loss, val_ser, current_lr, elapsed,
        )

        # Append to CSV log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                             f"{val_ser:.6f}", f"{current_lr:.2e}",
                             f"{time.time() - t_epoch:.1f}"])

        # Checkpoint best model
        if val_ser < best_ser:
            best_ser = val_ser
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                    "val_ser": val_ser,
                    "val_loss": val_loss,
                    "config": cfg,
                    "vocab_size": len(vocab),
                },
                best_ckpt,
            )
            log.info("  ✓ New best SER=%.4f — saved %s", val_ser, best_ckpt)
        else:
            patience_counter += 1

        # Also save latest (for resuming)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_ser": val_ser,
                "config": cfg,
            },
            cfg.model_dir / "latest_checkpoint.pt",
        )

        # Early stopping
        if cfg.early_stopping_patience > 0 and patience_counter >= cfg.early_stopping_patience:
            log.info(
                "Early stopping — val SER did not improve for %d epochs (best=%.4f)",
                cfg.early_stopping_patience, best_ser,
            )
            break

    total_time = time.time() - t0
    log.info(
        "Training complete. Best val SER=%.4f  Total time=%.0fs",
        best_ser, total_time,
    )
    return best_ckpt
