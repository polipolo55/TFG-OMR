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

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from .config import Config
from .dataset import collate_fn, make_splits
from .model import CRNN
from .training_utils import (
    create_run_dir,
    seed_everything,
    train_one_epoch,
    validate_ctc_epoch,
)
from .vocab import Vocabulary

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _resolve_run_dir(cfg: Config, resume_from: Path | str | None) -> Path:
    """Decide which run directory to use.

    * **Fresh run:** create a new timestamped directory.
    * **Resumed run:** reuse the directory containing the checkpoint.
    """
    if resume_from is not None:
        ckpt_path = Path(resume_from).resolve()
        # If the checkpoint lives inside a run_* subdirectory, reuse it.
        parent = ckpt_path.parent
        if parent.name.startswith("run_"):
            return parent
        # Legacy layout: checkpoint directly in model_dir — create new run
        # and copy the checkpoint there so the old file stays untouched.
        return create_run_dir(cfg.model_dir)

    return create_run_dir(cfg.model_dir)


def train(
    cfg: Config,
    resume_from: Path | str | None = None,
    init_weights: Path | str | None = None,
) -> Path:
    """Full training run. Returns path to the best checkpoint.

    Parameters
    ----------
    cfg : Config
        Centralised configuration (paths, hyperparameters, etc.).
    resume_from : path, optional
        Continue an interrupted run: restore model + optimiser + scheduler +
        epoch counter and keep training the SAME run in place.
    init_weights : path, optional
        Warm-start fine-tuning: load ONLY the model weights from this checkpoint,
        then start a FRESH run (new run dir, fresh optimiser, fresh OneCycle at
        ``cfg.lr`` for ``cfg.epochs``). Use with ``finetune_data_dirs`` to adapt
        a trained model to real data at a low LR. Mutually exclusive with
        ``resume_from``.
    """
    if resume_from is not None and init_weights is not None:
        raise ValueError("Pass either resume_from or init_weights, not both.")
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if use_amp:
        # Ampere+ TensorFloat-32: faster matmul on RTX 30xx with negligible accuracy impact
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    log.info("Device: %s | AMP: %s", device, use_amp)

    # ── Run directory (unique per training run) ────────────────────────────
    run_dir = _resolve_run_dir(cfg, resume_from)
    log.info("Run directory: %s", run_dir)

    # ── Vocabulary ─────────────────────────────────────────────────────────
    vocab = Vocabulary.from_file(cfg.vocab_path)
    log.info("Vocabulary: %d tokens (incl. blank + pad)", len(vocab))

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
        filter_non_leadsheet_clef=cfg.filter_non_leadsheet_clef,
        filter_unusual_time=cfg.filter_unusual_time,
        filter_multi_staff=cfg.filter_multi_staff,
        max_source_height=cfg.max_source_height,
        extra_data_dirs=cfg.extra_data_dirs or None,
        extra_scanned_dirs=(cfg.extra_scanned_dirs if cfg.use_scanned else None) or None,
        online_aug_prob=cfg.online_aug_prob,
        rare_lmx_oversample=cfg.rare_lmx_oversample,
        rare_lmx_tokens=frozenset(cfg.rare_lmx_tokens) if cfg.rare_lmx_tokens else frozenset(),
        finetune_data_dirs=cfg.finetune_data_dirs or None,
        finetune_scanned_dirs=(cfg.finetune_scanned_dirs if cfg.use_scanned else None) or None,
        variant_dirs=[Path(p) for p in cfg.scanned_variant_dirs] or None,
    )

    loader_kw: dict = {
        "batch_size": cfg.batch_size,
        "collate_fn": collate_fn,
        "pin_memory": device.type == "cuda",
        "persistent_workers": cfg.num_workers > 0,
    }
    if cfg.num_workers > 0:
        loader_kw["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        **loader_kw,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        num_workers=cfg.num_workers,
        **loader_kw,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = CRNN(
        vocab_size=len(vocab),
        cnn_out_channels=cfg.cnn_out_channels,
        rnn_hidden=cfg.rnn_hidden,
        rnn_layers=cfg.rnn_layers,
        dropout=cfg.dropout,
        cnn_dropout=cfg.cnn_dropout,
        backbone=cfg.backbone,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %s", f"{n_params:,}")

    # ── Optimiser & scheduler ──────────────────────────────────────────────
    criterion = nn.CTCLoss(blank=vocab.blank_idx, zero_infinity=True)
    try:
        optimiser = AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            fused=use_amp,
        )
    except TypeError, RuntimeError:
        optimiser = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ── Resume from checkpoint (if requested) ──────────────────────────────
    start_epoch = 1
    best_ser = float("inf")
    patience_counter = 0

    if resume_from is not None:
        ckpt_path = Path(resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        log.info("Resuming from checkpoint: %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimiser.load_state_dict(ckpt["optimiser_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_ser = ckpt.get("val_ser", float("inf"))
        patience_counter = ckpt.get("patience_counter", 0)
        log.info(
            "  Resumed at epoch %d  (best val_SER so far: %.4f, patience: %d)",
            start_epoch,
            best_ser,
            patience_counter,
        )

    remaining_epochs = cfg.epochs - (start_epoch - 1)
    if remaining_epochs <= 0:
        log.warning(
            "Checkpoint already reached epoch %d / %d — nothing left to train.",
            start_epoch - 1,
            cfg.epochs,
        )
        return run_dir / "best_model.pt"

    total_steps = cfg.epochs * len(train_loader)
    scheduler = OneCycleLR(
        optimiser,
        max_lr=cfg.lr,
        total_steps=total_steps,
        pct_start=cfg.warmup_frac,
        anneal_strategy="cos",
        last_epoch=-1,
    )

    if resume_from is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        # If the dataset size changed since the checkpoint was made, the saved
        # total_steps may not cover the remaining epochs.  Extend the budget so
        # the scheduler never runs out before epoch cfg.epochs.
        steps_remaining = (cfg.epochs - start_epoch + 1) * len(train_loader)
        new_total = scheduler.last_epoch + steps_remaining
        if new_total > scheduler.total_steps:
            scheduler.total_steps = new_total
            scheduler._schedule_phases[-1]["end_step"] = float(new_total) - 1
            log.info(
                "  Scheduler state restored (budget extended to %d steps — dataset size changed).",
                new_total,
            )
        else:
            log.info("  Scheduler state restored.")

    scaler = GradScaler("cuda", enabled=use_amp)
    if resume_from is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        log.info("  GradScaler state restored.")

    # ── Training log ───────────────────────────────────────────────────────
    log_path = run_dir / "training_log.csv"

    # Append to existing log when resuming, otherwise start fresh
    log_mode = "a" if resume_from is not None and log_path.exists() else "w"
    with open(log_path, log_mode, newline="") as f:
        writer = csv.writer(f)
        if log_mode == "w":
            writer.writerow(["epoch", "train_loss", "val_loss", "val_ser", "lr", "elapsed_s"])

    # ── Training loop ──────────────────────────────────────────────────────
    best_ckpt = run_dir / "best_model.pt"
    t0 = time.time()

    for epoch in range(start_epoch, cfg.epochs + 1):
        t_epoch = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimiser,
            scheduler,
            scaler,
            device,
            use_amp,
            max_grad_norm=cfg.max_grad_norm,
        )
        val_loss, val_ser = validate_ctc_epoch(
            model,
            val_loader,
            criterion,
            vocab,
            device,
            use_amp,
        )

        elapsed = time.time() - t0
        current_lr = optimiser.param_groups[0]["lr"]

        log.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_SER=%.4f  lr=%.2e  [%.0fs]",
            epoch,
            cfg.epochs,
            train_loss,
            val_loss,
            val_ser,
            current_lr,
            elapsed,
        )

        # Append to CSV log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_ser:.6f}",
                    f"{current_lr:.2e}",
                    f"{time.time() - t_epoch:.1f}",
                ]
            )

        # Checkpoint best model
        if val_ser < best_ser:
            best_ser = val_ser
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "val_ser": val_ser,
                    "val_loss": val_loss,
                    "patience_counter": 0,
                    "config": cfg,
                    "vocab_size": len(vocab),
                    "vocab_tokens": vocab._idx2tok[3:],
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
                "patience_counter": patience_counter,
                "config": cfg,
                "vocab_size": len(vocab),
                "vocab_tokens": vocab._idx2tok[3:],
            },
            run_dir / "latest_checkpoint.pt",
        )

        # Early stopping
        if cfg.early_stopping_patience > 0 and patience_counter >= cfg.early_stopping_patience:
            log.info(
                "Early stopping — val SER did not improve for %d epochs (best=%.4f)",
                cfg.early_stopping_patience,
                best_ser,
            )
            break

    total_time = time.time() - t0
    log.info(
        "Training complete. Best val SER=%.4f  Total time=%.0fs",
        best_ser,
        total_time,
    )
    return best_ckpt
