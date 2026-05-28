"""Shared helpers for CRNN training scripts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .evaluate import compute_ser_batch, greedy_decode
from .model import CRNN
from .vocab import Vocabulary


def seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_run_dir(base_dir: Path) -> Path:
    """Create a timestamped run directory and update the ``latest`` symlink."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    update_latest_symlink(base_dir, run_dir)
    return run_dir


def update_latest_symlink(model_dir: Path, run_dir: Path) -> None:
    """Point ``model_dir/latest`` at *run_dir* (relative symlink name)."""
    link = model_dir / "latest"
    tmp_link = model_dir / f"_latest_tmp_{run_dir.name}"
    try:
        tmp_link.symlink_to(run_dir.name)
        tmp_link.rename(link)
    except OSError:
        link.unlink(missing_ok=True)
        try:
            link.symlink_to(run_dir.name)
        except OSError:
            pass


def train_one_epoch(
    model: CRNN,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    optimiser: AdamW,
    scheduler: OneCycleLR | None,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    max_grad_norm: float = 5.0,
) -> float:
    """Run one training epoch; return mean CTC loss."""
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
            loss = criterion(log_probs, labels, output_lens, label_lens)

        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        scale_before = scaler.get_scale()
        scaler.step(optimiser)
        scaler.update()
        if scheduler is not None and scaler.get_scale() >= scale_before:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1
        bar.set_postfix(
            loss=f"{total_loss / n_batches:.4f}",
            lr=f"{optimiser.param_groups[0]['lr']:.2e}",
        )

    return total_loss / max(n_batches, 1)


@torch.inference_mode()
def validate_ctc_epoch(
    model: CRNN,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    vocab: Vocabulary,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    """Run validation; return (mean CTC loss, token error rate)."""
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

        preds = greedy_decode(log_probs, output_lens, vocab)
        gt_tokens: list[list[str]] = []
        offset = 0
        for length in label_lens:
            label_len = length.item()
            gt_tokens.append(vocab.decode(labels[offset : offset + label_len].tolist()))
            offset += label_len

        edit, ref_len = compute_ser_batch(preds, gt_tokens)
        total_edit += edit
        total_len += ref_len

    avg_loss = total_loss / max(n_batches, 1)
    error_rate = total_edit / max(total_len, 1)
    return avg_loss, error_rate
