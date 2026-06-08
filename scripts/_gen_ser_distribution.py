"""
_gen_ser_distribution.py
========================
Regenerate latex_documents/main/figures/fig_ser_distribution.png.

The previous version used a linear count axis: with ~94% of samples at
exactly SER=0, the zero bin is a ~4300-tall bar and the entire error
tail (counts of 1-40) is crushed flat, so the plot looked empty past
SER 0.02.  This version puts the count axis on a log scale so both the
zero spike and the thin right tail are legible, which is the honest
shape of a bimodal "mostly-perfect + sparse tail" distribution.

Computes the per-sample SER on the scanned test split (matching the
figure's label and the 0.23% aggregate) and redraws with the project
palette.

Run:  poetry run python scripts/_gen_ser_distribution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

import style
from CRNN_CTC.config import Config
from CRNN_CTC.dataset import collate_fn, make_splits
from CRNN_CTC.evaluate import _edit_distance, greedy_decode
from CRNN_CTC.model import CRNN
from CRNN_CTC.vocab import Vocabulary

FIG_DIR = REPO / "latex_documents/main/figures"


@torch.inference_mode()
def compute_sers(cfg: Config, checkpoint: Path) -> tuple[np.ndarray, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    vocab = Vocabulary.from_file(REPO / cfg.vocab_path)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = CRNN(
        vocab_size=ckpt.get("vocab_size") or len(vocab),
        cnn_out_channels=cfg.cnn_out_channels,
        rnn_hidden=cfg.rnn_hidden,
        rnn_layers=cfg.rnn_layers,
        dropout=0.0,
        cnn_dropout=cfg.cnn_dropout,
        backbone=cfg.backbone,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, _, test_ds = make_splits(
        data_dir=REPO / cfg.data_dir,
        vocab=vocab,
        img_height=cfg.img_height,
        max_image_width=cfg.max_image_width,
        scanned_dir=(REPO / cfg.scanned_dir) if cfg.use_scanned else None,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        seed=cfg.seed,
        filter_non_leadsheet_clef=cfg.filter_non_leadsheet_clef,
        filter_unusual_time=cfg.filter_unusual_time,
        filter_multi_staff=cfg.filter_multi_staff,
        max_source_height=cfg.max_source_height,
        online_aug_prob=0.0,
    )
    loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True)
    print(f"scanned test split: {len(test_ds)} samples")

    sers, tot_ed, tot_len = [], 0, 0
    for batch in loader:
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        label_lens = batch["label_lens"].to(device)
        widths = batch["image_widths"].to(device)
        with autocast("cuda", enabled=use_amp):
            log_probs, out_lens = model(images, widths)
        preds = greedy_decode(log_probs, out_lens, vocab)
        off = 0
        for i, ln in enumerate(label_lens):
            ln = ln.item()
            ref = vocab.decode(labels[off:off + ln].tolist())
            off += ln
            ed = _edit_distance(preds[i], ref)
            tot_ed += ed
            tot_len += len(ref)
            sers.append(ed / max(1, len(ref)))
    return np.asarray(sers), tot_ed / max(1, tot_len)


def plot(sers: np.ndarray, agg: float) -> None:
    style.apply()
    max_ser = float(sers.max()) if sers.max() > 0 else 1.0
    bins = np.linspace(0, max_ser, 49)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(sers, bins=bins, color=style.C["primary"], edgecolor="white",
            linewidth=0.4, alpha=0.85, label="Histogram", zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(0.7, len(sers) * 1.6)
    ax.set_xlim(0, max_ser)
    ax.set_xlabel("Per-sample SER")
    ax.set_ylabel("Count (log scale)")

    ax2 = ax.twinx()
    s = np.sort(sers)
    cdf = np.arange(1, len(s) + 1) / len(s)
    s = np.append(s, max_ser)
    cdf = np.append(cdf, cdf[-1])
    ax2.plot(s, cdf, color=style.C["secondary"], lw=1.6, label="CDF", zorder=3)
    ax2.set_ylabel("Cumulative fraction")
    ax2.set_ylim(0, 1.02)
    ax2.spines["right"].set_visible(True)

    perfect = (sers == 0).mean() * 100
    ax.set_title(f"SER distribution, scanned test split "
                 f"(aggregate SER = {agg:.4f})")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=7)

    fig.tight_layout()
    out = FIG_DIR / "fig_ser_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    print(f"agg={agg:.4f}  perfect={perfect:.1f}%  max_ser={max_ser:.4f}  "
          f"n_nonzero={(sers > 0).sum()}")


def main():
    cfg = Config()  # default: use_scanned=True
    sers, agg = compute_sers(cfg, REPO / "models/latest/best_model.pt")
    plot(sers, agg)


if __name__ == "__main__":
    main()
