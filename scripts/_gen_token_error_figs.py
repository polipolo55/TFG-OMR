"""
_gen_token_error_figs.py
========================
Regenerate two Ch. 6 error-analysis figures on the scanned test split,
reusing evaluate_full.run_eval so the numbers match the headline table.

  fig_error_pie.png        -> now a 100%-stacked horizontal bar of the
                              three Levenshtein operations (the former pie
                              chart; filename kept to avoid churn).
  fig_top_token_errors.png -> two legible panels (deleted / inserted
                              tokens).  Substitutions are sparse and are
                              tabulated separately (tab:top-subs), so they
                              are dropped from this figure.

Run:  poetry run python scripts/_gen_token_error_figs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib.pyplot as plt
import numpy as np

import style
from CRNN_CTC.config import Config
from evaluate_full import run_eval

FIG_DIR = REPO / "latex_documents/main/figures"
TOP_N = 15


def stacked_bar(tot_del: int, tot_ins: int, tot_sub: int) -> None:
    total = tot_del + tot_ins + tot_sub
    segs = [
        ("Deletions", tot_del, style.C["highlight"]),
        ("Insertions", tot_ins, style.C["secondary"]),
        ("Substitutions", tot_sub, style.C["primary"]),
    ]
    fig, ax = plt.subplots(figsize=(7, 1.9))
    left = 0.0
    for name, cnt, col in segs:
        pct = 100.0 * cnt / total
        ax.barh(0, pct, left=left, color=col, edgecolor="white", height=0.5)
        if pct >= 8:
            ax.text(left + pct / 2, 0, f"{name}\n{pct:.1f}% (n={cnt})",
                    ha="center", va="center", color="white",
                    fontsize=9.5, fontweight="bold")
        else:
            ax.annotate(f"{name}\n{pct:.1f}% (n={cnt})",
                        xy=(left + pct / 2, 0.25), xytext=(left + pct / 2, 0.78),
                        ha="center", va="bottom", fontsize=8.5,
                        color=style.C["neutral_dark"],
                        arrowprops=dict(arrowstyle="-",
                                        color=style.C["neutral_mid"], lw=0.8))
        left += pct
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 1.1)
    ax.set_yticks([])
    ax.set_xlabel("Share of total edit operations (%)")
    ax.set_title("Edit-operation distribution", fontsize=10, fontweight="bold")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "fig_error_pie.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}  (del={tot_del} ins={tot_ins} sub={tot_sub} total={total})")


def _barh(ax, counter, title, color):
    pairs = counter.most_common(TOP_N)
    labels = [lbl for lbl, _ in pairs]
    vals = [cnt for _, cnt in pairs]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=color, edgecolor="white", height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, family="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Count", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.margins(x=0.12)
    for yi, v in zip(y, vals):
        ax.text(v, yi, f" {v}", va="center", ha="left", fontsize=8,
                color=style.C["neutral_dark"])


def two_panel(del_counts, ins_counts) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * TOP_N + 1.2))
    _barh(axes[0], del_counts, f"Top-{TOP_N} deleted tokens (model missed)",
          style.C["highlight"])
    _barh(axes[1], ins_counts, f"Top-{TOP_N} inserted tokens (model hallucinated)",
          style.C["secondary"])
    fig.tight_layout(pad=1.6)
    out = FIG_DIR / "fig_top_token_errors.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


def main():
    style.apply()
    cfg = Config()  # scanned test split
    r = run_eval(cfg, REPO / "models/latest/best_model.pt", "test", 1)
    del_counts = r["del_tokens"]
    ins_counts = r["ins_tokens"]
    sub_pairs = r["sub_pairs"]
    tot_del = sum(del_counts.values())
    tot_ins = sum(ins_counts.values())
    tot_sub = sum(sub_pairs.values())
    stacked_bar(tot_del, tot_ins, tot_sub)
    two_panel(del_counts, ins_counts)


if __name__ == "__main__":
    main()
