"""
Generate original, deck-cohesive figures for the TFG defense slides.

Outputs to latex_documents/defense/figures/:
  def_hook.png      — clean LilyJAZZ phrase (slide 2, hook)
  def_paper.png     — scan-degraded phrase, warm paper tint (slide 3)
  def_augstrip.png  — 1 clean → 3 scan-simulated variants, labelled (slide 7)
  def_errorbar.png  — 100% stacked horizontal bar of edit ops (slide 10)
  def_training.png  — training curves, deck palette, Catalan (slide 18)

Run: poetry run python scripts/_gen_defense_figures.py
"""
import sys
import random
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from CRNN_CTC.lilypond_render import render_tokens
from data_processing.augment_scanned import build_pipeline, augment_sample

OUT = REPO / "latex_documents/defense/figures"
OUT.mkdir(parents=True, exist_ok=True)
CLEAN = REPO / "data/processed/primus/clean"

# ── Deck palette ───────────────────────────────────────────────────────
INK = "#14213D"; ACCENT = "#1F6FEB"; ACCENT_L = "#9CC2F6"
HIGH = "#E0701F"; MUTED = "#6B7280"; PAPER = "#FCFCFE"

# Try to use Adwaita Sans in the charts so they match the slides.
for cand in ("Adwaita Sans", "DejaVu Sans"):
    if any(cand in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams.update({
    "font.size": 13, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

SAMPLES = {"hook": "000101115-1_1_1", "paper": "000100447-1_1_1",
           "aug": "000100061-1_1_1"}


def toks(sid):
    return (CLEAN / sid / f"{sid}.lmx").read_text().split()


def render_clean(sid, dpi=220):
    arr = render_tokens(toks(sid), name=sid, dpi=dpi)
    return None if arr is None else arr.astype("uint8")


def scan(sid, seed, dpi=220):
    """Render clean then push through the real scan-simulation pipeline."""
    arr = render_clean(sid, dpi=dpi)
    if arr is None:
        return None
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "c.png"; dst = Path(td) / "s.png"
        Image.fromarray(arr).save(src)
        augment_sample(src, dst, build_pipeline(seed=seed), random.Random(seed))
        return np.array(Image.open(dst).convert("L"))


# ── 1. hook (slide 2): clean phrase, blended onto the deck paper ───────
def gen_hook():
    arr = render_clean(SAMPLES["hook"])
    v = arr.astype(float) / 255.0
    paper = np.array([0xFC, 0xFC, 0xFE]) / 255.0   # matches \definecolor{paper}
    ink = np.array([0x14, 0x21, 0x3D]) / 255.0      # deck ink, softer than pure black
    rgb = v[..., None] * paper + (1 - v[..., None]) * ink
    img = (np.clip(rgb, 0, 1) * 255).astype("uint8")
    Image.fromarray(img).save(OUT / "def_hook.png")
    print("def_hook.png", img.shape)


# ── 2. paper (slide 3): scan-degraded, warm cream tint ─────────────────
def gen_paper():
    g = scan(SAMPLES["paper"], seed=7)
    # tint the grayscale onto a cream paper for a "scanned page" feel
    v = g.astype(float) / 255.0
    cream = np.array([0xF3, 0xEC, 0xDA]) / 255.0   # warm paper
    inkc = np.array([0x20, 0x1C, 0x14]) / 255.0    # warm near-black ink
    rgb = (v[..., None] * cream + (1 - v[..., None]) * inkc)
    img = (np.clip(rgb, 0, 1) * 255).astype("uint8")
    Image.fromarray(img).save(OUT / "def_paper.png")
    print("def_paper.png", img.shape)


# ── 3. augmentation strip (slide 7): 1 clean → 3 scanned ───────────────
def gen_augstrip():
    sid = SAMPLES["aug"]
    rows = [("Net (LilyJAZZ)", render_clean(sid), False)]
    for i, seed in enumerate((11, 23, 42), 1):
        rows.append((f"Escaneig simulat {i}", scan(sid, seed=seed), True))
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 0.95 * n))
    for ax, (label, im, scanned) in zip(axes, rows):
        ax.imshow(im, cmap="gray", aspect="auto", vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(HIGH if scanned else ACCENT)
            s.set_linewidth(1.4)
        ax.set_ylabel(label, rotation=0, ha="right", va="center",
                      fontsize=11, color=(HIGH if scanned else ACCENT),
                      labelpad=12, fontweight="bold")
    fig.subplots_adjust(left=0.20, right=0.99, top=0.99, bottom=0.01, hspace=0.18)
    fig.savefig(OUT / "def_augstrip.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("def_augstrip.png")


# ── 4. error bar (slide 10): 100% stacked horizontal ───────────────────
def gen_errorbar():
    segs = [("Supressions", 57.8, ACCENT, "white"),
            ("Insercions", 38.8, ACCENT_L, INK),
            ("Substitucions", 3.4, HIGH, "white")]
    fig, ax = plt.subplots(figsize=(8.4, 2.2))
    left = 0.0
    for name, pct, col, txt in segs:
        ax.barh(0, pct, left=left, color=col, height=0.6,
                edgecolor="white", linewidth=2)
        if pct > 6:
            ax.text(left + pct / 2, 0, f"{name}\n{pct:.1f} %",
                    ha="center", va="center", color=txt, fontsize=12,
                    fontweight="bold")
        else:
            ax.annotate(f"{name}  {pct:.1f} %", xy=(left + pct / 2, -0.30),
                        xytext=(100, -0.78), ha="right",
                        fontsize=10.5, color=HIGH, fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=HIGH, lw=1))
        left += pct
    ax.set_xlim(0, 100); ax.set_ylim(-1.0, 0.95)
    ax.axis("off")
    ax.text(0, 0.66, "3.506 edicions  ·  ≈90 % en barres de compàs i lligadures",
            fontsize=11, color=MUTED, ha="left", va="bottom")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)
    fig.savefig(OUT / "def_errorbar.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("def_errorbar.png")


# ── 5. training curves (slide 18): deck palette, Catalan ───────────────
def gen_training():
    import csv
    ep, tl, vl, vs = [], [], [], []
    with open(REPO / "models/run_20260612_101637/training_log.csv") as f:
        for r in csv.DictReader(f):
            ep.append(int(r["epoch"])); tl.append(float(r["train_loss"]))
            vl.append(float(r["val_loss"])); vs.append(float(r["val_ser"]) * 100)
    best = 83
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.4))
    a1.plot(ep, tl, color=ACCENT, lw=2.2, label="Entrenament")
    a1.plot(ep, vl, color=HIGH, lw=2.2, label="Validació")
    a1.set_xlabel("Època"); a1.set_ylabel("Pèrdua CTC")
    a1.legend(frameon=False); a1.grid(True, color="#E5E7EB", lw=0.8)
    a2.plot(ep, vs, color=ACCENT, lw=2.2)
    a2.axvline(best, color=MUTED, ls="--", lw=1)
    bi = ep.index(best)
    a2.scatter([best], [vs[bi]], color=HIGH, zorder=5, s=45)
    a2.annotate(f"època {best}\\,→\\,{vs[bi]:.2f} %".replace("\\,", ""),
                xy=(best, vs[bi]), xytext=(best - 5, vs[bi] + 2.2),
                ha="right", fontsize=10.5, color=INK)
    a2.set_xlabel("Època"); a2.set_ylabel("SER de validació (%)")
    a2.grid(True, color="#E5E7EB", lw=0.8)
    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.15, wspace=0.28)
    fig.savefig(OUT / "def_training.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("def_training.png")


if __name__ == "__main__":
    gen_hook()
    gen_paper()
    gen_augstrip()
    gen_errorbar()
    gen_training()
    # tidy previews
    for p in OUT.glob("_prev_*.png"):
        p.unlink()
    print("done")
