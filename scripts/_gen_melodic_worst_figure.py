"""
Generate a *melodically* wrong worst-case figure for the defense slides.

Sample 220014195-1_1_1 has melodic edit distance 9 (wrong pitches in the final
measure), unlike the edit-distance "worst" cases which are barline shifts that
render melodically perfect.

Produces, in latex_documents/main/figures/worst/:
  melodic_worst_input.png  — scanned input strip
  melodic_worst_gt.png     — LilyPond render of the ground-truth LMX
  melodic_worst_pred.png   — LilyPond render of the model prediction

Run: poetry run python scripts/_gen_melodic_worst_figure.py
"""
import sys
import shutil
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from CRNN_CTC.config import Config
from CRNN_CTC.vocab import Vocabulary
from CRNN_CTC.model import CRNN
from CRNN_CTC.evaluate import greedy_decode
from CRNN_CTC.lilypond_render import render_tokens
from omr_pipeline.grammar_fix import fix_sequence

SID = "220014195-1_1_1"
CKPT = REPO / "models/run_20260612_101637/best_model.pt"
SCAN = REPO / "data/processed/primus/scanned" / SID / f"{SID}.png"
GTLMX = REPO / "data/processed/primus/clean" / SID / f"{SID}.lmx"
OUT = REPO / "latex_documents/main/figures/worst"
OUT.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = Config(use_scanned=True)
vocab = Vocabulary.from_file(cfg.vocab_path)
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
model = CRNN(
    vocab_size=ckpt.get("vocab_size") or len(vocab),
    cnn_out_channels=cfg.cnn_out_channels, rnn_hidden=cfg.rnn_hidden,
    rnn_layers=cfg.rnn_layers, dropout=0.0, cnn_dropout=cfg.cnn_dropout,
    backbone=cfg.backbone,
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()


def load_strip(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    new_h = cfg.img_height
    new_w = min(int(w * new_h / h), cfg.max_image_width)
    img = cv2.resize(img, (new_w, new_h)).astype("float32") / 255.0
    return (img - img.mean()) / (img.std() + 1e-6)


def save_png(arr: np.ndarray, path: Path, dpi: int = 150):
    h, w = arr.shape
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(arr, cmap="gray", aspect="auto", vmin=0, vmax=255)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# --- input ---
shutil.copy2(SCAN, OUT / "melodic_worst_input.png")
print("input  →", (OUT / "melodic_worst_input.png").name)

# --- prediction ---
img = load_strip(SCAN)
t = torch.tensor(img[None, None]).to(device)
w = torch.tensor([img.shape[1]], dtype=torch.long).to(device)
with torch.inference_mode():
    log_probs, out_lens = model(t, w)
pred_toks = greedy_decode(log_probs, out_lens, vocab)[0]
fixed, _, _ = fix_sequence(" ".join(pred_toks), global_key=None,
                           global_time=None, force_clef=True)
fixed_toks = fixed.split() if isinstance(fixed, str) else list(fixed)
pred_render = render_tokens(fixed_toks, name="melodic_worst_pred")
if pred_render is not None:
    save_png(pred_render, OUT / "melodic_worst_pred.png")
    print("pred   →", (OUT / "melodic_worst_pred.png").name)
else:
    print("PRED render FAILED")

# --- ground truth ---
gt_toks = GTLMX.read_text().split()
gt_fixed, _, _ = fix_sequence(" ".join(gt_toks), global_key=None,
                              global_time=None, force_clef=True)
gt_fixed_toks = gt_fixed.split() if isinstance(gt_fixed, str) else list(gt_fixed)
gt_render = render_tokens(gt_fixed_toks, name="melodic_worst_gt")
if gt_render is not None:
    save_png(gt_render, OUT / "melodic_worst_gt.png")
    print("gt     →", (OUT / "melodic_worst_gt.png").name)
else:
    print("GT render FAILED")

print("\nGT  :", " ".join(gt_toks[-22:]))
print("PRED:", " ".join(fixed_toks[-22:]))
