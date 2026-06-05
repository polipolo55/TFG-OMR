"""
Generate worst-case prediction figures for the thesis appendix.
For each of the top-5 worst samples:
  - Copy the scanned input PNG → figures/worst/worst_N_input.png
  - Run the CRNN, grammar-fix, render via LilyPond → figures/worst/worst_N_pred.png

Run: poetry run python scripts/_gen_worst_figures.py
"""
import sys, shutil
from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))

from CRNN_CTC.vocab import Vocabulary
from CRNN_CTC.model import CRNN
from CRNN_CTC.evaluate import greedy_decode
from CRNN_CTC.lilypond_render import render_tokens
from omr_pipeline.grammar_fix import fix_sequence
import cv2

REPO  = Path(__file__).parents[1]
CKPT  = REPO / "models/latest/best_model.pt"
SCAN  = REPO / "data/processed/primus/scanned"
OUT   = REPO / "latex_documents/main/figures/worst"
OUT.mkdir(parents=True, exist_ok=True)

# Top-5 worst: (sample_id, edit_distance, SER)
WORST = [
    ("210017589-1_51_1", 11, 0.1122),
    ("220034617-1_2_1",   9, 0.1268),
    ("220014195-1_1_1",   9, 0.0891),
    ("220011439-1_1_1",   9, 0.1667),
    ("000122226-1_1_1",   8, 0.1356),
]

# Load model
ckpt   = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg    = ckpt["config"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab  = Vocabulary.from_file(REPO / cfg.vocab_path)
model  = CRNN(vocab_size=len(vocab), cnn_out_channels=cfg.cnn_out_channels,
              rnn_hidden=cfg.rnn_hidden, backbone=cfg.backbone)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()

def load_strip(png_path: Path) -> np.ndarray:
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    new_h = cfg.img_height
    new_w = int(w * new_h / h)
    new_w = min(new_w, cfg.max_image_width)
    img = cv2.resize(img, (new_w, new_h))
    img = img.astype("float32") / 255.0
    img = (img - img.mean()) / (img.std() + 1e-6)
    return img

def predict(img: np.ndarray):
    t = torch.tensor(img[None, None]).to(device)
    w = torch.tensor([img.shape[1]], dtype=torch.long).to(device)
    with torch.inference_mode():
        log_probs, out_lens = model(t, w)
    preds = greedy_decode(log_probs, out_lens, vocab)
    return preds[0]

def save_img_as_png(arr: np.ndarray, path: Path, dpi: int = 150):
    h, w = arr.shape
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(arr, cmap="gray", aspect="auto", vmin=0, vmax=255)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

for rank, (sid, ed, ser) in enumerate(WORST, start=1):
    print(f"\nRank {rank}: {sid}  (ed={ed}, SER={ser:.4f})")

    # --- input ---
    src_png = SCAN / sid / f"{sid}.png"
    dst_input = OUT / f"worst_{rank}_input.png"
    shutil.copy2(src_png, dst_input)
    print(f"  input  → {dst_input.name}")

    # --- prediction render ---
    img   = load_strip(src_png)
    toks  = predict(img)
    fixed, _, _ = fix_sequence(
        " ".join(toks), global_key=None, global_time=None, force_clef=True
    )
    fixed_toks = fixed.split() if isinstance(fixed, str) else list(fixed)

    dst_pred = OUT / f"worst_{rank}_pred.png"
    rendered = render_tokens(fixed_toks, name=f"worst_{rank}")
    if rendered is not None:
        # rendered is an H×W uint8 array
        save_img_as_png(rendered, dst_pred, dpi=150)
        print(f"  render → {dst_pred.name}")
    else:
        print(f"  LilyPond render FAILED for rank {rank}")
        # Copy a blank placeholder so the build doesn't break
        blank = np.ones((64, 512), dtype=np.uint8) * 255
        save_img_as_png(blank, dst_pred, dpi=150)

    print(f"  tokens: {' '.join(fixed_toks[:12])} …")

print("\nDone.")
