"""
Find test-set samples whose *melodic* content is wrong (not just barline/tie
shifts). The thesis "worst" list ranks by total edit distance, which is
dominated by structural (measure/tie) errors that render melodically perfect.
For a slide we want a prediction that is *visibly* wrong: wrong pitches and/or
durations.

Ranking key = melodic edit distance (edit distance after stripping
measure/tie tokens) with a tie-break preferring pitch/octave/duration
substitutions.

Run: poetry run python scripts/_find_melodic_worst.py \
        --checkpoint models/run_20260612_101637/best_model.pt --top 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from CRNN_CTC.config import Config
from CRNN_CTC.dataset import collate_fn, make_splits
from CRNN_CTC.evaluate import _edit_distance, greedy_decode
from CRNN_CTC.model import CRNN
from CRNN_CTC.vocab import Vocabulary

STRUCT = {"measure", "tied:start", "tied:stop"}
MELODIC_PREFIX = ("pitch:", "octave:")
DURATIONS = {"whole", "half", "quarter", "eighth", "16th", "32nd",
             "64th", "breve", "longa", "dot"}


def melodic_only(seq: list[str]) -> list[str]:
    return [t for t in seq if t not in STRUCT]


def count_note_subs(hyp: list[str], ref: list[str]) -> int:
    """Cheap proxy: how many pitch/duration tokens differ in the melodic-only
    aligned multiset. Used only as a tie-break / sanity signal."""
    from collections import Counter
    h = Counter(t for t in hyp if t.startswith(MELODIC_PREFIX) or t in DURATIONS)
    r = Counter(t for t in ref if t.startswith(MELODIC_PREFIX) or t in DURATIONS)
    diff = (h - r) + (r - h)
    return sum(diff.values())


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=REPO / "models/run_20260612_101637/best_model.pt")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--use-scanned", action="store_true", default=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(use_scanned=True)
    vocab = Vocabulary.from_file(cfg.vocab_path)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
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
    print(f"Loaded {args.checkpoint} (epoch={ckpt.get('epoch')})")

    _, _, test_ds = make_splits(
        data_dir=cfg.data_dir, vocab=vocab,
        img_height=cfg.img_height, max_image_width=cfg.max_image_width,
        scanned_dir=cfg.scanned_dir if cfg.use_scanned else None,
        val_frac=cfg.val_frac, test_frac=cfg.test_frac, seed=cfg.seed,
        filter_non_leadsheet_clef=cfg.filter_non_leadsheet_clef,
        filter_unusual_time=cfg.filter_unusual_time,
        filter_multi_staff=cfg.filter_multi_staff,
        max_source_height=cfg.max_source_height,
        extra_data_dirs=cfg.extra_data_dirs or None,
        extra_scanned_dirs=cfg.extra_scanned_dirs or None,
        online_aug_prob=0.0,
    )
    # Sample IDs in DataLoader (shuffle=False) order
    full = test_ds.dataset
    sids = [full._samples[i][0] for i in test_ds.indices]

    loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=collate_fn)

    rows = []  # (mel_ed, note_subs, agg_ed, ref_len, sid, ref, hyp)
    idx = 0
    use_amp = device.type == "cuda"
    for batch in loader:
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        label_lens = batch["label_lens"].to(device)
        widths = batch["image_widths"].to(device)
        with autocast("cuda", enabled=use_amp):
            log_probs, out_lens = model(images, widths)
        preds = greedy_decode(log_probs, out_lens, vocab)
        offset = 0
        for i, length in enumerate(label_lens):
            l = length.item()
            ref = vocab.decode(labels[offset:offset + l].tolist())
            offset += l
            hyp = preds[i]
            agg_ed = _edit_distance(hyp, ref)
            mel_ed = _edit_distance(melodic_only(hyp), melodic_only(ref))
            nsubs = count_note_subs(hyp, ref)
            rows.append((mel_ed, nsubs, agg_ed, len(ref), sids[idx], ref, hyp))
            idx += 1

    # Rank by melodic edit distance, then by note substitutions
    rows.sort(key=lambda r: (r[0], r[1]), reverse=True)

    print(f"\nTop {args.top} by MELODIC edit distance "
          f"(wrong notes, not barlines):\n")
    for mel_ed, nsubs, agg_ed, rlen, sid, ref, hyp in rows[:args.top]:
        mel_ser = mel_ed / max(1, len(melodic_only(ref)))
        print(f"  {sid:24s}  mel_ed={mel_ed:2d}  note_diff={nsubs:2d}  "
              f"agg_ed={agg_ed:2d}  ref_len={rlen:3d}  mel_SER={mel_ser:.3f}")

    # Dump the single best candidate's full token diff for inspection
    if rows:
        mel_ed, nsubs, agg_ed, rlen, sid, ref, hyp = rows[0]
        print(f"\n=== Best candidate: {sid} (mel_ed={mel_ed}) ===")
        print("REF:", " ".join(ref))
        print("HYP:", " ".join(hyp))


if __name__ == "__main__":
    main()
