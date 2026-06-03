"""
evaluate_full.py
================
Run a comprehensive evaluation of a trained CRNN-CTC checkpoint and emit:

* aggregate SER (token edit-distance / total tokens)
* melodic SER (structural tokens stripped)
* per-sample SER distribution stats (mean, median, P90, P99, % perfect)
* per-token-category SER (pitch / octave / duration / accidental / tie / structural / clef / key / time)
* top-K substitutions / insertions / deletions
* clean-only vs scanned-only sub-split breakdown (when --both-splits)

Usage::

    poetry run python scripts/evaluate_full.py \\
        --checkpoint models/latest/best_model.pt \\
        --split test
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from CRNN_CTC.config import Config
from CRNN_CTC.dataset import collate_fn, make_splits
from CRNN_CTC.evaluate import (
    _edit_distance,
    beam_search_decode,
    greedy_decode,
    melodic_ser,
)
from CRNN_CTC.model import CRNN
from CRNN_CTC.vocab import Vocabulary


# ──────────────────────────────────────────────────────────────────────────
# Token categorisation
# ──────────────────────────────────────────────────────────────────────────

def categorise(tok: str) -> str:
    if tok.startswith("pitch:"):
        return "pitch"
    if tok.startswith("octave:"):
        return "octave"
    if tok in {"whole", "half", "quarter", "eighth",
               "16th", "32nd", "64th", "breve", "longa"}:
        return "duration"
    if tok == "dot":
        return "dot"
    if tok in {"flat", "sharp", "natural"}:
        return "accidental"
    if tok.startswith("tied:"):
        return "tie"
    if tok in {"measure"}:
        return "structural"
    if tok.startswith("clef:"):
        return "clef"
    if tok.startswith("key:"):
        return "key"
    if tok == "time" or tok.startswith("beats:") or tok.startswith("beat-type:"):
        return "time"
    if tok == "rest" or tok == "rest:measure":
        return "rest"
    if tok == "fermata":
        return "fermata"
    return "other"


# ──────────────────────────────────────────────────────────────────────────
# Detailed alignment with op tagging
# ──────────────────────────────────────────────────────────────────────────

def edit_ops(hyp: list[str], ref: list[str]) -> list[tuple[str, str, str]]:
    """Return list of (op, hyp_tok, ref_tok) where op ∈ {match, sub, ins, del}.

    For deletions hyp_tok='', for insertions ref_tok=''.
    """
    n, m = len(hyp), len(ref)
    mat = np.zeros((n + 1, m + 1), dtype=np.int32)
    mat[:, 0] = np.arange(n + 1)
    mat[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if hyp[i - 1] == ref[j - 1] else 1
            mat[i, j] = min(
                mat[i - 1, j] + 1,
                mat[i, j - 1] + 1,
                mat[i - 1, j - 1] + cost,
            )

    ops: list[tuple[str, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and hyp[i - 1] == ref[j - 1]:
            ops.append(("match", hyp[i - 1], ref[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and mat[i, j] == mat[i - 1, j - 1] + 1:
            ops.append(("sub", hyp[i - 1], ref[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and mat[i, j] == mat[i - 1, j] + 1:
            ops.append(("ins", hyp[i - 1], ""))
            i = i - 1
        else:
            ops.append(("del", "", ref[j - 1]))
            j = j - 1
    ops.reverse()
    return ops


# ──────────────────────────────────────────────────────────────────────────
# Main evaluation routine
# ──────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_eval(
    cfg: Config,
    checkpoint: Path,
    split: str,
    beam_width: int,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    vocab = Vocabulary.from_file(cfg.vocab_path)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    ckpt_vocab_size = ckpt.get("vocab_size") or len(vocab)
    if ckpt_vocab_size != len(vocab):
        raise SystemExit(
            f"Vocab mismatch: checkpoint={ckpt_vocab_size} vocab={len(vocab)}"
        )
    model = CRNN(
        vocab_size=ckpt_vocab_size,
        cnn_out_channels=cfg.cnn_out_channels,
        rnn_hidden=cfg.rnn_hidden,
        rnn_layers=cfg.rnn_layers,
        dropout=0.0,
        cnn_dropout=cfg.cnn_dropout,
        backbone=cfg.backbone,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(
        f"Loaded {checkpoint} (epoch={ckpt.get('epoch')} "
        f"val_ser={ckpt.get('val_ser'):.4f})"
    )

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
        extra_scanned_dirs=cfg.extra_scanned_dirs or None,
        online_aug_prob=0.0,
        rare_lmx_oversample=1,
    )
    ds = {"train": train_ds, "val": val_ds, "test": test_ds}[split]
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    print(f"Evaluating on '{split}' split ({len(ds)} samples) — beam={beam_width}")

    decode = (
        (lambda lp, ol, v: beam_search_decode(lp, ol, v, beam_width))
        if beam_width > 1
        else greedy_decode
    )

    total_ed = 0
    total_len = 0
    total_mel_ed = 0
    total_mel_len = 0

    sers: list[float] = []
    cat_ed: dict[str, int] = collections.Counter()
    cat_ref_count: dict[str, int] = collections.Counter()

    sub_pairs: collections.Counter = collections.Counter()
    ins_tokens: collections.Counter = collections.Counter()
    del_tokens: collections.Counter = collections.Counter()

    for batch in loader:
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        label_lens = batch["label_lens"].to(device)
        image_widths = batch["image_widths"].to(device)

        with autocast("cuda", enabled=use_amp):
            log_probs, output_lens = model(images, image_widths)

        preds = decode(log_probs, output_lens, vocab)

        offset = 0
        for i, length in enumerate(label_lens):
            l = length.item()
            ref = vocab.decode(labels[offset:offset + l].tolist())
            offset += l
            hyp = preds[i]

            ed = _edit_distance(hyp, ref)
            total_ed += ed
            total_len += len(ref)
            sers.append(ed / max(1, len(ref)))

            mel = melodic_ser(hyp, ref)
            total_mel_ed += int(round(mel * max(1, len([t for t in ref
                                                        if t not in {"measure", "tied:start", "tied:stop"}]))))
            total_mel_len += len([t for t in ref
                                  if t not in {"measure", "tied:start", "tied:stop"}])

            ops = edit_ops(hyp, ref)
            for op, h, r in ops:
                if op == "match":
                    cat_ref_count[categorise(r)] += 1
                elif op == "sub":
                    cat_ref_count[categorise(r)] += 1
                    cat_ed[categorise(r)] += 1
                    sub_pairs[(r, h)] += 1
                elif op == "del":
                    cat_ref_count[categorise(r)] += 1
                    cat_ed[categorise(r)] += 1
                    del_tokens[r] += 1
                elif op == "ins":
                    cat_ed[categorise(h)] += 1
                    ins_tokens[h] += 1

    sers_a = np.asarray(sers)
    return {
        "n": len(ds),
        "agg_ser": total_ed / max(1, total_len),
        "agg_mel_ser": total_mel_ed / max(1, total_mel_len),
        "p50": float(np.median(sers_a)),
        "p90": float(np.percentile(sers_a, 90)),
        "p99": float(np.percentile(sers_a, 99)),
        "perfect_pct": float((sers_a == 0).mean() * 100),
        "le05_pct": float((sers_a <= 0.05).mean() * 100),
        "cat_ed": dict(cat_ed),
        "cat_ref_count": dict(cat_ref_count),
        "sub_pairs": sub_pairs,
        "ins_tokens": ins_tokens,
        "del_tokens": del_tokens,
    }


# ──────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────

def _print_report(label: str, r: dict, top_k: int = 10) -> None:
    print()
    print("=" * 70)
    print(f"  {label}")
    print("=" * 70)
    print(f"  samples         : {r['n']}")
    print(f"  aggregate SER   : {r['agg_ser']:.4f}")
    print(f"  melodic   SER   : {r['agg_mel_ser']:.4f}")
    print(f"  median SER      : {r['p50']:.4f}")
    print(f"  P90 / P99 SER   : {r['p90']:.4f} / {r['p99']:.4f}")
    print(f"  perfect samples : {r['perfect_pct']:5.1f}%")
    print(f"  SER ≤ 0.05      : {r['le05_pct']:5.1f}%")

    print()
    print(f"  per-category errors / ref count / per-cat error rate:")
    cat_ed = r["cat_ed"]
    cat_ref = r["cat_ref_count"]
    rows = sorted(
        cat_ref.keys() | cat_ed.keys(),
        key=lambda c: cat_ed.get(c, 0),
        reverse=True,
    )
    for c in rows:
        ed = cat_ed.get(c, 0)
        ref = cat_ref.get(c, 0)
        rate = ed / max(1, ref)
        print(f"    {c:<12s} {ed:>7d} / {ref:>8d}    {rate * 100:>6.2f}%")

    print()
    print(f"  top-{top_k} substitutions (ref → pred):")
    for (ref_t, hyp_t), c in r["sub_pairs"].most_common(top_k):
        print(f"    {ref_t:>20s}  →  {hyp_t:<20s}  ×{c}")

    print()
    print(f"  top-{top_k} deleted (model missed):")
    for tok, c in r["del_tokens"].most_common(top_k):
        print(f"    {tok:>20s}  ×{c}")

    print()
    print(f"  top-{top_k} inserted (model hallucinated):")
    for tok, c in r["ins_tokens"].most_common(top_k):
        print(f"    {tok:>20s}  ×{c}")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--beam-width", type=int, default=1)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--both-splits", action="store_true",
                   help="Run separately on use_scanned=True and use_scanned=False")
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--scanned-dir", type=Path)
    p.add_argument("--vocab-path", type=Path)
    args = p.parse_args()

    cfg_kwargs: dict = {}
    if args.data_dir:
        cfg_kwargs["data_dir"] = args.data_dir
    if args.scanned_dir:
        cfg_kwargs["scanned_dir"] = args.scanned_dir
    if args.vocab_path:
        cfg_kwargs["vocab_path"] = args.vocab_path

    if args.both_splits:
        for use_scanned, label in [(True, "scanned (use_scanned=True)"),
                                   (False, "clean (use_scanned=False)")]:
            cfg = Config(use_scanned=use_scanned, **cfg_kwargs)
            r = run_eval(cfg, args.checkpoint, args.split, args.beam_width)
            _print_report(f"{args.split} — {label}", r, top_k=args.top_k)
    else:
        cfg = Config(**cfg_kwargs)
        r = run_eval(cfg, args.checkpoint, args.split, args.beam_width)
        _print_report(f"{args.split} (use_scanned={cfg.use_scanned})",
                      r, top_k=args.top_k)


if __name__ == "__main__":
    main()
