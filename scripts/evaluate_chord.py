"""
evaluate_chord.py
=================
Standalone evaluation of the chord-OCR CRNN for the thesis results chapter.

Reports, with no train-set leakage:

  1. Synthetic validation CER  — synthetic-pretrain checkpoint on
     ``data/chord_synth/val`` (the model never trains on val).
  2. Real zero-shot            — the same synthetic-pretrain checkpoint on
     ALL hand-labelled real strips (``status == done``); the pretrain model
     has never seen a real strip, so every one is held out.
  3. Real fine-tuned           — the fine-tuned checkpoint on its held-out
     real validation split (seed-42 90/10 split reproduced exactly), with the
     zero-shot model scored on the same split for a like-for-like comparison.

Plus a character-level confusion summary over the zero-shot real errors.

Run:  poetry run python scripts/evaluate_chord.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO / "src"))

from CRNN_CTC.chord_dataset import ChordDataset, collate_fn  # noqa: E402
from CRNN_CTC.chord_finetune import RealChordDataset  # noqa: E402
from CRNN_CTC.evaluate import compute_ser_batch, greedy_decode  # noqa: E402
from CRNN_CTC.model import CRNN  # noqa: E402
from CRNN_CTC.vocab import Vocabulary  # noqa: E402

PRETRAIN = REPO / "models/chord/run_20260601_111856/best_model.pt"
FINETUNE = REPO / "models/chord/latest/best_model.pt"
SYNTH = REPO / "data/chord_synth"
REAL_STRIPS = REPO / "data/chord_real/strips"
REAL_LABELS = REPO / "data/chord_real/labels.jsonl"
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ck.get("config", {})
    if not isinstance(cfg, dict):
        cfg = cfg.__dict__
    vocab = Vocabulary(list(ck["vocab_tokens"]))
    model = CRNN(
        vocab_size=len(vocab),
        cnn_out_channels=cfg.get("cnn_out_channels", 256),
        rnn_hidden=cfg.get("rnn_hidden", 192),
        rnn_layers=cfg.get("rnn_layers", 2),
        dropout=cfg.get("dropout", 0.2),
        cnn_dropout=cfg.get("cnn_dropout", 0.15),
        backbone=cfg.get("backbone", "resnet18"),
    ).to(DEVICE)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, vocab, cfg


@torch.no_grad()
def run(model, loader, vocab):
    """Return (cer, exact_match_frac, n, pairs) where pairs = [(pred_str, ref_str)]."""
    tot_edit = tot_len = 0
    exact = n = 0
    pairs: list[tuple[str, str]] = []
    for batch in loader:
        images = batch["images"].to(DEVICE)
        widths = batch["image_widths"].to(DEVICE)
        labels = batch["labels"]
        label_lens = batch["label_lens"]
        log_probs, out_lens = model(images, widths)
        preds = greedy_decode(log_probs, out_lens, vocab)
        gt: list[list[str]] = []
        off = 0
        for ln in label_lens:
            ln = ln.item()
            gt.append(vocab.decode(labels[off : off + ln].tolist()))
            off += ln
        e, l = compute_ser_batch(preds, gt)
        tot_edit += e
        tot_len += l
        for p, r in zip(preds, gt):
            ps, rs = "".join(p), "".join(r)
            pairs.append((ps, rs))
            exact += int(ps == rs)
            n += 1
    cer = tot_edit / max(tot_len, 1)
    return cer, exact / max(n, 1), n, pairs


def char_confusion(pairs):
    """Char-level substitution / insertion / deletion tally via edit traceback."""
    sub = Counter()
    ins = Counter()
    dele = Counter()
    for hyp, ref in pairs:
        a, b = ref, hyp  # ref -> hyp
        m, k = len(a), len(b)
        dp = [[0] * (k + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(k + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, k + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        i, j = m, k
        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
                if a[i - 1] != b[j - 1]:
                    sub[(a[i - 1], b[j - 1])] += 1
                i, j = i - 1, j - 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                dele[a[i - 1]] += 1
                i -= 1
            else:
                ins[b[j - 1]] += 1
                j -= 1
    return sub, ins, dele


def loader_for(ds, bs=32):
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, collate_fn=collate_fn)


def main():
    out = {}
    pre_model, pre_vocab, pre_cfg = load_model(PRETRAIN)
    ft_model, ft_vocab, ft_cfg = load_model(FINETUNE)
    ih = pre_cfg.get("img_height", 64)

    # 1. Synthetic val (pretrain)
    synth_val = ChordDataset(
        image_dir=SYNTH / "val",
        labels_csv=SYNTH / "val_labels.csv",
        vocab=pre_vocab,
        img_height=ih,
        augment=False,
    )
    cer, em, n, _ = run(pre_model, loader_for(synth_val), pre_vocab)
    out["synth_val"] = {"cer": cer, "exact_match": em, "n": n}
    print(f"[1] Synthetic val   (pretrain)  n={n:4d}  CER={cer:.4%}  exact={em:.2%}")

    # 2. Real zero-shot, ALL done strips (pretrain never saw real)
    real_all = RealChordDataset(REAL_STRIPS, REAL_LABELS, pre_vocab, img_height=ih, augment=False)
    cer, em, n, pairs = run(pre_model, loader_for(real_all), pre_vocab)
    out["real_zeroshot_all"] = {"cer": cer, "exact_match": em, "n": n}
    print(f"[2] Real zero-shot  (all done)  n={n:4d}  CER={cer:.4%}  exact={em:.2%}")

    sub, ins, dele = char_confusion(pairs)
    out["confusion"] = {
        "substitutions": [[f"{r}->{h}", c] for (r, h), c in sub.most_common(10)],
        "insertions": [[ch, c] for ch, c in ins.most_common(8)],
        "deletions": [[ch, c] for ch, c in dele.most_common(8)],
    }
    print("    top substitutions (ref->hyp):", out["confusion"]["substitutions"][:6])
    print("    top insertions:", out["confusion"]["insertions"][:6])
    print("    top deletions :", out["confusion"]["deletions"][:6])

    # 3. Held-out real val split (reproduce finetune's seed-42 90/10 split)
    n_real = len(real_all)
    n_val = max(10, n_real // 10)
    perm = torch.randperm(n_real, generator=torch.Generator().manual_seed(SEED)).tolist()
    val_idx = perm[:n_val]
    real_val = Subset(real_all, val_idx)
    cer_zs, em_zs, n_hv, _ = run(pre_model, loader_for(real_val), pre_vocab)
    cer_ft, em_ft, _, ft_pairs = run(ft_model, loader_for(real_val), ft_vocab)
    out["real_heldout_val"] = {
        "n": n_hv,
        "zeroshot": {"cer": cer_zs, "exact_match": em_zs},
        "finetuned": {"cer": cer_ft, "exact_match": em_ft},
    }
    print(f"[3] Real held-out val  n={n_hv}")
    print(f"      zero-shot : CER={cer_zs:.4%}  exact={em_zs:.2%}")
    print(f"      fine-tuned: CER={cer_ft:.4%}  exact={em_ft:.2%}")

    # Residual confusions of the DEPLOYED (fine-tuned) model on held-out val.
    fsub, fins, fdel = char_confusion(ft_pairs)
    out["finetuned_residual_confusion"] = {
        "substitutions": [[f"{r}->{h}", c] for (r, h), c in fsub.most_common(10)],
        "insertions": [[ch, c] for ch, c in fins.most_common(8)],
        "deletions": [[ch, c] for ch, c in fdel.most_common(8)],
        "wrong_pairs": [[p, r] for p, r in ft_pairs if p != r],
    }
    print("    fine-tuned residual subs:", out["finetuned_residual_confusion"]["substitutions"])
    print("    fine-tuned residual ins :", out["finetuned_residual_confusion"]["insertions"])
    print("    fine-tuned residual del :", out["finetuned_residual_confusion"]["deletions"])
    print("    fine-tuned wrong strips :")
    for p, r in out["finetuned_residual_confusion"]["wrong_pairs"]:
        print(f"        pred={p!r:30s} ref={r!r}")

    out["meta"] = {
        "pretrain_ckpt": str(PRETRAIN.relative_to(REPO)),
        "finetune_ckpt": str(FINETUNE.resolve().relative_to(REPO)),
        "vocab_size": len(pre_vocab),
        "device": str(DEVICE),
        "seed": SEED,
    }
    res_path = REPO / "scripts/_chord_eval_results.json"
    res_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {res_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
