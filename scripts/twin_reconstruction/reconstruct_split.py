"""Rebuild the June-1 train/val/test split: pool = filtered originals + filtered __nh twins.

Replicates the 7a5dc1c-era pipeline: _discover_samples ordering (sorted full-path
string — twins lived at clean/{sid}__nh/), token + height filters, randperm(seed 42),
and rare-token oversampling (factor 2 on tied:start/tied:stop) for the train size.

Fingerprint: the old best checkpoint recorded 185,327 optimizer steps over 59 epochs.
With batch 16 and AMP (GradScaler skips a few steps on overflow), the reconstructed
train virtual size L must satisfy 0 <= 59*ceil(L/16) - 185327 <= 200.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from CRNN_CTC.dataset import (  # noqa: E402
    _discover_samples,
    _image_source_height,
    _is_degenerate,
    _load_lmx_tokens,
)

CLEAN = _REPO / "data/processed/primus/clean"
TWINS = _REPO / "data/scratch/twin_recon/clean"
OUT = _REPO / "data/scratch/twin_recon/old_split.json"


def filtered(samples: list, label: str) -> list:
    kept = []
    for sid, png, lmx in samples:
        tokens = _load_lmx_tokens(lmx)
        if _is_degenerate(tokens, filter_non_leadsheet_clef=True, filter_unusual_time=True):
            continue
        if _image_source_height(png) > 180:
            continue
        kept.append((sid, png, lmx))
    print(f"{label}: {len(kept)} survive filtering (of {len(samples)})")
    return kept


def main() -> None:
    originals = filtered(_discover_samples(CLEAN, require_lmx=True), "originals")
    twins = filtered(_discover_samples(TWINS, require_lmx=True), "twins")
    assert len(originals) == 46089, f"originals drifted: {len(originals)}"

    # Old ordering: sorted(Path.rglob) over a single dir == full-path string sort.
    # Twins sat at clean/{sid}__nh/{sid}__nh.png — sort by that virtual path.
    def old_key(item) -> str:
        sid, png, _ = item
        if sid.endswith("__nh"):
            return str(CLEAN / sid / f"{sid}.png")
        return str(png)

    pool = sorted(originals + twins, key=old_key)
    n = len(pool)
    print("reconstructed old pool n =", n)

    rng = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=rng).tolist()
    n_test = max(1, int(n * 0.1))
    n_val = max(1, int(n * 0.1))
    n_train = n - n_val - n_test
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    rare = {"tied:start", "tied:stop"}
    L = 0
    for i in train_idx:
        L += 1
        if any(t in rare for t in _load_lmx_tokens(pool[i][2])):
            L += 1  # rare_lmx_oversample=2 -> one duplicate index
    steps = math.ceil(L / 16)
    skipped = 59 * steps - 185_327
    print(f"train virtual size L={L}  steps/epoch={steps}  implied AMP-skipped={skipped}")
    if not (0 <= skipped <= 200):
        raise SystemExit(
            "STEP-COUNT FINGERPRINT FAILED — reconstruction is NOT trustworthy. "
            "See plan Task 7 Step 4 contingency."
        )

    sid_of = lambda idxs: sorted(pool[i][0] for i in idxs)
    OUT.write_text(json.dumps({
        "n": n, "L": L, "steps_per_epoch": steps, "implied_skipped": skipped,
        "old_train": sid_of(train_idx),
        "old_val": sid_of(val_idx),
        "old_test": sid_of(test_idx),
    }))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
