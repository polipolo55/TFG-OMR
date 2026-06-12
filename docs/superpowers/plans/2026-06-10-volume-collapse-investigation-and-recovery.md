# Training Volume Collapse — Verified Findings & Recovery Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the *true* performance gap between the June-1 and June-8 CRNN checkpoints on uncontaminated data, then retrain with leakage-free diversity restored (multi-variant scan augmentation + rare-token oversampling) to get the best honest results.

**Architecture:** Phase 1 reconstructs the deleted `__nh` twin pool deterministically (hash-based membership, seeded DPI) to rebuild the June-1 train/val/test split, validates it against two independent fingerprints (optimizer step count, logged val SER), then evaluates both checkpoints on samples *neither* model trained on. Phase 2 makes no-regret training improvements and retrains. Phase 3 discharges post-retrain obligations (reject-gate recalibration, docs).

**Tech Stack:** PyTorch CRNN-CTC, LilyPond rendering, albumentations offline augmentation, Poetry. GPU: RTX 3060 12 GB.

---

## Investigation Findings (verified 2026-06-10)

These facts were established by direct measurement before this plan was written. They are the ground truth the plan builds on — do not re-litigate them, but do re-verify any number a task depends on.

### The original claim is mischaracterized in four ways

1. **The volume drop was ~27%, not ~50%.** From optimizer step counts inside the checkpoints
   (`best_model.pt` → `optimiser_state_dict.state[0].step`):
   - old run `run_20260601_134845`: 185,327 steps / 59 epochs ≈ 3,141 steps/epoch × batch 16 ≈ **50,260 train samples/epoch**
   - new run `run_20260608_102846`: 108,250 steps / 47 epochs ≈ 2,303 steps/epoch × batch 16 ≈ **36,850 train samples/epoch**
2. **The twins numbered 26,168, not 87,678** — `generate_headerless_twins.py` (deleted, recoverable at git `7a5dc1c`) twinned a deterministic 35% of treble samples (`md5(f"{seed}:{sid}")`, seed 42). 0.35 × 74,765 treble = 26,168, exactly matching the count recorded in `docs/superpowers/plans/2026-06-03-virtual-header-injection.md`. Only **~11,000 survived** the domain filters (arithmetic: 1.10 × 0.8 × (46,089 + T) = 50,260 → T ≈ 11,020).
3. **Scanned images never doubled anything.** `OMRDataset.__getitem__` ([dataset.py:374-385](src/CRNN_CTC/dataset.py#L374-L385)) *swaps* the clean PNG for the scanned one — "~350k training images" never existed.
4. **Neither run stopped early.** Both ran the full 60-epoch schedule (see `training_log.csv` in each run dir). "Epoch 47" vs "epoch 59" are the *best-checkpoint* epochs. The old run was still improving at epoch 59 (mildly undertrained); the new run plateaued at 47.

### The old metrics are inflated — the comparison is invalid

The June-1 `make_splits` (git `7a5dc1c`) put the `__nh` twins **into the split pool**: `torch.randperm` over originals+twins. Consequences:

- **Near-duplicate leakage:** a twin in val/test has ~80% probability that its parent (identical music, identical body glyphs, header removed) was in train — and vice versa. With ~11k twins in a ~57k pool, ~19% of the old val/test sets were quasi-leaked. The old model memorizes hard (train loss 0.0018), so those samples score near-perfect.
- **Different populations:** old test = randperm(≈57,109), new test = randperm(46,089) — disjoint memberships, different content mix (twins have shorter, header-less labels).

Therefore **0.98%/0.23%/94.1% (old) vs 1.28%/1.31%/72% (new) are measurements on different, differently-contaminated test sets**. The true regression is unknown until Phase 1 measures it; it is likely much smaller than it appears, possibly zero.

### What is genuinely confounded

Two changes landed between the runs (twins deleted June 3 in `75792a3`; oversampling removed June 8 09:56 in `2e4091f`, 32 min before the new run started). Verified: tie-containing samples are exactly **10.0%** (4,626 / 46,089) of the retained corpus, and per `docs/overview.md` ties account for ~13% of remaining edit errors.

### A real bug found along the way

`_AugSubset` ([dataset.py:469-499](src/CRNN_CTC/dataset.py#L469-L499)) permanently sets `_online_aug_prob` on the **shared** `OMRDataset` instance that the val/test `Subset`s also wrap. During training, in-loop validation images are therefore jittered ~50% of the time → in-loop val SER/loss are slightly inflated and best-checkpoint selection is noisy. (Offline `scripts/evaluate_full.py` is unaffected — it passes `online_aug_prob=0.0`.) Fixed in Task 9.

### Current dataset facts (re-verified)

- 87,678 sample dirs in `data/processed/primus/clean` (each has `.png .lmx .ly .semantic .agnostic .mid`) and 87,678 in `scanned/`.
- Filters: token filter removes 18,532; height filter (>180 px) removes 23,056 → **46,089 retained**; split = 36,873 train / 4,608 val / 4,608 test.
- `models/latest/` is a copy of the new run (best epoch 47).

---

## File Structure

```
scripts/twin_reconstruction/            # Phase 1 — throwaway forensic tooling (committed for thesis reproducibility)
├── legacy_generate_headerless_twins.py # recovered verbatim from git 7a5dc1c, import paths patched
├── legacy_generate_realbook.py         # recovered verbatim from git 7a5dc1c (provides omit_header_in_ly etc.)
├── render_twins.py                     # re-render twins into data/scratch (never touches live data dirs)
├── reconstruct_split.py                # rebuild June-1 split + step-count fingerprint
├── build_subsets.py                    # new-split ids, comparison subset S
└── eval_subset.py                      # evaluate a checkpoint on an explicit id list

data/scratch/twin_recon/                # NOT committed (gitignored) — clean/, scanned/, *.json manifests

src/CRNN_CTC/dataset.py                 # Task 9 (jitter fix), 10 (oversample restore), 13 (variant sampling)
src/CRNN_CTC/config.py                  # Tasks 10, 11, 13 (fields + ensure_config_defaults)
src/CRNN_CTC/train.py                   # Tasks 10, 13 (pass new cfg fields to make_splits)
src/cli.py                              # Task 14 (--scanned-variant-dirs)
tests/test_dataset_aug.py               # Tasks 9, 10, 13 (new test module)
docs/experiments/2026-06-10-volume-collapse-findings.md   # Task 8 (verdict write-up)
```

---

# Phase 1 — Measure the true regression (mostly CPU; one short GPU eval)

### Task 1: Triage — evaluate the old checkpoint on the current test split

Cheap sanity anchor before the heavy reconstruction. ~80% of the current test set was in the old model's training set, so this eval is biased *in the old model's favor* — it bounds the gap from above.

**Files:** none (read-only run)

- [ ] **Step 1: Run the old checkpoint through the standard evaluator**

```bash
poetry run python scripts/evaluate_full.py \
  --checkpoint models/run_20260601_134845/best_model.pt \
  --split test --beam-width 1
```

Expected: loads cleanly (vocab unchanged between runs — both checkpoints store `vocab_size`; a mismatch raises `SystemExit`). Record `aggregate SER` and `perfect samples`.

- [ ] **Step 2: Run the new checkpoint identically (current reference numbers)**

```bash
poetry run python scripts/evaluate_full.py \
  --checkpoint models/run_20260608_102846/best_model.pt \
  --split test --beam-width 1
```

Expected: ≈1.31% aggregate SER, ≈72% perfect (matches `docs/overview.md`).

- [ ] **Step 3: Record both result blocks** in a scratch note for Task 8. Interpretation rule: if old ≤ new here *despite* its leakage advantage, the regression is already disproven and Tasks 2–7 become optional (skip to Task 8, then Phase 2). Otherwise (expected) proceed.

### Task 2: Recover the legacy twin tooling from git

**Files:**
- Create: `scripts/twin_reconstruction/legacy_generate_headerless_twins.py`
- Create: `scripts/twin_reconstruction/legacy_generate_realbook.py`

- [ ] **Step 1: Extract the deleted modules verbatim**

```bash
mkdir -p scripts/twin_reconstruction data/scratch
git show 7a5dc1c:src/data_processing/generate_headerless_twins.py \
  > scripts/twin_reconstruction/legacy_generate_headerless_twins.py
git show 7a5dc1c:src/data_processing/generate_realbook.py \
  > scripts/twin_reconstruction/legacy_generate_realbook.py
```

- [ ] **Step 2: Patch import paths in `legacy_generate_headerless_twins.py`**

The file computes `_SRC = Path(__file__).resolve().parent.parent` (which now resolves to `scripts/`, not the repo) and imports the deleted `data_processing.generate_realbook`. Apply both edits:

Replace:
```python
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
```
with:
```python
_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))
```

Replace:
```python
from data_processing.generate_realbook import (
    headerless_label_tokens,
    omit_header_in_ly,
)
```
with:
```python
from legacy_generate_realbook import (
    headerless_label_tokens,
    omit_header_in_ly,
)
```

- [ ] **Step 3: Apply the same `_SRC` patch to `legacy_generate_realbook.py`** if it contains a similar `sys.path` block pointing at `parent.parent` (inspect the file; it must be able to import `CRNN_CTC.lilypond_render` from the live `src/`).

- [ ] **Step 4: Verify both legacy modules import**

```bash
cd scripts/twin_reconstruction && poetry run python -c "
import legacy_generate_headerless_twins as t
import legacy_generate_realbook as r
print('imports ok:', t._TWIN_SUFFIX, callable(r.omit_header_in_ly))
" && cd ../..
```

Expected: `imports ok: __nh True`

- [ ] **Step 5: Verify deterministic membership reproduces exactly 26,168 twins**

```bash
cd scripts/twin_reconstruction && poetry run python -c "
from pathlib import Path
from legacy_generate_headerless_twins import _in_fraction
root = Path('../../data/processed/primus/clean')
n = 0
for d in sorted(root.iterdir()):
    if not d.is_dir():
        continue
    lmx = d / (d.name + '.lmx')
    if not lmx.exists() or not _in_fraction(d.name, 42, 0.35):
        continue
    if 'clef:G2' in lmx.read_text(encoding='utf-8').split():
        n += 1
print('expected twin count:', n)
" && cd ../..
```

Expected: `expected twin count: 26168`. **If this does not match exactly, STOP** — the June-1 generation used different (seed, fraction) parameters and the reconstruction premise is wrong; fall back to the contingency in Task 7 Step 4.

- [ ] **Step 6: Gitignore the scratch dir and commit the tooling**

```bash
echo "data/scratch/" >> .gitignore
git add scripts/twin_reconstruction/ .gitignore
git commit -m "forensics: recover legacy twin generator from 7a5dc1c for split reconstruction"
```

### Task 3: Re-render the twin images into scratch

**Files:**
- Create: `scripts/twin_reconstruction/render_twins.py`

- [ ] **Step 1: Write the render driver.** It replicates `process_one` from the legacy script exactly (same membership test, same per-sid DPI from `(180, 200, 220)`, same `omit_header_in_ly` → `run_lilypond` → `crop_content` chain) but writes twin dirs under `--output` instead of next to the originals, so the live dataset is never touched.

```python
"""Re-render the deleted __nh twin samples into a scratch dir (forensic reconstruction).

Replicates generate_headerless_twins.process_one (git 7a5dc1c) byte-for-byte in
behavior, except twins are written under --output instead of into the live
data/processed/primus/clean tree. Resumable: existing twin PNGs are skipped.
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing
import sys
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO / "src"))

from legacy_generate_headerless_twins import _dpi_for, _in_fraction  # noqa: E402
from legacy_generate_realbook import headerless_label_tokens, omit_header_in_ly  # noqa: E402
from CRNN_CTC.lilypond_render import crop_content, run_lilypond  # noqa: E402

SEED = 42
FRACTION = 0.35
DPI_CHOICES = (180, 200, 220)


def process_one(sample_dir: Path, out_root: Path) -> str:
    sid = sample_dir.name
    if sid.endswith("__nh") or not _in_fraction(sid, SEED, FRACTION):
        return "skip-fraction"
    ly_path = sample_dir / f"{sid}.ly"
    lmx_path = sample_dir / f"{sid}.lmx"
    if not ly_path.exists() or not lmx_path.exists():
        return "fail"
    tokens = lmx_path.read_text(encoding="utf-8").split()
    if "clef:G2" not in tokens:
        return "skip-nontreble"

    twin_id = f"{sid}__nh"
    twin_dir = out_root / twin_id
    twin_png = twin_dir / f"{twin_id}.png"
    if twin_png.exists():
        return "ok"  # resumable re-run

    ly = ly_path.read_text(encoding="utf-8")
    twin_ly = omit_header_in_ly(ly)
    if twin_ly == ly:
        return "fail"
    dpi = _dpi_for(sid, DPI_CHOICES)
    with tempfile.TemporaryDirectory(prefix="twin_") as tmp:
        png = run_lilypond(twin_ly, twin_id, Path(tmp), dpi=dpi, timeout=30)
        if png is None:
            return "fail"
        try:
            cropped = crop_content(np.array(Image.open(png).convert("L")))
        except Exception:
            return "fail"
        if cropped.size == 0 or np.all(cropped == 255):
            return "fail"
        twin_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(cropped).save(twin_png)
    (twin_dir / f"{twin_id}.lmx").write_text(
        " ".join(headerless_label_tokens(tokens)), encoding="utf-8"
    )
    return "ok"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=_REPO / "data/processed/primus/clean")
    p.add_argument("--output", type=Path, default=_REPO / "data/scratch/twin_recon/clean")
    p.add_argument("--workers", type=int, default=max(1, (multiprocessing.cpu_count() or 2) - 2))
    p.add_argument("--limit", type=int, default=0, help="Process at most N dirs (smoke test).")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    dirs = sorted(d for d in args.data_dir.iterdir() if d.is_dir())
    if args.limit:
        dirs = dirs[: args.limit]
    worker = partial(process_one, out_root=args.output)
    counts: dict[str, int] = {}
    with multiprocessing.Pool(args.workers) as pool:
        for status in tqdm(pool.imap_unordered(worker, dirs, chunksize=16), total=len(dirs)):
            counts[status] = counts.get(status, 0) + 1
    print(counts)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test on 200 dirs** (requires LilyPond on PATH)

```bash
poetry run python scripts/twin_reconstruction/render_twins.py --limit 200
ls data/scratch/twin_recon/clean | head -3
```

Expected: `{'skip-fraction': ~130, 'ok': ~55, 'skip-nontreble': ~15}` (proportions approximate) and twin dirs named `*__nh` containing a `.png` + `.lmx`.

- [ ] **Step 3: Full run** (~26k LilyPond renders; expect 1–3 h on all cores; resumable if interrupted)

```bash
poetry run python scripts/twin_reconstruction/render_twins.py
```

Expected final counts: `ok` ≈ 26,168 (a handful of `fail` is tolerable — the June-1 run had failures too; >100 fails means LilyPond version drift, note it for Task 5's tolerance).

- [ ] **Step 4: Verify count and commit the driver**

```bash
ls data/scratch/twin_recon/clean | wc -l   # expect ≈26168
git add scripts/twin_reconstruction/render_twins.py
git commit -m "forensics: twin re-render driver (writes to scratch, never live data)"
```

### Task 4: Re-create the scanned twin variants

The June-1 val set used *scanned* twin images (`__getitem__` swaps to scanned). Needed for the Task 7 fingerprint.

**Files:** none new (uses existing `augment_scanned.py`)

- [ ] **Step 1: Spot-check augmenter determinism** — re-augment 5 live samples and byte-compare against the existing scanned dir:

```bash
poetry run python src/data_processing/augment_scanned.py \
  --source data/processed/primus/clean --output /tmp/aug_check \
  --copies 1 --seed 42 --limit 5 --workers 1
for d in $(ls /tmp/aug_check); do
  cmp -s /tmp/aug_check/$d/$d.png data/processed/primus/scanned/$d/$d.png \
    && echo "$d IDENTICAL" || echo "$d DIFFERS"
done
```

If IDENTICAL: augmentation is per-sample deterministic → Task 7 fingerprint can be tight. If DIFFERS: note it — the fingerprint tolerance in Task 7 widens (twins get *a* plausible scan distortion, just not the original one).

- [ ] **Step 2: Augment the reconstructed twins**

```bash
poetry run python src/data_processing/augment_scanned.py \
  --source data/scratch/twin_recon/clean \
  --output data/scratch/twin_recon/scanned \
  --copies 1 --seed 42
ls data/scratch/twin_recon/scanned | wc -l   # expect same count as clean twins
```

### Task 5: Reconstruct the June-1 split and check the step-count fingerprint

**Files:**
- Create: `scripts/twin_reconstruction/reconstruct_split.py`

- [ ] **Step 1: Write the reconstruction script**

```python
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
            "STEP-COUNT FINGERPRINT FAILED — reconstruction is not trustworthy. "
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
```

- [ ] **Step 2: Run it**

```bash
poetry run python scripts/twin_reconstruction/reconstruct_split.py
```

Expected: `n` ≈ 57,100 (46,089 + ~11,000 surviving twins), fingerprint passes (`0 ≤ skipped ≤ 200`).

- [ ] **Step 3: Commit**

```bash
git add scripts/twin_reconstruction/reconstruct_split.py
git commit -m "forensics: reconstruct June-1 split with step-count fingerprint"
```

### Task 6: Build the comparison subset and the per-id evaluator

**Files:**
- Create: `scripts/twin_reconstruction/build_subsets.py`
- Create: `scripts/twin_reconstruction/eval_subset.py`

- [ ] **Step 1: Write `build_subsets.py`** — derive the current split ids the same way `make_splits` does, then intersect:

```python
"""Compute current split ids and the clean comparison subset S.

S = (current test set) ∩ (June-1 val ∪ June-1 test), originals only.
No sample in S was in either model's training set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from CRNN_CTC.dataset import OMRDataset  # noqa: E402
from CRNN_CTC.vocab import Vocabulary  # noqa: E402

OLD = json.loads((_REPO / "data/scratch/twin_recon/old_split.json").read_text())
OUT = _REPO / "data/scratch/twin_recon/subsets.json"

vocab = Vocabulary.from_file(_REPO / "data/vocab/primus_lmx.txt")
ds = OMRDataset(
    _REPO / "data/processed/primus/clean", vocab,
    filter_non_leadsheet_clef=True, filter_unusual_time=True,
    filter_multi_staff=True, max_source_height=180,
)
n = len(ds)
assert n == 46089, n
rng = torch.Generator().manual_seed(42)
perm = torch.randperm(n, generator=rng).tolist()
n_test = max(1, int(n * 0.1))
n_val = max(1, int(n * 0.1))
n_train = n - n_val - n_test
sids = ds.sample_ids
new_test = {sids[i] for i in perm[n_train + n_val :]}
new_val = {sids[i] for i in perm[n_train : n_train + n_val]}

old_held = set(OLD["old_val"]) | set(OLD["old_test"])
S = sorted(s for s in new_test & old_held if not s.endswith("__nh"))
print(f"new_test={len(new_test)}  old_heldout={len(old_held)}  |S|={len(S)}")

OUT.write_text(json.dumps({
    "S": S,
    "new_test": sorted(new_test),
    "new_val": sorted(new_val),
}))
print("wrote", OUT)
```

- [ ] **Step 2: Run it.** Expected `|S|` ≈ 850–950 (4,608 × ~19–20% of the old pool held out).

```bash
poetry run python scripts/twin_reconstruction/build_subsets.py
```

- [ ] **Step 3: Write `eval_subset.py`** — evaluates any checkpoint on an explicit id list, resolving images exactly like `OMRDataset` (scanned preferred), with twin ids resolving into scratch:

```python
"""Evaluate a CRNN-CTC checkpoint on an explicit list of sample ids (greedy decode).

Image resolution mirrors OMRDataset: the scanned variant is preferred when it
exists. Twin ids (*__nh) resolve into data/scratch/twin_recon/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from CRNN_CTC.dataset import _load_image, _load_lmx_tokens, collate_fn  # noqa: E402
from CRNN_CTC.evaluate import _edit_distance, greedy_decode  # noqa: E402
from CRNN_CTC.model import CRNN  # noqa: E402
from CRNN_CTC.vocab import Vocabulary  # noqa: E402

CLEAN = _REPO / "data/processed/primus/clean"
SCANNED = _REPO / "data/processed/primus/scanned"
TWIN_CLEAN = _REPO / "data/scratch/twin_recon/clean"
TWIN_SCANNED = _REPO / "data/scratch/twin_recon/scanned"


class IdListDataset(Dataset):
    def __init__(self, sids: list[str], vocab: Vocabulary, img_height: int, max_w: int) -> None:
        self.vocab, self.h, self.max_w = vocab, img_height, max_w
        self.items = []
        for sid in sids:
            croot = TWIN_CLEAN if sid.endswith("__nh") else CLEAN
            sroot = TWIN_SCANNED if sid.endswith("__nh") else SCANNED
            png = sroot / sid / f"{sid}.png"
            if not png.exists():
                png = croot / sid / f"{sid}.png"
            self.items.append((sid, png, croot / sid / f"{sid}.lmx"))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        sid, png, lmx = self.items[i]
        img = _load_image(png, self.h, self.max_w)
        img = (img - img.mean()) / (img.std() + 1e-6)
        tokens = _load_lmx_tokens(lmx)
        return {
            "sample_id": sid,
            "image": torch.from_numpy(img).unsqueeze(0),
            "label": torch.tensor(self.vocab.encode(tokens), dtype=torch.long),
            "tokens": tokens,
        }


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--ids-json", type=Path, required=True)
    ap.add_argument("--ids-key", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    vocab = Vocabulary.from_file(_REPO / "data/vocab/primus_lmx.txt")
    assert ckpt["vocab_size"] == len(vocab), "vocab mismatch"

    model = CRNN(
        vocab_size=len(vocab),
        cnn_out_channels=cfg.cnn_out_channels,
        rnn_hidden=cfg.rnn_hidden,
        rnn_layers=cfg.rnn_layers,
        dropout=0.0,
        cnn_dropout=cfg.cnn_dropout,
        backbone=cfg.backbone,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sids = json.loads(args.ids_json.read_text())[args.ids_key]
    ds = IdListDataset(sids, vocab, cfg.img_height, cfg.max_image_width)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=8, collate_fn=collate_fn)

    total_ed = total_len = perfect = count = 0
    for batch in loader:
        images = batch["images"].to(device)
        widths = batch["image_widths"].to(device)
        log_probs, out_lens = model(images, widths)
        preds = greedy_decode(log_probs, out_lens, vocab)
        offset = 0
        for i, ll in enumerate(batch["label_lens"]):
            l = int(ll)
            ref = vocab.decode(batch["labels"][offset : offset + l].tolist())
            offset += l
            ed = _edit_distance(preds[i], ref)
            total_ed += ed
            total_len += len(ref)
            perfect += ed == 0
            count += 1
    print(f"checkpoint={args.checkpoint}")
    print(f"ids={args.ids_key} n={count}")
    print(f"aggregate SER = {total_ed / max(1, total_len):.4f}")
    print(f"perfect = {100 * perfect / max(1, count):.1f}%")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit**

```bash
git add scripts/twin_reconstruction/build_subsets.py scripts/twin_reconstruction/eval_subset.py
git commit -m "forensics: comparison subset builder and per-id evaluator"
```

### Task 7: Validate the reconstruction, then run the head-to-head

- [ ] **Step 1: Fingerprint #2 — old checkpoint on reconstructed June-1 val set**

```bash
poetry run python scripts/twin_reconstruction/eval_subset.py \
  --checkpoint models/run_20260601_134845/best_model.pt \
  --ids-json data/scratch/twin_recon/old_split.json --ids-key old_val
```

The training log records val SER 0.9826% at the best epoch (with the in-loop jitter bug mildly inflating it). Acceptance: **aggregate SER in [0.75%, 1.10%]**. A wrong reconstruction fails loudly: if our "old_val" actually contains old-*train* samples, the memorizing old model scores far *below* this band.

- [ ] **Step 2: Negative control — old checkpoint on reconstructed old_train sample** (confirm memorization signature). Build a 2,000-id slice and evaluate:

```bash
poetry run python -c "
import json, pathlib
p = pathlib.Path('data/scratch/twin_recon/old_split.json')
d = json.loads(p.read_text()); d['old_train_slice'] = d['old_train'][:2000]
p.write_text(json.dumps(d))
"
poetry run python scripts/twin_reconstruction/eval_subset.py \
  --checkpoint models/run_20260601_134845/best_model.pt \
  --ids-json data/scratch/twin_recon/old_split.json --ids-key old_train_slice
```

Expected: SER clearly below the Step 1 value (memorized). This confirms split sides are not swapped.

- [ ] **Step 3: THE comparison — both checkpoints on S** (neither trained on these):

```bash
poetry run python scripts/twin_reconstruction/eval_subset.py \
  --checkpoint models/run_20260601_134845/best_model.pt \
  --ids-json data/scratch/twin_recon/subsets.json --ids-key S
poetry run python scripts/twin_reconstruction/eval_subset.py \
  --checkpoint models/run_20260608_102846/best_model.pt \
  --ids-json data/scratch/twin_recon/subsets.json --ids-key S
```

Verdict rule: this is the true old-vs-new gap. Δ(aggregate SER) ≤ 0.15 points and Δ(perfect) ≤ 3 points → the reported regression was predominantly a measurement artifact. Larger → genuine regression from lost diversity/oversampling, magnitude now quantified.

- [ ] **Step 4 (CONTINGENCY — only if Step 1 or Task 5 fingerprints fail):** the reconstruction is untrustworthy (most likely LilyPond version drift changed render heights → different filter outcomes → different `n`). Do not force it. Instead answer the question by ablation: re-run training with *only* oversampling restored (after Task 10) on the current data — `poetry run python src/cli.py train --epochs 60` — and compare to `run_20260608_102846` on the current test split. ~9 h GPU. Document whichever path was taken.

### Task 8: Write up the verdict

**Files:**
- Create: `docs/experiments/2026-06-10-volume-collapse-findings.md`

- [ ] **Step 1: Write the findings doc.** Contents: the "Investigation Findings" section from the top of this plan, plus the measured numbers from Tasks 1 and 7 in one table (old/new × current-test/S), plus the verdict sentence. State explicitly that pre-June-3 evaluation numbers (0.23% SER / 94.1% perfect) are invalid for comparison due to twin leakage and must not be cited against post-June-3 numbers.

- [ ] **Step 2: Fix the stale performance block in `docs/overview.md`** — it currently says "epoch 37"; the live checkpoint is best-epoch 47 of `run_20260608_102846`. Correct the epoch and add one line: "Numbers from runs before 2026-06-03 used a twin-contaminated split and are not comparable (see docs/experiments/2026-06-10-volume-collapse-findings.md)."

- [ ] **Step 3: Commit**

```bash
git add docs/experiments/2026-06-10-volume-collapse-findings.md docs/overview.md
git commit -m "docs: volume-collapse verdict; mark pre-June-3 metrics non-comparable"
```

---

# Phase 2 — No-regret training improvements + retrain

These are worth doing regardless of the Phase 1 verdict: they restore (and exceed) the diversity the twins incidentally provided, without reintroducing leakage.

### Task 9: Fix val/test jitter contamination (TDD)

`_AugSubset` mutates `_online_aug_prob` on the shared `OMRDataset`; val/test `Subset`s wrap the same instance, so in-loop validation is jittered. Fix: never mutate the shared dataset — route augmentation through an explicit `get_item` parameter.

**Files:**
- Modify: `src/CRNN_CTC/dataset.py` (`OMRDataset.__getitem__`, `_AugSubset`, `make_splits`)
- Create: `tests/test_dataset_aug.py`

- [ ] **Step 1: Write the failing test** (`tests/test_dataset_aug.py`):

```python
"""Tests for training-only augmentation isolation in OMRDataset/make_splits."""
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

from CRNN_CTC.dataset import make_splits  # noqa: E402
from CRNN_CTC.vocab import Vocabulary  # noqa: E402

LMX = "clef:G2 key:fifths:0 time beats:4 beat-type:4 pitch:C octave:4 quarter measure"


def _make_corpus(root: Path, n: int = 12) -> None:
    rng = np.random.default_rng(0)
    for i in range(n):
        sid = f"s{i:03d}"
        d = root / sid
        d.mkdir(parents=True)
        img = (rng.random((100, 400)) * 255).astype(np.uint8)
        cv2.imwrite(str(d / f"{sid}.png"), img)
        (d / f"{sid}.lmx").write_text(LMX)


@pytest.fixture()
def corpus(tmp_path):
    data = tmp_path / "clean"
    _make_corpus(data)
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text("\n".join(sorted(set(LMX.split()))))
    return data, Vocabulary.from_file(vocab_file)


def test_val_items_are_deterministic_with_online_aug_enabled(corpus):
    data, vocab = corpus
    _, val_ds, test_ds = make_splits(
        data, vocab, img_height=64, val_frac=0.2, test_frac=0.2,
        filter_multi_staff=False, online_aug_prob=1.0,
    )
    for ds in (val_ds, test_ds):
        a = ds[0]["image"]
        b = ds[0]["image"]
        assert torch.equal(a, b), "val/test item changed between reads — aug leaked"


def test_train_items_are_actually_augmented(corpus):
    data, vocab = corpus
    train_ds, _, _ = make_splits(
        data, vocab, img_height=64, val_frac=0.2, test_frac=0.2,
        filter_multi_staff=False, online_aug_prob=1.0,
    )
    a = train_ds[0]["image"]
    b = train_ds[0]["image"]
    assert not torch.equal(a, b), "train aug (prob=1.0) produced identical reads"
```

- [ ] **Step 2: Run it to verify the val test fails on current code**

```bash
poetry run pytest tests/test_dataset_aug.py -v
```

Expected: `test_val_items_are_deterministic_with_online_aug_enabled` FAILS (shared-instance mutation), the train test passes.

- [ ] **Step 3: Implement the fix in `src/CRNN_CTC/dataset.py`.**

(a) Split `__getitem__` so augmentation is an explicit per-call decision instead of shared state. Replace the body of `__getitem__` with a delegating call and add `get_item`:

```python
    def __getitem__(self, idx: int) -> dict[str, Tensor | list[str] | str]:
        return self.get_item(idx)

    def get_item(
        self,
        idx: int,
        *,
        online_aug_prob: float | None = None,
    ) -> dict[str, Tensor | list[str] | str]:
```

…then inside, where the old code read `self._online_aug_prob`, use:

```python
        prob = self._online_aug_prob if online_aug_prob is None else online_aug_prob
        if prob > 0 and random.random() < prob:
            img = _online_jitter(img)
```

(b) Replace `_AugSubset` entirely (delete the mutation-based class and its stale comment):

```python
class _AugSubset(Dataset):
    """Train-split wrapper enabling per-call augmentation.

    Must NOT mutate the shared OMRDataset: the val/test Subsets wrap the same
    instance, so writing flags onto it bleeds augmentation into evaluation
    (this inflated in-loop val SER in all runs before 2026-06-10).
    """

    def __init__(self, subset, online_aug_prob: float) -> None:
        self._ds: OMRDataset = subset.dataset  # type: ignore[attr-defined]
        self._indices: list[int] = list(subset.indices)  # type: ignore[attr-defined]
        self._online_prob = online_aug_prob

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._ds.get_item(self._indices[idx], online_aug_prob=self._online_prob)
```

(c) In `make_splits`, the two `_AugSubset(...)` call sites keep the same signature (`_AugSubset(Subset(...), online_aug_prob)`) — verify both still construct correctly.

- [ ] **Step 4: Run the tests — all pass**

```bash
poetry run pytest tests/test_dataset_aug.py -v
poetry run pytest tests/ -v
```

- [ ] **Step 5: Commit**

```bash
git add src/CRNN_CTC/dataset.py tests/test_dataset_aug.py
git commit -m "fix: stop online jitter bleeding into val/test items during training"
```

### Task 10: Restore rare-token oversampling (TDD)

Removed in `2e4091f` without ablation; ties are ~13% of remaining errors and 10.0% of samples. Restore exactly as it was.

**Files:**
- Modify: `src/CRNN_CTC/dataset.py`, `src/CRNN_CTC/config.py`, `src/CRNN_CTC/train.py`
- Test: `tests/test_dataset_aug.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_dataset_aug.py`):

```python
def test_rare_token_oversampling_duplicates_tied_samples(corpus, tmp_path):
    data, _ = corpus
    # Make 2 of the 12 samples contain a tie token
    for sid in ("s000", "s001"):
        f = data / sid / f"{sid}.lmx"
        f.write_text(f.read_text() + " pitch:C octave:4 quarter tied:start")
    vocab_file = tmp_path / "vocab2.txt"
    vocab_file.write_text("\n".join(sorted(set((LMX + " tied:start").split()))))
    vocab = Vocabulary.from_file(vocab_file)

    train_plain, _, _ = make_splits(
        data, vocab, img_height=64, val_frac=0.2, test_frac=0.2,
        filter_multi_staff=False, rare_lmx_oversample=1,
    )
    train_over, _, _ = make_splits(
        data, vocab, img_height=64, val_frac=0.2, test_frac=0.2,
        filter_multi_staff=False, rare_lmx_oversample=2,
        rare_lmx_tokens=frozenset({"tied:start", "tied:stop"}),
    )
    n_tied_in_train = sum(
        1 for i in range(len(train_plain))
        if "tied:start" in train_plain[i]["tokens"]
    )
    assert len(train_over) == len(train_plain) + n_tied_in_train
```

- [ ] **Step 2: Run to verify it fails** (`make_splits` has no `rare_lmx_oversample` param):

```bash
poetry run pytest tests/test_dataset_aug.py::test_rare_token_oversampling_duplicates_tied_samples -v
```

Expected: FAIL with `TypeError: make_splits() got an unexpected keyword argument`.

- [ ] **Step 3: Restore the deleted code in `src/CRNN_CTC/dataset.py`** (verbatim from git `e82ee18`). Add near the top of the file (after the imports, before the domain-filter section):

```python
# Tokens that are rare at the *sample* level but error-prone at evaluation.
# PrIMuS ties appear in ~10% of in-domain samples yet account for a large
# share of remaining edit errors. Keep this set narrow so we do not up-weight
# most of the corpus (e.g. every sample with a key signature).
_DEFAULT_RARE_LMX_TOKENS: frozenset[str] = frozenset({"tied:start", "tied:stop"})


def _train_indices_with_rare_oversample(
    full_ds: OMRDataset,
    train_idx: list[int],
    *,
    oversample: int,
    rare_tokens: frozenset[str],
) -> list[int]:
    """Duplicate training indices for samples whose LMX contains *rare_tokens*."""
    if oversample <= 1 or not rare_tokens:
        return train_idx
    expanded: list[int] = []
    for i in train_idx:
        expanded.append(i)
        _sid, _png, lmx_path = full_ds._samples[i]
        tokens = _load_lmx_tokens(lmx_path)
        if any(t in rare_tokens for t in tokens):
            for _ in range(oversample - 1):
                expanded.append(i)
    return expanded
```

NOTE: `_train_indices_with_rare_oversample` references `OMRDataset` in its annotation — place it *after* the class definition, or quote the annotation (`full_ds: "OMRDataset"`). Quote it and keep it near the top.

In `make_splits`, add the parameters:

```python
    online_aug_prob: float = 0.0,
    rare_lmx_oversample: int = 1,
    rare_lmx_tokens: frozenset[str] | None = None,
```

and after `train_idx = perm[:n_train]` insert:

```python
    rare_set = rare_lmx_tokens if rare_lmx_tokens is not None else _DEFAULT_RARE_LMX_TOKENS
    train_idx = _train_indices_with_rare_oversample(
        full_ds, train_idx, oversample=rare_lmx_oversample, rare_tokens=rare_set,
    )
    if rare_lmx_oversample > 1 and rare_set:
        log.info(
            "Rare-token oversample: factor=%d tokens=%s → train virtual size %d (unique %d)",
            rare_lmx_oversample, sorted(rare_set), len(train_idx), n_train,
        )
```

- [ ] **Step 4: Restore the Config fields in `src/CRNN_CTC/config.py`** (where `2e4091f` removed them, after `online_aug_prob`):

```python
    # rare_lmx_oversample: repeat training indices for samples containing
    # rare_lmx_tokens (N-1 extra times).  1 = disabled.
    rare_lmx_oversample: int = 2
    rare_lmx_tokens: tuple[str, ...] = (
        "tied:start",
        "tied:stop",
    )
```

- [ ] **Step 5: Restore the pass-through in `src/CRNN_CTC/train.py`** `make_splits(...)` call:

```python
        online_aug_prob=cfg.online_aug_prob,
        rare_lmx_oversample=cfg.rare_lmx_oversample,
        rare_lmx_tokens=frozenset(cfg.rare_lmx_tokens) if cfg.rare_lmx_tokens else frozenset(),
```

- [ ] **Step 6: Run tests + the standard validation checks**

```bash
poetry run pytest tests/test_dataset_aug.py -v
poetry run python -c "from src.CRNN_CTC.config import Config; Config()"
```

- [ ] **Step 7: Commit**

```bash
git add src/CRNN_CTC/dataset.py src/CRNN_CTC/config.py src/CRNN_CTC/train.py
git commit -m "feat: restore rare-token (tie) oversampling removed without ablation in 2e4091f"
```

### Task 11: Config backward compatibility for pickled checkpoints (TDD)

Checkpoints pickle `Config` *instances* (CLAUDE.md constraint #3). The June-8 checkpoint's config now lacks `rare_lmx_oversample`/`rare_lmx_tokens` (and will lack `scanned_variant_dirs` from Task 13); attribute access on a loaded old config raises `AttributeError`.

**Files:**
- Modify: `src/CRNN_CTC/config.py`
- Test: `tests/test_dataset_aug.py`

- [ ] **Step 1: Write the failing test** (append):

```python
def test_ensure_config_defaults_backfills_missing_fields():
    from CRNN_CTC.config import Config, ensure_config_defaults

    cfg = Config()
    del cfg.__dict__["rare_lmx_oversample"]  # simulate a pre-restore pickled instance
    assert not hasattr(cfg, "rare_lmx_oversample")
    ensure_config_defaults(cfg)
    assert cfg.rare_lmx_oversample == 2
```

- [ ] **Step 2: Run to verify it fails** (no `ensure_config_defaults`):

```bash
poetry run pytest tests/test_dataset_aug.py::test_ensure_config_defaults_backfills_missing_fields -v
```

- [ ] **Step 3: Implement in `src/CRNN_CTC/config.py`** (module level, after the dataclass):

```python
def ensure_config_defaults(cfg: "Config") -> "Config":
    """Backfill fields added after a checkpoint's Config was pickled.

    Checkpoints serialize Config instances; unpickling an old checkpoint yields
    an instance missing fields added since. Backfill them with current defaults
    so downstream code can use plain attribute access.
    """
    from dataclasses import MISSING, fields

    for f in fields(Config):
        if not hasattr(cfg, f.name):
            if f.default is not MISSING:
                setattr(cfg, f.name, f.default)
            elif f.default_factory is not MISSING:  # type: ignore[misc]
                setattr(cfg, f.name, f.default_factory())  # type: ignore[misc]
    return cfg
```

- [ ] **Step 4: Wire it at every `ckpt["config"]` consumer.** Find them:

```bash
grep -rn '\["config"\]\|\[.config.\]' src/ scripts/ --include='*.py'
```

For each hit that *uses* the loaded config (e.g. `src/omr_pipeline/inference.py`, `src/CRNN_CTC/evaluate.py` if they build models from checkpoint config), wrap: `cfg = ensure_config_defaults(ckpt["config"])`. Skip sites that only construct a fresh `Config()`.

- [ ] **Step 5: Tests + smoke-load both real checkpoints**

```bash
poetry run pytest tests/test_dataset_aug.py -v
poetry run python -c "
import sys; sys.path.insert(0, 'src')
import torch
from CRNN_CTC.config import ensure_config_defaults
for run in ['run_20260601_134845', 'run_20260608_102846']:
    cfg = ensure_config_defaults(torch.load(f'models/{run}/best_model.pt', map_location='cpu', weights_only=False)['config'])
    print(run, cfg.rare_lmx_oversample, cfg.rare_lmx_tokens)
"
```

Expected: both print without `AttributeError` (old prints its stored `2`, new prints the backfilled default `2`).

- [ ] **Step 6: Commit**

```bash
git add src/CRNN_CTC/config.py src/ tests/test_dataset_aug.py
git commit -m "feat: ensure_config_defaults backfills Config fields on old pickled checkpoints"
```

### Task 12: Generate extra offline scan variants

This is the volume/diversity restoration: every train sample gets 2 extra scan-distorted renders (different seed than the existing set), tripling the visual variants the model can see per sample.

**Files:** none new (data generation)

- [ ] **Step 1: Check disk headroom** — need ~2× the current scanned dir:

```bash
du -sh data/processed/primus/scanned && df -h .
```

Abort and reclaim space if free < 2× scanned size.

- [ ] **Step 2: Smoke-run 20 samples**

```bash
poetry run python src/data_processing/augment_scanned.py \
  --source data/processed/primus/clean \
  --output data/processed/primus/scanned_extra \
  --copies 2 --seed 4242 --limit 20
ls data/processed/primus/scanned_extra | head -4
```

Expected: dirs named `{sid}_aug00`, `{sid}_aug01`, each containing `{dir}.png` and `{dir}.lmx`.

- [ ] **Step 3: Full run** (87,678 samples × 2 copies; expect several hours, resumable — re-run skips existing):

```bash
poetry run python src/data_processing/augment_scanned.py \
  --source data/processed/primus/clean \
  --output data/processed/primus/scanned_extra \
  --copies 2 --seed 4242
ls data/processed/primus/scanned_extra | wc -l   # expect 175356
```

### Task 13: Train-only variant sampling in the dataset (TDD)

Each train `__getitem__` picks uniformly among {existing scanned PNG, extra variants}. Val/test always get the original scanned PNG. Leakage-free by construction: variants are keyed by sample id and the split is over ids — a variant can never cross splits.

**Files:**
- Modify: `src/CRNN_CTC/dataset.py`, `src/CRNN_CTC/config.py`, `src/CRNN_CTC/train.py`
- Test: `tests/test_dataset_aug.py`

- [ ] **Step 1: Write the failing tests** (append):

```python
def _add_variants(tmp_path: Path, data: Path, n_variants: int = 2) -> Path:
    vroot = tmp_path / "scanned_extra"
    rng = np.random.default_rng(7)
    for d in data.iterdir():
        sid = d.name
        for k in range(n_variants):
            vid = f"{sid}_aug{k:02d}"
            vd = vroot / vid
            vd.mkdir(parents=True)
            img = (rng.random((100, 400)) * 255).astype(np.uint8)
            cv2.imwrite(str(vd / f"{vid}.png"), img)
    return vroot


def test_variant_sampling_train_only(corpus, tmp_path):
    import random as _random

    data, vocab = corpus
    vroot = _add_variants(tmp_path, data)
    train_ds, val_ds, _ = make_splits(
        data, vocab, img_height=64, val_frac=0.2, test_frac=0.2,
        filter_multi_staff=False, variant_dirs=[vroot],
    )
    # val: two reads identical (never samples variants)
    assert torch.equal(val_ds[0]["image"], val_ds[0]["image"])
    # train: across many reads, at least two distinct images must appear
    _random.seed(0)
    reads = [train_ds[0]["image"] for _ in range(20)]
    assert any(not torch.equal(reads[0], r) for r in reads[1:]), \
        "train item never sampled a variant"
```

- [ ] **Step 2: Run to verify failure** (`make_splits` has no `variant_dirs`):

```bash
poetry run pytest tests/test_dataset_aug.py::test_variant_sampling_train_only -v
```

- [ ] **Step 3: Implement in `src/CRNN_CTC/dataset.py`.**

(a) `OMRDataset.__init__`: add parameter `variant_dirs: list[Path] | None = None` and build the per-sid index at the end of `__init__`:

```python
        # Per-sid extra scan variants ({sid}_aug00, {sid}_aug01, …) used for
        # train-time variant sampling. Keyed by sample id so variants can never
        # cross the train/val/test split (which is over ids).
        self._variants: dict[str, list[Path]] = {}
        for vd in [Path(p) for p in (variant_dirs or [])]:
            if not vd.is_dir():
                log.warning("variant dir missing, skipping: %s", vd)
                continue
            for sub in vd.iterdir():
                vid = sub.name
                if "_aug" not in vid:
                    continue
                png = sub / f"{vid}.png"
                if png.exists():
                    self._variants.setdefault(vid.rsplit("_aug", 1)[0], []).append(png)
        if self._variants:
            log.info(
                "Variant sampling: %d sids have extra scan variants",
                len(self._variants),
            )
```

(b) `get_item`: add keyword `variant_sampling: bool = False`; replace the scanned-swap block:

```python
        # Image source: prefer the scanned render; with variant_sampling
        # (train only), pick uniformly among scanned + extra variants.
        if self.scanned_dir is not None:
            pool: list[Path] = []
            alt_png = self.scanned_dir / sid / f"{sid}.png"
            if alt_png.exists():
                pool.append(alt_png)
            else:
                for sd in self.extra_scanned_dirs:
                    alt2 = sd / sid / f"{sid}.png"
                    if alt2.exists():
                        pool.append(alt2)
                        break
            if variant_sampling:
                pool.extend(self._variants.get(sid, []))
            if pool:
                png_path = pool[0] if len(pool) == 1 or not variant_sampling else random.choice(pool)
```

(c) `_AugSubset`: add `variant_sampling: bool = False` to `__init__`, store it, and pass through:

```python
    def __getitem__(self, idx: int):
        return self._ds.get_item(
            self._indices[idx],
            online_aug_prob=self._online_prob,
            variant_sampling=self._variant_sampling,
        )
```

(d) `make_splits`: add parameter `variant_dirs: list[Path] | None = None`; pass it to the `OMRDataset(...)` constructor (`variant_dirs=variant_dirs`); change the train-wrapper condition:

```python
    train_ds: Dataset = Subset(full_ds, train_idx)
    if online_aug_prob > 0 or variant_dirs:
        train_ds = _AugSubset(train_ds, online_aug_prob, variant_sampling=bool(variant_dirs))
```

Apply the same condition change to the fine-tune wrapper block.

- [ ] **Step 4: Config + train plumbing.** `src/CRNN_CTC/config.py`, after the rare-token fields:

```python
    # Extra offline scan-variant roots (dirs of {sid}_augNN sample dirs made by
    # augment_scanned.py --copies N). Train-time only: each __getitem__ picks
    # uniformly among scanned + variants. Empty = disabled.
    scanned_variant_dirs: tuple[str, ...] = ()
```

`src/CRNN_CTC/train.py` `make_splits(...)` call:

```python
        variant_dirs=[Path(p) for p in cfg.scanned_variant_dirs] or None,
```

- [ ] **Step 5: Run all tests + validation checks**

```bash
poetry run pytest tests/test_dataset_aug.py -v
poetry run python -c "from src.CRNN_CTC.config import Config; Config()"
poetry run python -c "from src.CRNN_CTC.model import CRNN; print('ok')"
```

- [ ] **Step 6: Commit**

```bash
git add src/CRNN_CTC/dataset.py src/CRNN_CTC/config.py src/CRNN_CTC/train.py tests/test_dataset_aug.py
git commit -m "feat: train-only scan-variant sampling (leakage-free volume restoration)"
```

### Task 14: CLI flag for variant dirs

**Files:**
- Modify: `src/cli.py` (`_add_common_data_args` or the train parser group, and `_build_config_from_args`)

- [ ] **Step 1: Add the argument** next to the existing `--scanned-dir`-style args:

```python
    g.add_argument(
        "--scanned-variant-dirs",
        nargs="*",
        default=None,
        help="Extra scan-variant roots from augment_scanned.py --copies N "
        "(train-time variant sampling; default: none)",
    )
```

- [ ] **Step 2: Map it in `_build_config_from_args`** following the existing pattern for optional args:

```python
    if getattr(args, "scanned_variant_dirs", None):
        kwargs["scanned_variant_dirs"] = tuple(args.scanned_variant_dirs)
```

- [ ] **Step 3: Verify**

```bash
poetry run python src/cli.py train --help | grep -A2 scanned-variant-dirs
```

- [ ] **Step 4: Commit**

```bash
git add src/cli.py
git commit -m "feat: --scanned-variant-dirs train flag"
```

### Task 15: Retrain

- [ ] **Step 1: Launch** (90 epochs: the old run was still improving at 59 and the added variant diversity delays overfitting; early stopping patience 12 still guards). Expect ~14–16 h on the RTX 3060:

```bash
poetry run python src/cli.py train \
  --epochs 90 \
  --scanned-variant-dirs data/processed/primus/scanned_extra
```

- [ ] **Step 2: Verify startup log lines before walking away:** `Rare-token oversample: factor=2 … train virtual size ~40,560 (unique 36,873)`, `Variant sampling: ~87k sids have extra scan variants`, and steps/epoch ≈ 2,535 (= ceil(40,560/16)).

- [ ] **Step 3: After completion, record the run dir name and best epoch** from `models/run_*/training_log.csv`.

### Task 16: Evaluate the new model

- [ ] **Step 1: Standard evaluation**

```bash
poetry run python scripts/evaluate_full.py \
  --checkpoint models/<NEW_RUN>/best_model.pt --split test --both-splits
```

- [ ] **Step 2: Subset-S evaluation** (comparable to Phase 1 head-to-head):

```bash
poetry run python scripts/twin_reconstruction/eval_subset.py \
  --checkpoint models/<NEW_RUN>/best_model.pt \
  --ids-json data/scratch/twin_recon/subsets.json --ids-key S
```

- [ ] **Step 3: Acceptance:** on the current test split, the retrained model must beat `run_20260608_102846` (1.31% SER / 72% perfect). On S it should at least match the better of the two Phase-1 contenders. If it regresses, do not promote it; investigate before proceeding (suspect the variant-sampling pool or oversample interaction first — check the startup log numbers from Task 15 Step 2).

- [ ] **Step 4: Append the results table to `docs/experiments/2026-06-10-volume-collapse-findings.md`** and commit.

```bash
git add docs/experiments/2026-06-10-volume-collapse-findings.md
git commit -m "docs: retrain results vs both prior runs"
```

---

# Phase 3 — Post-retrain obligations

### Task 17: Recalibrate the staff-reject gate (CLAUDE.md hard constraint #7)

- [ ] **Step 1: Confirm `models/latest/` points at the new run** (check how `train.py` populates it; if it is a manual copy, copy `best_model.pt` + `training_log.csv` from the new run dir).

- [ ] **Step 2: Recalibrate and commit**

```bash
poetry run python src/cli.py calibrate-reject \
  --fixtures data/staff_reject --out models/staff_reject/thresholds.json
git add models/staff_reject/thresholds.json
git commit -m "calib: staff-reject thresholds for retrained CRNN"
```

### Task 18: End-to-end smoke test

- [ ] **Step 1: API smoke** per CLAUDE.md validation section:

```bash
poetry run python src/cli.py api &
sleep 5 && curl -s http://localhost:8000/ | python -m json.tool && kill %1
```

- [ ] **Step 2: Run one real lead-sheet PDF through the pipeline** (use any sample the project already uses for manual checks) and eyeball that segments/chords/tokens come out sane — header injection is untouched by this plan, but it consumes the new checkpoint.

### Task 19: Documentation updates (CLAUDE.md docs protocol)

- [ ] **Step 1: `docs/overview.md`** — replace the Performance block with the new model's numbers (provenance line: run dir, epoch, split, decoding).
- [ ] **Step 2: `docs/training.md`** — document restored oversampling, variant sampling, the jitter-bleed fix, and the 90-epoch schedule.
- [ ] **Step 3: `docs/data_pipeline.md`** — add the `scanned_extra` generation stage (`augment_scanned.py --copies 2 --seed 4242`).
- [ ] **Step 4: `docs/configuration.md`** — add `rare_lmx_oversample`, `rare_lmx_tokens`, `scanned_variant_dirs`, and note `ensure_config_defaults`.
- [ ] **Step 5: `CLAUDE.md`** — add one pitfall: "**`extra_data_dirs` joins the split pool.** Never add per-sample derived data (twins, variants, re-renders of existing samples) via `extra_data_dirs` — derived copies of a train sample can land in val/test and leak (this inflated all pre-June-3 metrics). Use `scanned_variant_dirs`, which is train-only by construction."
- [ ] **Step 6: Commit**

```bash
git add docs/ CLAUDE.md
git commit -m "docs: new training pipeline, performance numbers, leakage pitfall"
```

### Task 20: Thesis touch-points (do NOT build the PDF)

- [ ] **Step 1: List every place `latex_documents/main/chapters/` cites the invalidated numbers** (0.98%, 0.23%, 94.1%) or describes the twin approach as a training-data feature:

```bash
grep -rn "94\.1\|0\.23\|0\.98" latex_documents/main/chapters/ | head -20
```

- [ ] **Step 2: Report the list to Pol with the replacement numbers and a 2-sentence suggested framing** (the leakage finding is thesis-worthy: "initial evaluation was inflated by near-duplicate leakage from header-stripped twins; after removing them and adding leakage-free variant augmentation, the honest numbers are X"). Do not rewrite chapters unprompted — thesis edits go through the `writing-tfg-fib` skill at Pol's direction.

---

## Future work (separate plans — do not fold into this one)

1. **Recover the 23,056 height-filtered samples** (+50% volume, the single biggest remaining data lever): re-render long incipits with single-line LilyPond layout (`ragged-right`, no system breaks) so they pass the multi-staff filter. Requires renderer changes, a `max_image_width` strategy (2048 clamp distorts very wide staves), batch-width bucketing for VRAM, and a new split (breaks metric comparability — schedule after this plan's numbers are locked).
2. **Online albumentations augmentation** (infinite variants instead of K=3) if the dataloader can sustain ~70 img/s with the scan pipeline.
3. **Triage the 18,532 token-filtered samples**: some may be label-side artifacts (e.g. non-G2 clef tokens whose renders were already normalized to treble) — if so they are free in-domain data after relabeling. Verify against `lilypond_render.py` clef tables before believing this.

## Self-review notes

- Spec coverage: investigation verdict (Tasks 1–8), root-cause isolation (Tasks 5–7 reconstruction + Task 7 Step 4 ablation contingency), best-results retrain (Tasks 9–16), system consistency obligations (Tasks 17–19), thesis (Task 20). ✓
- The numbers 46,089 / 4,626 / 26,168 / 185,327 / 108,250 were measured directly on 2026-06-10, not estimated. Scripts assert on them where they are load-bearing (`reconstruct_split.py`, `build_subsets.py`).
- Type consistency: `get_item(idx, *, online_aug_prob, variant_sampling)` is introduced in Task 9 and extended in Task 13; `_AugSubset(subset, online_aug_prob, variant_sampling=False)` signature consistent across Tasks 9/13. `make_splits` gains `rare_lmx_oversample`/`rare_lmx_tokens` (Task 10) and `variant_dirs` (Task 13); `train.py` passes all three.
