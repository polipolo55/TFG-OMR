# Training "Volume Collapse" — Investigation Verdict (2026-06-10)

## TL;DR

The reported regression between the June-1 CRNN (`run_20260601_134845`) and the
June-8 CRNN (`run_20260608_102846`) is **largely a measurement artifact, not a
real loss of model quality**. The old run's headline numbers (0.23 % SER,
94.1 % perfect) were inflated by **near-duplicate train/test leakage** caused by
putting header-stripped "twin" samples into the same `randperm` split pool as
their parents. The honest comparison the original analysis assumed exists
**cannot be reconstructed** (proven below), so the only valid path forward is to
retrain on the current leakage-free split and compare new-vs-new.

The original "volume collapse" framing was also quantitatively wrong on three
counts (see *Corrections* below).

---

## The decisive evidence: the old model's own checkpoint contradicts the leak-free story

Every checkpoint stores the validation SER it was selected against. Compare that
to each model evaluated on the **current** test split (`scripts/evaluate_full.py
--split test`, greedy):

| Run | Stored `val_ser` (its own held-out val) | Aggregate SER on **current** test split | Perfect on current test |
|-----|------|------|------|
| `run_20260601_134845` (old) | **0.98 %** | **0.23 %** | 94.1 % |
| `run_20260608_102846` (new) | **1.28 %** | **1.31 %** | 72.0 % |

Read the old row carefully. The old model scores **4× better on the "test" set
than on the validation set it was actually checkpointed against.** A model
cannot generalize *better* to held-out data than to its own validation set —
unless the "test" set isn't held out. It isn't: the current test split is drawn
from the 46,089 filtered originals, and ~80 % of those originals were in the old
model's training pool (the old 80/10/10 split was over a *different, larger* pool
of originals + twins, so membership over originals is effectively random w.r.t.
the current split). The old model is being graded largely on its own training
data, and it memorizes hard (final train loss 0.0018). **0.23 % / 94.1 % is a
memorization readout, not a generalization metric.**

The new model is the control: stored val 1.28 % vs current-test 1.31 % — nearly
identical, exactly what an honest, leak-free split looks like.

## Why a clean head-to-head cannot be reconstructed

The plan's Phase 1 attempted to rebuild the old split, re-render the deleted
twins, and evaluate both checkpoints on samples *neither* trained on. That
attempt **failed its validation fingerprint and was abandoned** — which is
itself an informative result:

- The deleted twins were regenerated deterministically (hash-based membership,
  seed 42, fraction 0.35 → exactly **26,168** twins, matching the count recorded
  in `docs/superpowers/plans/2026-06-03-virtual-header-injection.md`).
- But re-rendered headerless twins **crop systematically shorter** than the
  June-1 renders did (no clef/time glyphs → tighter content box). Only **838**
  of 26,168 re-rendered twins exceed the 180 px multi-staff height filter,
  versus the ~15,000 that must have been filtered in June to make the numbers
  add up. **25,208** twins survive filtering today vs the ~11,000 the old
  optimizer step count (185,327 steps / 59 epochs / batch 16, with rare-token
  oversampling) demands.
- The step-count fingerprint requires a reconstructed train virtual size of
  L ≈ 50,257–50,304. The re-render produces L = 62,674. **Off by ~12,000
  samples**, far outside the ±200 AMP-skip tolerance. Two alternate
  filtering hypotheses were tested and also failed (54,619 and 55,864).

Conclusion: the exact June-1 split depended on LilyPond render geometry that the
current toolchain no longer reproduces. The old metrics are not just
contaminated — they are **not reproducible**, so they must never be cited
against any post-2026-06-03 number.

## Corrections to the original "volume collapse" analysis

1. **The volume drop was ~27 %, not ~50 %.** From optimizer step counts:
   old ≈ 50,260 train samples/epoch, new ≈ 36,850. The twins added ~11,000
   *surviving* training samples (after filters), not a full duplicate of the
   corpus.
2. **The twins numbered 26,168, not 87,678**, and only ~11,000 survived the
   domain filters into the old training set.
3. **Scanned images never doubled anything.** `OMRDataset.get_item` *swaps* the
   clean PNG for the scanned one; it does not add a second sample. The
   "~350,000 training images" never existed.
4. **Neither run stopped early.** Both ran the full 60-epoch schedule
   (`training_log.csv` in each run dir). "Epoch 47" vs "epoch 59" are the
   *best-checkpoint* epochs, not stopping points. The old run was still
   improving at epoch 59 (mildly undertrained).

## A real bug found along the way

`_AugSubset` mutated `_online_aug_prob` on the **shared** `OMRDataset` instance,
so the val/test `Subset`s (which wrap the same instance) had online jitter
applied during in-loop validation in every run before 2026-06-10. This inflated
in-loop val SER and added noise to best-checkpoint selection (it does **not**
affect offline `scripts/evaluate_full.py`, which passes `online_aug_prob=0`).
Fixed in commit `e786a40`.

## What was done in response (this branch)

Leakage-free quality improvements, all committed on branch `fixes`:

- **`e786a40`** — fix the val/test jitter bleed.
- **`1f1e670` / `1f015e0`** — restore rare-token (tie) oversampling that was
  removed without ablation in `2e4091f` (ties are ~13 % of remaining edits and
  10.0 % of samples), plus its CLI flags and docs.
- **`50e6bf0`** — `ensure_config_defaults` so old pickled checkpoints still load
  after Config fields change.
- **Task 12** — generated 175,354 extra offline scan variants
  (`data/processed/primus/scanned_extra`, `augment_scanned.py --copies 2
  --seed 4242`).
- **`374e61a` / `af84c26`** — **train-only scan-variant sampling**: each train
  `__getitem__` picks uniformly among the base scanned image plus that sample's
  variants. This restores (and exceeds) the training-image diversity the twins
  incidentally provided, **without** the leakage that twins caused — variants
  are keyed by sample id and the split is over ids, so a variant can never cross
  the train/val/test boundary. Enabled with `--scanned-variant-dirs`.

The retrain that uses these (`Task 15`) is run by the thesis author on the GPU;
its results table will be appended below.

## Results

Greedy CTC, current leakage-free test split (4,608 samples), `scripts/evaluate_full.py
--split test --both-splits`:

| Model | Train data | Scanned SER | Scanned perfect | Clean SER | Clean perfect | Notes |
|-------|-----------|------|------|------|------|-------|
| `run_20260601_134845` | originals + twins (leaked split) | 0.23 % | 94.1 % | — | — | **invalid — memorization / leakage; not reproducible** |
| `run_20260608_102846` | originals only | 1.31 % | 72.0 % | 1.19 % | 73 % | honest baseline |
| **`run_20260612_101637`** (now `models/latest`, best epoch 83) | originals + variant sampling + tie oversampling 2× | **1.23 %** | **72.7 %** | **1.17 %** | **73.7 %** | leakage-free; best honest result |

The retrain (variant sampling + jitter-bleed fix + tie oversampling) gives a
**modest, real improvement** over the honest baseline on both splits — and is now
the shipped checkpoint. The gap to the old 0.23 % is *not* a quality gap; that
number was leakage (see above).

**Per-category errors (scanned split, retrain).** Errors are almost entirely
structural: barlines (`measure`) 14.7 % and ties 31.0 % together are ~87 % of all
edits; pitch 0.06 %, duration 0.04 %, octave 0.02 %, key 0.00 %.

**Tie oversampling did not pay off.** Despite 2× oversampling of tie-containing
samples, the tie category still showed **31.0 %** error (scanned) / 30.2 %
(clean) — duplicating whole staff images cannot teach the model to disambiguate
a faint tie arc on a degraded scan. Oversampling was therefore **disabled by
default** afterward (commit `a2eb073`; `rare_lmx_oversample=1`), reclaiming ~10 %
train time at no measured tie cost. The tie problem wants a targeted fix
(tie-specific augmentation or a structural post-processor), not sample
duplication — logged as future work.

## Reject-gate recalibration (CLAUDE.md #7)

After the retrain the staff-reject gate was recalibrated against a hand-labelled
fixture set of **276 music / 123 non-music** strips, harvested from the Real
Book PDFs with `scripts/build_reject_label_set.py` and sorted in the browser at
`/reject-labeler`.

**The `calibrate-reject` tool's geometric sweep was rejected** — it pushed
`min_line_span_frac` to 0.69 and would have **rejected 60 % of real music
(FP=165/276)**. Root cause: with clean Real Book input the detector's
false-positives are *hard negatives* (partial staves, full-width text/chord rows)
that are geometrically staff-like — wider line-span than many genuinely short
real staves — so the geometry features do not separate the classes. (Separately,
the tool only sweeps geometry; it copies `min_mean_logprob` from the default and
never calibrates the CTC gate — a tool gap, logged as future work.)

The CTC confidence axis separates only weakly too (music mean-logprob median
−0.032 but tails to −0.172; non-music median −0.053). Given the strong cost
asymmetry — losing a real staff is far worse than passing a garbage strip that
downstream can ignore — the gate was set **music-preserving** by hand:

| `min_mean_logprob` | music kept | non-music rejected (combined w/ geometry) |
|------|------|------|
| −0.05 (old) | 93.3 % | — |
| **−0.15 (chosen)** | **98.6 %** (272/276) | **30.1 %** (37/123) |

Final `models/staff_reject/thresholds.json`: permissive geometry (the prior
production values, unchanged) + `min_mean_logprob=-0.15`,
`confident_override_logprob=-0.05`. This keeps 98.6 % of real staves (vs 93.3 %
under the old −0.05) while the geometry gates still drop no-staff-line garbage
and the CTC gate drops the lowest-confidence fragments. The 4 false-rejected
music strips are degenerate partial staves (<5 detectable lines).

**Future work:** the hard-negative overlap means a geometry+confidence gate
cannot cleanly filter clean-Real-Book false-detections; meaningfully better
rejection would need a small learned classifier on the strip, not threshold
sweeps. And `calibrate-reject` should sweep `min_mean_logprob` with a
cost-asymmetric objective rather than per-feature Youden-J on geometry only.
