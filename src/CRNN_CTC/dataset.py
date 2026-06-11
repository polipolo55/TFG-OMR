"""
dataset.py
==========
PyTorch ``Dataset`` and collation utilities for the monophonic OMR pipeline.

Each *sample* is a directory under ``data_dir`` containing at least:
    {sample_id}.png          — grayscale staff-line image
    {sample_id}.lmx          — space-separated LMX token sequence

The scanned (augmented) variant stores images under a separate root but
shares the ``.lmx`` labels with the clean (original) directory.

Image pre-processing
--------------------
1. Load as grayscale, float32 in [0, 1].
2. Resize height to ``img_height`` (default 128), width scaled proportionally.
3. Normalise to zero-mean, unit-variance (channel-wise).

The custom ``collate_fn`` pads images to the widest sample in each mini-batch
(right-padding with zeros after normalisation) and packs CTC targets.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .vocab import Vocabulary

log = logging.getLogger(__name__)

# Tokens that are rare at the *sample* level but error-prone at evaluation.
# PrIMuS ties appear in ~10% of in-domain samples yet account for a large
# share of remaining edit errors. Keep this set narrow so we do not up-weight
# most of the corpus (e.g. every sample with a key signature).
_DEFAULT_RARE_LMX_TOKENS: frozenset[str] = frozenset({"tied:start", "tied:stop"})


def _train_indices_with_rare_oversample(
    full_ds: "OMRDataset",
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


# ---------------------------------------------------------------------------
# Domain filters
# ---------------------------------------------------------------------------
#
# These filters define the lead-sheet target domain — see
# ``docs/overview.md`` → "Domain Specification".  They do not exist to clean
# up "edge cases"; they are the contract of the system.
#

# Jazz lead-sheet target domain: only treble clef.
# C3 (alto), C4 (tenor), G1 (French violin), F4 (bass) appear in orchestral
# PrIMuS but never in Real Book melody lines.
_CLEF_LEADSHEET = frozenset({"clef:G2"})

# Common jazz time signatures — must match ``_COMMON_TIME_SIGS`` in
# ``src/omr_pipeline/grammar_fix.py`` (the inference-side counterpart).
# Exotic meters (7/4, 9/8, 11/8 …) appear in classical PrIMuS but not Real Book.
# These two constants are kept in separate modules to avoid coupling CRNN_CTC
# (training) to omr_pipeline (inference).
_COMMON_TIME_SIGS: frozenset[tuple[str, str]] = frozenset(
    {
        ("beats:4", "beat-type:4"),
        ("beats:3", "beat-type:4"),
        ("beats:2", "beat-type:4"),
        ("beats:2", "beat-type:2"),
        ("beats:6", "beat-type:8"),
        ("beats:6", "beat-type:4"),
        ("beats:5", "beat-type:4"),
        ("beats:12", "beat-type:8"),
    }
)


def _is_degenerate(
    tokens: list[str],
    *,
    filter_non_leadsheet_clef: bool = False,
    filter_unusual_time: bool = False,
) -> bool:
    """Return *True* if a sample falls outside the lead-sheet domain.

    Criteria:

    non-leadsheet-clef
        Any clef token not in ``_CLEF_LEADSHEET`` (i.e. not ``clef:G2``).
        Drops C3 (alto), C4 (tenor), G1, F4 etc. that appear in orchestral
        PrIMuS excerpts but never in jazz lead sheets.

    unusual-time
        A time signature not in ``_COMMON_TIME_SIGS`` (common jazz meters).
        Drops 7/4, 9/8, 11/8 etc. that appear in classical PrIMuS but not
        Real Book.
    """
    if not tokens:
        return True

    if filter_non_leadsheet_clef:
        for t in tokens:
            if t.startswith("clef:") and t not in _CLEF_LEADSHEET:
                return True

    if filter_unusual_time:
        for i, t in enumerate(tokens):
            if t == "time" and i + 2 < len(tokens):
                pair = (tokens[i + 1], tokens[i + 2])
                if pair not in _COMMON_TIME_SIGS:
                    return True

    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _discover_samples(
    data_dir: Path,
    *,
    require_lmx: bool = True,
) -> list[tuple[str, Path, Path]]:
    """Return ``(sample_id, png_path, lmx_path)`` triples discovered recursively.

    ``data_dir`` may contain arbitrary nested package structure; any directory
    that contains a ``{name}.png`` (and optionally ``{name}.lmx``) is treated
    as a sample directory with ``name`` as the sample_id.
    """
    samples: list[tuple[str, Path, Path]] = []
    if not data_dir.is_dir():
        return samples

    for png in sorted(data_dir.rglob("*.png")):
        sid = png.stem
        sub = png.parent
        lmx = sub / f"{sid}.lmx"
        if png.stat().st_size == 0:
            continue
        if require_lmx and not lmx.exists():
            log.debug("Skipping %s — no .lmx file", sid)
            continue
        samples.append((sid, png, lmx))
    return samples


def _image_source_height(path: Path) -> int:
    """Return the original pixel height of a PNG without decoding pixel data.

    Uses Pillow's lazy header read — only the IHDR chunk is parsed, so this
    is ~25× faster than ``cv2.imread`` for the multi-staff filter.  On the
    87k-sample PrIMuS corpus this drops dataset init by ~40 seconds.
    """
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.size[1]  # (width, height) → height
    except Exception:
        return 0


def _load_image(
    path: Path,
    img_height: int,
    max_width: int = 0,
) -> np.ndarray:
    """Load a grayscale image, resize to ``img_height``, return float32 [0, 1].

    If *max_width* > 0 and the proportionally-scaled width exceeds it, the
    image is clamped to ``(img_height, max_width)`` — a safety valve against
    OOM from outlier-wide samples.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    h, w = img.shape
    new_w = max(1, round(w * img_height / h))
    if max_width > 0 and new_w > max_width:
        new_w = max_width
    img = cv2.resize(img, (new_w, img_height), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def _load_lmx_tokens(path: Path) -> list[str]:
    """Read a ``.lmx`` file and return its whitespace-separated tokens."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return text.split()


# ---------------------------------------------------------------------------
# Online augmentation — applied at __getitem__ for epoch-to-epoch diversity
# ---------------------------------------------------------------------------


def _online_jitter(img: np.ndarray) -> np.ndarray:
    """Apply cheap per-sample jitter to a [0, 1] grayscale image.

    Combines three lightweight effects whose individual impact is small but
    whose composition gives meaningful epoch-to-epoch variation on top of the
    offline-augmented PNG:

    1. Brightness/contrast jitter — multiplicative gain ±5%, additive bias ±3%.
    2. Gaussian pixel noise (σ ≈ 0.005–0.015) — sub-perceptual sensor noise.
    3. Random horizontal jitter (±2 px shift, edge-replicated) — staff position
       variability without disturbing CTC alignment.

    All three are vectorized numpy ops; combined cost is ~50 µs per sample on
    a typical staff image, far below dataloader throughput limits.
    """
    h, w = img.shape

    gain = 1.0 + (random.random() - 0.5) * 0.10  # [0.95, 1.05]
    bias = (random.random() - 0.5) * 0.06  # [-0.03, 0.03]
    img = img * gain + bias

    sigma = random.uniform(0.005, 0.015)
    img = img + np.random.normal(0.0, sigma, size=img.shape).astype(np.float32)

    shift = random.randint(-2, 2)
    if shift != 0:
        img = np.roll(img, shift, axis=1)
        if shift > 0:
            img[:, :shift] = img[:, shift : shift + 1]
        else:
            img[:, shift:] = img[:, shift - 1 : shift]

    return np.clip(img, 0.0, 1.0)



# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class OMRDataset(Dataset):
    """PNG + LMX dataset for CTC-based monophonic OMR.

    Parameters
    ----------
    data_dir : Path | str
        Directory with per-sample sub-folders (must contain ``.lmx``).
    vocab : Vocabulary
        Token ↔ index mapping.
    img_height : int
        Target image height in pixels (width is scaled proportionally).
    scanned_dir : Path | str | None
        If given, load *images* from this directory instead of *data_dir*.
        Labels are always read from *data_dir*.
    filter_multi_staff : bool
        If *True*, discard images whose source height exceeds
        ``max_source_height``.  Multi-staff renders (LilyPond wrapping onto
        two lines) are ~2-3× taller than single staves (normal range
        84–152 px, gap, then ≥200 px for double staves).
    max_source_height : int
        Upper bound on original image height used by the multi-staff filter.
        A value of 180 px cleanly separates the entire normal population
        (p95 = 152 px) from all multi-staff images (≥200 px).

    Note: ``filter_non_leadsheet_clef`` and ``filter_unusual_time`` default
    to *False* here for backward compatibility.  ``Config`` sets them True
    (the intended default for jazz lead-sheet training); ``train.py`` and
    ``evaluate.py`` always pass these flags explicitly from ``Config``.
    """

    def __init__(
        self,
        data_dir: Path | str,
        vocab: Vocabulary,
        img_height: int = 128,
        max_image_width: int = 0,
        scanned_dir: Path | str | None = None,
        filter_non_leadsheet_clef: bool = False,  # Config default: True
        filter_unusual_time: bool = False,  # Config default: True
        filter_multi_staff: bool = True,
        max_source_height: int = 180,
        extra_data_dirs: list[Path] | None = None,
        extra_scanned_dirs: list[Path] | None = None,
        online_aug_prob: float = 0.0,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.vocab = vocab
        self.img_height = img_height
        self.max_image_width = max_image_width
        self.scanned_dir = Path(scanned_dir) if scanned_dir else None
        self.extra_scanned_dirs = [Path(p) for p in (extra_scanned_dirs or [])]
        self._online_aug_prob = online_aug_prob
        self._oov_counts: dict[str, int] = {}  # token → occurrence count

        # Discover all valid (png + lmx) samples from primary + extra dirs
        raw_samples = _discover_samples(self.data_dir, require_lmx=True)
        for extra in extra_data_dirs or []:
            extra_path = Path(extra)
            if extra_path.is_dir():
                extra_samples = _discover_samples(extra_path, require_lmx=True)
                log.info("Extra data dir %s: %d samples", extra_path, len(extra_samples))
                raw_samples.extend(extra_samples)
        if not raw_samples:
            raise RuntimeError(f"No valid samples found in {self.data_dir}")

        # Token-level filter + OOV scan in a single pass.  Each sample's .lmx
        # file is read exactly once instead of twice (filter pass + later OOV
        # pass) — halves init I/O for the 87k-sample corpus.
        do_token_filter = filter_non_leadsheet_clef or filter_unusual_time
        oov: dict[str, int] = {}
        after_token_filter: list[tuple[str, Path, Path]] = []
        for sid, png, lmx in raw_samples:
            tokens = _load_lmx_tokens(lmx)
            if do_token_filter and _is_degenerate(
                tokens,
                filter_non_leadsheet_clef=filter_non_leadsheet_clef,
                filter_unusual_time=filter_unusual_time,
            ):
                continue
            after_token_filter.append((sid, png, lmx))
            for t in tokens:
                if t not in self.vocab:
                    oov[t] = oov.get(t, 0) + 1

        n_removed = len(raw_samples) - len(after_token_filter)
        if do_token_filter and n_removed:
            log.info(
                "Token filter removed %d out-of-domain samples",
                n_removed,
            )

        # Image-height filter — rejects multi-staff renders.  Uses PIL's
        # header-only read (see ``_image_source_height``) so this is fast
        # even on tens of thousands of files.
        if filter_multi_staff:
            self._samples = [
                (sid, png, lmx)
                for sid, png, lmx in after_token_filter
                if _image_source_height(png) <= max_source_height
            ]
            n_tall = len(after_token_filter) - len(self._samples)
            if n_tall:
                log.info(
                    "Height filter (max %dpx) removed %d multi-staff images",
                    max_source_height,
                    n_tall,
                )
        else:
            self._samples = after_token_filter

        if self._samples:
            log.info(
                "OMRDataset: %d samples retained from %s%s",
                len(self._samples),
                self.data_dir,
                f" (images from {self.scanned_dir})" if self.scanned_dir else "",
            )

        if not self._samples:
            raise RuntimeError(f"No samples remain in {self.data_dir} after filtering.")

        if oov:
            total = sum(oov.values())
            top5 = sorted(oov.items(), key=lambda x: -x[1])[:5]
            log.warning(
                "OOV tokens in dataset: %d unique, %d occurrences. Top: %s",
                len(oov),
                total,
                ", ".join(f"{tok!r} ×{cnt}" for tok, cnt in top5),
            )

    # -- Dataset protocol ---------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor | list[str] | str]:
        return self.get_item(idx)

    def get_item(
        self,
        idx: int,
        *,
        online_aug_prob: float | None = None,
    ) -> dict[str, Tensor | list[str] | str]:
        """Load one sample; *online_aug_prob* overrides the instance default.

        ``None`` (the default) falls back to the constructor's
        ``online_aug_prob``.  Train-split wrappers pass it explicitly so the
        shared dataset instance never has to be mutated.
        """
        sid, png_path, lmx_path = self._samples[idx]

        # Optionally swap the image source to the scanned directory
        if self.scanned_dir is not None:
            alt_png = self.scanned_dir / sid / f"{sid}.png"
            if alt_png.exists():
                png_path = alt_png
            else:
                # Try extra scanned dirs
                for sd in self.extra_scanned_dirs:
                    alt2 = sd / sid / f"{sid}.png"
                    if alt2.exists():
                        png_path = alt2
                        break

        # Image → (H, W) float32 [0, 1]
        img = _load_image(png_path, self.img_height, self.max_image_width)
        tokens = _load_lmx_tokens(lmx_path)

        # Online augmentation (training only): light per-epoch jitter on top
        # of the offline-augmented scanned PNG.  Without this every epoch
        # sees the exact same pixel grid for each sample → the model overfits
        # to those specific augmentations.  Cheap to compute (no morphology,
        # no albumentations) so it does not bottleneck the dataloader.
        prob = self._online_aug_prob if online_aug_prob is None else online_aug_prob
        if prob > 0 and random.random() < prob:
            img = _online_jitter(img)

        # Normalise to zero-mean, unit-variance
        img = (img - img.mean()) / (img.std() + 1e-6)
        img_t = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)

        label = self.vocab.encode(tokens)

        return {
            "sample_id": sid,
            "image": img_t,  # (1, H, W)
            "label": torch.tensor(label, dtype=torch.long),  # (L,)
            "tokens": tokens,  # raw strings (debug)
        }

    # -- Convenience --------------------------------------------------------

    @property
    def sample_ids(self) -> list[str]:
        return [s[0] for s in self._samples]



# ---------------------------------------------------------------------------
# Collate function — pad images to uniform width, pack CTC targets
# ---------------------------------------------------------------------------


def collate_fn(
    batch: list[dict[str, Tensor | list[str] | str]],
) -> dict[str, Tensor | list[str]]:
    """Collate samples into a padded mini-batch for ``CTCLoss``.

    Returns
    -------
    dict with keys:
        images      : (B, 1, H, W_max) — right-padded with 0
        labels      : (sum(L_i),)       — flat-packed label indices
        label_lens  : (B,)              — individual label lengths
        image_widths: (B,)              — original (unpadded) widths
        sample_ids  : list[str]
    """
    images: list[Tensor] = [s["image"] for s in batch]
    labels: list[Tensor] = [s["label"] for s in batch]
    sample_ids: list[str] = [s["sample_id"] for s in batch]

    # Pad images to the widest in the batch
    max_w = max(im.shape[2] for im in images)
    padded: list[Tensor] = []
    widths: list[int] = []
    for im in images:
        w = im.shape[2]
        widths.append(w)
        if w < max_w:
            pad = torch.zeros(1, im.shape[1], max_w - w, dtype=im.dtype)
            im = torch.cat([im, pad], dim=2)
        padded.append(im)

    return {
        "images": torch.stack(padded, dim=0),  # (B, 1, H, W_max)
        "labels": torch.cat(labels, dim=0),  # (sum L_i,)
        "label_lens": torch.tensor([label.size(0) for label in labels], dtype=torch.long),  # (B,)
        "image_widths": torch.tensor(widths, dtype=torch.long),  # (B,)
        "sample_ids": sample_ids,
    }


# ---------------------------------------------------------------------------
# Train / val split helper
# ---------------------------------------------------------------------------


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


def make_splits(
    data_dir: Path | str,
    vocab: Vocabulary,
    img_height: int = 128,
    max_image_width: int = 0,
    scanned_dir: Path | str | None = None,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    filter_non_leadsheet_clef: bool = True,
    filter_unusual_time: bool = True,
    filter_multi_staff: bool = True,
    max_source_height: int = 180,
    extra_data_dirs: list[Path] | None = None,
    extra_scanned_dirs: list[Path] | None = None,
    online_aug_prob: float = 0.0,
    rare_lmx_oversample: int = 1,
    rare_lmx_tokens: frozenset[str] | None = None,
    finetune_data_dirs: list[Path] | None = None,
    finetune_scanned_dirs: list[Path] | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
    """Create train / val / test splits from a single data directory.

    The split is deterministic (seeded) and stratified at the sample-id level
    (no data leakage across splits).

    Returns
    -------
    train_ds, val_ds, test_ds
    """
    from torch.utils.data import Subset

    # NOTE: fine-tune (real-domain) dirs are deliberately NOT merged into the
    # split pool here. They are real Real Book pages used only to adapt the
    # model; letting them fall into val/test would leak real data into the
    # held-out sets and invalidate the synthetic-to-real domain-gap measurement.
    # They are appended to the TRAIN set only, after the split (see below).
    extra_clean = list(extra_data_dirs or [])
    extra_scanned_combined = list(extra_scanned_dirs or [])

    full_ds = OMRDataset(
        data_dir,
        vocab,
        img_height=img_height,
        max_image_width=max_image_width,
        scanned_dir=scanned_dir,
        filter_non_leadsheet_clef=filter_non_leadsheet_clef,
        filter_unusual_time=filter_unusual_time,
        filter_multi_staff=filter_multi_staff,
        max_source_height=max_source_height,
        extra_data_dirs=extra_clean or None,
        extra_scanned_dirs=extra_scanned_combined or None,
    )
    n = len(full_ds)

    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng).tolist()

    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val - n_test

    train_idx = perm[:n_train]
    rare_set = rare_lmx_tokens if rare_lmx_tokens is not None else _DEFAULT_RARE_LMX_TOKENS
    train_idx = _train_indices_with_rare_oversample(
        full_ds, train_idx, oversample=rare_lmx_oversample, rare_tokens=rare_set,
    )
    if rare_lmx_oversample > 1 and rare_set:
        log.info(
            "Rare-token oversample: factor=%d tokens=%s → train virtual size %d (unique %d)",
            rare_lmx_oversample, sorted(rare_set), len(train_idx), n_train,
        )
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    train_ds: Dataset = Subset(full_ds, train_idx)
    if online_aug_prob > 0:
        train_ds = _AugSubset(train_ds, online_aug_prob)

    # Append fine-tune (real-domain) samples to the TRAIN set only. Built as a
    # separate OMRDataset so none of its samples can land in val/test.
    ft_dirs = list(finetune_data_dirs or [])
    if ft_dirs:
        ft_scanned = list(finetune_scanned_dirs or [])
        finetune_ds = OMRDataset(
            ft_dirs[0],
            vocab,
            img_height=img_height,
            max_image_width=max_image_width,
            scanned_dir=ft_scanned[0] if ft_scanned else None,
            filter_non_leadsheet_clef=filter_non_leadsheet_clef,
            filter_unusual_time=filter_unusual_time,
            filter_multi_staff=filter_multi_staff,
            max_source_height=max_source_height,
            extra_data_dirs=ft_dirs[1:] or None,
            extra_scanned_dirs=ft_scanned[1:] or None,
        )
        ft_train: Dataset = finetune_ds
        if online_aug_prob > 0:
            ft_train = _AugSubset(
                Subset(finetune_ds, list(range(len(finetune_ds)))),
                online_aug_prob,
            )
        from torch.utils.data import ConcatDataset

        train_ds = ConcatDataset([train_ds, ft_train])
        log.info(
            "Fine-tune: appended %d real samples to TRAIN only (val/test stay synthetic)",
            len(finetune_ds),
        )

    log.info(
        "Split: train_loader=%d (unique ids %d)  val=%d  test=%d",
        len(train_idx),
        n_train,
        n_val,
        n_test,
    )
    return train_ds, Subset(full_ds, val_idx), Subset(full_ds, test_idx)
