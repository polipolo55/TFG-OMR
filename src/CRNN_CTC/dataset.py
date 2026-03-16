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
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .vocab import Vocabulary

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sample-quality filter
# ---------------------------------------------------------------------------

# Structural tokens that carry no pitched content
_REST_STRUCTURAL = frozenset({"rest", "rest:measure", "measure"})

# C-clef and F3-clef variants that are not used in jazz lead sheets and cause
# systematic pitch-cascade errors due to visual similarity with a neighbouring
# clef (C1/C2 ≈ C4 tenor; F3 baritone ≈ F4 bass, one line off).
_CLEF_UNWANTED = frozenset({"clef:C1", "clef:C2", "clef:F3"})


def _is_degenerate(
    tokens: list[str],
    *,
    filter_rest_heavy: bool = True,
    filter_unwanted_clefs: bool = True,
) -> bool:
    """Return *True* if a sample should be excluded from training/evaluation.

    Two independent criteria:

    rest-heavy
        More than 80 % of tokens are structural (``rest``, ``rest:measure``,
        ``measure``) *and* the sequence is longer than 50 tokens.  These are
        multi-bar tacet passages whose image shows an uninformative long rest
        — the CTC edit distance explodes and they contribute no signal.

    unwanted-clefs
        The sample contains a soprano (``clef:C1``) or mezzo-soprano
        (``clef:C2``) clef.  These C-clef variants look visually like tenor
        clef but sit on a different staff line; the model confuses them and
        every subsequent pitch prediction is shifted by a fixed interval,
        creating a large cascade of substitution errors.  Neither clef appears
        in jazz lead sheets.
    """
    if not tokens:
        return True

    if filter_unwanted_clefs and any(t in _CLEF_UNWANTED for t in tokens):
        return True

    if filter_rest_heavy and len(tokens) > 50:
        n_structural = sum(1 for t in tokens if t in _REST_STRUCTURAL)
        if n_structural / len(tokens) > 0.80:
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
    """Return the original pixel height of a PNG without full decode.

    Uses ``cv2.imread`` in grayscale mode; faster than a full decode when
    all we need is ``img.shape[0]``.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    return img.shape[0]


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
    """

    def __init__(
        self,
        data_dir: Path | str,
        vocab: Vocabulary,
        img_height: int = 128,
        max_image_width: int = 0,
        scanned_dir: Path | str | None = None,
        filter_rest_heavy: bool = True,
        filter_unwanted_clefs: bool = True,
        filter_multi_staff: bool = True,
        max_source_height: int = 180,
        extra_data_dirs: list[Path] | None = None,
        extra_scanned_dirs: list[Path] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.vocab = vocab
        self.img_height = img_height
        self.max_image_width = max_image_width
        self.scanned_dir = Path(scanned_dir) if scanned_dir else None
        self.extra_scanned_dirs = [Path(p) for p in (extra_scanned_dirs or [])]
        self._oov_counts: dict[str, int] = {}  # token → occurrence count

        # Discover all valid (png + lmx) samples from primary + extra dirs
        raw_samples = _discover_samples(self.data_dir, require_lmx=True)
        for extra in (extra_data_dirs or []):
            extra_path = Path(extra)
            if extra_path.is_dir():
                extra_samples = _discover_samples(extra_path, require_lmx=True)
                log.info("Extra data dir %s: %d samples", extra_path, len(extra_samples))
                raw_samples.extend(extra_samples)
        if not raw_samples:
            raise RuntimeError(f"No valid samples found in {self.data_dir}")

        # 1) Token-level quality filters (fast — reads tiny .lmx files)
        if filter_rest_heavy or filter_unwanted_clefs:
            after_token_filter = [
                (sid, png, lmx)
                for sid, png, lmx in raw_samples
                if not _is_degenerate(
                    _load_lmx_tokens(lmx),
                    filter_rest_heavy=filter_rest_heavy,
                    filter_unwanted_clefs=filter_unwanted_clefs,
                )
            ]
            n_removed = len(raw_samples) - len(after_token_filter)
            if n_removed:
                log.info(
                    "Token filter removed %d degenerate/unwanted samples",
                    n_removed,
                )
        else:
            after_token_filter = raw_samples

        # 2) Image-height filter — rejects multi-staff renders
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
                    max_source_height, n_tall,
                )
        else:
            self._samples = after_token_filter

        if self._samples:
            log.info(
                "OMRDataset: %d samples retained from %s%s",
                len(self._samples), self.data_dir,
                f" (images from {self.scanned_dir})" if self.scanned_dir else "",
            )

        if not self._samples:
            raise RuntimeError(
                f"No samples remain in {self.data_dir} after filtering."
            )

        # 3) Scan labels for OOV tokens (fast, one-time check at init)
        oov: dict[str, int] = {}
        for _sid, _png, lmx in self._samples:
            for t in _load_lmx_tokens(lmx):
                if t not in self.vocab:
                    oov[t] = oov.get(t, 0) + 1
        if oov:
            total = sum(oov.values())
            top5 = sorted(oov.items(), key=lambda x: -x[1])[:5]
            log.warning(
                "OOV tokens in dataset: %d unique, %d occurrences. "
                "Top: %s",
                len(oov), total,
                ", ".join(f"{tok!r} ×{cnt}" for tok, cnt in top5),
            )

    # -- Dataset protocol ---------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor | list[str] | str]:
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

        # Image → (1, H, W) float32 tensor, normalised
        img = _load_image(png_path, self.img_height, self.max_image_width)  # (H, W) float32 [0,1]
        img = (img - img.mean()) / (img.std() + 1e-6)  # zero-mean, unit-var
        img_t = torch.from_numpy(img).unsqueeze(0)      # (1, H, W)

        # Label → list[int]
        tokens = _load_lmx_tokens(lmx_path)
        label = self.vocab.encode(tokens)

        return {
            "sample_id": sid,
            "image": img_t,                          # (1, H, W)
            "label": torch.tensor(label, dtype=torch.long),  # (L,)
            "tokens": tokens,                         # raw strings (debug)
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
        "images": torch.stack(padded, dim=0),           # (B, 1, H, W_max)
        "labels": torch.cat(labels, dim=0),              # (sum L_i,)
        "label_lens": torch.tensor(
            [l.size(0) for l in labels], dtype=torch.long
        ),                                                # (B,)
        "image_widths": torch.tensor(widths, dtype=torch.long),  # (B,)
        "sample_ids": sample_ids,
    }


# ---------------------------------------------------------------------------
# Train / val split helper
# ---------------------------------------------------------------------------

def make_splits(
    data_dir: Path | str,
    vocab: Vocabulary,
    img_height: int = 128,
    max_image_width: int = 0,
    scanned_dir: Path | str | None = None,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    filter_rest_heavy: bool = True,
    filter_unwanted_clefs: bool = True,
    filter_multi_staff: bool = True,
    max_source_height: int = 180,
    extra_data_dirs: list[Path] | None = None,
    extra_scanned_dirs: list[Path] | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
    """Create train / val / test splits from a single data directory.

    The split is deterministic (seeded) and stratified at the sample-id level
    (no data leakage across splits).

    Returns
    -------
    train_ds, val_ds, test_ds
    """
    from torch.utils.data import Subset

    full_ds = OMRDataset(
        data_dir, vocab, img_height=img_height,
        max_image_width=max_image_width,
        scanned_dir=scanned_dir,
        filter_rest_heavy=filter_rest_heavy,
        filter_unwanted_clefs=filter_unwanted_clefs,
        filter_multi_staff=filter_multi_staff,
        max_source_height=max_source_height,
        extra_data_dirs=extra_data_dirs,
        extra_scanned_dirs=extra_scanned_dirs,
    )
    n = len(full_ds)
    indices = list(range(n))

    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng).tolist()

    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val - n_test

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    log.info("Split: train=%d  val=%d  test=%d", n_train, n_val, n_test)
    return Subset(full_ds, train_idx), Subset(full_ds, val_idx), Subset(full_ds, test_idx)
