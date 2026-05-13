"""
chord_dataset.py
================
PyTorch Dataset + collation for the chord-recognition CRNN.

Mirrors ``OMRDataset`` but operates on (PNG, *visual chord string*) pairs
discovered from a ``labels.csv`` file rather than ``(PNG, .lmx)`` directory
pairs.  Labels are tokenized **character-by-character**, so the CRNN
predicts arbitrary chord strings without being constrained to a fixed chord
vocabulary.

The character vocabulary is built from the union of characters appearing in
any training label; see :func:`build_chord_vocab`.

Augmentation strategy
---------------------
Synthetic chord crops are clean and uniform — the model would overfit
without aggressive augmentation.  We apply Albumentations *on every training
sample* with:

* Geometric: small rotation / scale / shift / perspective warp.
* Photometric: brightness / contrast jitter, multiplicative noise.
* Quality: motion blur, Gaussian blur, salt-and-pepper noise, downscale.
* Morphological: occasional erosion / dilation to vary stroke thickness.

These together simulate the visual gap between LilyJAZZ renders and real
Real Book scans (paper texture, scan noise, pen variations, slight skew).
"""
from __future__ import annotations

import csv
import logging
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .vocab import Vocabulary

# ---------------------------------------------------------------------------
# Real Book "visual clutter" simulator
# ---------------------------------------------------------------------------
# Real Real Book chord strips contain non-chord visual elements that the
# CRNN never sees in pure LilyJAZZ renders:
#
#   * Slash repeat marks  "/."  scattered between chords.
#   * Staff line bleed at the bottom of the strip (top of the next staff).
#   * Binder hole shadow on the left margin.
#   * Page numbers / annotation text in the corners.
#
# Without exposure to these during training the model decodes them as
# garbage chord characters.  We synthesize them with cv2 primitives and
# overlay them onto chord strips before the Albumentations pipeline so the
# downstream geometric / photometric jitter applies to clutter and chord
# text alike.

def _add_realbook_clutter(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Overlay random Real Book-style clutter onto a chord-strip image."""
    h, w = img.shape
    out = img.copy()

    # Slash repeat marks "/." between chords (40% of samples)
    if rng.random() < 0.40:
        for _ in range(rng.randint(1, 3)):
            cx = rng.randint(w // 6, max(w // 6 + 1, w - w // 6))
            cy = rng.randint(h // 3, 3 * h // 4)
            slash_len = rng.randint(max(6, h // 4), max(8, h // 2))
            thickness = rng.randint(2, 4)
            pt1 = (cx + slash_len // 2, cy - slash_len // 2)
            pt2 = (cx - slash_len // 2, cy + slash_len // 2)
            cv2.line(out, pt1, pt2, 0, thickness, lineType=cv2.LINE_AA)
            # Optional trailing dot (the "." of "/.")
            if rng.random() < 0.7:
                dot_x = cx + slash_len // 2 + rng.randint(3, 8)
                dot_y = cy + slash_len // 2
                if 0 <= dot_x < w and 0 <= dot_y < h:
                    cv2.circle(out, (dot_x, dot_y), rng.randint(1, 3), 0, -1, lineType=cv2.LINE_AA)

    # Staff-line bleed at bottom of strip (30%)
    if rng.random() < 0.30:
        line_y = h - rng.randint(2, max(3, h // 6))
        thickness = rng.randint(1, 3)
        cv2.line(out, (0, line_y), (w, line_y), 0, thickness)
        # Sometimes add a couple more parallel lines (more of the staff visible)
        if rng.random() < 0.4:
            for k in range(1, rng.randint(2, 4)):
                yy = line_y + k * rng.randint(3, 8)
                if 0 <= yy < h:
                    cv2.line(out, (0, yy), (w, yy), 0, max(1, thickness - 1))

    # Binder-hole shadow on the left margin (15%)
    if rng.random() < 0.15:
        hole_w = rng.randint(8, max(10, w // 30))
        hole_top = rng.randint(0, max(1, h // 3))
        hole_bot = rng.randint(2 * h // 3, h)
        cv2.rectangle(out, (0, hole_top), (hole_w, hole_bot), 0, -1)

    # Top-of-strip staff bleed — top edge of the previous staff's bottom (15%)
    if rng.random() < 0.15:
        top_y = rng.randint(1, max(2, h // 6))
        cv2.line(out, (0, top_y), (w, top_y), 0, rng.randint(1, 2))

    return out

log = logging.getLogger(__name__)

# Default image height for the chord CRNN.  Chord text is simpler than music
# notation so 64 px is enough to resolve every superscript and accidental;
# half the music CRNN's 128 px → ~4× faster inference.
DEFAULT_IMG_HEIGHT = 64


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def build_chord_vocab(label_files: list[Path]) -> Vocabulary:
    """Scan labels CSVs and build a character-level chord vocabulary.

    Every unique character (including spaces between chords) becomes one
    token.  Indices 0/1/2 are reserved for ``<blank>``/``<pad>``/``<unk>``
    by the ``Vocabulary`` class.
    """
    chars: set[str] = set()
    for path in label_files:
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) < 2:
                    continue
                chars.update(row[1])
    if not chars:
        raise RuntimeError(f"No characters found in {label_files}")
    return Vocabulary(sorted(chars))


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_chord_image(
    path: Path,
    img_height: int,
    max_width: int = 0,
) -> np.ndarray:
    """Load grayscale PNG, resize to ``img_height``, return float32 in [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    h, w = img.shape
    new_w = max(1, round(w * img_height / h))
    if max_width > 0 and new_w > max_width:
        new_w = max_width
    img = cv2.resize(img, (new_w, img_height), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
# We instantiate the Albumentations pipeline lazily so importing this module
# does not require Albumentations (e.g. during inference-only deployments).

_AUG_PIPELINE = None


def _get_aug_pipeline():
    """Return a cached Albumentations pipeline for chord-strip augmentation."""
    global _AUG_PIPELINE
    if _AUG_PIPELINE is not None:
        return _AUG_PIPELINE

    import albumentations as A

    _AUG_PIPELINE = A.Compose([
        # Geometric — preserve chord legibility (small distortions only)
        A.Affine(
            scale=(0.92, 1.08),
            translate_percent=(-0.02, 0.02),
            rotate=(-3, 3),
            shear=(-3, 3),
            border_mode=cv2.BORDER_CONSTANT,
            fill=255,
            p=0.7,
        ),
        A.Perspective(
            scale=(0.01, 0.04),
            border_mode=cv2.BORDER_CONSTANT,
            fill=255,
            p=0.3,
        ),

        # Photometric — variations in scan exposure
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.7),

        # Blur / noise — simulate scan resolution and ink bleed
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
            A.Downscale(scale_range=(0.5, 0.85), p=1.0),
        ], p=0.4),
        A.GaussNoise(std_range=(0.02, 0.12), p=0.4),

        # Morphological — varies pen stroke thickness
        A.OneOf([
            A.Morphological(scale=(1, 2), operation="dilation", p=1.0),
            A.Morphological(scale=(1, 2), operation="erosion", p=1.0),
        ], p=0.3),
    ])
    return _AUG_PIPELINE


def _apply_augmentation(img_u8: np.ndarray) -> np.ndarray:
    """Apply Albumentations pipeline; expects uint8 grayscale (H, W)."""
    pipe = _get_aug_pipeline()
    out = pipe(image=img_u8)["image"]
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChordDataset(Dataset):
    """Synthetic Real Book chord-strip dataset.

    Parameters
    ----------
    image_dir
        Directory containing the chord-strip PNGs.
    labels_csv
        CSV with header ``filename,label``.
    vocab
        Character-level :class:`Vocabulary`.
    img_height
        Target image height; widths scale proportionally.
    max_image_width
        Optional safety cap on per-sample width (prevents OOM on outliers).
    augment
        If *True*, apply the Albumentations pipeline at ``__getitem__``.
        Disabled on validation/test splits.
    """

    def __init__(
        self,
        image_dir: Path | str,
        labels_csv: Path | str,
        vocab: Vocabulary,
        *,
        img_height: int = DEFAULT_IMG_HEIGHT,
        max_image_width: int = 2048,
        augment: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.vocab = vocab
        self.img_height = img_height
        self.max_image_width = max_image_width
        self.augment = augment

        self._samples: list[tuple[str, str]] = []
        with open(labels_csv, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                filename, label = row[0], row[1]
                if (self.image_dir / filename).exists():
                    self._samples.append((filename, label))

        if not self._samples:
            raise RuntimeError(
                f"No usable samples in {labels_csv} (image_dir={self.image_dir})"
            )

        log.info(
            "ChordDataset: %d samples from %s (height=%d, augment=%s)",
            len(self._samples), labels_csv, img_height, augment,
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor | str]:
        filename, label = self._samples[idx]
        img = _load_chord_image(
            self.image_dir / filename,
            self.img_height,
            self.max_image_width,
        )

        if self.augment:
            img_u8 = (img * 255).astype(np.uint8)
            # Real Book visual clutter FIRST (before Albumentations) so the
            # downstream geometric/photometric jitter applies to clutter too.
            img_u8 = _add_realbook_clutter(img_u8, random)
            img_u8 = _apply_augmentation(img_u8)
            img = img_u8.astype(np.float32) / 255.0

        # Normalise to zero-mean, unit-variance (matches OMRDataset)
        img = (img - img.mean()) / (img.std() + 1e-6)
        img_t = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)

        # Character-level label encoding
        label_indices = self.vocab.encode(list(label))

        return {
            "sample_id": filename,
            "image": img_t,
            "label": torch.tensor(label_indices, dtype=torch.long),
            "tokens": list(label),  # raw chars for debug
        }


# Re-export the existing collate function — it operates on the dict layout
# above (which matches OMRDataset's) so no chord-specific changes needed.
from .dataset import collate_fn  # noqa: E402  (intentional re-export)


__all__ = [
    "ChordDataset",
    "DEFAULT_IMG_HEIGHT",
    "build_chord_vocab",
    "collate_fn",
]
