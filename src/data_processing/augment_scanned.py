"""Simple realbook scan simulation.

Read clean lilyjazz PNGs and write distorted copies. Metadata files
(.semantic, .agnostic, .mid) are propagated unchanged.

CLI options mirror generate_realbook.py; see --help for details.
"""

import argparse
import logging
import multiprocessing
import os
import random
import shutil
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)

PAPER_BRIGHT  = 242
PAPER_DARK    = 22
INK_DILATE_ITERATIONS = 2
INK_DILATE_KERNEL     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
VIGNETTE_STRENGTH = 0.22


# Albumentations pipeline (scan-like distortions)

def build_pipeline(seed: int | None = None) -> A.Compose:
    """Return an albumentations Compose pipeline for scan simulation."""
    return A.Compose(
        [
            A.ElasticTransform(alpha=28, sigma=5, p=0.90),
            A.GridDistortion(num_steps=5, distort_limit=(-0.10, 0.10), p=0.80),
            A.Affine(
                translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                scale=(0.98, 1.02),
                rotate=(-3.0, 3.0),
                shear=(-1.5, 1.5),
                border_mode=cv2.BORDER_CONSTANT,
                fill=255,
                p=0.9,
            ),
            A.GaussianBlur(blur_limit=0, sigma_limit=(0.3, 0.8), p=0.70),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.85, 1.0), p=0.55),
            A.RandomToneCurve(scale=0.15, p=0.80),
            A.GaussNoise(std_range=(0.02, 0.06), mean_range=(0.0, 0.0),
                         per_channel=False, p=0.85),
            A.RandomBrightnessContrast(brightness_limit=(-0.10, 0.05),
                                       contrast_limit=(0.05, 0.20),
                                       p=0.90),
        ],
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Individual augmentation steps applied outside albumentations
# ---------------------------------------------------------------------------

def dilate_ink(img_gray: np.ndarray, iterations: int = INK_DILATE_ITERATIONS) -> np.ndarray:
    """Erode to simulate ink bleed."""
    if iterations == 0:
        return img_gray
    return cv2.erode(img_gray, INK_DILATE_KERNEL, iterations=iterations)


def remap_tones(img_gray: np.ndarray) -> np.ndarray:
    """Linear map white/black to PAPER_BRIGHT/DARK."""
    f = img_gray.astype(np.float32) / 255.0
    out = PAPER_DARK + f * (PAPER_BRIGHT - PAPER_DARK)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_vignette(img_gray: np.ndarray, strength: float = VIGNETTE_STRENGTH) -> np.ndarray:
    """Darken corners by a radial mask."""
    h, w = img_gray.shape
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    xn = (X - cx) / cx
    yn = (Y - cy) / cy
    dist = np.sqrt(xn ** 2 + yn ** 2)
    dist_norm = np.clip(dist / 1.414, 0.0, 1.0)
    mask = 1.0 - strength * dist_norm ** 2
    out = np.clip(img_gray.astype(np.float32) * mask, 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Per-sample augmentation
# ---------------------------------------------------------------------------

def augment_sample(
    src_png: Path,
    dst_png: Path,
    pipeline: A.Compose,
    rng: random.Random,
) -> None:
    """Augment a single grayscale PNG and write result."""
    img = np.array(Image.open(src_png).convert("L"))
    img = dilate_ink(img)
    img3 = np.stack([img, img, img], axis=-1)
    result = pipeline(image=img3)["image"]
    img = result[:, :, 0]
    img = remap_tones(img)
    img = add_vignette(img)

    dst_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(dst_png)


# ---------------------------------------------------------------------------
# Top-level worker (must be module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

_LABEL_EXTS = (".semantic", ".agnostic", ".mid", ".lmx")


def _copy_labels(src_dir: Path, out_dir: Path, sample_id: str, out_id: str) -> None:
    """Copy annotation files from source to output, unconditionally."""
    for ext in _LABEL_EXTS:
        src_ann = src_dir / f"{sample_id}{ext}"
        if src_ann.exists():
            shutil.copy2(src_ann, out_dir / f"{out_id}{ext}")


def _validate_source_png(src_png: Path) -> tuple[bool, str]:
    """Validate that a source PNG exists, is non-empty, and decodes cleanly."""
    if not src_png.exists():
        return False, "missing source PNG"
    try:
        if src_png.stat().st_size == 0:
            return False, "empty source PNG (0 bytes)"
        with Image.open(src_png) as img:
            img.load()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        return False, f"unreadable source PNG: {exc}"
    return True, ""


def _worker(args: tuple[Path, Path, int, int]) -> list[tuple[bool, str, str]]:
    """
    Multiprocessing worker function.
    Given (sample_dir, root_out_dir, num_copies, base_seed), generates
    all augmented copies for that sample.
    """
    src_dir, output_dir, copies, seed_base = args
    sample_id = src_dir.name
    src_png = src_dir / f"{sample_id}.png"
    results: list[tuple[bool, str, str]] = []
    for copy_idx in range(copies):
        copy_seed = seed_base + copy_idx
        out_id = sample_id if copies == 1 else f"{sample_id}_aug{copy_idx:02d}"
        out_dir = output_dir / out_id
        out_png = out_dir / f"{out_id}.png"

        if out_png.exists():
            # Image already augmented — just re-sync labels (they may
            # have been regenerated since the last augmentation run).
            _copy_labels(src_dir, out_dir, sample_id, out_id)
            results.append((True, sample_id, ""))
            continue
        try:
            pipeline = build_pipeline(seed=copy_seed)
            augment_sample(src_png, out_png, pipeline, random.Random(copy_seed))
            out_dir.mkdir(parents=True, exist_ok=True)
            _copy_labels(src_dir, out_dir, sample_id, out_id)
            results.append((True, sample_id, ""))
        except Exception as exc:
            log.warning("Augment failed for %s: %s", sample_id, exc)
            results.append((False, sample_id, f"{src_png}: {exc}"))
    return results


# ---------------------------------------------------------------------------
# Standalone label-sync (no image re-augmentation)
# ---------------------------------------------------------------------------

def sync_labels(
    source: Path,
    output: Path,
) -> tuple[int, int]:
    """Copy label files from *source* to every matching sample in *output*.

    Returns ``(synced, skipped)`` counts.
    """
    synced = skipped = 0
    for out_sub in sorted(output.iterdir()):
        if not out_sub.is_dir():
            continue
        out_id = out_sub.name
        # strip _augNN suffix to find the original sample
        base_id = out_id.rsplit("_aug", 1)[0]
        src_sub = source / base_id
        if not src_sub.is_dir():
            skipped += 1
            continue
        _copy_labels(src_sub, out_sub, base_id, out_id)
        synced += 1
    return synced, skipped


def _init_worker(nice_val: int) -> None:
    """Lower the OS scheduling priority of each worker process."""
    try:
        os.nice(nice_val)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply scan-simulation augmentations to the clean realbook_primus dataset."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/realbook_primus"),
        help="Clean dataset root (default: data/realbook_primus)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/realbook_primus_augmented"),
        help="Output dataset root (default: data/realbook_primus_augmented)",
    )
    parser.add_argument(
        "--copies",
        type=int,
        default=1,
        help="Number of augmented versions to create per sample (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed (default: 42)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 2),
        help="Parallel workers (default: cpu_count - 2)",
    )
    parser.add_argument(
        "--maxtasks",
        type=int,
        default=32,
        help="Tasks per worker before it is recycled to free memory (default: 32)",
    )
    parser.add_argument(
        "--nice",
        type=int,
        default=10,
        help="OS nice value for worker processes, 0–19 (default: 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N samples (for testing)",
    )
    parser.add_argument(
        "--sync-labels",
        action="store_true",
        help="Only copy/sync label files (.lmx, .semantic, .agnostic, .mid) "
             "from source to output without re-augmenting images.  Use after "
             "regenerating .lmx files in the source directory.",
    )
    args = parser.parse_args()

    # Configure logging only if the root logger has no handlers yet
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )

    # --- Label-only sync mode (fast, no images) ---
    if args.sync_labels:
        log.info("Syncing labels from %s → %s ...", args.source, args.output)
        synced, skipped = sync_labels(args.source, args.output)
        log.info("Done. Synced: %d  Skipped: %d", synced, skipped)
        return

    candidate_dirs = sorted(
        d for d in args.source.rglob("*")
        if d.is_dir() and not d.name.startswith(".")
        and (d / f"{d.name}.png").exists()
    )

    if not candidate_dirs:
        log.error("No samples found in %s", args.source)
        sys.exit(1)

    if args.limit:
        candidate_dirs = candidate_dirs[: args.limit]

    args.output.mkdir(parents=True, exist_ok=True)

    error_log = args.output / "errors_augment.log"
    if error_log.exists():
        error_log.unlink()

    sample_dirs: list[Path] = []
    invalid_sources: list[tuple[str, Path, str]] = []
    for d in candidate_dirs:
        src_png = d / f"{d.name}.png"
        is_valid, reason = _validate_source_png(src_png)
        if is_valid:
            sample_dirs.append(d)
        else:
            invalid_sources.append((d.name, src_png, reason))

    if invalid_sources:
        with open(error_log, "a") as err_f:
            for sample_id, src_png, reason in invalid_sources:
                err_f.write(f"SKIP_INVALID_SOURCE: {sample_id} - {src_png} - {reason}\n")
        log.warning("Skipping %d unreadable source images.", len(invalid_sources))

    if not sample_dirs:
        log.error("No valid source images left after validation in %s", args.source)
        sys.exit(1)

    log.info("Augmenting %d/%d valid samples × %d cop%s → %s (workers: %d, nice: %d)",
             len(sample_dirs), len(candidate_dirs), args.copies,
             "y" if args.copies == 1 else "ies",
             args.output, args.workers, args.nice)

    work_items = [
        (d, args.output, args.copies, args.seed + i * 1000)
        for i, d in enumerate(sample_dirs)
    ]

    ok = fail = 0
    with multiprocessing.Pool(
        processes=args.workers,
        maxtasksperchild=args.maxtasks,
        initializer=_init_worker,
        initargs=(args.nice,),
    ) as pool:
        with tqdm(total=len(work_items), desc="Augmenting") as pbar:
            for i, results in enumerate(pool.imap(_worker, work_items)):
                for success, sample_id, detail in results:
                    if success:
                        ok += 1
                    else:
                        fail += 1
                        with open(error_log, "a") as err_f:
                            err_f.write(f"FAILED: {sample_id} - {detail}\n")
                pbar.update(1)

    log.info("Done. Success: %d  Failed: %d", ok, fail)
    if fail > 0:
        log.warning(f"See {error_log} for list of failed samples.")


if __name__ == "__main__":
    main()
