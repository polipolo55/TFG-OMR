#!/usr/bin/env python3
"""A/B music recognition on a PDF: greedy vs beam and tiling on/off.

Runs the same staff crops through ``recognize_music`` with different
``beam_width`` settings.  Tiling is **off** by default in inference; this script
compares that mode vs ``OMR_ENABLE_TILING=1`` (legacy multi-tile merge).

  poetry run python scripts/omr_pipeline_ab.py \\
      --pdf data/real_book/some.pdf --checkpoint models/latest/best_model.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
for _p in (_SRC, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> None:
    from omr_pipeline.inference import recognize_music
    from omr_pipeline.preprocess import load_pdf_page, pdf_load_dpi, preprocess_page
    from omr_pipeline.staff_detect import detect_systems

    ap = argparse.ArgumentParser(description="Pipeline decode A/B (beam × tiling).")
    ap.add_argument("--pdf", type=Path, required=True, help="Input PDF path")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/latest/best_model.pt"),
        help="CRNN checkpoint",
    )
    ap.add_argument(
        "--beams",
        type=str,
        default="1,10",
        help="Comma-separated beam widths (default: 1,10)",
    )
    ap.add_argument("--page", type=int, default=0, help="PDF page index (default: 0)")
    args = ap.parse_args()

    pdf = args.pdf
    if not pdf.is_file():
        print(f"PDF not found: {pdf}", file=sys.stderr)
        sys.exit(1)
    ckpt = args.checkpoint
    if not ckpt.is_file():
        print(f"Checkpoint not found: {ckpt}", file=sys.stderr)
        sys.exit(1)

    data = pdf.read_bytes()
    gray = load_pdf_page(data, page=args.page, dpi=pdf_load_dpi())
    page = preprocess_page(gray)
    systems = detect_systems(page.grayscale, page.binary)
    if not systems:
        print("No staff systems detected.", file=sys.stderr)
        sys.exit(2)

    music_imgs = []
    staff_positions = []
    bbox_y0s = []
    music_bins = []
    for sys in systems:
        if sys.music_image is not None and sys.music_image.size > 0:
            music_imgs.append(sys.music_image)
            staff_positions.append(sys.staff.line_ys)
            bbox_y0s.append(sys.music_bbox[1])
            music_bins.append(sys.music_binary)

    beams = [int(x.strip()) for x in args.beams.split(",") if x.strip()]
    if not beams:
        print("No beam widths parsed.", file=sys.stderr)
        sys.exit(1)

    for tiling_on in (False, True):
        if tiling_on:
            os.environ["OMR_ENABLE_TILING"] = "1"
            os.environ.pop("OMR_DISABLE_TILING", None)
        else:
            os.environ.pop("OMR_ENABLE_TILING", None)
            os.environ["OMR_DISABLE_TILING"] = "1"
        mode = "tiling_on" if tiling_on else "tiling_off (default)"
        print(f"\n=== {mode} ===")
        for bw in beams:
            preds = recognize_music(
                music_imgs,
                checkpoint_path=ckpt,
                staff_line_positions=staff_positions,
                music_bbox_y0s=bbox_y0s,
                music_binaries=music_bins,
                beam_width=bw,
            )
            ntok = sum(len(s.split()) for s in preds)
            print(f"  beam={bw:3d}  strips={len(preds)}  total_tokens={ntok}")
            for i, s in enumerate(preds[:3]):
                preview = s[:120] + ("…" if len(s) > 120 else "")
                print(f"    [{i}] {preview}")


if __name__ == "__main__":
    main()
