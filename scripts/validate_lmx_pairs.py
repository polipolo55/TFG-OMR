#!/usr/bin/env python3
"""Validate a directory tree for fine-tuning: each sample folder needs PNG + LMX.

Place hand-labelled Real Book (or other) crops under e.g. data/real_book_ft/
as::

    {sample_id}/{sample_id}.png
    {sample_id}/{sample_id}.lmx

Then train with::

  poetry run python src/cli.py train \\
      --finetune-data-dir data/real_book_ft \\
      ... other flags ...

  poetry run python src/cli.py train --use-scanned \\
      --finetune-data-dir data/real_book_ft_clean \\
      --finetune-scanned-dir data/real_book_ft_scanned \\
      ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Count valid PNG+LMX sample folders for fine-tuning.",
    )
    ap.add_argument("root", type=Path, help="Dataset root (nested sample dirs)")
    args = ap.parse_args()
    root: Path = args.root
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    ok = missing_png = missing_lmx = 0
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        sid = d.name
        png = d / f"{sid}.png"
        lmx = d / f"{sid}.lmx"
        if not png.is_file():
            missing_png += 1
            continue
        if not lmx.is_file():
            missing_lmx += 1
            continue
        ok += 1

    print(f"Root: {root.resolve()}")
    print(f"Valid samples (PNG+LMX): {ok}")
    print(f"Missing PNG: {missing_png}  Missing LMX: {missing_lmx}")
    if ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
