#!/usr/bin/env python3
"""Count LMX token frequencies over a recursive dataset tree.

  poetry run python scripts/audit_lmx_corpus.py data/processed/primus/clean
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from CRNN_CTC.dataset import _load_lmx_tokens  # noqa: E402

_FOCUS = (
    "tied:start",
    "tied:stop",
    "sharp",
    "flat",
    "natural",
    "measure",
    "dot",
)


def main() -> None:
    p = argparse.ArgumentParser(description="Audit LMX token counts under a data root.")
    p.add_argument(
        "data_dir",
        type=Path,
        help="Root directory searched recursively for *.lmx",
    )
    p.add_argument(
        "--top",
        type=int,
        default=40,
        help="How many most-frequent tokens to print (default: 40)",
    )
    args = p.parse_args()
    root: Path = args.data_dir
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    counts: Counter[str] = Counter()
    samples_with_tie = 0
    samples_with_accid = 0
    n_lmx = 0

    for lmx_path in sorted(root.rglob("*.lmx")):
        try:
            tokens = _load_lmx_tokens(lmx_path)
        except OSError as exc:
            print(f"Skip {lmx_path}: {exc}", file=sys.stderr)
            continue
        n_lmx += 1
        counts.update(tokens)
        if "tied:start" in tokens or "tied:stop" in tokens:
            samples_with_tie += 1
        if "sharp" in tokens or "flat" in tokens or "natural" in tokens:
            samples_with_accid += 1

    print(f"Files: {n_lmx} .lmx under {root.resolve()}")
    if n_lmx == 0:
        sys.exit(0)

    print(
        f"Samples with tie tokens:     {samples_with_tie} ({100.0 * samples_with_tie / n_lmx:.2f}%)"
    )
    print(
        f"Samples with sharp/flat/nat: {samples_with_accid} ({100.0 * samples_with_accid / n_lmx:.2f}%)"
    )
    print()
    print("Focused token totals (corpus-wide):")
    for tok in _FOCUS:
        print(f"  {tok:16s} {counts.get(tok, 0):>10d}")
    print()
    print(f"Top {args.top} tokens by count:")
    for tok, c in counts.most_common(args.top):
        print(f"  {tok:24s} {c:>10d}")


if __name__ == "__main__":
    main()
