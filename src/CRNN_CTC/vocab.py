"""
vocab.py
========
LMX vocabulary management: token ↔ index mapping for the CTC output layer.

Index 0 is reserved for the CTC *blank* symbol.  The remaining indices map
one-to-one to LMX tokens discovered from the training data (or loaded from
a pre-built vocabulary file).

Usage::

    vocab = Vocabulary.from_file("src/CRNN-CTC/vocabulary.txt")
    indices = vocab.encode(["measure", "key:fifths:-3", "clef:G2", "B5", "quarter"])
    tokens  = vocab.decode(indices)
"""

from __future__ import annotations

import logging
import multiprocessing
from itertools import chain
from pathlib import Path

from tqdm import tqdm

log = logging.getLogger(__name__)


class Vocabulary:
    """Bidirectional token ↔ integer mapping with a CTC blank at index 0."""

    BLANK = "<blank>"
    PAD = "<pad>"

    def __init__(self, tokens: list[str]) -> None:
        """
        Parameters
        ----------
        tokens : list[str]
            Ordered list of LMX tokens (without blank/pad — those are added
            automatically at indices 0 and 1).
        """
        self._idx2tok: list[str] = [self.BLANK, self.PAD] + list(tokens)
        self._tok2idx: dict[str, int] = {t: i for i, t in enumerate(self._idx2tok)}

    # -- Properties ---------------------------------------------------------

    @property
    def blank_idx(self) -> int:
        return 0

    @property
    def pad_idx(self) -> int:
        return 1

    def __len__(self) -> int:
        """Total size including blank + pad."""
        return len(self._idx2tok)

    # -- Encode / Decode ----------------------------------------------------

    def encode(self, tokens: list[str]) -> list[int]:
        """Convert a list of LMX token strings to integer indices.

        Unknown tokens are dropped with a warning so that data-quality
        issues surface in the logs instead of being silently hidden.
        """
        indices: list[int] = []
        for t in tokens:
            if t in self._tok2idx:
                indices.append(self._tok2idx[t])
            else:
                log.warning("OOV token dropped during encode: %r", t)
        return indices

    def decode(self, indices: list[int]) -> list[str]:
        """Convert integer indices back to token strings, skipping blank/pad."""
        return [
            self._idx2tok[i]
            for i in indices
            if 0 <= i < len(self._idx2tok) and i not in (self.blank_idx, self.pad_idx)
        ]

    def __contains__(self, token: str) -> bool:
        return token in self._tok2idx

    # -- I/O ----------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> "Vocabulary":
        """Load vocabulary from a text file (one token per line)."""
        path = Path(path)
        tokens = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        return cls(tokens)

    def save(self, path: str | Path) -> None:
        """Save vocabulary to a text file (one token per line, no blank/pad)."""
        path = Path(path)
        # Skip blank and pad (indices 0, 1)
        lines = self._idx2tok[2:]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _read_tokens(lmx_file: Path) -> set[str]:
        text = lmx_file.read_text(encoding="utf-8").strip()
        if text:
            return set(text.split())
        return set()

    @classmethod
    def build_from_lmx_dirs(cls, data_dirs: list[str | Path], workers: int | None = None) -> "Vocabulary":
        """
        Scan all ``.lmx`` files under the given *data_dirs* and build a
        vocabulary from the union of all observed tokens, sorted alphabetically,
        using multiprocessing for speed.
        """
        import os
        if workers is None:
            workers = max(1, (os.cpu_count() or 4) // 2)

        all_lmx_files = []
        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            all_lmx_files.extend(list(data_dir.rglob("*.lmx")))

        token_set: set[str] = set()

        if not all_lmx_files:
            log.warning("No .lmx files found in provided directories.")
            return cls([])

        log.info("Scanning %d .lmx files using %d workers...", len(all_lmx_files), workers)

        if workers <= 1:
            with tqdm(total=len(all_lmx_files), desc="Building Vocab") as pbar:
                for lmx_file in all_lmx_files:
                    token_set.update(cls._read_tokens(lmx_file))
                    pbar.update(1)
        else:
            with multiprocessing.Pool(processes=workers) as pool:
                with tqdm(total=len(all_lmx_files), desc="Building Vocab") as pbar:
                    for batch_tokens in pool.imap_unordered(cls._read_tokens, all_lmx_files, chunksize=100):
                        token_set.update(batch_tokens)
                        pbar.update(1)

        tokens = sorted(token_set)
        return cls(tokens)
