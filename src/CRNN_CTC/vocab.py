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
    """Bidirectional token ↔ integer mapping with a CTC blank at index 0.

    Layout::

        index 0: <blank>  (CTC blank)
        index 1: <pad>    (sequence padding)
        index 2: <unk>    (unknown / OOV fallback)
        index 3…: music tokens (sorted alphabetically)
    """

    BLANK = "<blank>"
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, tokens: list[str]) -> None:
        """
        Parameters
        ----------
        tokens : list[str]
            Ordered list of LMX tokens (without blank/pad/unk — those are
            added automatically at indices 0, 1, 2).
        """
        # Strip special tokens if accidentally present in the input list
        cleaned = [t for t in tokens if t not in (self.BLANK, self.PAD, self.UNK)]
        self._idx2tok: list[str] = [self.BLANK, self.PAD, self.UNK] + list(cleaned)
        self._tok2idx: dict[str, int] = {t: i for i, t in enumerate(self._idx2tok)}

    # -- Properties ---------------------------------------------------------

    @property
    def blank_idx(self) -> int:
        return 0

    @property
    def pad_idx(self) -> int:
        return 1

    @property
    def unk_idx(self) -> int:
        return 2

    def __len__(self) -> int:
        """Total size including blank + pad + unk."""
        return len(self._idx2tok)

    # -- Encode / Decode ----------------------------------------------------

    def encode(self, tokens: list[str]) -> list[int]:
        """Convert a list of LMX token strings to integer indices.

        Unknown tokens are mapped to the ``<unk>`` index so that
        OOV symbols are preserved in the label sequence rather than
        silently dropped.
        """
        indices: list[int] = []
        for t in tokens:
            idx = self._tok2idx.get(t)
            if idx is not None:
                indices.append(idx)
            else:
                log.warning("OOV token mapped to <unk>: %r", t)
                indices.append(self.unk_idx)
        return indices

    def decode(self, indices: list[int]) -> list[str]:
        """Convert integer indices back to token strings, skipping blank/pad."""
        _skip = (self.blank_idx, self.pad_idx)
        return [
            self._idx2tok[i]
            for i in indices
            if 0 <= i < len(self._idx2tok) and i not in _skip
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
        """Save vocabulary to a text file (one token per line, no blank/pad/unk)."""
        path = Path(path)
        # Skip blank, pad, and unk (indices 0, 1, 2)
        lines = self._idx2tok[3:]
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

        Ensures the full ``pitch:A``–``pitch:G`` and ``octave:0``–``octave:8``
        ranges are present even if not all combinations appear in the data,
        so the model can generalise to unseen pitch/octave combinations.
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
            raise RuntimeError("No .lmx files found in provided directories.")

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

        # Ensure full pitch/octave ranges for OOV robustness
        for step in "ABCDEFG":
            token_set.add(f"pitch:{step}")
        for octave in range(0, 9):
            token_set.add(f"octave:{octave}")

        tokens = sorted(token_set)
        return cls(tokens)
