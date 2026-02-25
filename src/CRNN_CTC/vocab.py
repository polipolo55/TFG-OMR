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

from pathlib import Path


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
        """Convert a list of LMX token strings to integer indices."""
        return [self._tok2idx[t] for t in tokens if t in self._tok2idx]

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

    @classmethod
    def build_from_lmx_dir(cls, data_dir: str | Path) -> "Vocabulary":
        """
        Scan all ``.lmx`` files under *data_dir* and build a vocabulary from
        the union of all observed tokens, sorted alphabetically.
        """
        data_dir = Path(data_dir)
        token_set: set[str] = set()
        for lmx_file in data_dir.rglob("*.lmx"):
            text = lmx_file.read_text(encoding="utf-8").strip()
            if text:
                token_set.update(text.split())
        tokens = sorted(token_set)
        return cls(tokens)
