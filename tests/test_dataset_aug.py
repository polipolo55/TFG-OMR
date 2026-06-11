"""Tests for training-only augmentation isolation in OMRDataset/make_splits."""
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

from CRNN_CTC.dataset import make_splits  # noqa: E402
from CRNN_CTC.vocab import Vocabulary  # noqa: E402

LMX = "clef:G2 key:fifths:0 time beats:4 beat-type:4 pitch:C octave:4 quarter measure"


def _make_corpus(root: Path, n: int = 12) -> None:
    rng = np.random.default_rng(0)
    for i in range(n):
        sid = f"s{i:03d}"
        d = root / sid
        d.mkdir(parents=True)
        img = (rng.random((100, 400)) * 255).astype(np.uint8)
        cv2.imwrite(str(d / f"{sid}.png"), img)
        (d / f"{sid}.lmx").write_text(LMX)


@pytest.fixture()
def corpus(tmp_path):
    data = tmp_path / "clean"
    _make_corpus(data)
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text("\n".join(sorted(set(LMX.split()))))
    return data, Vocabulary.from_file(vocab_file)


def test_val_items_are_deterministic_with_online_aug_enabled(corpus):
    data, vocab = corpus
    _, val_ds, test_ds = make_splits(
        data, vocab, img_height=64, val_frac=0.2, test_frac=0.2,
        filter_multi_staff=False, online_aug_prob=1.0,
    )
    for ds in (val_ds, test_ds):
        a = ds[0]["image"]
        b = ds[0]["image"]
        assert torch.equal(a, b), "val/test item changed between reads — aug leaked"


def test_train_items_are_actually_augmented(corpus):
    data, vocab = corpus
    train_ds, _, _ = make_splits(
        data, vocab, img_height=64, val_frac=0.2, test_frac=0.2,
        filter_multi_staff=False, online_aug_prob=1.0,
    )
    a = train_ds[0]["image"]
    b = train_ds[0]["image"]
    assert not torch.equal(a, b), "train aug (prob=1.0) produced identical reads"


def test_rare_token_oversampling_duplicates_tied_samples(corpus, tmp_path):
    data, _ = corpus
    # Make 2 of the 12 samples contain a tie token
    for sid in ("s000", "s001"):
        f = data / sid / f"{sid}.lmx"
        f.write_text(f.read_text() + " pitch:C octave:4 quarter tied:start")
    vocab_file = tmp_path / "vocab2.txt"
    vocab_file.write_text("\n".join(sorted(set((LMX + " tied:start").split()))))
    vocab = Vocabulary.from_file(vocab_file)

    train_plain, _, _ = make_splits(
        data, vocab, img_height=64, val_frac=0.2, test_frac=0.2,
        filter_multi_staff=False, rare_lmx_oversample=1,
    )
    train_over, _, _ = make_splits(
        data, vocab, img_height=64, val_frac=0.2, test_frac=0.2,
        filter_multi_staff=False, rare_lmx_oversample=2,
        rare_lmx_tokens=frozenset({"tied:start", "tied:stop"}),
    )
    n_tied_in_train = sum(
        1 for i in range(len(train_plain))
        if "tied:start" in train_plain[i]["tokens"]
    )
    assert len(train_over) == len(train_plain) + n_tied_in_train
