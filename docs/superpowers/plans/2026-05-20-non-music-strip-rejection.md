# Non-Music Strip Rejection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the OMR pipeline from returning hallucinated music for non-music page regions (titles, footers) by inserting a hybrid pre-CRNN + post-CRNN rejection gate.

**Architecture:** New module `src/omr_pipeline/staff_reject.py` exposing `evaluate_pre_crnn` and `evaluate_post_crnn`. Geometric gates (line span, spacing CoV, inter-line ink) and an OCR text-density gate run before CRNN. CTC mean-log-prob gate runs after. Geometry-impossible strips are dropped entirely; OCR/CTC failures keep their bbox with a `rejected` reason and empty tokens. Thresholds come from a calibration CLI (`calibrate-reject`) that sweeps a labeled fixture set.

**Tech Stack:** Python 3.14, Poetry, NumPy, OpenCV, PyTorch, EasyOCR (re-pinned), pytest (newly introduced for this feature).

**Spec:** `docs/superpowers/specs/2026-05-20-non-music-strip-rejection-design.md`

---

## File Structure

| File | Status | Responsibility |
|------|--------|----------------|
| `src/omr_pipeline/staff_reject.py` | **new** | All rejection logic: signal computations, thresholds, evaluate functions |
| `src/omr_pipeline/staff_detect.py` | modify | Call `evaluate_pre_crnn` in `detect_systems`, drop geometry-rejected strips, store `pre_result` on `System` |
| `src/omr_pipeline/inference.py` | modify | `recognize_music` returns `(tokens, log_probs, out_lens)` |
| `src/omr_pipeline/pipeline.py` | modify | Wire `evaluate_post_crnn`, populate `rejected` + `reject_diagnostics`, add `num_rejected` meta |
| `src/cli.py` | modify | Add `calibrate-reject` and `harvest-reject-fixtures` subcommands |
| `pyproject.toml` | modify | Pin `easyocr` and `pytest` |
| `docs/inference_pipeline.md` | modify | Document the new gate layer |
| `docs/api.md` | modify | Document `rejected` and `reject_diagnostics` fields |
| `docs/cli.md` | modify | Document the two new CLI subcommands |
| `CLAUDE.md` | modify | One-line note: re-run calibration after retraining the CRNN |
| `tests/__init__.py` | **new** | empty |
| `tests/test_staff_reject.py` | **new** | Unit tests for every gate function |
| `tests/integration/__init__.py` | **new** | empty |
| `tests/integration/test_pipeline_rejects_titles.py` | **new** | End-to-end on Satin Doll page |
| `tests/fixtures/pages/satin_doll.png` | **new** | Single-page render of Satin Doll for the integration test |
| `models/staff_reject/thresholds.json` | **new** | Calibrated thresholds output by the CLI |

---

## Task 1: Add `easyocr` and `pytest` to dependencies, set up `tests/` package

**Files:**
- Modify: `pyproject.toml` (dependencies block, dev dependencies block)
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`

- [ ] **Step 1: Edit `pyproject.toml` to add `easyocr` to `dependencies` and `pytest` to dev dependencies**

In `[project]` `dependencies`, append:
```toml
    "easyocr (>=1.7.2,<2.0.0)",
```

In `[dependency-groups]` `dev`, append:
```toml
    "pytest (>=8.3.0,<9.0.0)",
    "pytest-mock (>=3.14.0,<4.0.0)",
```

- [ ] **Step 2: Run `poetry lock` and `poetry install --with dev`**

Run:
```bash
poetry lock --no-update
poetry install --with dev
```

Expected: both succeed; `poetry run pytest --version` prints a version.

- [ ] **Step 3: Create empty `tests/__init__.py`**

```python
```

- [ ] **Step 4: Create `tests/conftest.py` with a project-root sys.path hook**

```python
"""Make `src.` imports work from tests without installing the project."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
```

- [ ] **Step 5: Smoke test — pytest discovers no tests yet but does not error**

Run: `poetry run pytest tests/ -q`
Expected: exit code 5 ("no tests ran") OR exit code 0 with "no tests ran" message. Either is fine.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml poetry.lock tests/__init__.py tests/conftest.py
git commit -m "test: scaffold pytest, pin easyocr and pytest deps"
```

---

## Task 2: Scaffold `staff_reject.py` with dataclasses and threshold loader

**Files:**
- Create: `src/omr_pipeline/staff_reject.py`
- Create: `tests/test_staff_reject.py`

- [ ] **Step 1: Write failing test `test_default_thresholds_load` and `test_thresholds_json_overrides`**

Create `tests/test_staff_reject.py`:
```python
"""Unit tests for src/omr_pipeline/staff_reject.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_default_thresholds_load(monkeypatch):
    from omr_pipeline.staff_reject import DEFAULT_THRESHOLDS, load_thresholds

    monkeypatch.delenv("OMR_REJECT_THRESHOLDS", raising=False)
    assert load_thresholds() == DEFAULT_THRESHOLDS


def test_thresholds_json_overrides(monkeypatch, tmp_path):
    from omr_pipeline.staff_reject import load_thresholds

    cfg = tmp_path / "t.json"
    cfg.write_text(json.dumps({
        "min_line_span_frac": 0.85,
        "max_text_area_frac": 0.20,
    }))
    monkeypatch.setenv("OMR_REJECT_THRESHOLDS", str(cfg))
    th = load_thresholds()
    assert th.min_line_span_frac == 0.85
    assert th.max_text_area_frac == 0.20
    # untouched fields keep defaults
    assert th.min_mean_logprob == -1.2


def test_thresholds_missing_file_falls_back(monkeypatch, tmp_path):
    from omr_pipeline.staff_reject import DEFAULT_THRESHOLDS, load_thresholds

    monkeypatch.setenv("OMR_REJECT_THRESHOLDS", str(tmp_path / "does_not_exist.json"))
    assert load_thresholds() == DEFAULT_THRESHOLDS
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `poetry run pytest tests/test_staff_reject.py -v`
Expected: ImportError or ModuleNotFoundError on `omr_pipeline.staff_reject`.

- [ ] **Step 3: Create the module with dataclasses and loader**

Create `src/omr_pipeline/staff_reject.py`:
```python
"""Non-music strip rejection.

Three gates filter out detected "staves" that aren't real music:

* Geometric gates (line span, spacing coefficient of variation, inter-line ink
  density) — run on the binary strip before the CRNN. Geometry-impossible
  strips are dropped from the pipeline entirely.
* OCR text-area gate — runs EasyOCR's text detector on the grayscale strip.
  Strips whose text area exceeds a fraction threshold are flagged "rejected"
  but kept (bbox preserved, tokens emptied).
* CTC mean-log-prob gate — runs on the CRNN output. Low confidence is a
  catch-all for hallucinations that slipped past pre-CRNN gates.

Thresholds come from the ``calibrate-reject`` CLI subcommand; see
``docs/superpowers/specs/2026-05-20-non-music-strip-rejection-design.md``.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RejectThresholds:
    min_line_span_frac: float = 0.70
    max_spacing_cov: float = 0.18
    min_interline_ink_frac: float = 0.005
    max_text_area_frac: float = 0.35
    min_mean_logprob: float = -1.2


DEFAULT_THRESHOLDS = RejectThresholds()


@dataclass
class RejectionResult:
    passed: bool
    reason: str | None
    diagnostics: dict[str, float] = field(default_factory=dict)


def load_thresholds() -> RejectThresholds:
    """Read thresholds from ``$OMR_REJECT_THRESHOLDS`` (JSON path) or fall back to defaults."""
    path = os.environ.get("OMR_REJECT_THRESHOLDS", "").strip()
    if not path:
        return DEFAULT_THRESHOLDS
    try:
        raw = json.loads(open(path, "r", encoding="utf-8").read())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("OMR_REJECT_THRESHOLDS=%r unreadable (%s); using defaults", path, exc)
        return DEFAULT_THRESHOLDS
    known = {f.name for f in fields(RejectThresholds)}
    merged = {**asdict(DEFAULT_THRESHOLDS), **{k: float(v) for k, v in raw.items() if k in known}}
    return RejectThresholds(**merged)
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `poetry run pytest tests/test_staff_reject.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/omr_pipeline/staff_reject.py tests/test_staff_reject.py
git commit -m "feat(reject): scaffold RejectThresholds and load_thresholds"
```

---

## Task 3: Geometric signal functions (line span, spacing CoV, inter-line ink)

**Files:**
- Modify: `src/omr_pipeline/staff_reject.py`
- Modify: `tests/test_staff_reject.py`

- [ ] **Step 1: Append failing tests for the three private signal functions**

Append to `tests/test_staff_reject.py`:
```python
import numpy as np


def _staff_binary(h=64, w=300, line_ys=(8, 20, 32, 44, 56), line_span=1.0, interline_ink=False):
    """Build a synthetic binary staff strip.

    line_span: fraction of x columns that have ink on each line (1.0 = full span).
    interline_ink: if True, add a notehead blob between lines 2 and 3.
    """
    img = np.zeros((h, w), dtype=np.uint8)
    span_px = int(w * line_span)
    for y in line_ys:
        img[y, :span_px] = 1
    if interline_ink:
        img[24:30, 100:108] = 1   # 6x8 notehead
    return img


def test_line_span_min_full_lines():
    from omr_pipeline.staff_reject import _line_span_min

    img = _staff_binary(line_span=1.0)
    assert _line_span_min(img, [8, 20, 32, 44, 56]) == pytest.approx(1.0, abs=1e-3)


def test_line_span_min_short_lines():
    from omr_pipeline.staff_reject import _line_span_min

    img = _staff_binary(line_span=0.40)
    assert _line_span_min(img, [8, 20, 32, 44, 56]) == pytest.approx(0.40, abs=1e-2)


def test_spacing_cov_uniform():
    from omr_pipeline.staff_reject import _spacing_cov

    assert _spacing_cov([8, 20, 32, 44, 56]) == pytest.approx(0.0, abs=1e-6)


def test_spacing_cov_irregular():
    from omr_pipeline.staff_reject import _spacing_cov

    # gaps [5, 12, 12, 12] — first gap is half the others
    cov = _spacing_cov([0, 5, 17, 29, 41])
    assert cov > 0.20


def test_interline_ink_frac_empty():
    from omr_pipeline.staff_reject import _interline_ink_frac

    img = _staff_binary(line_span=1.0, interline_ink=False)
    assert _interline_ink_frac(img, [8, 20, 32, 44, 56]) < 0.001


def test_interline_ink_frac_with_notehead():
    from omr_pipeline.staff_reject import _interline_ink_frac

    img = _staff_binary(line_span=1.0, interline_ink=True)
    assert _interline_ink_frac(img, [8, 20, 32, 44, 56]) > 0.001
```

- [ ] **Step 2: Run the new tests to confirm they fail**

Run: `poetry run pytest tests/test_staff_reject.py -v -k "line_span or spacing_cov or interline_ink"`
Expected: 6 failures with ImportError on `_line_span_min` / `_spacing_cov` / `_interline_ink_frac`.

- [ ] **Step 3: Implement the three private functions in `staff_reject.py`**

Add to `src/omr_pipeline/staff_reject.py` (after the dataclass section):
```python
# ---------------------------------------------------------------------------
# Geometric signals
# ---------------------------------------------------------------------------

import numpy as np


def _line_span_min(binary: np.ndarray, line_ys: list[int]) -> float:
    """Minimum across the 5 lines of the fraction of columns with ink on that line.

    Each line uses a ±1-row tolerance band to cope with sub-pixel detection drift.
    """
    if binary.size == 0 or not line_ys:
        return 0.0
    h, w = binary.shape[:2]
    if w == 0:
        return 0.0
    spans: list[float] = []
    for y in line_ys:
        y0 = max(0, y - 1)
        y1 = min(h, y + 2)
        band = binary[y0:y1]
        if band.size == 0:
            spans.append(0.0)
            continue
        has_ink = np.any(band > 0, axis=0)
        spans.append(float(has_ink.sum()) / float(w))
    return min(spans) if spans else 0.0


def _spacing_cov(line_ys: list[int]) -> float:
    """Coefficient of variation of inter-line gaps (std / mean)."""
    if len(line_ys) < 2:
        return float("inf")
    gaps = np.diff(np.asarray(line_ys, dtype=np.float64))
    mean = float(gaps.mean())
    if mean <= 0:
        return float("inf")
    return float(gaps.std()) / mean


def _interline_ink_frac(binary: np.ndarray, line_ys: list[int]) -> float:
    """Fraction of pixels in the inter-line region that are ink.

    Excludes ±1-row bands around every line (so the lines themselves don't count).
    """
    if binary.size == 0 or len(line_ys) < 5:
        return 0.0
    h, w = binary.shape[:2]
    y_top = max(0, line_ys[0] + 1)
    y_bot = min(h, line_ys[-1])
    if y_bot - y_top <= 0:
        return 0.0
    band = binary[y_top:y_bot].astype(bool)
    mask = np.ones_like(band, dtype=bool)
    for y in line_ys:
        lo = max(0, y - 1 - y_top)
        hi = min(band.shape[0], y + 2 - y_top)
        if hi > lo:
            mask[lo:hi] = False
    inter = band & mask
    denom = int(mask.sum())
    if denom == 0:
        return 0.0
    return float(inter.sum()) / float(denom)
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `poetry run pytest tests/test_staff_reject.py -v -k "line_span or spacing_cov or interline_ink"`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/omr_pipeline/staff_reject.py tests/test_staff_reject.py
git commit -m "feat(reject): geometric signals — line span, spacing CoV, interline ink"
```

---

## Task 4: OCR text-density signal with lazy singleton reader

**Files:**
- Modify: `src/omr_pipeline/staff_reject.py`
- Modify: `tests/test_staff_reject.py`

- [ ] **Step 1: Append failing tests for `_text_area_frac` using a mocked reader**

Append to `tests/test_staff_reject.py`:
```python
def test_text_area_frac_no_text(mocker):
    from omr_pipeline import staff_reject

    fake_reader = mocker.Mock()
    fake_reader.detect.return_value = ([[]], [[]])     # no horizontal, no free-form boxes
    mocker.patch.object(staff_reject, "_get_ocr_reader", return_value=fake_reader)

    gray = np.full((64, 300), 255, dtype=np.uint8)
    assert staff_reject._text_area_frac(gray) == 0.0


def test_text_area_frac_with_text(mocker):
    from omr_pipeline import staff_reject

    # EasyOCR detect() returns (horizontal_list, free_list).
    # horizontal_list[0] is a list of [x_min, x_max, y_min, y_max] boxes.
    boxes = [[0, 100, 0, 50], [120, 220, 10, 60]]   # area = 5000 + 5000 = 10000
    fake_reader = mocker.Mock()
    fake_reader.detect.return_value = ([boxes], [[]])
    mocker.patch.object(staff_reject, "_get_ocr_reader", return_value=fake_reader)

    gray = np.full((100, 300), 255, dtype=np.uint8)  # area = 30000
    frac = staff_reject._text_area_frac(gray)
    assert frac == pytest.approx(10000 / 30000, abs=1e-3)


def test_text_area_frac_empty_strip(mocker):
    from omr_pipeline import staff_reject

    fake_reader = mocker.Mock()
    mocker.patch.object(staff_reject, "_get_ocr_reader", return_value=fake_reader)

    assert staff_reject._text_area_frac(np.zeros((0, 0), dtype=np.uint8)) == 0.0
    fake_reader.detect.assert_not_called()
```

- [ ] **Step 2: Run the new tests to confirm they fail**

Run: `poetry run pytest tests/test_staff_reject.py -v -k "text_area_frac"`
Expected: 3 failures with AttributeError on `_text_area_frac` / `_get_ocr_reader`.

- [ ] **Step 3: Implement `_get_ocr_reader` and `_text_area_frac`**

Append to `src/omr_pipeline/staff_reject.py`:
```python
# ---------------------------------------------------------------------------
# OCR text signal
# ---------------------------------------------------------------------------

_OCR_READER = None


def _get_ocr_reader():
    """Lazy-load a single EasyOCR Reader; cached for the lifetime of the process."""
    global _OCR_READER
    if _OCR_READER is None:
        import easyocr   # lazy import — heavy
        _OCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _OCR_READER


def _text_area_frac(grayscale: np.ndarray) -> float:
    """Fraction of strip area covered by EasyOCR text-detector bounding boxes.

    Detector-only (no recognition) is roughly 5–10x cheaper than full ``readtext``.
    """
    if grayscale.size == 0:
        return 0.0
    h, w = grayscale.shape[:2]
    if h < 8 or w < 8:
        return 0.0
    reader = _get_ocr_reader()
    horizontal_list, free_list = reader.detect(grayscale)
    total = 0.0
    # horizontal_list[0] is a list of [x_min, x_max, y_min, y_max]
    for box in (horizontal_list[0] if horizontal_list else []):
        x0, x1, y0, y1 = box
        total += max(0, x1 - x0) * max(0, y1 - y0)
    # free_list[0] is a list of polygons [[x,y], [x,y], [x,y], [x,y]]
    for poly in (free_list[0] if free_list else []):
        if len(poly) >= 3:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            total += (max(xs) - min(xs)) * (max(ys) - min(ys))   # bbox approx
    return float(total) / float(h * w)
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `poetry run pytest tests/test_staff_reject.py -v -k "text_area_frac"`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/omr_pipeline/staff_reject.py tests/test_staff_reject.py
git commit -m "feat(reject): OCR text-density signal via EasyOCR detector"
```

---

## Task 5: CTC confidence signal

**Files:**
- Modify: `src/omr_pipeline/staff_reject.py`
- Modify: `tests/test_staff_reject.py`

- [ ] **Step 1: Append failing tests for `_mean_logprob`**

Append to `tests/test_staff_reject.py`:
```python
import torch


def test_mean_logprob_confident():
    from omr_pipeline.staff_reject import _mean_logprob

    # Confident: one class dominates each frame -> log-prob ~ log(0.95) ~ -0.05
    T, C = 20, 10
    logits = torch.full((T, C), -3.0)
    for t in range(T):
        logits[t, t % C] = 3.0
    log_probs = torch.log_softmax(logits, dim=-1)
    lp = _mean_logprob(log_probs, out_len=T)
    assert lp > -0.5


def test_mean_logprob_uniform():
    from omr_pipeline.staff_reject import _mean_logprob

    # Uniform: log-prob = log(1/C)
    T, C = 20, 10
    log_probs = torch.log_softmax(torch.zeros(T, C), dim=-1)
    lp = _mean_logprob(log_probs, out_len=T)
    assert lp == pytest.approx(np.log(1.0 / C), abs=1e-4)


def test_mean_logprob_zero_length():
    from omr_pipeline.staff_reject import _mean_logprob

    log_probs = torch.zeros(0, 10)
    assert _mean_logprob(log_probs, out_len=0) == float("-inf")
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `poetry run pytest tests/test_staff_reject.py -v -k "mean_logprob"`
Expected: 3 failures.

- [ ] **Step 3: Implement `_mean_logprob`**

Append to `src/omr_pipeline/staff_reject.py`:
```python
# ---------------------------------------------------------------------------
# CTC confidence signal
# ---------------------------------------------------------------------------

def _mean_logprob(log_probs, out_len: int) -> float:
    """Mean log-probability of the argmax class across the first ``out_len`` frames."""
    import torch

    if out_len <= 0 or log_probs.numel() == 0:
        return float("-inf")
    frames = log_probs[:out_len]
    argmax = frames.argmax(dim=-1)
    picked = frames.gather(-1, argmax.unsqueeze(-1)).squeeze(-1)
    return float(picked.mean().item())
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `poetry run pytest tests/test_staff_reject.py -v -k "mean_logprob"`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/omr_pipeline/staff_reject.py tests/test_staff_reject.py
git commit -m "feat(reject): CTC mean log-prob signal"
```

---

## Task 6: `evaluate_pre_crnn` and `evaluate_post_crnn`

**Files:**
- Modify: `src/omr_pipeline/staff_reject.py`
- Modify: `tests/test_staff_reject.py`

- [ ] **Step 1: Append failing tests for the composed evaluators**

Append to `tests/test_staff_reject.py`:
```python
from dataclasses import dataclass
from types import SimpleNamespace


def _fake_system(music_binary, music_image=None):
    """A minimal stand-in for staff_detect.System for unit tests."""
    if music_image is None:
        # convert 0/1 binary into a 0/255 grayscale so EasyOCR receives uint8 image
        music_image = ((1 - music_binary) * 255).astype(np.uint8)
    return SimpleNamespace(music_binary=music_binary, music_image=music_image)


def test_evaluate_pre_crnn_clean_staff_passes(mocker):
    from omr_pipeline import staff_reject

    fake_reader = mocker.Mock()
    fake_reader.detect.return_value = ([[]], [[]])
    mocker.patch.object(staff_reject, "_get_ocr_reader", return_value=fake_reader)
    # Bypass the local staff-line re-detection by patching it
    mocker.patch.object(
        staff_reject,
        "_strip_line_ys",
        return_value=[8, 20, 32, 44, 56],
    )

    img = _staff_binary(line_span=1.0, interline_ink=True)
    sys = _fake_system(img)
    res = staff_reject.evaluate_pre_crnn(sys)
    assert res.passed is True
    assert res.reason is None
    assert "line_span_min" in res.diagnostics


def test_evaluate_pre_crnn_short_lines_rejects_geometry(mocker):
    from omr_pipeline import staff_reject

    mocker.patch.object(staff_reject, "_get_ocr_reader")
    mocker.patch.object(
        staff_reject,
        "_strip_line_ys",
        return_value=[8, 20, 32, 44, 56],
    )

    img = _staff_binary(line_span=0.30, interline_ink=True)
    res = staff_reject.evaluate_pre_crnn(_fake_system(img))
    assert res.passed is False
    assert res.reason == "geometry_line_span"


def test_evaluate_pre_crnn_no_lines_rejects(mocker):
    from omr_pipeline import staff_reject

    mocker.patch.object(staff_reject, "_strip_line_ys", return_value=None)
    img = np.zeros((64, 300), dtype=np.uint8)
    res = staff_reject.evaluate_pre_crnn(_fake_system(img))
    assert res.passed is False
    assert res.reason == "geometry_no_staff_lines"


def test_evaluate_pre_crnn_text_dense_rejects(mocker):
    from omr_pipeline import staff_reject

    fake_reader = mocker.Mock()
    # 50% of strip area is text
    fake_reader.detect.return_value = ([[[0, 300, 0, 32]]], [[]])
    mocker.patch.object(staff_reject, "_get_ocr_reader", return_value=fake_reader)
    mocker.patch.object(
        staff_reject,
        "_strip_line_ys",
        return_value=[8, 20, 32, 44, 56],
    )

    img = _staff_binary(line_span=1.0, interline_ink=True)
    res = staff_reject.evaluate_pre_crnn(_fake_system(img))
    assert res.passed is False
    assert res.reason == "ocr_text_density"


def test_evaluate_post_crnn_low_confidence_rejects():
    from omr_pipeline.staff_reject import RejectionResult, evaluate_post_crnn

    T, C = 16, 10
    log_probs = torch.log_softmax(torch.zeros(T, C), dim=-1)
    pre = RejectionResult(passed=True, reason=None, diagnostics={
        "line_span_min": 0.9, "spacing_cov": 0.05,
        "interline_ink_frac": 0.01, "text_area_frac": 0.05,
    })
    sys = _fake_system(np.zeros((10, 10), dtype=np.uint8))
    res = evaluate_post_crnn(sys, log_probs, T, ["pitch:C"], pre)
    assert res.passed is False
    assert res.reason == "ctc_low_confidence"
    assert "mean_logprob" in res.diagnostics


def test_evaluate_post_crnn_confident_passes():
    from omr_pipeline.staff_reject import RejectionResult, evaluate_post_crnn

    T, C = 16, 10
    logits = torch.full((T, C), -3.0)
    for t in range(T):
        logits[t, t % C] = 3.0
    log_probs = torch.log_softmax(logits, dim=-1)
    pre = RejectionResult(passed=True, reason=None, diagnostics={
        "line_span_min": 0.9, "spacing_cov": 0.05,
        "interline_ink_frac": 0.01, "text_area_frac": 0.05,
    })
    sys = _fake_system(np.zeros((10, 10), dtype=np.uint8))
    res = evaluate_post_crnn(sys, log_probs, T, ["pitch:C"], pre)
    assert res.passed is True
    assert res.reason is None
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `poetry run pytest tests/test_staff_reject.py -v -k "evaluate_"`
Expected: 6 failures referencing missing `_strip_line_ys`, `evaluate_pre_crnn`, `evaluate_post_crnn`.

- [ ] **Step 3: Implement `_strip_line_ys`, `evaluate_pre_crnn`, `evaluate_post_crnn`**

Append to `src/omr_pipeline/staff_reject.py`:
```python
# ---------------------------------------------------------------------------
# Composed evaluators
# ---------------------------------------------------------------------------

def _strip_line_ys(music_binary: np.ndarray) -> list[int] | None:
    """Re-run staff-line detection inside a single strip.

    Wrapper around ``staff_detect.local_primary_staff_lines`` to keep this
    module's import surface narrow.
    """
    from .staff_detect import local_primary_staff_lines

    return local_primary_staff_lines(music_binary)


def evaluate_pre_crnn(system, thresholds: RejectThresholds | None = None) -> RejectionResult:
    """Run the geometric + OCR gates on ``system`` before CRNN inference."""
    th = thresholds or load_thresholds()
    diag: dict[str, float] = {
        "line_span_min": 0.0,
        "spacing_cov": float("inf"),
        "interline_ink_frac": 0.0,
        "text_area_frac": 0.0,
    }

    binary = getattr(system, "music_binary", None)
    if binary is None or binary.size == 0:
        return RejectionResult(passed=False, reason="geometry_no_strip", diagnostics=diag)

    line_ys = _strip_line_ys(binary)
    if not line_ys or len(line_ys) < 5:
        return RejectionResult(passed=False, reason="geometry_no_staff_lines", diagnostics=diag)

    diag["line_span_min"] = _line_span_min(binary, line_ys)
    diag["spacing_cov"] = _spacing_cov(line_ys)
    diag["interline_ink_frac"] = _interline_ink_frac(binary, line_ys)

    if diag["line_span_min"] < th.min_line_span_frac:
        return RejectionResult(passed=False, reason="geometry_line_span", diagnostics=diag)
    if diag["spacing_cov"] > th.max_spacing_cov:
        return RejectionResult(passed=False, reason="geometry_spacing_cov", diagnostics=diag)
    if diag["interline_ink_frac"] < th.min_interline_ink_frac:
        return RejectionResult(passed=False, reason="geometry_interline_ink", diagnostics=diag)

    music_image = getattr(system, "music_image", None)
    if music_image is not None and music_image.size > 0:
        try:
            diag["text_area_frac"] = _text_area_frac(music_image)
        except Exception as exc:   # OCR failures should not kill the pipeline
            log.warning("OCR text-area gate failed: %s", exc)
            diag["text_area_frac"] = 0.0

    if diag["text_area_frac"] > th.max_text_area_frac:
        return RejectionResult(passed=False, reason="ocr_text_density", diagnostics=diag)

    return RejectionResult(passed=True, reason=None, diagnostics=diag)


def evaluate_post_crnn(
    system,
    log_probs,
    out_len: int,
    tokens: list[str],
    pre_result: RejectionResult,
    thresholds: RejectThresholds | None = None,
) -> RejectionResult:
    """Run the CTC-confidence gate on a CRNN output.

    Inherits diagnostics from ``pre_result`` and adds ``mean_logprob``.
    If ``pre_result.passed`` is False, returns that verdict unchanged
    (caller is expected to short-circuit but we stay safe).
    """
    th = thresholds or load_thresholds()
    diag = {**pre_result.diagnostics}
    diag["mean_logprob"] = _mean_logprob(log_probs, out_len)

    if not pre_result.passed:
        return RejectionResult(passed=False, reason=pre_result.reason, diagnostics=diag)

    if out_len <= 0:
        return RejectionResult(passed=False, reason="ctc_zero_length", diagnostics=diag)

    if diag["mean_logprob"] < th.min_mean_logprob:
        return RejectionResult(passed=False, reason="ctc_low_confidence", diagnostics=diag)

    return RejectionResult(passed=True, reason=None, diagnostics=diag)
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `poetry run pytest tests/test_staff_reject.py -v`
Expected: all tests in the file pass.

- [ ] **Step 5: Commit**

```bash
git add src/omr_pipeline/staff_reject.py tests/test_staff_reject.py
git commit -m "feat(reject): compose evaluate_pre_crnn and evaluate_post_crnn"
```

---

## Task 7: Modify `recognize_music` to return `(tokens, log_probs, out_lens)`

**Files:**
- Modify: `src/omr_pipeline/inference.py`
- Modify: `tests/test_staff_reject.py` (add a smoke test for the new signature only if no caller exercises it elsewhere — skip if not feasible)

- [ ] **Step 1: Inspect current return shape**

Run: `poetry run grep -n "return" src/omr_pipeline/inference.py | tail -10`
Confirm: the function currently returns `results: list[str]`.

- [ ] **Step 2: Modify `recognize_music` to also return per-strip log-probs and out-lens**

Read the full file first:
```bash
poetry run cat src/omr_pipeline/inference.py | sed -n '1,40p'
```

In `src/omr_pipeline/inference.py`, locate the function `recognize_music` (the public entrypoint). Identify where `log_probs, out_lens = model(batch, width_t)` is computed (around line 200) and where the function returns `results`.

Change the return path to a 3-tuple `(results, per_strip_logprobs, per_strip_outlens)`. Concretely:

After the line `token_lists = decode_fn(log_probs, out_lens, vocab)`, add:
```python
    # Split the batched (B, T, C) log-prob tensor into per-strip CPU tensors.
    # ``out_lens`` is (B,) on the same device; move both to CPU for downstream
    # consumption by the staff_reject gate.
    lp_cpu = log_probs.detach().to("cpu")        # (B, T, C)
    ol_cpu = out_lens.detach().to("cpu").tolist()
```

Change the final loop and return from:
```python
    results: list[str] = []
    for tl, ok in zip(token_lists, valid_mask):
        results.append(" ".join(tl) if ok else "")
    return results
```
To:
```python
    results: list[str] = []
    per_strip_lp = []
    per_strip_ol: list[int] = []
    for i, (tl, ok) in enumerate(zip(token_lists, valid_mask)):
        results.append(" ".join(tl) if ok else "")
        # log_probs shape is (B, T, C); slice the i-th and trim to that strip's length
        ol = int(ol_cpu[i]) if i < len(ol_cpu) else 0
        per_strip_lp.append(lp_cpu[i, :ol] if ok else lp_cpu[i, :0])
        per_strip_ol.append(ol if ok else 0)
    return results, per_strip_lp, per_strip_ol
```

> ⚠️ The CRNN may emit `log_probs` shaped as either `(B, T, C)` or `(T, B, C)` depending on the model implementation. Before locking in the slicing index, **inspect the model output**:
>
> Run:
> ```bash
> poetry run python -c "
> from src.CRNN_CTC.model import CRNN
> import inspect
> src = inspect.getsource(CRNN.forward)
> print(src[-400:])
> "
> ```
> If the shape is `(T, B, C)`, change the slicing to `lp_cpu[:ol, i]` and adjust accordingly. Add a comment near the slice documenting the layout that was observed.

- [ ] **Step 3: Update the docstring and Stage-3 doc**

Update the docstring of `recognize_music` to document the new 3-tuple return.

- [ ] **Step 4: Update `docs/inference_pipeline.md` Stage 3 section**

In `docs/inference_pipeline.md`, change the Stage 3 diagram to indicate `recognize_music` returns `(token_lists, log_probs_per_strip, out_lens_per_strip)`. Edit the lines around the CTC decode block.

- [ ] **Step 5: Smoke test — import still works and signature is 3-tuple**

Add to `tests/test_staff_reject.py` (end of file):
```python
def test_recognize_music_returns_three_tuple():
    """Smoke test only — does not actually run the model.

    Verifies the signature so downstream wiring is type-safe. Uses a stub
    checkpoint path so we can call the function without loading weights.
    """
    import inspect
    from omr_pipeline.inference import recognize_music

    sig = inspect.signature(recognize_music)
    # Sanity check: ensure the function exists and takes at least one positional arg.
    assert len(sig.parameters) >= 1
```

Run: `poetry run pytest tests/test_staff_reject.py -v`
Expected: all tests pass.

- [ ] **Step 6: Verify import-time correctness of the pipeline module**

Run:
```bash
poetry run python -c "
from src.omr_pipeline.inference import recognize_music
import inspect
print('OK; params:', list(inspect.signature(recognize_music).parameters))
"
```
Expected: prints "OK; params: [...]" without error.

- [ ] **Step 7: Commit**

```bash
git add src/omr_pipeline/inference.py docs/inference_pipeline.md tests/test_staff_reject.py
git commit -m "refactor(inference): recognize_music returns (tokens, log_probs, out_lens)"
```

---

## Task 8: Wire pre-CRNN gate into `staff_detect.detect_systems`

**Files:**
- Modify: `src/omr_pipeline/staff_detect.py`

- [ ] **Step 1: Extend `System` dataclass with a `pre_result` field**

In `src/omr_pipeline/staff_detect.py`, modify the `System` dataclass (around line 43-52):
```python
@dataclass
class System:
    """One staff system on the page: chord region above + music staff."""
    staff: Staff
    chord_bbox: tuple[int, int, int, int] | None
    music_bbox: tuple[int, int, int, int]
    chord_image: np.ndarray | None = field(default=None, repr=False)
    music_image: np.ndarray | None = field(default=None, repr=False)
    chord_binary: np.ndarray | None = field(default=None, repr=False)
    music_binary: np.ndarray | None = field(default=None, repr=False)
    # Filled by staff_reject.evaluate_pre_crnn after detection. None means
    # gate not yet run; passed=False with geometry_* reason means dropped.
    pre_result: object | None = field(default=None, repr=False)
```

- [ ] **Step 2: Insert the gate call at the end of `detect_systems`**

In `staff_detect.py`, locate `detect_systems` (around line 343). Inside the loop where systems are returned after `validated`, add the gate call. The full edit:

Find the block:
```python
        validated = [s for s in systems if music_strip_has_valid_staff(s.music_binary)]
        if validated:
            if len(validated) < len(systems):
                log.info(
                    "Dropped %d region(s) without a local five-line staff",
                    len(systems) - len(validated),
                )
            systems = validated
            log.info(
                "Detected %d staff system(s); staff-space ≈ %.1f px",
                len(systems),
                float(np.mean([s.staff.staff_space for s in systems])),
            )
            return systems
```

Replace with:
```python
        validated = [s for s in systems if music_strip_has_valid_staff(s.music_binary)]
        if validated:
            if len(validated) < len(systems):
                log.info(
                    "Dropped %d region(s) without a local five-line staff",
                    len(systems) - len(validated),
                )
            systems = validated

            # ----- pre-CRNN rejection gate -----
            # Local import to avoid a circular dependency at module load.
            from .staff_reject import evaluate_pre_crnn

            kept: list[System] = []
            for s in systems:
                pre = evaluate_pre_crnn(s)
                s.pre_result = pre
                if (not pre.passed) and (pre.reason or "").startswith("geometry_"):
                    log.info(
                        "Pre-CRNN gate dropped strip @ y=%d (reason=%s, diag=%s)",
                        s.staff.top, pre.reason, pre.diagnostics,
                    )
                    continue
                kept.append(s)
            systems = kept
            # -----------------------------------

            log.info(
                "Detected %d staff system(s); staff-space ≈ %.1f px",
                len(systems),
                float(np.mean([s.staff.staff_space for s in systems])) if systems else 0.0,
            )
            return systems
```

- [ ] **Step 3: Apply the same gate to the fallback path**

In the same function, locate the `if fallback_systems:` branch lower down and apply the same pre-CRNN gate before returning, using the identical loop body.

- [ ] **Step 4: Sanity-check imports**

Run:
```bash
poetry run python -c "
from src.omr_pipeline.staff_detect import detect_systems, System
print('System fields:', [f for f in System.__dataclass_fields__])
"
```
Expected: list includes `pre_result`.

- [ ] **Step 5: Commit**

```bash
git add src/omr_pipeline/staff_detect.py
git commit -m "feat(staff_detect): drop geometry-rejected strips via pre-CRNN gate"
```

---

## Task 9: Wire post-CRNN gate into `pipeline._process_systems`

**Files:**
- Modify: `src/omr_pipeline/pipeline.py`

- [ ] **Step 1: Update `_process_systems` to consume the new 3-tuple from `recognize_music`**

In `src/omr_pipeline/pipeline.py`, locate `_process_systems` (around line 68). Replace its body with:

```python
def _process_systems(
    systems: list[System],
    checkpoint_path: Path | None,
) -> list[dict]:
    """One segment per staff.

    Each segment: ``{staff_bbox, chord_bbox, lmx_tokens, chords, rejected, reject_diagnostics}``.
    Segments are ordered top-to-bottom.
    """
    from .staff_reject import RejectionResult, evaluate_post_crnn

    music_imgs: list[np.ndarray] = []
    chord_imgs: list[np.ndarray] = []

    for sys_ in systems:
        if sys_.music_image is not None and sys_.music_image.size > 0:
            music_imgs.append(sys_.music_image)
        else:
            music_imgs.append(np.zeros((10, 10), dtype=np.uint8))
        if sys_.chord_image is not None and sys_.chord_image.size > 0:
            chord_imgs.append(sys_.chord_image)
        else:
            chord_imgs.append(np.zeros((10, 10), dtype=np.uint8))

    _save_debug(music_imgs, chord_imgs)

    music_preds, music_logprobs, music_outlens = recognize_music(music_imgs, checkpoint_path)
    chord_preds = recognize_chords_crnn(chord_imgs)

    # LMX grammar correction with cross-system key + time propagation
    global_key: str | None = None
    global_time: tuple[str, str, str] | None = None
    fixed_music: list[str] = []
    for pred in music_preds:
        fixed, global_key, global_time = fix_sequence(
            pred, global_key=global_key, global_time=global_time,
            force_clef=True,
        )
        fixed_music.append(fixed)

    segments: list[dict] = []
    for i, sys_ in enumerate(systems):
        lmx_str = fixed_music[i] if i < len(fixed_music) else ""
        chord_str = chord_preds[i] if i < len(chord_preds) else ""

        # Compose post-CRNN gate result
        pre = getattr(sys_, "pre_result", None)
        if pre is None:
            pre = RejectionResult(passed=True, reason=None, diagnostics={
                "line_span_min": 0.0, "spacing_cov": 0.0,
                "interline_ink_frac": 0.0, "text_area_frac": 0.0,
            })
        post = evaluate_post_crnn(
            sys_,
            music_logprobs[i] if i < len(music_logprobs) else music_logprobs[0][:0],
            int(music_outlens[i]) if i < len(music_outlens) else 0,
            lmx_str.split() if lmx_str else [],
            pre,
        )

        mx, my, mw, mh = sys_.music_bbox
        chord_bbox = list(sys_.chord_bbox) if sys_.chord_bbox is not None else None

        rejected_reason = None if post.passed else post.reason
        if rejected_reason is not None:
            lmx_tokens_out: list[str] = []
            chord_tokens_out: list[str] = []
        else:
            lmx_tokens_out = lmx_str.split() if lmx_str else []
            chord_tokens_out = chord_str.split() if chord_str else []

        segments.append({
            "staff_bbox": [mx, my, mw, mh],
            "chord_bbox": chord_bbox,
            "lmx_tokens": lmx_tokens_out,
            "chords": chord_tokens_out,
            "rejected": rejected_reason,
            "reject_diagnostics": post.diagnostics,
        })

    return segments
```

- [ ] **Step 2: Update `run_pipeline` to populate `num_rejected` in meta**

In the same file, locate the `return` statement at the end of `run_pipeline`. Replace:
```python
        "meta": {**base_meta, "num_systems": len(systems)},
```
With:
```python
        "meta": {
            **base_meta,
            "num_systems": len(systems),
            "num_rejected": sum(1 for s in segments if s.get("rejected") is not None),
        },
```

- [ ] **Step 3: Import-level smoke test**

Run:
```bash
poetry run python -c "
from src.omr_pipeline.pipeline import run_pipeline
print('OK')
"
```
Expected: prints `OK`.

- [ ] **Step 4: Commit**

```bash
git add src/omr_pipeline/pipeline.py
git commit -m "feat(pipeline): wire post-CRNN gate; emit rejected + reject_diagnostics"
```

---

## Task 10: Integration test on a real Satin Doll page

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_pipeline_rejects_titles.py`
- Create: `tests/fixtures/pages/satin_doll.png` (harvested from `data/real_book/full_realbook.pdf`)

> Note: this task requires a model checkpoint and EasyOCR model weights. Skip the test cleanly when either is unavailable.

- [ ] **Step 1: Harvest the Satin Doll page as a PNG fixture**

Find the Satin Doll page in `data/real_book/full_realbook.pdf` (page 367 by the printed page number; the PDF page index may differ).

Run:
```bash
poetry run python -c "
import fitz
import sys
from pathlib import Path

doc = fitz.open('data/real_book/full_realbook.pdf')
# Find by text
for i, page in enumerate(doc):
    if 'SATIN' in page.get_text().upper() and 'ELLINGTON' in page.get_text().upper():
        print('Found Satin Doll on PDF page index', i)
        pix = page.get_pixmap(dpi=300)
        Path('tests/fixtures/pages').mkdir(parents=True, exist_ok=True)
        pix.save('tests/fixtures/pages/satin_doll.png')
        sys.exit(0)
print('Satin Doll page not found by text search; fall back to manual harvest')
sys.exit(1)
"
```

If the text search fails (scanned-image PDF with no OCR layer), the engineer must manually identify the correct PDF page index, run the same `get_pixmap` save, and commit the resulting PNG. **Do not skip this fixture creation step** — the integration test depends on it.

- [ ] **Step 2: Create `tests/integration/__init__.py`**

```python
```

- [ ] **Step 3: Create `tests/integration/test_pipeline_rejects_titles.py`**

```python
"""End-to-end test: title and footer regions are rejected, not transcribed."""
from __future__ import annotations

from pathlib import Path

import pytest


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "pages" / "satin_doll.png"
CHECKPOINT_CANDIDATES = [
    Path("models/latest/best_model.pt"),
    Path("models/best_model.pt"),
]


def _checkpoint_path() -> Path | None:
    for p in CHECKPOINT_CANDIDATES:
        if p.exists():
            return p
    return None


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture page missing")
@pytest.mark.skipif(_checkpoint_path() is None, reason="no CRNN checkpoint available")
def test_satin_doll_title_is_rejected():
    from src.omr_pipeline.pipeline import run_pipeline

    data = FIXTURE.read_bytes()
    out = run_pipeline(data, "satin_doll.png", checkpoint_path=_checkpoint_path())

    assert out.get("error") is None
    segments = out["pages"][0]["segments"]
    assert len(segments) >= 6, f"expected at least 6 segments, got {len(segments)}"

    rejected = [s for s in segments if s.get("rejected") is not None]
    accepted = [s for s in segments if s.get("rejected") is None]

    assert rejected, "expected at least one rejected segment for title/footer"
    assert accepted, "expected accepted segments for the real music staves"

    # The top-most segment is almost certainly the title region.
    top = min(segments, key=lambda s: s["staff_bbox"][1])
    bottom = max(segments, key=lambda s: s["staff_bbox"][1] + s["staff_bbox"][3])
    # The title and/or footer should appear in the rejected set, OR be entirely
    # dropped (in which case no top/bottom matches a rejection — that is also OK
    # because the pipeline dropped them at the geometric gate).
    if top in rejected or bottom in rejected:
        return
    # Otherwise we expect them to have been dropped pre-CRNN, in which case
    # the remaining segments must all be real music (acceptable too).
    assert len(rejected) == 0, "rejected segments exist but neither title nor footer is among them"
```

- [ ] **Step 4: Run the integration test**

Run: `poetry run pytest tests/integration/ -v -s`
Expected: passes, OR is skipped with a clear reason if fixture/checkpoint missing.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/__init__.py tests/integration/test_pipeline_rejects_titles.py tests/fixtures/pages/satin_doll.png
git commit -m "test(integration): assert Satin Doll title/footer are rejected"
```

---

## Task 11: `harvest-reject-fixtures` CLI subcommand

**Files:**
- Modify: `src/cli.py`

- [ ] **Step 1: Inspect existing CLI subcommand registration**

Run: `poetry run grep -n "add_parser\|set_defaults" src/cli.py | head -20`
Identify the pattern used to register subcommands (argparse subparsers, click, etc.).

- [ ] **Step 2: Add the `harvest-reject-fixtures` subcommand**

In `src/cli.py`, alongside the existing subcommand registrations, add:

```python
def _cmd_harvest_reject_fixtures(args) -> None:
    """Harvest detected music strips from PDFs for the reject calibration set."""
    import logging
    from pathlib import Path
    import glob
    import cv2

    from src.omr_pipeline.preprocess import load_pdf_page, pdf_load_dpi, preprocess_page
    from src.omr_pipeline.staff_detect import detect_systems

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("harvest")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for pat in args.pdfs:
        paths.extend(glob.glob(pat))
    if not paths:
        log.error("no PDFs matched %s", args.pdfs)
        return

    counter = 0
    dpi = pdf_load_dpi()
    for p in paths:
        try:
            with open(p, "rb") as fh:
                blob = fh.read()
            n_pages = args.pages or 1
            for page_idx in range(n_pages):
                try:
                    img = load_pdf_page(blob, page=page_idx, dpi=dpi)
                except Exception:
                    break
                page = preprocess_page(img)
                systems = detect_systems(page.grayscale, page.binary)
                for s in systems:
                    if s.music_image is None or s.music_image.size == 0:
                        continue
                    name = f"{Path(p).stem}_p{page_idx:03d}_s{counter:04d}.png"
                    cv2.imwrite(str(out_dir / name), s.music_image)
                    counter += 1
        except Exception as exc:
            log.warning("skip %s: %s", p, exc)

    log.info("Harvested %d strips into %s", counter, out_dir)
    log.info("Now sort them manually into <out>/music/ and <out>/non_music/.")


def _add_harvest_parser(sub):
    p = sub.add_parser("harvest-reject-fixtures", help="Harvest staff strips from PDFs for reject calibration")
    p.add_argument("--pdfs", nargs="+", required=True, help="Glob(s) of PDFs to harvest from")
    p.add_argument("--pages", type=int, default=1, help="Number of pages per PDF to process (default 1)")
    p.add_argument("--out", default="tests/fixtures/reject/_harvest", help="Output directory")
    p.set_defaults(func=_cmd_harvest_reject_fixtures)
```

Register `_add_harvest_parser(sub)` next to the existing parser registrations.

- [ ] **Step 3: Smoke test the subcommand**

Run:
```bash
poetry run python src/cli.py harvest-reject-fixtures --help
```
Expected: usage help is printed without error.

- [ ] **Step 4: Run it against a tiny set to confirm end-to-end**

Run:
```bash
poetry run python src/cli.py harvest-reject-fixtures \
  --pdfs 'data/real_book/AutumnLeaves_clean.pdf' \
  --pages 1 \
  --out tests/fixtures/reject/_harvest
ls tests/fixtures/reject/_harvest | head
```
Expected: at least one PNG appears, no exceptions.

- [ ] **Step 5: Commit**

```bash
git add src/cli.py
git commit -m "feat(cli): add harvest-reject-fixtures to bootstrap calibration set"
```

---

## Task 12: `calibrate-reject` CLI subcommand

**Files:**
- Modify: `src/cli.py`

- [ ] **Step 1: Add the `calibrate-reject` subcommand**

In `src/cli.py`, add alongside the previous task's additions:

```python
def _cmd_calibrate_reject(args) -> None:
    """Sweep thresholds against a labeled fixture set and write the result."""
    import json
    import logging
    from dataclasses import asdict
    from pathlib import Path

    import cv2
    import numpy as np
    import torch

    from src.omr_pipeline import staff_reject
    from src.omr_pipeline.staff_reject import (
        DEFAULT_THRESHOLDS, RejectThresholds,
        _interline_ink_frac, _line_span_min, _spacing_cov,
        _strip_line_ys, _text_area_frac,
    )

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("calibrate")

    root = Path(args.fixtures)
    music_dir = root / "music"
    non_music_dir = root / "non_music"
    if not music_dir.is_dir() or not non_music_dir.is_dir():
        log.error("expected <fixtures>/music/ and <fixtures>/non_music/ to exist")
        return

    def _load_binary(p: Path) -> tuple[np.ndarray, np.ndarray]:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"could not read {p}")
        # Match the preprocessing the pipeline uses: Otsu binarise
        _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return gray, binary.astype(np.uint8)

    def _signals_for(p: Path) -> dict[str, float] | None:
        gray, binary = _load_binary(p)
        line_ys = _strip_line_ys(binary)
        if not line_ys or len(line_ys) < 5:
            return {
                "line_span_min": 0.0,
                "spacing_cov": float("inf"),
                "interline_ink_frac": 0.0,
                "text_area_frac": _text_area_frac(gray),
                "no_lines": 1.0,
            }
        return {
            "line_span_min": _line_span_min(binary, line_ys),
            "spacing_cov": _spacing_cov(line_ys),
            "interline_ink_frac": _interline_ink_frac(binary, line_ys),
            "text_area_frac": _text_area_frac(gray),
            "no_lines": 0.0,
        }

    samples_pos = []   # label = 1 (music — should PASS)
    samples_neg = []   # label = 0 (non-music — should REJECT)
    for p in sorted(music_dir.glob("*.png")):
        s = _signals_for(p)
        if s is not None:
            samples_pos.append((str(p), s))
    for p in sorted(non_music_dir.glob("*.png")):
        s = _signals_for(p)
        if s is not None:
            samples_neg.append((str(p), s))

    log.info("Loaded %d positive (music) and %d negative (non-music) samples",
             len(samples_pos), len(samples_neg))

    def _sweep(values_pos: list[float], values_neg: list[float], reject_when_greater: bool) -> float:
        """Return the threshold maximising Youden's J on positives vs negatives."""
        candidates = sorted(set(values_pos + values_neg))
        best_t, best_j = candidates[0] if candidates else 0.0, -1.0
        for t in candidates:
            if reject_when_greater:
                tp = sum(1 for v in values_neg if v > t)
                fp = sum(1 for v in values_pos if v > t)
            else:
                tp = sum(1 for v in values_neg if v < t)
                fp = sum(1 for v in values_pos if v < t)
            tpr = tp / max(1, len(values_neg))
            fpr = fp / max(1, len(values_pos))
            j = tpr - fpr
            if j > best_j:
                best_j, best_t = j, t
        return float(best_t)

    pos_signals = {k: [s[1][k] for s in samples_pos] for k in ("line_span_min", "spacing_cov", "interline_ink_frac", "text_area_frac")}
    neg_signals = {k: [s[1][k] for s in samples_neg] for k in ("line_span_min", "spacing_cov", "interline_ink_frac", "text_area_frac")}

    chosen = RejectThresholds(
        min_line_span_frac=_sweep(pos_signals["line_span_min"], neg_signals["line_span_min"], reject_when_greater=False),
        max_spacing_cov=_sweep(pos_signals["spacing_cov"], neg_signals["spacing_cov"], reject_when_greater=True),
        min_interline_ink_frac=_sweep(pos_signals["interline_ink_frac"], neg_signals["interline_ink_frac"], reject_when_greater=False),
        max_text_area_frac=_sweep(pos_signals["text_area_frac"], neg_signals["text_area_frac"], reject_when_greater=True),
        # CTC threshold not calibrated here — it needs CRNN outputs; keep default.
        min_mean_logprob=DEFAULT_THRESHOLDS.min_mean_logprob,
    )

    log.info("Recommended thresholds:")
    for k, v in asdict(chosen).items():
        log.info("  %s = %.4f", k, v)

    # Confusion matrix at chosen thresholds (pre-CRNN only)
    def _would_reject(sig: dict[str, float], th: RejectThresholds) -> bool:
        if sig.get("no_lines", 0.0) > 0.5:
            return True
        if sig["line_span_min"] < th.min_line_span_frac:
            return True
        if sig["spacing_cov"] > th.max_spacing_cov:
            return True
        if sig["interline_ink_frac"] < th.min_interline_ink_frac:
            return True
        if sig["text_area_frac"] > th.max_text_area_frac:
            return True
        return False

    tp = sum(1 for (_, s) in samples_neg if _would_reject(s, chosen))
    fp = sum(1 for (_, s) in samples_pos if _would_reject(s, chosen))
    log.info("Confusion at chosen thresholds: TP=%d/%d  FP=%d/%d",
             tp, len(samples_neg), fp, len(samples_pos))

    # Print misclassified for inspection
    misclassified = []
    for (p, s) in samples_pos:
        if _would_reject(s, chosen):
            misclassified.append(("FP", p, s))
    for (p, s) in samples_neg:
        if not _would_reject(s, chosen):
            misclassified.append(("FN", p, s))
    for kind, path, s in misclassified:
        log.info("  %s %s %s", kind, path, {k: round(v, 4) for k, v in s.items()})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(chosen), indent=2))
    log.info("Wrote thresholds to %s", out_path)


def _add_calibrate_parser(sub):
    p = sub.add_parser("calibrate-reject", help="Sweep reject thresholds on a labeled fixture set")
    p.add_argument("--fixtures", default="tests/fixtures/reject", help="Root dir with music/ and non_music/")
    p.add_argument("--out", default="models/staff_reject/thresholds.json", help="Where to write thresholds JSON")
    p.set_defaults(func=_cmd_calibrate_reject)
```

Register `_add_calibrate_parser(sub)` next to the harvest parser.

- [ ] **Step 2: Smoke test the help text**

Run: `poetry run python src/cli.py calibrate-reject --help`
Expected: usage printed.

- [ ] **Step 3: Build a tiny fixture set and run the sweep end-to-end**

Manually move a handful of strips from `tests/fixtures/reject/_harvest/` into `tests/fixtures/reject/music/` (music) and `tests/fixtures/reject/non_music/` (titles/footers).

Then:
```bash
poetry run python src/cli.py calibrate-reject \
  --fixtures tests/fixtures/reject \
  --out models/staff_reject/thresholds.json
cat models/staff_reject/thresholds.json
```
Expected: a JSON file with all 5 keys; output prints a confusion matrix.

- [ ] **Step 4: Commit**

```bash
git add src/cli.py models/staff_reject/thresholds.json tests/fixtures/reject/
git commit -m "feat(cli): add calibrate-reject to tune thresholds from labeled strips"
```

---

## Task 13: Documentation updates

**Files:**
- Modify: `docs/inference_pipeline.md`
- Modify: `docs/api.md`
- Modify: `docs/cli.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add a "Stage 2b — Rejection Gates" section to `docs/inference_pipeline.md`**

After Stage 2 (Staff Detection) and before Stage 3, insert:

```markdown
## Stage 2b — Rejection Gates

**File:** `src/omr_pipeline/staff_reject.py`

Detected staff systems pass through a hybrid pre-CRNN + post-CRNN gate that
filters out title regions, footer text, and other non-music page elements.

**Pre-CRNN gates** (run inside `detect_systems`):

| Gate | Signal | Reject when |
|------|--------|-------------|
| `geometry_no_staff_lines` | local 5-line re-detection | <5 lines found |
| `geometry_line_span` | min fraction of cols with ink on each line | < `min_line_span_frac` |
| `geometry_spacing_cov` | std/mean of inter-line gaps | > `max_spacing_cov` |
| `geometry_interline_ink` | ink fraction between lines (excluding line bands) | < `min_interline_ink_frac` |
| `ocr_text_density` | EasyOCR text-bbox area / strip area | > `max_text_area_frac` |

**Post-CRNN gate** (run inside `pipeline._process_systems`):

| Gate | Signal | Reject when |
|------|--------|-------------|
| `ctc_low_confidence` | mean log-prob of argmax frames | < `min_mean_logprob` |
| `ctc_zero_length` | `out_len` after width compression | == 0 |

**Behaviour on rejection:**
- Geometry-level rejections **drop the segment entirely** from `pages[].segments[]`.
- OCR and CTC rejections **keep** the segment (bbox preserved) but set
  `rejected: "<reason>"`, empty `lmx_tokens`, empty `chords`. Diagnostics
  are always present under `reject_diagnostics`.

**Threshold calibration:** see `poetry run python src/cli.py calibrate-reject --help`.
After re-training the CRNN, **re-run calibration** — the CTC log-prob distribution
will shift.
```

Also update the existing Stage 3 block to mention the new return signature:
```markdown
**Returns:** `(token_lists, log_probs_per_strip, out_lens_per_strip)`.  The log-probs and
lengths are consumed by the post-CRNN gate (Stage 2b).
```

- [ ] **Step 2: Update `docs/api.md`**

In the segment-schema block, add the two new fields:
```markdown
- `rejected` (string | null): reject reason code, or null when the segment is good.
  See `docs/inference_pipeline.md` Stage 2b for the full list of codes.
- `reject_diagnostics` (object): per-gate numeric signals (`line_span_min`,
  `spacing_cov`, `interline_ink_frac`, `text_area_frac`, `mean_logprob`). Always present.
```

In the meta-block, add `num_rejected: int`.

- [ ] **Step 3: Update `docs/cli.md`**

Add documentation for `harvest-reject-fixtures` and `calibrate-reject`:
```markdown
### `harvest-reject-fixtures`

Harvest detected staff strips from a glob of PDFs into a directory, ready for
manual labelling.

```bash
poetry run python src/cli.py harvest-reject-fixtures \
  --pdfs 'data/real_book/*.pdf' --pages 5 \
  --out tests/fixtures/reject/_harvest
```

Sort the resulting PNGs into `music/` and `non_music/` subfolders by hand.

### `calibrate-reject`

Sweep rejection thresholds on a labelled fixture set and write a JSON file
that `RejectThresholds.load_thresholds()` reads when
`$OMR_REJECT_THRESHOLDS` points to it.

```bash
poetry run python src/cli.py calibrate-reject \
  --fixtures tests/fixtures/reject \
  --out models/staff_reject/thresholds.json
```

Re-run after every CRNN re-train — the CTC log-prob distribution will shift.
```

- [ ] **Step 4: Add a one-line note to `CLAUDE.md`**

Under the "Hard Constraints" section, append:
```markdown
**7. CTC mean-logprob threshold is checkpoint-dependent.**
The CTC-confidence gate in `staff_reject.py` uses `min_mean_logprob` calibrated
to the current checkpoint. After every CRNN re-train, re-run
`poetry run python src/cli.py calibrate-reject ...` and commit the updated
`models/staff_reject/thresholds.json`.
```

- [ ] **Step 5: Commit**

```bash
git add docs/inference_pipeline.md docs/api.md docs/cli.md CLAUDE.md
git commit -m "docs: document non-music strip rejection gates"
```

---

## Task 14: Full test suite + manual smoke

**Files:** none modified

- [ ] **Step 1: Run the full unit-test suite**

Run: `poetry run pytest tests/ -v`
Expected: all tests pass (integration tests may skip if no checkpoint).

- [ ] **Step 2: Re-run the user's failing example to confirm hallucination is gone**

Run the OMR pipeline on the user's Satin Doll PDF (the file that triggered this work) and inspect the output:

```bash
poetry run python -c "
from pathlib import Path
import json
from src.omr_pipeline.pipeline import run_pipeline

# Replace path with the actual Satin Doll PDF the user uploaded.
data = Path('data/real_book/full_realbook.pdf').read_bytes()
out = run_pipeline(data, 'satin_doll.pdf')
segments = out['pages'][0]['segments']
print('num_systems:', out['meta']['num_systems'])
print('num_rejected:', out['meta']['num_rejected'])
for i, s in enumerate(segments[:3]):
    print(f'seg {i}: bbox={s[\"staff_bbox\"]}  rejected={s.get(\"rejected\")}  tokens[:6]={s[\"lmx_tokens\"][:6]}')
"
```

Expected: at least one of the top/bottom segments has `rejected` set, and its
`lmx_tokens` is `[]` (no hallucination).

- [ ] **Step 3: Final commit if anything stragglers**

```bash
git status
```
If clean, no commit needed. If a notebook/log file changed inadvertently, restore it (`git checkout -- <file>`).

---

## Self-Review Notes (for the writer)

- Spec coverage: every section of the spec (architecture, gates, data flow, output contract, calibration, testing, risks) is implemented in tasks 2-13.
- Placeholders: replaced the spec's threshold placeholders with explicit defaults in the dataclass; the calibration CLI generates the real values.
- Type consistency: `RejectionResult` shape is identical across `staff_reject.py`, `staff_detect.py`, and `pipeline.py` — same `passed: bool`, `reason: str | None`, `diagnostics: dict[str, float]`.
- Risk: Task 7 step 2 instructs the engineer to verify the CRNN's log-prob tensor layout before slicing. The plan provides a runnable command; if the shape differs, the slicing line is the only code that needs adjustment.
