# Virtual Header Injection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken headerless-twin training approach with inference-time virtual header injection — prepending prerendered clef+key+time glyphs to continuation staff images so the CRNN always sees what it was trained on.

**Architecture:** Remove all twin generation infrastructure. Prerender 120 header-strip templates (15 keys × 8 time signatures) offline with LilyPond+LilyJAZZ and store them in `data/header_templates/`. At inference, after the first CRNN pass extracts the page's key+time from the first music staff, prepend the matching template to each continuation staff image and run a second CRNN pass — so continuation staves look identical to the full-header staves the model was trained on.

**Tech Stack:** LilyPond (system), LilyJAZZ font (existing), PyTorch, OpenCV/numpy, Pillow, pytest

---

## File Map

**Delete:**
- `src/data_processing/generate_headerless_twins.py`

**Create:**
- `src/data_processing/generate_header_templates.py`
- `src/omr_pipeline/header_injector.py`
- `tests/test_header_injector.py`

**Modify:**
- `src/data_processing/generate_realbook.py` — remove `_NEW_STAFF_RE`, `_TWIN_DROP_EXACT`, `_TWIN_DROP_PREFIX`, `omit_header_in_ly()`, `headerless_label_tokens()` and their comment block
- `src/CRNN_CTC/dataset.py` — remove `_HEADER_PREFIXES`, `_strip_header_tokens()`, `strip_header_prob` from `_AugSubset` and `make_splits()`
- `src/CRNN_CTC/train.py` — remove `strip_header_prob=cfg.strip_header_prob` from `make_splits` call
- `scripts/evaluate_full.py` — remove `strip_header_prob=0.0` from `make_splits` call
- `src/cli.py` — remove `cmd_headerless_twins`, `--force-twins`, `--no-headerless-twins`, `--headerless-*` args, pipeline stage 3; add `cmd_generate_header_templates` and `generate-header-templates` subcommand; update pipeline stage 3 description
- `src/omr_pipeline/pipeline.py` — two-pass CRNN with header injection
- `CLAUDE.md` — update twin description
- `docs/data_pipeline.md` — replace Stage 3 twins section
- `docs/training.md` — update continuation-staff section
- `docs/configuration.md` — update `strip_header_prob` entry
- `docs/cli.md` — remove twin flags, add `generate-header-templates`
- `docs/inference_pipeline.md` — add header injection to Stage 3
- `docs/overview.md` — update repo structure listing

---

## Background for New Engineers

**What is PrIMuS?** A dataset of ~87k single-staff music images rendered with LilyPond. Every image has a full header: treble clef glyph + key signature glyphs + time signature glyphs. The CRNN was trained exclusively on these.

**What is the problem?** Real Book pages have one full-header staff (line 1) followed by continuation staves (lines 2-N) with NO clef, key, or time glyphs. The CRNN has never seen this during training, so it may hallucinate or mis-recognise notes.

**The twin approach (being deleted):** Generated synthetic "no-header" variants of PrIMuS samples and added them to training. Flawed because: (a) max 50% headerless in training vs ~86% at inference, (b) degrades first-staff header detection.

**The new approach:** At inference, detect the key+time from the first music staff (it has a real header), then prepend a prerendered header image to each continuation staff before the CRNN sees it. The model always sees a full-header staff — exactly what it was trained on.

**Key source-of-truth files to read before coding:**
- `src/CRNN_CTC/lilypond_render.py` — `KEY_LY`, `LY_TEMPLATE`, `run_lilypond()`, `crop_content()` (used in Tasks 7 and 8)
- `src/omr_pipeline/grammar_fix.py` — `fix_sequence()`, `_COMMON_TIME_SIGS` (used in Task 9)
- `src/omr_pipeline/pipeline.py` — `_process_systems()` (modified in Task 9)
- `src/omr_pipeline/inference.py` — `recognize_music()` signature (used in Task 9)

---

## Task 1: Delete Twin Data from Disk

**Files:** data directories only

- [ ] **Step 1: Count existing twin samples before deletion**

```bash
echo "Clean twins:  $(ls data/processed/primus/clean/ | grep '__nh' | wc -l)"
echo "Scanned twins: $(ls data/processed/primus/scanned/ | grep '__nh' | wc -l)"
```

Expected: both show ~26168.

- [ ] **Step 2: Delete twin sample directories**

```bash
find data/processed/primus/clean   -maxdepth 1 -name '*__nh' -type d | wc -l
find data/processed/primus/scanned -maxdepth 1 -name '*__nh' -type d | wc -l
```

Confirm counts match ~26168 each, then delete:

```bash
find data/processed/primus/clean   -maxdepth 1 -name '*__nh' -type d -exec rm -rf {} +
find data/processed/primus/scanned -maxdepth 1 -name '*__nh' -type d -exec rm -rf {} +
```

- [ ] **Step 3: Verify deletion**

```bash
echo "Clean twins remaining:   $(ls data/processed/primus/clean/ | grep '__nh' | wc -l)"
echo "Scanned twins remaining: $(ls data/processed/primus/scanned/ | grep '__nh' | wc -l)"
```

Expected: both show `0`.

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "data: delete __nh headerless twin sample directories"
```

---

## Task 2: Remove Twin Code from `generate_realbook.py`

**Files:**
- Modify: `src/data_processing/generate_realbook.py:297-344`

- [ ] **Step 1: Delete the twin block**

Open `src/data_processing/generate_realbook.py`. Delete everything from line 297 (the `# ----` separator before the twin comment block) through line 344 (end of `headerless_label_tokens()`). This removes:
- The entire comment block explaining twins
- `_NEW_STAFF_RE`
- `_TWIN_DROP_EXACT`
- `_TWIN_DROP_PREFIX`
- `omit_header_in_ly()`
- `headerless_label_tokens()`

The next line after the deletion should be `# ---------------------------------------------------------------------------` followed by `# Per-sample processing`.

- [ ] **Step 2: Verify no twin symbols remain in the file**

```bash
grep -n "omit_header_in_ly\|headerless_label\|_TWIN_DROP\|_NEW_STAFF_RE\|__nh\|continuation" \
    src/data_processing/generate_realbook.py
```

Expected: no output.

- [ ] **Step 3: Syntax check**

```bash
poetry run python -c "import sys; sys.path.insert(0,'src'); from data_processing.generate_realbook import process_sample; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add src/data_processing/generate_realbook.py
git commit -m "refactor: remove twin generation functions from generate_realbook"
```

---

## Task 3: Delete `generate_headerless_twins.py`

**Files:**
- Delete: `src/data_processing/generate_headerless_twins.py`

- [ ] **Step 1: Confirm nothing else imports from it**

```bash
grep -r "generate_headerless_twins\|run_headerless_twins" src/ scripts/ --include="*.py"
```

Expected: only `src/cli.py` references it (will be fixed in Task 6).

- [ ] **Step 2: Delete the file**

```bash
rm src/data_processing/generate_headerless_twins.py
```

- [ ] **Step 3: Commit**

```bash
git add src/data_processing/generate_headerless_twins.py
git commit -m "refactor: delete generate_headerless_twins.py"
```

---

## Task 4: Clean `dataset.py`

**Files:**
- Modify: `src/CRNN_CTC/dataset.py`

**Context:** `strip_header_prob` is noted in the code as "inert" (never does anything — see line ~454). `_strip_header_tokens` and `_HEADER_PREFIXES` are dead code. `_AugSubset` still handles `online_aug_prob` which IS used, so we keep `_AugSubset` but strip the `strip_header_prob` parameter from it.

- [ ] **Step 1: Remove dead code — `_HEADER_PREFIXES` and `_strip_header_tokens`**

Delete lines 260–275 (the constants and function):

```python
# DELETE these lines:
_HEADER_PREFIXES = ("clef:", "key:fifths:", "time", "beats:", "beat-type:")


def _strip_header_tokens(tokens: list[str]) -> list[str]:
    """Remove leading header tokens (clef, key, time) from an LMX sequence.
    ...
    """
    if not tokens or tokens[0] != "measure":
        return tokens

    i = 1
    while i < len(tokens) and any(tokens[i] == p or tokens[i].startswith(p) for p in _HEADER_PREFIXES):
        i += 1
    return ["measure"] + tokens[i:]
```

Also delete the comment block at lines 278–284 that explains the old crop approach.

- [ ] **Step 2: Simplify `_AugSubset` — remove `strip_header_prob`**

Replace the `_AugSubset` class (lines ~535–576) with this stripped version:

```python
class _AugSubset(Dataset):
    """Thin wrapper around a ``Subset`` that enables training-only online augmentation.

    Temporarily sets ``_online_aug_prob`` on the underlying ``OMRDataset``
    before each ``__getitem__`` call. Safe across DataLoader workers because
    each worker gets an independent copy via fork/pickle.
    """

    def __init__(self, subset: Dataset, online_aug_prob: float) -> None:
        self._subset = subset
        self._online_prob = online_aug_prob

    def __len__(self) -> int:
        return len(self._subset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        ds: OMRDataset = self._subset.dataset  # type: ignore[attr-defined]
        old_online = ds._online_aug_prob
        ds._online_aug_prob = self._online_prob
        try:
            return self._subset[idx]
        finally:
            ds._online_aug_prob = old_online
```

- [ ] **Step 3: Remove `strip_header_prob` from `make_splits` signature and body**

In `make_splits()` (around line 594):
- Remove `strip_header_prob: float = 0.0` from the parameter list
- Remove its docstring entry
- Change `if strip_header_prob > 0 or online_aug_prob > 0:` → `if online_aug_prob > 0:`
- Change `_AugSubset(train_ds, strip_header_prob, online_aug_prob)` → `_AugSubset(train_ds, online_aug_prob)`
- Same for the finetune block (~line 690–695)

- [ ] **Step 4: Also remove `strip_header_prob` from `OMRDataset.__init__` (line ~335) and `self._strip_header_prob` assignment (line ~344)**

The field is set in `__init__` but never read (it's inert). Remove both lines:
```python
# DELETE:
strip_header_prob: float = 0.0,   # in __init__ signature
# ...
self._strip_header_prob = strip_header_prob  # in __init__ body
```

**Note:** Do NOT touch `config.py` — `strip_header_prob` stays there for checkpoint serialization compatibility.

- [ ] **Step 5: Verify**

```bash
poetry run python -c "
import sys; sys.path.insert(0,'src')
from CRNN_CTC.vocab import Vocabulary
from CRNN_CTC.dataset import make_splits
v = Vocabulary.from_file('data/vocab/primus_lmx.txt')
print('make_splits signature ok')
import inspect
sig = inspect.signature(make_splits)
assert 'strip_header_prob' not in sig.parameters, 'strip_header_prob still present'
print('ok')
"
```

Expected: `ok`

- [ ] **Step 6: Commit**

```bash
git add src/CRNN_CTC/dataset.py
git commit -m "refactor: remove strip_header_prob dead code from dataset.py"
```

---

## Task 5: Clean `train.py` and `evaluate_full.py`

**Files:**
- Modify: `src/CRNN_CTC/train.py:121`
- Modify: `scripts/evaluate_full.py:175`

- [ ] **Step 1: Remove `strip_header_prob` from `train.py`**

Find the `make_splits(...)` call in `src/CRNN_CTC/train.py` (around line 121). Delete the line:

```python
strip_header_prob=cfg.strip_header_prob,   # DELETE this line
```

- [ ] **Step 2: Remove `strip_header_prob` from `evaluate_full.py`**

Find the `make_splits(...)` call in `scripts/evaluate_full.py` (around line 175). Delete:

```python
strip_header_prob=0.0,   # DELETE this line
```

- [ ] **Step 3: Verify both files parse and import cleanly**

```bash
poetry run python -c "import sys; sys.path.insert(0,'src'); from CRNN_CTC import train; print('train ok')"
poetry run python -c "import ast; ast.parse(open('scripts/evaluate_full.py').read()); print('evaluate_full ok')"
```

Expected: both print `ok`.

- [ ] **Step 4: Commit**

```bash
git add src/CRNN_CTC/train.py scripts/evaluate_full.py
git commit -m "refactor: remove strip_header_prob from train and evaluate_full"
```

---

## Task 6: Rework `cli.py`

**Files:**
- Modify: `src/cli.py`

Remove all twin plumbing; add `generate-header-templates` subcommand; update the `pipeline` command's stage 3 to call header-template generation instead.

- [ ] **Step 1: Remove twin constants (top of file, ~lines 53–55)**

Delete:
```python
_FULL_RUN_HEADERLESS_FRACTION = 0.35
_FULL_RUN_HEADERLESS_DPI: tuple[int, ...] = (180, 200, 220)
_FULL_RUN_HEADERLESS_SEED = 42
```

- [ ] **Step 2: Remove `--force-twins` from `_add_pipeline_rebuild_args` (~lines 95–99)**

Delete the `g_force.add_argument("--force-twins", ...)` block.

Also remove `"force_twins"` from `_resolve_pipeline_force()` return dict.

- [ ] **Step 3: Remove the entire `g_twins` argument group (~lines 105–129)**

Delete everything from `g_twins = parser.add_argument_group("header-less twins (__nh)")` through the last `g_twins.add_argument(...)`.

- [ ] **Step 4: Delete `cmd_headerless_twins` function (~lines 425–438)**

Delete the entire function including its comment header.

- [ ] **Step 5: Replace pipeline stage 3 in `cmd_pipeline` (~lines 495–510)**

Replace:
```python
    # 3. Header-less continuation-staff twins (__nh) — before augment so twins get scanned variants
    if not getattr(args, "no_headerless_twins", False):
        log.info("--- Generating header-less (__nh) twin samples ---")
        twins_args = argparse.Namespace(...)
        cmd_headerless_twins(twins_args)
    else:
        log.info("--- Skipping header-less twins (--no-headerless-twins) ---")
```

With:
```python
    # 3. Prerender header templates for virtual header injection at inference
    log.info("--- Generating header templates for virtual header injection ---")
    tmpl_args = argparse.Namespace(output=Path("data/header_templates"), force=getattr(args, "force_all", False))
    cmd_generate_header_templates(tmpl_args)
```

- [ ] **Step 6: Add `cmd_generate_header_templates` and wire up the subcommand**

Add after the existing `cmd_convert` block (keep the style consistent):

```python
# ── generate-header-templates ──────────────────────────────────────────────


def cmd_generate_header_templates(args: argparse.Namespace) -> None:
    """Prerender clef+key+time header strip templates for virtual header injection."""
    from data_processing.generate_header_templates import generate_all_templates

    generate_all_templates(
        output_dir=Path(args.output),
        force=getattr(args, "force", False),
    )
```

In the `main()` subparser registration block, add:

```python
    p_tmpl = sub.add_parser(
        "generate-header-templates",
        help="Prerender 120 header-strip templates (15 keys × 8 time sigs) for virtual header injection.",
    )
    p_tmpl.add_argument(
        "--output",
        type=Path,
        default=Path("data/header_templates"),
        help="Directory to write template PNGs (default: data/header_templates).",
    )
    p_tmpl.add_argument(
        "--force",
        action="store_true",
        help="Re-render even if templates already exist.",
    )
    p_tmpl.set_defaults(func=cmd_generate_header_templates)
```

- [ ] **Step 7: Update the module docstring** (top of cli.py, the `Subcommands` list)

Change:
```
pipeline    render → convert → header-less twins → augment → vocab.
pipeline-train  pipeline, then train (--force-all for full rebuild).
```
To:
```
generate-header-templates  Prerender 120 clef+key+time templates for inference.
pipeline    render → convert → header-templates → augment → vocab.
pipeline-train  pipeline, then train (--force-all for full rebuild).
```

- [ ] **Step 8: Verify CLI loads and shows new subcommand**

```bash
poetry run python src/cli.py --help | grep -E "generate-header|pipeline|twins"
```

Expected: see `generate-header-templates`, no mention of `twins`.

```bash
poetry run python src/cli.py generate-header-templates --help
```

Expected: shows `--output` and `--force` flags.

- [ ] **Step 9: Commit**

```bash
git add src/cli.py
git commit -m "refactor: remove twin CLI plumbing, add generate-header-templates subcommand"
```

---

## Task 7: Create `generate_header_templates.py` and Run It

**Files:**
- Create: `src/data_processing/generate_header_templates.py`
- Creates data: `data/header_templates/key_{N}_time_{beats}_{beat_type}.png` (120 files)

**Context:** Uses `KEY_LY` (maps fifths→`\key ... \major` string) and `run_lilypond` + `crop_content` from `src/CRNN_CTC/lilypond_render.py`. Keys go from -6 to +7 (13 values per the domain spec; we add -7 for completeness = 15 total). Time sigs come from `grammar_fix._COMMON_TIME_SIGS`.

- [ ] **Step 1: Create the file**

```python
# src/data_processing/generate_header_templates.py
"""
generate_header_templates.py
============================
Offline script: prerender 120 header-strip PNG templates (15 key signatures ×
8 time signatures) using LilyPond + LilyJAZZ.  The resulting images are stored
in ``data/header_templates/`` and loaded at inference time by
``src/omr_pipeline/header_injector.py`` to prepend clef+key+time glyphs to
continuation staff images before the CRNN.

Run once before inference (included in ``cli.py pipeline`` stage 3)::

    poetry run python src/cli.py generate-header-templates
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from CRNN_CTC.lilypond_render import KEY_LY, crop_content, run_lilypond

log = logging.getLogger(__name__)

# All key signatures the system supports (fifths -7 … +7)
_ALL_FIFTHS: tuple[int, ...] = tuple(range(-7, 8))  # -7 to +7 inclusive = 15 keys

# All time signatures the system supports (must match grammar_fix._COMMON_TIME_SIGS)
_ALL_TIMES: tuple[tuple[int, int], ...] = (
    (4, 4), (3, 4), (2, 4), (2, 2),
    (6, 8), (6, 4), (5, 4), (12, 8),
)

_DPI = 200
_STAFF_SIZE = 17  # matches training renders in generate_realbook.py

_LY_TEMPLATE = r"""
\version "2.26.0"
#(set-global-staff-size {staff_size})
\include "lilyjazz.ily"
\header {{ tagline = ##f }}
\paper {{
  indent = 0
  ragged-right = ##t
  top-margin = 6\mm
  bottom-margin = 6\mm
  left-margin = 4\mm
  right-margin = 4\mm
  paper-height = 55\mm
}}
\score {{
  \new Staff {{
    \clef treble
    {key_cmd}
    \time {beats}/{beat_type}
    s1
  }}
  \layout {{ \context {{ \Score \omit BarNumber }} }}
}}
""".strip()


def template_filename(fifths: int, beats: int, beat_type: int) -> str:
    """Canonical filename for a given key+time combination."""
    return f"key_{fifths}_time_{beats}_{beat_type}.png"


def _render_one(fifths: int, beats: int, beat_type: int, output_dir: Path) -> bool:
    """Render a single template. Returns True on success."""
    out_path = output_dir / template_filename(fifths, beats, beat_type)
    key_cmd = KEY_LY[fifths]
    ly_src = _LY_TEMPLATE.format(
        staff_size=_STAFF_SIZE,
        key_cmd=key_cmd,
        beats=beats,
        beat_type=beat_type,
    )
    with tempfile.TemporaryDirectory(prefix="tmpl_") as tmp:
        png = run_lilypond(ly_src, f"tmpl_{fifths}_{beats}_{beat_type}", Path(tmp), dpi=_DPI)
        if png is None:
            log.warning("LilyPond failed for key=%d time=%d/%d", fifths, beats, beat_type)
            return False
        try:
            img = np.array(Image.open(png).convert("L"))
            cropped = crop_content(img)
        except Exception as exc:
            log.warning("Crop failed for key=%d time=%d/%d: %s", fifths, beats, beat_type, exc)
            return False
        if cropped.size == 0 or np.all(cropped == 255):
            log.warning("Empty render for key=%d time=%d/%d", fifths, beats, beat_type)
            return False
        Image.fromarray(cropped).save(out_path)
    return True


def generate_all_templates(
    output_dir: Path = Path("data/header_templates"),
    force: bool = False,
) -> dict[str, int]:
    """Generate all 120 templates. Returns {'ok': N, 'skip': N, 'fail': N}."""
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "skip": 0, "fail": 0}
    total = len(_ALL_FIFTHS) * len(_ALL_TIMES)
    done = 0
    for fifths in _ALL_FIFTHS:
        for beats, beat_type in _ALL_TIMES:
            path = output_dir / template_filename(fifths, beats, beat_type)
            if path.exists() and not force:
                counts["skip"] += 1
            elif _render_one(fifths, beats, beat_type, output_dir):
                counts["ok"] += 1
            else:
                counts["fail"] += 1
            done += 1
            if done % 20 == 0 or done == total:
                log.info("Templates: %d/%d (ok=%d skip=%d fail=%d)",
                         done, total, counts["ok"], counts["skip"], counts["fail"])
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    generate_all_templates(force="--force" in sys.argv)
```

- [ ] **Step 2: Run the generator**

```bash
poetry run python src/cli.py generate-header-templates
```

Expected: logs ~120 templates rendered, ends with `ok=120 skip=0 fail=0`.

- [ ] **Step 3: Verify output**

```bash
ls data/header_templates/ | wc -l
ls data/header_templates/ | head -5
```

Expected: `120` files, names like `key_-2_time_4_4.png`, `key_0_time_3_4.png`, etc.

```bash
poetry run python -c "
from PIL import Image
import os
p = 'data/header_templates'
imgs = [f for f in os.listdir(p) if f.endswith('.png')]
for f in imgs[:3]:
    img = Image.open(os.path.join(p, f))
    print(f, img.size)
"
```

Expected: each image is wider than tall (e.g., `(180, 60)`) — showing the header glyphs on a short staff strip.

- [ ] **Step 4: Commit**

```bash
git add src/data_processing/generate_header_templates.py
# do NOT add data/header_templates/ to git — add to .gitignore instead
echo "data/header_templates/" >> .gitignore
git add .gitignore
git commit -m "feat: add generate_header_templates.py for virtual header injection"
```

---

## Task 8: Create `header_injector.py` with Tests (TDD)

**Files:**
- Create: `src/omr_pipeline/header_injector.py`
- Create: `tests/test_header_injector.py`

**Context:** The injector loads a prerendered template (grayscale PNG), resizes it to match the staff image height, then horizontally concatenates it to the left of the staff image. If no template is found, it returns the staff image unchanged (graceful fallback). `global_key` from `grammar_fix` is a string like `"key:fifths:-2"`. `global_time` is a 3-tuple like `("time", "beats:4", "beat-type:4")`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_header_injector.py
"""Tests for the virtual header injector."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

# Adjust path so we can import from src/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from omr_pipeline.header_injector import inject_header, load_template, _template_path


TEMPLATES_DIR = Path("data/header_templates")


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

def test_template_path_encodes_key_and_time():
    """Filename must embed fifths, beats, beat_type."""
    p = _template_path("key:fifths:-2", ("time", "beats:4", "beat-type:4"))
    assert p.name == "key_-2_time_4_4.png"

def test_template_path_positive_key():
    p = _template_path("key:fifths:3", ("time", "beats:3", "beat-type:4"))
    assert p.name == "key_3_time_3_4.png"

def test_template_path_zero_key():
    p = _template_path("key:fifths:0", ("time", "beats:6", "beat-type:8"))
    assert p.name == "key_0_time_6_8.png"

@pytest.mark.skipif(not TEMPLATES_DIR.exists(), reason="data/header_templates not generated")
def test_all_120_templates_exist():
    """All key × time combinations must have a template file."""
    from data_processing.generate_header_templates import _ALL_FIFTHS, _ALL_TIMES, template_filename
    missing = []
    for fifths in _ALL_FIFTHS:
        for beats, beat_type in _ALL_TIMES:
            p = TEMPLATES_DIR / template_filename(fifths, beats, beat_type)
            if not p.exists():
                missing.append(p.name)
    assert not missing, f"Missing templates: {missing[:5]}"

@pytest.mark.skipif(not TEMPLATES_DIR.exists(), reason="data/header_templates not generated")
def test_load_template_returns_grayscale_array():
    template = load_template("key:fifths:0", ("time", "beats:4", "beat-type:4"))
    assert template is not None
    assert template.ndim == 2
    assert template.dtype == np.uint8
    assert template.shape[0] > 0 and template.shape[1] > 0

def test_load_template_returns_none_when_missing(tmp_path, monkeypatch):
    """Missing template → None, no exception."""
    import omr_pipeline.header_injector as hi
    monkeypatch.setattr(hi, "TEMPLATES_DIR", tmp_path)
    result = load_template("key:fifths:0", ("time", "beats:4", "beat-type:4"))
    assert result is None


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def test_inject_header_widens_image(monkeypatch):
    """Injected image must be wider than the original staff."""
    import omr_pipeline.header_injector as hi
    fake_template = np.full((50, 30), 255, dtype=np.uint8)
    monkeypatch.setattr(hi, "load_template", lambda k, t: fake_template)
    staff = np.full((50, 200), 128, dtype=np.uint8)
    result = hi.inject_header(staff, "key:fifths:0", ("time", "beats:4", "beat-type:4"))
    assert result.shape[1] > staff.shape[1]

def test_inject_header_height_unchanged(monkeypatch):
    """Injecting must not change the staff height."""
    import omr_pipeline.header_injector as hi
    fake_template = np.full((80, 40), 255, dtype=np.uint8)
    monkeypatch.setattr(hi, "load_template", lambda k, t: fake_template)
    staff = np.full((100, 300), 128, dtype=np.uint8)
    result = hi.inject_header(staff, "key:fifths:0", ("time", "beats:4", "beat-type:4"))
    assert result.shape[0] == 100

def test_inject_header_template_resized_to_staff_height(monkeypatch):
    """Template is scaled to match the staff image height before concat."""
    import omr_pipeline.header_injector as hi
    fake_template = np.full((50, 30), 255, dtype=np.uint8)
    monkeypatch.setattr(hi, "load_template", lambda k, t: fake_template)
    staff = np.full((100, 200), 128, dtype=np.uint8)
    result = hi.inject_header(staff, "key:fifths:0", ("time", "beats:4", "beat-type:4"))
    # Template at 50px scaled to 100px → width becomes 60px → total = 260
    assert result.shape == (100, 260)

def test_inject_header_falls_back_when_no_template(monkeypatch):
    """When template is missing, return staff image unchanged."""
    import omr_pipeline.header_injector as hi
    monkeypatch.setattr(hi, "load_template", lambda k, t: None)
    staff = np.full((100, 300), 42, dtype=np.uint8)
    result = hi.inject_header(staff, "key:fifths:0", ("time", "beats:4", "beat-type:4"))
    assert result is staff  # same object returned unchanged
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_header_injector.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'omr_pipeline.header_injector'` or similar import errors — tests fail because the module doesn't exist yet.

- [ ] **Step 3: Implement `header_injector.py`**

```python
# src/omr_pipeline/header_injector.py
"""
Virtual header injector for continuation-staff CRNN inference.

Loads prerendered clef+key+time strip templates from ``data/header_templates/``
(generated by ``src/data_processing/generate_header_templates.py``) and
prepends them to staff images before the CRNN so the model always sees a
full-header staff — matching its training distribution exactly.

If a template is missing (templates not generated yet), ``inject_header``
returns the staff image unchanged and logs a warning.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "header_templates"


def _template_path(key: str, time_tuple: tuple[str, str, str]) -> Path:
    """Map (key string, time tuple) → template PNG path.

    Args:
        key: e.g. ``"key:fifths:-2"``
        time_tuple: e.g. ``("time", "beats:4", "beat-type:4")``
    """
    fifths = int(key.split(":")[2])
    beats = int(time_tuple[1].split(":")[1])
    beat_type = int(time_tuple[2].split(":")[1])
    return TEMPLATES_DIR / f"key_{fifths}_time_{beats}_{beat_type}.png"


def load_template(key: str, time_tuple: tuple[str, str, str]) -> np.ndarray | None:
    """Load the header template for *key* + *time_tuple*.

    Returns a grayscale uint8 ndarray, or ``None`` if the template file is
    missing (templates not yet generated).
    """
    path = _template_path(key, time_tuple)
    if not path.exists():
        log.warning("Header template not found: %s — run 'cli.py generate-header-templates'", path.name)
        return None
    return np.array(Image.open(path).convert("L"))


def inject_header(
    staff_img: np.ndarray,
    key: str,
    time_tuple: tuple[str, str, str],
) -> np.ndarray:
    """Prepend the matching header template to *staff_img*.

    The template is resized to match the staff image height (preserving aspect
    ratio) before horizontal concatenation. If the template is missing, the
    original *staff_img* is returned unchanged.

    Args:
        staff_img: Grayscale uint8 ndarray, shape (H, W).
        key: e.g. ``"key:fifths:-2"``
        time_tuple: e.g. ``("time", "beats:4", "beat-type:4")``
    """
    template = load_template(key, time_tuple)
    if template is None:
        return staff_img

    h = staff_img.shape[0]
    th, tw = template.shape[:2]
    new_w = max(1, round(tw * h / th))
    resized = cv2.resize(template, (new_w, h), interpolation=cv2.INTER_AREA)
    return np.concatenate([resized, staff_img], axis=1)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_header_injector.py -v
```

Expected: all tests pass (the `skipif` tests are skipped if templates not generated, which is fine).

- [ ] **Step 5: Commit**

```bash
git add src/omr_pipeline/header_injector.py tests/test_header_injector.py
git commit -m "feat: add header_injector module with tests"
```

---

## Task 9: Two-Pass CRNN in `pipeline.py`

**Files:**
- Modify: `src/omr_pipeline/pipeline.py`

**How it works:**
1. **First pass:** Run all detected staff images through the CRNN (unchanged — exactly the current behaviour).
2. **Grammar fix pass 1:** Identify `first_keyed_idx` — the index of the first staff that produced a time signature (i.e., the first real music staff).
3. **Second pass (continuation staves only):** For all staves after `first_keyed_idx` that are not geometry-unrecoverable, inject the header template and re-run the CRNN.
4. **Grammar fix pass 2:** Re-run grammar fixing with the corrected second-pass predictions.

The post-CRNN rejection gate uses `music_logprobs` and `music_outlens`; update those arrays too after the second pass.

- [ ] **Step 1: Add the import at the top of `pipeline.py`**

Add after the existing imports:

```python
from .header_injector import inject_header
```

- [ ] **Step 2: Replace the grammar-fix block in `_process_systems` to track `first_keyed_idx`**

Find the current grammar-fix block (~lines 111–140):

```python
    # LMX grammar correction with cross-system key + time propagation
    global_key: str | None = None
    global_time: tuple[str, str, str] | None = None
    fixed_music: list[str] = []
    for pred in music_preds:
        fixed, global_key, global_time = fix_sequence(
            pred,
            global_key=global_key,
            global_time=global_time,
            force_clef=True,
        )
        fixed_music.append(fixed)

    # If no staff produced a time signature, fall back to 4/4 ...
    if global_time is None:
        ...
```

Replace with:

```python
    # LMX grammar correction — pass 1: find key+time from first music staff
    global_key: str | None = None
    global_time: tuple[str, str, str] | None = None
    first_keyed_idx: int = -1
    fixed_music: list[str] = []
    for i, pred in enumerate(music_preds):
        fixed, global_key, global_time = fix_sequence(
            pred,
            global_key=global_key,
            global_time=global_time,
            force_clef=True,
        )
        if global_time is not None and first_keyed_idx < 0:
            first_keyed_idx = i
        fixed_music.append(fixed)

    # Virtual header injection: re-run CRNN on continuation staves with the
    # key+time header prepended so the model sees its training distribution.
    if first_keyed_idx >= 0 and global_key is not None and global_time is not None:
        continuation_idxs = [
            i for i in range(first_keyed_idx + 1, len(systems))
            if not crnn_skip_mask[i]
        ]
        if continuation_idxs:
            log.info(
                "Header injection: re-running CRNN on %d continuation staff(s) "
                "(key=%s time=%s %s)",
                len(continuation_idxs), global_key, global_time[1], global_time[2],
            )
            cont_imgs = [
                inject_header(music_imgs[i], global_key, global_time)
                for i in continuation_idxs
            ]
            cont_preds, cont_lp, cont_ol = recognize_music(cont_imgs, checkpoint_path)
            for j, i in enumerate(continuation_idxs):
                music_preds[i] = cont_preds[j]
                music_logprobs[i] = cont_lp[j]
                music_outlens[i] = cont_ol[j]

            # Grammar fix pass 2: re-fix with corrected continuation predictions
            global_key = None
            global_time = None
            fixed_music = []
            for pred in music_preds:
                fixed, global_key, global_time = fix_sequence(
                    pred,
                    global_key=global_key,
                    global_time=global_time,
                    force_clef=True,
                )
                fixed_music.append(fixed)

    # If no time signature was detected on any staff (first staff was title/rejected
    # and subsequent staves had no visible time glyph), fall back to 4/4.
    if global_time is None:
        _DEFAULT_TIME = ("time", "beats:4", "beat-type:4")
        log.info("No time signature detected on any staff; injecting 4/4 default")
        fixed_music = []
        for pred in music_preds:
            fixed, _, _ = fix_sequence(
                pred,
                global_key=global_key,
                global_time=_DEFAULT_TIME,
                force_clef=True,
            )
            fixed_music.append(fixed)
```

**Important:** `music_preds` must be a mutable list so the second pass can update it. `recognize_music` already returns a `list[str]`, so this works without changes to `inference.py`.

- [ ] **Step 3: Verify the pipeline still imports cleanly**

```bash
poetry run python -c "
import sys; sys.path.insert(0,'src')
from omr_pipeline.pipeline import run_pipeline
print('pipeline import ok')
"
```

Expected: `pipeline import ok`

- [ ] **Step 4: Smoke-test the pipeline on the Satin Doll image**

If you have the Satin Doll scan available:

```bash
poetry run python -c "
import sys; sys.path.insert(0, 'src')
from pathlib import Path
from omr_pipeline.pipeline import run_pipeline

data = Path('path/to/satin_doll.png').read_bytes()
result = run_pipeline(data, 'satin_doll.png')
segs = result['pages'][0]['segments']
accepted = [s for s in segs if not s['rejected']]
print(f'Total segments: {len(segs)}, accepted: {len(accepted)}')
for s in accepted:
    toks = s['lmx_tokens']
    has_time = 'time' in toks
    print(f'  bbox={s[\"staff_bbox\"][:2]}, time={has_time}, tokens={len(toks)}')
"
```

Expected: all accepted segments have `time=True` (time signature present). First segment has it from its own header; continuation segments have it from injected headers. No segment should have `time=False`.

- [ ] **Step 5: Commit**

```bash
git add src/omr_pipeline/pipeline.py
git commit -m "feat: two-pass CRNN with virtual header injection for continuation staves"
```

---

## Task 10: Update Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/data_pipeline.md`
- Modify: `docs/training.md`
- Modify: `docs/configuration.md`
- Modify: `docs/cli.md`
- Modify: `docs/inference_pipeline.md`
- Modify: `docs/overview.md`

- [ ] **Step 1: `CLAUDE.md`**

Replace the `__nh` twin bullet (~lines 105–112):

```markdown
- **Continuation staves use virtual header injection, not training twins.**
  Real Book continuation staves (all staves after the first) have no clef, key,
  or time signature. At inference, `src/omr_pipeline/header_injector.py` prepends
  a prerendered template (clef+key+time glyphs, from `data/header_templates/`) to
  each continuation staff image before the CRNN, so the model always sees a
  full-header staff matching its training distribution. Templates are generated
  once by `cli.py generate-header-templates` and are stage 3 of `cli.py pipeline`.
  `strip_header_prob` is DEPRECATED/inert (kept on `Config` only for checkpoint
  compatibility).
```

- [ ] **Step 2: `docs/data_pipeline.md`**

Find the "Stage 3 — Header-less twins (`__nh`)" section (~lines 105–120) and replace with:

```markdown
## Stage 3 — Header template generation

**Script:** `src/data_processing/generate_header_templates.py`
**CLI:** `poetry run python src/cli.py generate-header-templates`
**Output:** `data/header_templates/key_{N}_time_{beats}_{beat_type}.png` (120 files)

Prerenders 120 header-strip images (15 key signatures × 8 time signatures) using
LilyPond + LilyJAZZ at 200 DPI. These are used at inference time by
`header_injector.py` to prepend the correct clef+key+time glyphs to continuation
staff images before the CRNN. Run once; re-run with `--force` if LilyJAZZ
styling changes.
```

- [ ] **Step 3: `docs/training.md`**

Find the continuation-staves / strip_header_prob section (~lines 71–86) and replace with:

```markdown
### Continuation staves

Real Book continuation staves (all staves after the first on a page) carry no
header. The model is trained only on full-header PrIMuS staves — this is by
design. At inference, `header_injector.py` prepends a prerendered header to
each continuation staff before the CRNN (see `docs/inference_pipeline.md`), so
the model always receives its training distribution.

`strip_header_prob` appears in `Config` but is **DEPRECATED/inert** — it is
kept only so existing checkpoints deserialise without error.
```

In the config table, update the `strip_header_prob` row to: `DEPRECATED/inert — kept for checkpoint compatibility only`.

- [ ] **Step 4: `docs/configuration.md`**

Find `strip_header_prob` entry and update to:

```
| strip_header_prob | 0.0 | **DEPRECATED/inert.** Kept for checkpoint compatibility. Has no effect. |
```

- [ ] **Step 5: `docs/cli.md`**

- Remove entries for `--no-headerless-twins`, `--force-twins`, `--headerless-fraction`, `--headerless-dpi`, `--headerless-seed`.
- Add entry for `generate-header-templates` subcommand with `--output` and `--force` flags.
- Update `pipeline` description to say stage 3 is "header template generation".

- [ ] **Step 6: `docs/inference_pipeline.md`**

Add a new subsection after Stage 3 (Music Recognition) or update Stage 2b to describe the header injection:

```markdown
## Stage 3b — Virtual header injection (continuation staves)

**File:** `src/omr_pipeline/header_injector.py`

After the first CRNN pass, `_process_systems` identifies the first music staff
that produced a key and time signature. For all subsequent non-rejected staves,
it prepends the matching prerendered header template and re-runs the CRNN:

```
continuation staff image (bare):  [notes...]
after injection:                   [clef][key][time][notes...]
```

The CRNN then sees a full-header staff identical to its training data. Grammar
fix pass 2 runs on the corrected predictions. If no time signature was detected
at all (first staff was title text / rejected), the pipeline falls back to
inserting a 4/4 default.

Templates live in `data/header_templates/` (generated by
`cli.py generate-header-templates`). If a template is missing at runtime,
that staff is processed without injection and a warning is logged.
```

- [ ] **Step 7: `docs/overview.md`**

In the repository structure listing, replace:

```
│   ├── generate_headerless_twins.py  __nh continuation-staff twins
```

With:

```
│   ├── generate_header_templates.py  prerender 120 header-strip templates
```

- [ ] **Step 8: Verify no stale twin references remain in docs or code**

```bash
grep -rn "headerless.twin\|__nh\|generate_headerless\|no.headerless\|force.twins\|headerless.fraction" \
    docs/ CLAUDE.md src/ scripts/ --include="*.py" --include="*.md" \
    | grep -v "test_\|\.pyc\|DEPRECATED\|checkpoint compat"
```

Expected: no output (or only checkpoint-compat notes already updated above).

- [ ] **Step 9: Commit docs**

```bash
git add CLAUDE.md docs/
git commit -m "docs: replace twin documentation with virtual header injection"
```

---

## Self-Review

### Spec coverage check

| Requirement | Task |
|---|---|
| Delete twin data from disk | Task 1 |
| Remove `omit_header_in_ly`, `headerless_label_tokens` | Task 2 |
| Delete `generate_headerless_twins.py` | Task 3 |
| Remove dead dataset code (`_strip_header_tokens`, `_AugSubset.strip_header_prob`) | Task 4 |
| Remove `strip_header_prob` from training | Task 5 |
| Remove twin CLI subcommand and pipeline stage | Task 6 |
| Add `generate-header-templates` CLI subcommand | Task 6 |
| Prerender 120 templates | Task 7 |
| Header injector with tests | Task 8 |
| Two-pass pipeline with injection | Task 9 |
| Update all documentation | Task 10 |

### Type consistency check

- `inject_header(staff_img: np.ndarray, key: str, time_tuple: tuple[str,str,str])` — used in Tasks 8 and 9 with same signature ✓
- `load_template(key: str, time_tuple: tuple[str,str,str]) → np.ndarray | None` — consistent across Tasks 8 and 9 ✓
- `global_key: str | None` (e.g. `"key:fifths:-2"`) — consistent across Task 9 and injector parsing ✓
- `global_time: tuple[str, str, str] | None` (e.g. `("time", "beats:4", "beat-type:4")`) — consistent ✓
- `music_preds: list[str]` — returned by `recognize_music`, mutated in second pass ✓

### Placeholder scan

No TBDs, no "similar to above", no missing test code — ✓

---

**Plan complete and saved to `docs/superpowers/plans/2026-06-03-virtual-header-injection.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks

**2. Inline Execution** — execute tasks in this session using executing-plans

Which approach?
