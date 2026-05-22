# Non-Music Strip Rejection — Design

**Date:** 2026-05-20
**Author:** Pol Casanovas (with Claude)
**Status:** Approved, ready for implementation plan

## Problem

On full-page Real Book scans, the morphological staff finder in `src/omr_pipeline/staff_detect.py` reports false staves over **title regions** (e.g. "SATIN DOLL — DUKE ELLINGTON") and **footer text** (e.g. "Ellingtonia, Vol. 2"). The reasons:

1. Title text contains horizontal strokes (underline, letter bars) that morphological opening picks up as candidate staff lines.
2. The five-line grouper accepts any 5 row-centroids whose inter-line spacings agree within tolerance — there is no requirement that those lines actually *span* the strip or that there be musical content *between* them.
3. The local validator `music_strip_has_valid_staff` only re-runs the same detection inside the crop; if 5 spurious lines were grouped at page level, they will be re-grouped at strip level too.

The downstream consequence: the CRNN dutifully decodes LMX tokens for these strips because it is never trained to emit "no music". Example output for a Satin Doll title strip:

```
measure clef:G2 key:fifths:0 time beats:2 beat-type:2 rest 32nd rest half
pitch:A octave:4 32nd rest 64th pitch:A octave:5 32nd sharp pitch:D octave:3 ...
```

This is pure hallucination. The transcription JSON returned by `run_pipeline` is corrupted by these phantom segments.

## Goal

Reject non-music strips before they reach the user-visible transcription, while preserving the bounding box and a structured reason so the UI / downstream tooling can render or hide them as it sees fit.

## Non-Goals

- Improving CRNN accuracy on real music staves.
- Re-training the CRNN with a "no-music" class.
- Rewriting the staff-line detector. The geometric/morphological detector stays; we add a gate layer on top.
- Handling multi-page scans differently (this design applies per-page identically).

## Design

### Architecture

One new module: `src/omr_pipeline/staff_reject.py`. Three rejection gates:

1. **Geometric gate** — pre-CRNN. Validates that the strip looks like a staff at the pixel level. Geometry-impossible strips drop entirely (no CRNN call, no segment in output).
2. **OCR text gate** — pre-CRNN. Runs EasyOCR's text detector on the strip. Strips where text bounding boxes occupy too much area drop into the "kept but rejected" path (bbox preserved, `rejected` set).
3. **CTC-confidence gate** — post-CRNN. Mean log-probability of the argmax frames. Catches hallucinations that slipped past the pre-CRNN gates.

A `RejectionResult` dataclass carries the per-strip verdict, the first failing gate's name, and a `diagnostics` dict with every signal value (passed or not). Diagnostics surface in the API response under `reject_diagnostics` so the calibration CLI can ingest API output directly.

### Module surface

```python
# src/omr_pipeline/staff_reject.py

@dataclass
class RejectionResult:
    passed: bool
    reason: str | None
    diagnostics: dict[str, float]

@dataclass
class RejectThresholds:
    min_line_span_frac: float = 0.70
    max_spacing_cov: float = 0.18
    min_interline_ink_frac: float = 0.005
    max_text_area_frac: float = 0.35
    min_mean_logprob: float = -1.2
    confident_override_logprob: float = -0.1   # see "CTC-confidence override"

DEFAULT_THRESHOLDS = RejectThresholds()

def load_thresholds() -> RejectThresholds: ...
    # Reads OMR_REJECT_THRESHOLDS (path to JSON); falls back to DEFAULT_THRESHOLDS.

def evaluate_pre_crnn(system: System, thresholds: RejectThresholds | None = None) -> RejectionResult: ...
def evaluate_post_crnn(
    system: System,
    log_probs: torch.Tensor,   # (T, C) for this strip
    out_len: int,
    pre_result: RejectionResult,
    thresholds: RejectThresholds | None = None,
) -> RejectionResult: ...
```

Threshold values above are placeholders. Real ones come from the calibration sweep (see "Calibration").

### Gate signals (precise definitions)

#### Geometric gate

Inputs: `system.music_binary` (uint8 0/1 mask, shape `(H, W)`).

1. **Line-span fraction**
   - Subtract a horizontal-opening of the binary (kernel width = `max(25, W // 6)`) from the binary itself to remove staff-line columns and keep notation ink. Then re-take long horizontal runs (kernel width = `max(40, W // 4)`) to get clean staff lines.
   - Re-run `local_primary_staff_lines` from `staff_detect.py` to get the 5 line y-coordinates.
   - For each line at y₀, define a 3-row band `[y₀-1, y₀+1]`. Count columns in that band with any ink. Divide by `W`.
   - `line_span_min = min(span_i for i in 0..4)`.
   - Reject if `line_span_min < thresholds.min_line_span_frac`.

2. **Spacing coefficient of variation (CoV)**
   - `gaps = [line_ys[i+1] - line_ys[i] for i in 0..3]`.
   - `spacing_cov = std(gaps) / mean(gaps)`.
   - Reject if `spacing_cov > thresholds.max_spacing_cov`.

3. **Inter-line ink density**
   - Define inter-line region = pixels strictly between `line_ys[0] + 1` and `line_ys[-1] - 1`, minus the ±1-row bands around each interior line.
   - `interline_ink_frac = ink_pixels_in_region / total_pixels_in_region`.
   - Reject if `interline_ink_frac < thresholds.min_interline_ink_frac`.

If `local_primary_staff_lines` returns `None`, set `line_span_min = 0`, `spacing_cov = ∞`, `interline_ink_frac = 0` and reject with reason `"geometry_no_staff_lines"`.

#### OCR text gate

Input: `system.music_image` (grayscale uint8, shape `(H, W)`).

- Lazy-load EasyOCR reader (singleton, same instance as `chord_recognizer`'s).
- `boxes = reader.readtext(image, detail=1, paragraph=False)` — only the detector output is needed; we ignore the recognized strings to save time.
  - If `readtext` is too slow with full recognition, switch to `reader.detect(image)` which is detector-only and faster.
- `text_area = sum(polygon_area(box) for box in boxes)`.
- `text_area_frac = text_area / (H * W)`.
- Reject if `text_area_frac > thresholds.max_text_area_frac`.

#### CTC-confidence gate

Inputs: `log_probs` shape `(T, C)`, `out_len` (effective frames after width compression).

- `frames = log_probs[:out_len]`.
- `argmax_frames = argmax(frames, dim=-1)`.
- `mean_logprob = mean(frames[i, argmax_frames[i]] for i in range(out_len))`.
- Reject if `mean_logprob < thresholds.min_mean_logprob`.

Edge case: if `out_len == 0` (strip was zero-width padding from preprocess failure), set `mean_logprob = -inf` and reject as `"ctc_zero_length"`.

### Combination rule

Within `evaluate_pre_crnn`: gates run in order **geometry → OCR**. First failure wins; remaining diagnostics still computed and recorded.

`evaluate_post_crnn` only runs the CTC gate; the pre-result is carried forward so the post result inherits any pre-gate failure (in the rare case the caller decides to keep a pre-rejected strip alive through CRNN for debugging).

Final segment status:
- `pre_result.passed AND post_result.passed` → segment is normal output.
- Any rejection (geometry, OCR, CTC) → keep segment with `rejected = reason`,
  empty `lmx_tokens`, empty `chords`. The UI hides them by default; the API
  consumer can inspect the diagnostics to debug.
- Geometry-rejected segments skip the CRNN call (so `mean_logprob` is `null`),
  but their bbox and full diagnostics are returned.

**CTC-confidence override (post-implementation addition).** A *geometry*
rejection is overridden if the CRNN's mean argmax log-prob is at or above
`thresholds.confident_override_logprob`. Rationale: sparse music (e.g. a
single whole rest) under-shoots `interline_ink_frac` and gets a false
geometry rejection — high CRNN confidence is the most reliable rescue
signal. **OCR text-density rejections are never overridden** (text in a
strip means it's not music, regardless of how confidently the CRNN
hallucinates tokens from the text strokes).

When an override fires, the returned `RejectionResult` has `passed=True`
and `reason=None` (so downstream treats the segment as a normal pass),
but `diagnostics["override_reason"]` carries the original failing gate
for audit. The field is absent on segments that were not overridden.

**Update (2026-05-20):** Earlier drafts of this spec dropped geometry-rejected
segments entirely. We reversed that during testing — silent drops make
detector regressions invisible. Always-include keeps the API debuggable.

### Import structure (avoid cycle)

`staff_reject.py` imports `System` and `local_primary_staff_lines` from `staff_detect.py`. To call `evaluate_pre_crnn` from inside `staff_detect.detect_systems`, the import lives at function-local scope (`from .staff_reject import evaluate_pre_crnn` inside the function body). This keeps `staff_detect.py` import-clean while still letting it call the gate.

### Data flow

```
staff_detect.detect_systems()
   └─ _build_systems() returns raw list of System
       └─ for each system: pre_result = evaluate_pre_crnn(system)
             ├─ if not pre_result.passed AND reason.startswith("geometry_"):
             │      log + drop
             └─ otherwise: system.pre_result = pre_result; keep

inference.recognize_music(strip_images, ckpt)
   ├─ returns (tokens_list, log_probs_list, out_lens_list)
   └─ shape change: previously returned tokens_list only.
       Updates Stage-3 doc in docs/inference_pipeline.md.

pipeline._process_systems(systems, ckpt)
   ├─ pulls images, runs recognize_music
   └─ for each system:
         post_result = evaluate_post_crnn(system, log_probs[i], out_lens[i], system.pre_result)
         segment["rejected"] = post_result.reason   # None when passed
         segment["reject_diagnostics"] = post_result.diagnostics
         if post_result.reason is not None:
             segment["lmx_tokens"] = []
             segment["chords"] = []
```

### Output contract (per segment)

```json
{
  "staff_bbox": [x, y, w, h],
  "chord_bbox": [x, y, w, h] | null,
  "lmx_tokens": [...] | [],
  "chords": [...] | [],
  "rejected": "ocr_text_density" | "ctc_low_confidence" | null,
  "reject_diagnostics": {
    "line_span_min": 0.81,
    "spacing_cov": 0.12,
    "interline_ink_frac": 0.0007,
    "text_area_frac": 0.62,
    "mean_logprob": null
  }
}
```

`reject_diagnostics` is always populated. Numeric fields use `null` only when the gate did not run (e.g. CRNN was skipped on a geometry-rejected strip — but those segments are dropped, so in practice all kept segments have all five numbers).

Page-level meta gains one field:
```json
"meta": {
  ...,
  "num_systems": 9,         // existing — total detected (post pre-CRNN drop)
  "num_rejected": 2          // new — count of kept-but-flagged segments
}
```

### Calibration CLI

New CLI subcommand: `poetry run python src/cli.py calibrate-reject --fixtures <dir> [--out thresholds.json]`.

**Fixture layout:**
```
tests/fixtures/reject/
├── music/        # real staff strips (PNG) — must pass
└── non_music/    # title regions, footer text, blank strips, page noise — must reject
```

Initial set:
- ~30 music strips harvested from `data/real_book/full_realbook.pdf` (random page sample) + a few from the synthetic eval set.
- ~15 non-music strips harvested from the same PDF (titles, footers, page numbers).

A harvesting helper subcommand `poetry run python src/cli.py harvest-reject-fixtures --pdfs <glob>` runs preprocessing + staff detection on a glob of PDFs and saves all detected music strips into `harvest_out/strips/NNNN.png`. The user then manually moves them into `music/` and `non_music/` by visual inspection.

**Sweep behavior:**
- For each gate's signal in isolation, computes precision/recall on the labeled set and recommends the threshold at Youden's J (max(TPR - FPR)).
- At the recommended thresholds, prints the combined-rule confusion matrix and a per-strip diagnostics table for misclassified strips.
- Writes the chosen thresholds to `--out` as JSON; defaults to `models/staff_reject/thresholds.json`.

**Loading:** `RejectThresholds.load_thresholds()` checks env `OMR_REJECT_THRESHOLDS` (path to JSON); falls back to `DEFAULT_THRESHOLDS`.

### Testing

**Unit tests** (`tests/test_staff_reject.py`):
- Each gate function fed synthetic strips and assert pass/reject:
  - Pure staff (5 evenly spaced lines, noteheads between) → passes geometry.
  - Title-text mock (5 short horizontal strokes, no inter-line ink) → fails on `interline_ink_frac`.
  - Noisy crop (random noise, no lines) → fails on `geometry_no_staff_lines`.
  - Text-heavy strip → fails on `text_area_frac` (mock EasyOCR).
  - Low-entropy CRNN output (uniform log-probs) → fails on `mean_logprob`.
- Threshold loading: env var, JSON file, fallback to defaults.

**Integration test** (`tests/integration/test_pipeline_rejects_titles.py`):
- Run `run_pipeline` on a saved Satin Doll page (PNG fixture in `tests/fixtures/pages/satin_doll.png`).
- Assert that:
  - The first detected strip (title region) is either dropped or has `rejected = "ocr_text_density"` (depending on which gate fires).
  - The middle strips (real music) are not rejected.
  - The last strip (footer) is either dropped or rejected.
  - `meta["num_systems"]` and `len(segments)` are consistent.

**Regression:** existing `tests/` must still pass. The API shape change is additive (new `rejected` and `reject_diagnostics` keys); no existing key changes shape.

### Touched files (summary)

| File | Change |
|------|--------|
| `src/omr_pipeline/staff_reject.py` | **new** |
| `src/omr_pipeline/staff_detect.py` | call `evaluate_pre_crnn` after `_build_systems`; drop geometry-rejected strips |
| `src/omr_pipeline/inference.py` | `recognize_music` returns `(tokens, log_probs, out_lens)` |
| `src/omr_pipeline/pipeline.py` | wire `evaluate_post_crnn`; populate `rejected` and `reject_diagnostics` |
| `src/cli.py` | new `calibrate-reject` and `harvest-reject-fixtures` subcommands |
| `docs/inference_pipeline.md` | update Stage-2 (mention the pre-CRNN gate), Stage-3 (mention CRNN returns log-probs), add new section for rejection |
| `docs/api.md` | document new `rejected` / `reject_diagnostics` fields |
| `docs/cli.md` | document the two new subcommands |
| `tests/test_staff_reject.py` | **new** — unit tests |
| `tests/integration/test_pipeline_rejects_titles.py` | **new** — integration test |
| `tests/fixtures/reject/{music,non_music}/` | **new** — labeled strips |
| `tests/fixtures/pages/satin_doll.png` | **new** — integration fixture |
| `models/staff_reject/thresholds.json` | **new** — calibrated thresholds |

### Risks

1. **EasyOCR latency on the music strip.** Adds an OCR pass per detected staff. Mitigation: detector-only call, lazy-singleton reader, and the geometric gate filters most negatives before OCR even runs.
2. **CTC threshold is checkpoint-dependent.** A retrained CRNN will shift the mean-logprob distribution. Mitigation: the calibration script is reproducible; re-run after every retrain. Document this in CLAUDE.md.
3. **Real staves with sparse content (e.g. whole rest only).** Could under-shoot `min_interline_ink_frac`. Mitigation: the calibration set must include sparse-music examples. If still problematic, relax to `min_interline_ink_frac = 0.002` or below.
4. **Fixture harvesting is manual.** ~45 strips need human labeling. Mitigation: the `harvest-reject-fixtures` helper writes labeled PNGs with predictable filenames; sorting takes <30 minutes.

### Open questions

None at the time of writing — surfaced during brainstorming and resolved:
- Hybrid signals: confirmed.
- Output behavior: keep bbox + flag for OCR/CTC failures; drop geometry-impossible strips entirely.
- Calibration: sweep script.
- Module structure: dedicated `staff_reject.py`.
