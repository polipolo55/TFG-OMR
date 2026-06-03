# Inference Pipeline

The OMR pipeline processes a PDF or image through five sequential stages to produce a structured JSON transcription.

**Entry point:** `src/omr_pipeline/pipeline.py` → `run_pipeline(data: bytes, filename: str) → dict`

## Stage 1 — Preprocessing

**File:** `src/omr_pipeline/preprocess.py`

```
bytes / Path / ndarray
        │
        ▼
  load_image()         → grayscale uint8
        │
        ▼ (if PDF)
  load_pdf_page()      → rasterize via PyMuPDF at pdf_load_dpi()
                        (env OMR_PDF_DPI, default 300)
        │
        ▼
  binarize()           CLAHE → Otsu threshold → binary
        │
        ▼
  deskew()             projection-profile rotation correction
```

**Binarization:** CLAHE (Contrast Limited Adaptive Histogram Equalization) applied first to normalize local contrast, then Otsu thresholding on the CLAHE output. Produces a clean binary image robust to uneven scanner illumination.

**Deskewing:** Tests a range of rotation angles; selects the angle that maximizes variance of the horizontal projection profile (sharpest row peaks = best alignment).

## Stage 2 — Staff Detection

**File:** `src/omr_pipeline/staff_detect.py`

```
binary image
        │
        ▼
  _extract_line_mask()       morphological opening (wide horizontal kernel)
        │
        ▼
  _line_row_centroids()      find row centroids, merge thick lines
        │
        ▼
  _cluster_into_5_lines()    group centroids into consecutive 5-line sets
        │
        ▼
  _associate_chords()        chord region = vertical gap above each staff
        │
        ▼
  [Staff objects]            y-coords, staff-space, x-span, cropped images
```

**Data structures:**

```python
@dataclass
class Staff:
    line_ys: list[int]     # 5 y-coordinates (px)
    staff_space: float     # avg distance between adjacent lines
    x_start: int
    x_end: int

@dataclass
class System:
    staff: Staff
    chord_bbox: tuple[int,int,int,int] | None  # (x, y, w, h) or None
    music_bbox: tuple[int,int,int,int]         # (x, y, w, h)
    chord_image: ndarray | None                # region above staff (chord symbols)
    music_image: ndarray | None                # cropped grayscale staff region
    chord_binary: ndarray | None               # binarized chord strip
    music_binary: ndarray | None               # binarized music strip
    pre_result: object | None                  # set by pre-CRNN gate; None = gate not yet run
```

## Stage 2b — Rejection Gates

**File:** `src/omr_pipeline/staff_reject.py`

Detected staff systems pass through a hybrid pre-CRNN + post-CRNN gate that
filters out title regions, footer text, and other non-music page elements.

**Pre-CRNN gates** (run inside `detect_systems`):

| Gate | Signal | Reject when |
|------|--------|-------------|
| `geometry_no_image` | music image present | no image extracted |
| `geometry_no_strip` | binary strip present | no binary strip extracted |
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

**CTC confidence override.** Geometric pre-CRNN gates sometimes mis-fire on
sparse music (e.g. a single whole note on a staff where `interline_ink_frac`
ends up below the threshold). To recover those, every strip with a borderline
geometric verdict still goes through the CRNN. If the resulting `mean_logprob`
is at or above `confident_override_logprob`, the geometric rejection is
overridden and the strip is accepted.

**The override only applies to `geometry_*` rejections.** `ocr_text_density`
is treated as a stronger signal — when EasyOCR finds significant text in the
strip, that means it's not music, regardless of how confidently the CRNN
hallucinates tokens from the text strokes. This avoids a class of false
positives where footer text under an empty staff produces high-confidence
CRNN output.

**Behaviour on rejection:**
- All rejections (geometry, OCR, CTC) **keep** the segment in `pages[].segments[]`
  with `rejected: "<reason>"`, empty `lmx_tokens`, empty `chords`, and full
  `reject_diagnostics`. The UI is expected to hide segments with non-null
  `rejected`. Keeping them in the response makes debugging tractable — you can
  see exactly which strips were tried and why each verdict was reached.
- Geometry-rejected strips still **skip the CRNN call** (so `mean_logprob` is
  `null` for them) — only their diagnostics and bbox survive.

**Threshold calibration:** see `poetry run python src/cli.py calibrate-reject --help`.
After re-training the CRNN, **re-run calibration** — the CTC log-prob
distribution will shift.

## Stage 3 — Music Recognition (CRNN)

**File:** `src/omr_pipeline/inference.py`

```
[System.music_img]
        │
        ▼
  Staff-aware normalization:
    - resize height → 128 px
    - center staff vertically
    - per-image zero-mean / unit-var
        │
        ▼
  Batch padding → (B, 1, 128, W_max)
        │
        ▼
  CRNN forward pass
        │
        ▼
  CTC decode (greedy or beam search)
        │
        ▼
  list[list[str]]    one token sequence per staff
```

**Returns:** `(token_lists, log_probs_per_strip, out_lens_per_strip)`. The log-probs (shape `(T_i, C)` per strip, on CPU) and lengths are consumed by the post-CRNN gate in Stage 2b.

**Model caching:** The CRNN is loaded once from checkpoint into memory and reused for all subsequent calls in the same process.

**Width clamping:** Images wider than `max_image_width=2048` px are clamped to avoid OOM errors.

**Decoding modes:**
- **Greedy** (default, also when `OMR_BEAM_WIDTH` is unset): argmax per time step, collapse duplicates, remove blank.  Greedy is fast and ~as accurate as beam search on monophonic Real Book staves.
- **Beam search** (`OMR_BEAM_WIDTH=N` env var with N>1, or CLI `--beam-width N`): slightly higher accuracy on dense passages, but slower (~N× slower in the worst case).

**Optional tiling** (env `OMR_ENABLE_TILING=1`): splits wide images into 50%-overlapping tiles (~850 px), processes separately, merges sequences. Legacy feature, not recommended by default.

## Stage 4 — Chord OCR

**File:** `src/omr_pipeline/chord_recognizer.py`

Processes `System.chord_image` (the strip above the staff) with a dedicated
CRNN-CTC model trained on synthetic Real Book-style chord images.

**Inference steps:**
1. Trim leading columns with high ink density (binder-hole shadow removal — up to 10 % of strip width, ≥ 50 % ink density threshold)
2. Resize strip to training height (64 px), preserving aspect ratio; clamp to `max_image_width=2048`
3. Per-image zero-mean / unit-variance normalization
4. Right-pad batch to common width
5. CRNN forward pass → log-probabilities
6. Greedy CTC decode → character sequence
7. `<unk>` tokens (from clutter the synthetic model never saw) treated as word separators
8. `clean_chord_line()` from `chord_postprocess.py` validates against the jazz chord grammar, dropping non-chord fragments
9. Single-root false-positives (`"B"`, `"Eb"`, etc. with no quality/extension) filtered out

**Chord vocabulary** (character-level, 26 content tokens + blank/pad/unk):

| Category | Characters |
|----------|-----------|
| Roots | `A B C D E F G` |
| Accidentals | `# b` |
| Separators / bass | ` ` (space), `/` |
| Quality / extension | `- + ø` ; `d i m` → `dim` ; `m a j` → `maj` ; `s u` → `sus` |
| Numbers | `1 3 6 7 9` |

**Chord notation conventions** (Real Book style):

| Symbol | Meaning |
|--------|---------|
| `-` | minor (e.g. `G-7`) |
| `maj` | major 7 (e.g. `Fmaj7`) |
| `ø` | half-diminished — the CRNN predicts the `ø` token, which `chord_postprocess` expands to the Real Book printed form `-7b5` in the final output (e.g. `B-7b5`) |
| `dim` / `dim7` | diminished (e.g. `Cdim7`) |
| `+` | augmented (e.g. `C+7`) |
| `/` | slash bass (e.g. `Fmaj7/A`) |

**Model checkpoint resolution** (in priority order):
1. `OMR_CHORD_CHECKPOINT` env var
2. `<project_root>/models/chord/latest/best_model.pt`

If no checkpoint is found, chord recognition is skipped silently and `chords` is `[]` for every segment.

**Model caching:** loaded once per process and reused for all subsequent calls.

**Chord postprocessing** (`src/omr_pipeline/chord_postprocess.py`):
- Validates against jazz chord grammar:
  ```
  ROOT [ACC] [QUALITY] [EXTEN] [ALT]* [SLASH]
  ```
- Rejects non-chord tokens (numbers, punctuation, page numbers, tempo markings)

## Stage 5 — Grammar Fixing

**File:** `src/omr_pipeline/grammar_fix.py`

The CRNN may produce malformed token sequences (missing durations, wrong accidental order, etc.). This stateful validator walks the raw token stream and enforces LMX grammar.

**Validation rules:**
- Notes must appear in order: `pitch` → `octave` → duration → optional dot/acc/tie
- Rests must have a duration
- Accidentals only after pitch + octave (`flat` / `sharp` / `natural` — double accidentals are not in the LMX vocabulary)
- Octaves restricted to [3, 7] (covers Real Book altissimo without leaving room for outliers)
- Time signature beats/types within the allowed set (see `docs/configuration.md`)
- Clef must be G2 (other clefs are silently rewritten)

**Barline regularization:**
- Counts cumulative beats
- Inserts implicit `measure` boundaries when beat count exceeds time-signature numerator
- Guards against incomplete final measure

Invalid tokens are discarded (not replaced); partial notes are dropped entirely.

## Output Structure

```json
{
  "pages": [
    {
      "image_data": "data:image/png;base64,<b64>",
      "segments": [
        {
          "staff_bbox": [x, y, w, h],
          "lmx_tokens": [
            "clef:G2",
            "key:fifths:0",
            "time", "beats:4", "beat-type:4",
            "pitch:C", "octave:5", "quarter",
            "pitch:E", "octave:5", "quarter",
            "measure",
            "..."
          ],
          "chords": ["Cmaj7", "Am7", "Dm7", "G7"]
        }
      ]
    }
  ]
}
```

On error:
```json
{"error": "<description>"}
```

## Error Handling

- If no staves are detected → `{"error": "no staves detected"}`
- If CRNN fails on a segment → that segment's `lmx_tokens` is `[]`
- If chord OCR fails → that segment's `chords` is `[]`
- Pipeline failures propagate as `{"error": "..."}`

## Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `OMR_PDF_DPI` | `300` | PyMuPDF rasterisation DPI for PDFs (clamped 72–600). |
| `OMR_BEAM_WIDTH` | `1` (greedy) | CTC beam width for music CRNN.  Set to >1 to enable beam search. |
| `OMR_ENABLE_TILING` | unset | Set `=1` to enable legacy tiling mode (rarely useful). |
| `OMR_CHORD_CHECKPOINT` | `models/chord/latest/best_model.pt` | Path to the chord CRNN checkpoint. Absolute or project-root-relative. |
| `OMR_REJECT_THRESHOLDS` | unset | Path to a JSON thresholds file for staff rejection gates. Falls back to `models/staff_reject/thresholds.json` if it exists, then to baked-in defaults. |
