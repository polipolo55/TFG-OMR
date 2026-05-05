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
    lines: list[int]       # 5 y-coordinates (px)
    staff_space: float     # avg distance between adjacent lines
    x_start: int
    x_end: int

@dataclass
class System:
    staff: Staff
    music_img: ndarray     # cropped grayscale staff region
    music_bin: ndarray     # binarized version
    chord_img: ndarray     # region above staff (chord symbols)
    chord_bin: ndarray
    bbox: tuple[int,int,int,int]  # (x, y, w, h) in page
```

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

**Model caching:** The CRNN is loaded once from checkpoint into memory and reused for all subsequent calls in the same process.

**Width clamping:** Images wider than `max_image_width=2048` px are clamped to avoid OOM errors.

**Decoding modes:**
- **Greedy** (default, also when `OMR_BEAM_WIDTH` is unset): argmax per time step, collapse duplicates, remove blank.  Greedy is fast and ~as accurate as beam search on monophonic Real Book staves.
- **Beam search** (`OMR_BEAM_WIDTH=N` env var with N>1, or CLI `--beam-width N`): slightly higher accuracy on dense passages, but slower (~N× slower in the worst case).

**Optional tiling** (env `OMR_ENABLE_TILING=1`): splits wide images into 50%-overlapping tiles (~850 px), processes separately, merges sequences. Legacy feature, not recommended by default.

## Stage 4 — Chord OCR

**File:** `src/omr_pipeline/ocr_chords.py`

Processes `System.chord_img` (the strip above the staff).

**Default backend: `contour`**
1. Ensure light background (auto-invert if mean < 128)
2. Upscale to min height 200 px
3. CLAHE (clipLimit=3.0, tileGridSize=4×4) + unsharp mask
4. Connected-component isolation — finds individual symbols
5. Per-component EasyOCR → string

**Alternative backends:**
- `easyocr` — whole-strip EasyOCR (less accurate on multi-symbol strips)
- `vlm` — vision-language model (GPT-4o / Gemini) for complex/ambiguous chords

**Chord postprocessing** (`src/omr_pipeline/chord_postprocess.py`):
- Corrects common OCR confusions: `majl→maj7`, `susl→sus4`, trailing `0→o`
- Validates against jazz chord grammar:
  ```
  ROOT [ACC] [QUALITY] [EXTEN] [ALT]* [SLASH]
  ```
- Rejects non-chord tokens (numbers, punctuation, page numbers)

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
| `OMR_BEAM_WIDTH` | `1` (greedy) | CTC beam width.  Set to >1 to enable beam search. |
| `OMR_ENABLE_TILING` | unset | Set `=1` to enable legacy tiling mode (rarely useful). |
| `OMR_CHORD_BACKEND` | `contour` | `contour` / `easyocr` / `vlm` for the chord OCR backend. |
