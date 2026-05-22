# Web API

**File:** `src/api/main.py`
**Server:** FastAPI + Uvicorn

## Starting the Server

```bash
poetry run python src/cli.py api --host 0.0.0.0 --port 8000
```

Or directly:
```bash
poetry run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

The upload UI is served at `http://localhost:8000`.

## Endpoints

### `GET /health`

Returns model loading status.  Call this after startup to confirm the checkpoint
is readable before sending inference requests.

**Response (ok):** `200 OK`
```json
{"status": "ok", "checkpoint": "/abs/path/to/models/latest/best_model.pt"}
```

**Response (missing checkpoint):** `503 Service Unavailable`
```json
{
  "status": "degraded",
  "checkpoint": null,
  "error": "Checkpoint file not found — set OMR_CHECKPOINT or place model at models/latest/best_model.pt"
}
```

### `GET /`

Returns the upload UI (`static/index.html`) if it exists, otherwise a JSON hint.

```json
{"message": "OMR API running. POST to /api/omr/lead-sheet with a file."}
```

### `POST /api/omr/lead-sheet`

Upload a PDF or image for OMR transcription.

**Request:** `multipart/form-data` with field `file` (PDF, PNG, JPG, etc.)

**Response (success):** `200 OK`
```json
{
  "error": null,
  "pages": [
    {
      "index": 0,
      "page_image_data_url": "data:image/png;base64,<b64-encoded page>",
      "segments": [
        {
          "staff_bbox": [x, y, w, h],
          "chord_bbox": [x, y, w, h],
          "lmx_tokens": ["measure", "clef:G2", "key:fifths:0", "time", "beats:4", "beat-type:4", "pitch:C", "octave:5", "quarter", "..."],
          "chords": ["Cmaj7", "Am7", "Dm7", "G7"],
          "rejected": null,
          "reject_diagnostics": {
            "line_span_min": 0.83,
            "spacing_cov": 0.04,
            "interline_ink_frac": 0.17,
            "text_area_frac": 0.02,
            "mean_logprob": -0.03
          }
        }
      ]
    }
  ],
  "meta": {
    "filename": "leadsheet.pdf",
    "page_width": 2480,
    "page_height": 3508,
    "deskew_angle_deg": 0.2,
    "num_systems": 8,
    "num_rejected": 0,
    "pdf_render_dpi": 300
  }
}
```

Notes on the segment fields:

| Field | Type | Description |
|-------|------|-------------|
| `staff_bbox` | `[x, y, w, h]` | Bounding box of the music staff region (page pixels) |
| `chord_bbox` | `[x, y, w, h] \| null` | Bounding box of the chord strip above the staff, or `null` if no chord region was found |
| `lmx_tokens` | `list[str]` | Flat LMX token sequence after grammar fixing. Empty for rejected segments. |
| `chords` | `list[str]` | Jazz chord symbols, left-to-right order. Empty for rejected segments. |
| `rejected` | `str \| null` | Reject reason code, or `null` when the segment is good. Codes: `geometry_no_strip`, `geometry_no_staff_lines`, `geometry_line_span`, `geometry_spacing_cov`, `geometry_interline_ink`, `ocr_text_density`, `ctc_low_confidence`, `ctc_zero_length`. Every detected system is included in the response (rejected ones too, so the UI can render or filter them); see `docs/inference_pipeline.md` Stage 2b. |
| `reject_diagnostics` | `object` | Per-gate numeric signals: `line_span_min`, `spacing_cov`, `interline_ink_frac`, `text_area_frac`, `mean_logprob`. Always present. `mean_logprob` is `null` for geometry-rejected segments (no CRNN call was made). May also include `override_reason` (string) when a geometry rejection was overridden by high CRNN confidence — the segment is treated as a clean pass but the original failing gate is preserved here for audit. |

Top-level `meta` gains:
- `num_rejected` — total number of segments with a non-null `rejected` reason.
- `num_rejected_by_gate` — `{geometry, ocr_text_density, ctc}` counts.
- `staff_detection` — per-pass detector statistics (`horizontal_lines_found`, `raw_systems`, `local_validated_systems`, `after_pre_crnn_gate`, `pre_crnn_geometry_rejected`). Use this to diagnose **missing** staves: if `horizontal_lines_total` is low or `raw_systems` is short relative to the expected staff count, the morphological detector itself is the bottleneck, not the rejection gate.

**Response (no staff detected):** `200 OK` with `error` set and empty `segments`
```json
{
  "error": "No staff systems detected in the image.",
  "pages": [{"index": 0, "segments": [], "page_image_data_url": "..."}],
  "meta": {...}
}
```

**Response (processing error):** `422 Unprocessable Entity`
```json
{"detail": "<error message from pipeline>"}
```

**Response (model missing):** `503 Service Unavailable`
```json
{"detail": "Model checkpoint not found — ..."}
```

## Static Files

The `static/` directory is mounted at `/static`. The upload UI (`static/index.html`)
provides a drag-and-drop interface that shows:

- The rendered page image with bounding boxes overlaid (blue = staff, orange dashed = chord strip)
- Per-staff cards with chord pills and the LMX token sequence (collapsible for long sequences)

## Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `OMR_CHECKPOINT` | `models/latest/best_model.pt` | Path to music CRNN checkpoint. Absolute or relative to project root. |
| `OMR_CHORD_CHECKPOINT` | `models/chord/latest/best_model.pt` | Path to chord CRNN checkpoint. Absolute or relative to project root. |
| `OMR_PDF_DPI` | `300` | DPI for PyMuPDF PDF rasterisation (72–600). |
| `OMR_BEAM_WIDTH` | `1` (greedy) | CTC beam width for music CRNN decoding. |
| `OMR_DEBUG_DIR` | _(unset)_ | If set, intermediate crops (music strips, chord strips) are saved here. |

## Chord Strip Labeling Endpoints

These endpoints power `static/chord_labeler.html`, the hand-labeling UI used to
build fine-tuning data for the chord CRNN.  Data is stored in
`data/chord_real/labels.jsonl` (one JSON record per line).

### `GET /labeler`

Serves `static/chord_labeler.html`.

### `GET /api/labeler/stats`

Returns label counts by status.

```json
{"done": 292, "skip": 603, "pending": 0, "total": 895}
```

### `GET /api/labeler/next`

Returns the next record with `status == "pending"`, or `404` if none remain.

```json
{"filename": "page0030_staff2.png", "predicted": "G-7 C7 Fmaj7", "label": null, "status": "pending"}
```

### `GET /api/labeler/strip/{filename}`

Serves the chord-strip PNG from `data/chord_real/strips/`.  Rejects paths containing `/` or `..`.

### `POST /api/labeler/save`

Persist a label or skip decision.

**Request body:**
```json
{"filename": "page0030_staff2.png", "label": "G-7 C7 Fmaj7", "status": "done"}
```
`status` must be `"done"` or `"skip"`.  `label` may be `null` when skipping.

**Response:** `{"ok": true}`

## Integration Notes

- The CRNN model is loaded once at startup (lazy, on first request) and cached.
- `lmx_tokens` is a **list** of token strings; join with spaces to get the raw LMX string.
- `chords` is a **list** of canonical chord symbol strings.
- Multi-page PDFs: only page 0 is processed today.
- The server runs on a single worker by default. For production, use gunicorn + uvicorn workers.
