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

### `GET /`

Returns the upload UI (`static/index.html`) if it exists, otherwise a JSON message.

```json
{"message": "TFG-OMR API — POST /api/omr/lead-sheet to transcribe"}
```

### `POST /api/omr/lead-sheet`

Upload a PDF or image for OMR transcription.

**Request:** `multipart/form-data` with field `file` (PDF, PNG, JPG, etc.)

**Response (success):** `200 OK`
```json
{
  "pages": [
    {
      "image_data": "data:image/png;base64,<b64-encoded page>",
      "segments": [
        {
          "staff_bbox": [x, y, w, h],
          "lmx_tokens": ["clef:G2", "key:0", "time:4:4", "pitch:C", "octave:5", "quarter", ...],
          "chords": ["Cmaj7", "Am7", "Dm7", "G7"]
        }
      ]
    }
  ]
}
```

**Response (no file):** `400 Bad Request`
```json
{"detail": "No file provided"}
```

**Response (processing error):** `422 Unprocessable Entity`
```json
{"detail": "<error message from pipeline>"}
```

## Static Files

The `static/` directory is mounted at `/static`. The upload UI (`static/index.html`) provides a drag-and-drop interface for quick testing.

## Integration Notes

- The CRNN model is loaded once at startup and cached in memory for all requests.
- Multi-page PDFs return one entry per page in the `pages` array.
- `image_data` contains a base64-encoded PNG of the full page with staff bounding boxes overlaid.
- `lmx_tokens` is the raw validated token sequence; clients may parse it into MusicXML or other formats.
- The server runs on a single worker by default. For production, use `--workers N` with gunicorn + uvicorn workers.
