"""
FastAPI server for OMR lead sheet upload and processing.

Run: poetry run python src/cli.py api
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Resolve project root early so load_dotenv can find .env
_SRC = Path(__file__).resolve().parent.parent
_ROOT = _SRC.parent
load_dotenv(_ROOT / ".env", override=True)  # .env wins over shell env

for p in (str(_SRC), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import json
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from omr_pipeline.pipeline import run_pipeline

app = FastAPI(title="TFG-OMR Lead Sheet API", version="0.2.0")

# Static frontend
_STATIC = _ROOT / "static"
if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


def _checkpoint_path() -> Path | None:
    """Resolve checkpoint from OMR_CHECKPOINT env var or the default location."""
    env = os.environ.get("OMR_CHECKPOINT", "").strip()
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = _ROOT / p
        return p
    default = _ROOT / "models" / "latest" / "best_model.pt"
    return default if default.exists() else None


@app.get("/")
async def root():
    """Serve the upload UI."""
    index = _STATIC / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "OMR API running. POST to /api/omr/lead-sheet with a file."}


@app.get("/health")
async def health():
    """Return model loading status and checkpoint info."""
    ckpt = _checkpoint_path()
    if ckpt is None or not ckpt.exists():
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "checkpoint": str(ckpt) if ckpt else None,
                "error": "Checkpoint file not found — set OMR_CHECKPOINT or place model at models/latest/best_model.pt",
            },
        )

    # Attempt a lazy load to verify the checkpoint is readable
    try:
        from omr_pipeline.inference import _load_model

        _load_model(ckpt)
        return {
            "status": "ok",
            "checkpoint": str(ckpt),
        }
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "checkpoint": str(ckpt),
                "error": str(exc),
            },
        )


@app.post("/api/omr/lead-sheet")
async def process_lead_sheet(
    file: UploadFile = File(..., description="PDF or image of a lead sheet"),
):
    """Process uploaded PDF or image; return transcribed segments."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")

    ckpt = _checkpoint_path()
    if ckpt is None or not ckpt.exists():
        raise HTTPException(
            503,
            "Model checkpoint not found — start the server with OMR_CHECKPOINT set, "
            "or place the model at models/latest/best_model.pt",
        )

    result = run_pipeline(data, file.filename or "upload", checkpoint_path=ckpt)
    if result.get("error") and not result.get("pages"):
        raise HTTPException(422, result["error"])
    return result


# ---------------------------------------------------------------------------
# Chord-strip labeling endpoints (used by static/chord_labeler.html)
# ---------------------------------------------------------------------------

_LABELER_ROOT = _ROOT / "data" / "chord_real"
_LABELS_PATH = _LABELER_ROOT / "labels.jsonl"
_STRIPS_DIR = _LABELER_ROOT / "strips"
_labels_lock = Lock()


class _JsonlLabelStore:
    """Atomic read/write for a JSONL label file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def save(self, records: list[dict]) -> None:
        tmp = self.path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(self.path)


_chord_labels = _JsonlLabelStore(_LABELS_PATH)


class SaveRequest(BaseModel):
    filename: str
    label: str | None = None
    status: str  # "done" | "skip"


@app.get("/labeler")
async def labeler_page():
    """Serve the chord-labeling UI."""
    page = _STATIC / "chord_labeler.html"
    if not page.exists():
        raise HTTPException(404, "chord_labeler.html missing")
    return FileResponse(page)


@app.get("/api/labeler/stats")
async def labeler_stats():
    """Return counts of labels by status."""
    with _labels_lock:
        records = _chord_labels.load()
    counts = {"done": 0, "skip": 0, "pending": 0, "total": len(records)}
    for r in records:
        counts[r.get("status", "pending")] = counts.get(r.get("status", "pending"), 0) + 1
    return counts


@app.get("/api/labeler/next")
async def labeler_next():
    """Return the next pending strip, or 404 if none remain."""
    with _labels_lock:
        records = _chord_labels.load()
    for r in records:
        if r.get("status") == "pending":
            return r
    raise HTTPException(404, "No pending strips")


@app.get("/api/labeler/strip/{filename}")
async def labeler_strip(filename: str):
    """Serve a chord-strip PNG by filename."""
    # Defensive filename check (prevent path traversal)
    if "/" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")
    path = _STRIPS_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Strip not found")
    return FileResponse(path, media_type="image/png")


@app.post("/api/labeler/save")
async def labeler_save(req: SaveRequest):
    """Persist a label / skip decision for one strip."""
    if req.status not in ("done", "skip"):
        raise HTTPException(400, "status must be 'done' or 'skip'")
    with _labels_lock:
        records = _chord_labels.load()
        for r in records:
            if r["filename"] == req.filename:
                r["label"] = req.label
                r["status"] = req.status
                _chord_labels.save(records)
                return {"ok": True}
    raise HTTPException(404, "Filename not in labels.jsonl")


# ---------------------------------------------------------------------------
# Music-strip labeling endpoints (used by static/music_labeler.html)
# ---------------------------------------------------------------------------

_MUSIC_LABELER_ROOT = _ROOT / "data" / "music_real"
_MUSIC_LABELS_PATH = _MUSIC_LABELER_ROOT / "labels.jsonl"
_MUSIC_STRIPS_DIR = _MUSIC_LABELER_ROOT / "strips"
_music_labels_lock = Lock()
_music_labels = _JsonlLabelStore(_MUSIC_LABELS_PATH)


class MusicSaveRequest(BaseModel):
    filename: str
    label: str | None = None
    status: str  # "done" | "skip"


class RenderRequest(BaseModel):
    lmx: str
    # Optional layout-only time signature for continuation staves whose
    # label has no `time` token. Example: "3/4", "2/2", "6/8".
    # When the LMX already contains a time token, this is ignored.
    render_time: str | None = None


@app.get("/music-labeler")
async def music_labeler_page():
    """Serve the music-strip labeling UI."""
    page = _STATIC / "music_labeler.html"
    if not page.exists():
        raise HTTPException(404, "music_labeler.html missing")
    return FileResponse(page)


@app.get("/api/music-labeler/stats")
async def music_labeler_stats():
    with _music_labels_lock:
        records = _music_labels.load()
    counts = {"done": 0, "skip": 0, "pending": 0, "total": len(records)}
    for r in records:
        s = r.get("status", "pending")
        counts[s] = counts.get(s, 0) + 1
    return counts


@app.get("/api/music-labeler/next")
async def music_labeler_next():
    """Return the next pending strip, or 404 if none remain."""
    with _music_labels_lock:
        records = _music_labels.load()
    for r in records:
        if r.get("status") == "pending":
            return r
    raise HTTPException(404, "No pending strips")


@app.get("/api/music-labeler/strip/{filename}")
async def music_labeler_strip(filename: str):
    if "/" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")
    path = _MUSIC_STRIPS_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Strip not found")
    return FileResponse(path, media_type="image/png")


@app.post("/api/music-labeler/render")
async def music_labeler_render(req: RenderRequest):
    """Render an LMX token string with LilyPond; return PNG as base64 data URL."""
    import base64

    import cv2 as _cv2

    try:
        from CRNN_CTC.lilypond_render import render_tokens
    except ImportError:
        raise HTTPException(500, "lilypond_render not available")

    tokens = req.lmx.split()
    if not tokens:
        raise HTTPException(422, "Empty LMX string")

    render_time_hint: tuple[int, int] | None = None
    if req.render_time:
        try:
            b_str, bt_str = req.render_time.split("/", 1)
            render_time_hint = (int(b_str), int(bt_str))
        except ValueError, AttributeError:
            raise HTTPException(400, f"Bad render_time {req.render_time!r}; expected 'beats/beat-type' e.g. '3/4'")

    arr = render_tokens(tokens, render_time_hint=render_time_hint)
    if arr is None:
        raise HTTPException(422, "LilyPond render failed — check token syntax")

    ok, buf = _cv2.imencode(".png", arr)
    if not ok:
        raise HTTPException(500, "PNG encode failed")

    data_url = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()
    return {"image": data_url}


@app.post("/api/music-labeler/save")
async def music_labeler_save(req: MusicSaveRequest):
    if req.status not in ("done", "skip"):
        raise HTTPException(400, "status must be 'done' or 'skip'")
    with _music_labels_lock:
        records = _music_labels.load()
        for r in records:
            if r["filename"] == req.filename:
                r["label"] = req.label
                r["status"] = req.status
                _music_labels.save(records)
                return {"ok": True}
    raise HTTPException(404, "Filename not in labels.jsonl")


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
