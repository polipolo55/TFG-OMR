"""
FastAPI server for OMR lead sheet upload and processing.

Run: poetry run python src/api/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure src/ on path (api lives in src/api/)
_SRC = Path(__file__).resolve().parent.parent
_ROOT = _SRC.parent
for p in (str(_SRC), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from omr_pipeline.pipeline import run_pipeline

app = FastAPI(title="TFG-OMR Lead Sheet API", version="0.1.0")

# Static frontend
_STATIC = _ROOT / "static"
if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/")
async def root():
    """Serve the upload UI."""
    index = _STATIC / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "OMR API running. POST to /api/omr/lead-sheet with a file."}


@app.post("/api/omr/lead-sheet")
async def process_lead_sheet(file: UploadFile = File(..., description="PDF or image of a lead sheet")):
    """Process uploaded PDF or image; return transcribed segments."""
    if not file.filename:
        raise HTTPException(400, "No filename")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    result = run_pipeline(data, file.filename or "upload")
    if result.get("error"):
        raise HTTPException(422, result["error"])
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
