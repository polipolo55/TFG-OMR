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
load_dotenv(_ROOT / ".env")  # no-op if file absent

for p in (str(_SRC), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from omr_pipeline.pipeline import run_pipeline

log = logging.getLogger(__name__)

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


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
