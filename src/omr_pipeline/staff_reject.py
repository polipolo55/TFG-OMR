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

import numpy as np

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
    # If a strip fails a pre-CRNN gate but the CRNN's mean argmax log-prob is
    # at or above this value, the gate verdict is overridden and the strip is
    # accepted. Geometry/OCR signals can't always tell sparse music from empty
    # text-bearing staves; high CTC confidence is the ground truth.
    confident_override_logprob: float = -0.1


DEFAULT_THRESHOLDS = RejectThresholds()


@dataclass
class RejectionResult:
    passed: bool
    reason: str | None
    diagnostics: dict[str, float] = field(default_factory=dict)


_DEFAULT_THRESHOLDS_PATH = "models/staff_reject/thresholds.json"


def load_thresholds() -> RejectThresholds:
    """Read thresholds from ``$OMR_REJECT_THRESHOLDS`` if set, else from
    ``models/staff_reject/thresholds.json``, else fall back to baked-in defaults."""
    path = os.environ.get("OMR_REJECT_THRESHOLDS", "").strip()
    if not path and os.path.exists(_DEFAULT_THRESHOLDS_PATH):
        path = _DEFAULT_THRESHOLDS_PATH
    if not path:
        return DEFAULT_THRESHOLDS
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.loads(fh.read())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("threshold file %r unreadable (%s); using defaults", path, exc)
        return DEFAULT_THRESHOLDS
    known = {f.name for f in fields(RejectThresholds)}
    merged = {**asdict(DEFAULT_THRESHOLDS), **{k: float(v) for k, v in raw.items() if k in known}}
    return RejectThresholds(**merged)


# ---------------------------------------------------------------------------
# Geometric signals
# ---------------------------------------------------------------------------


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
    h, _ = binary.shape[:2]
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


# ---------------------------------------------------------------------------
# OCR text signal
# ---------------------------------------------------------------------------

_OCR_READER = None


def _get_ocr_reader():
    """Lazy-load a single EasyOCR Reader; cached for the lifetime of the process."""
    global _OCR_READER
    if _OCR_READER is None:
        import easyocr  # lazy import — heavy

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
    for box in horizontal_list[0] if horizontal_list else []:
        x0, x1, y0, y1 = box
        total += max(0, x1 - x0) * max(0, y1 - y0)
    # free_list[0] is a list of polygons [[x,y], [x,y], [x,y], [x,y]]
    for poly in free_list[0] if free_list else []:
        if len(poly) >= 3:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            total += (max(xs) - min(xs)) * (max(ys) - min(ys))  # bbox approx
    return float(total) / float(h * w)


# ---------------------------------------------------------------------------
# CTC confidence signal
# ---------------------------------------------------------------------------


def _mean_logprob(log_probs, out_len: int) -> float:
    """Mean log-probability of the argmax class across the first ``out_len`` frames."""
    if out_len <= 0 or log_probs.numel() == 0:
        return float("-inf")
    frames = log_probs[:out_len]
    argmax = frames.argmax(dim=-1)
    picked = frames.gather(-1, argmax.unsqueeze(-1)).squeeze(-1)
    return float(picked.mean().item())


# ---------------------------------------------------------------------------
# Composed evaluators
# ---------------------------------------------------------------------------


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

    from .staff_detect import local_primary_staff_lines

    line_ys = local_primary_staff_lines(binary)
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
        except Exception as exc:  # OCR failures should not kill the pipeline
            log.warning("OCR text-area gate failed: %s", exc)
            diag["text_area_frac"] = 0.0

    if diag["text_area_frac"] > th.max_text_area_frac:
        return RejectionResult(passed=False, reason="ocr_text_density", diagnostics=diag)

    return RejectionResult(passed=True, reason=None, diagnostics=diag)


def evaluate_post_crnn(
    system,
    log_probs,
    out_len: int,
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

    if out_len <= 0:
        return RejectionResult(passed=False, reason="ctc_zero_length", diagnostics=diag)

    # CTC-confidence override: geometric gates sometimes mis-fire on sparse
    # music (e.g. a single whole note). High CRNN confidence rescues those.
    # OCR text-density is a stronger signal — text in the strip means it's
    # not music, regardless of how confidently the CRNN hallucinates tokens
    # from the text strokes. Geometry rejections are overridable; OCR is not.
    if not pre_result.passed:
        overridable = (pre_result.reason or "").startswith("geometry_")
        if overridable and diag["mean_logprob"] >= th.confident_override_logprob:
            log.info(
                "CTC override @ logprob=%.4f rescues %s rejection",
                diag["mean_logprob"],
                pre_result.reason,
            )
            # Surface the overridden reason in diagnostics so the API consumer
            # can tell a rescued segment apart from a clean pass.
            diag["override_reason"] = pre_result.reason
            return RejectionResult(passed=True, reason=None, diagnostics=diag)
        return RejectionResult(passed=False, reason=pre_result.reason, diagnostics=diag)

    if diag["mean_logprob"] < th.min_mean_logprob:
        return RejectionResult(passed=False, reason="ctc_low_confidence", diagnostics=diag)

    return RejectionResult(passed=True, reason=None, diagnostics=diag)
