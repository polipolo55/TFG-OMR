# tests/test_header_injector.py
"""Tests for the virtual header injector."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from omr_pipeline.header_injector import inject_header, load_template, _template_path


TEMPLATES_DIR = Path("data/header_templates")


# ---------------------------------------------------------------------------
# Template path encoding
# ---------------------------------------------------------------------------

def test_template_path_negative_key():
    p = _template_path("key:fifths:-2", ("time", "beats:4", "beat-type:4"))
    assert p.name == "key_-2_time_4_4.png"

def test_template_path_positive_key():
    p = _template_path("key:fifths:3", ("time", "beats:3", "beat-type:4"))
    assert p.name == "key_3_time_3_4.png"

def test_template_path_zero_key():
    p = _template_path("key:fifths:0", ("time", "beats:6", "beat-type:8"))
    assert p.name == "key_0_time_6_8.png"


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TEMPLATES_DIR.exists(), reason="data/header_templates not generated")
def test_all_120_templates_exist():
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

def test_load_template_returns_none_when_missing(tmp_path, monkeypatch):
    import omr_pipeline.header_injector as hi
    monkeypatch.setattr(hi, "TEMPLATES_DIR", tmp_path)
    result = load_template("key:fifths:0", ("time", "beats:4", "beat-type:4"))
    assert result is None


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def test_inject_header_widens_image(monkeypatch):
    import omr_pipeline.header_injector as hi
    fake_template = np.full((50, 30), 255, dtype=np.uint8)
    monkeypatch.setattr(hi, "load_template", lambda k, t: fake_template)
    staff = np.full((50, 200), 128, dtype=np.uint8)
    result = hi.inject_header(staff, "key:fifths:0", ("time", "beats:4", "beat-type:4"))
    assert result.shape[1] > staff.shape[1]

def test_inject_header_height_unchanged(monkeypatch):
    import omr_pipeline.header_injector as hi
    fake_template = np.full((80, 40), 255, dtype=np.uint8)
    monkeypatch.setattr(hi, "load_template", lambda k, t: fake_template)
    staff = np.full((100, 300), 128, dtype=np.uint8)
    result = hi.inject_header(staff, "key:fifths:0", ("time", "beats:4", "beat-type:4"))
    assert result.shape[0] == 100

def test_inject_header_template_resized_to_staff_height(monkeypatch):
    import omr_pipeline.header_injector as hi
    fake_template = np.full((50, 30), 255, dtype=np.uint8)
    monkeypatch.setattr(hi, "load_template", lambda k, t: fake_template)
    staff = np.full((100, 200), 128, dtype=np.uint8)
    result = hi.inject_header(staff, "key:fifths:0", ("time", "beats:4", "beat-type:4"))
    # Template at h=50 scaled to h=100 → width doubles from 30 to 60 → total = 260
    assert result.shape == (100, 260)

def test_inject_header_falls_back_when_no_template(monkeypatch):
    import omr_pipeline.header_injector as hi
    monkeypatch.setattr(hi, "load_template", lambda k, t: None)
    staff = np.full((100, 300), 42, dtype=np.uint8)
    result = hi.inject_header(staff, "key:fifths:0", ("time", "beats:4", "beat-type:4"))
    assert result is staff  # same object returned unchanged
