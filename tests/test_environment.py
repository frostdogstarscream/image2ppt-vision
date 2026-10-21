"""Environment evidence is reported, never treated as task failure."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image2ppt_vision import environment, typography
from image2ppt_vision.element_calibration import calibrate_elements
from image2ppt_vision.storage import digest, write_json


def test_missing_build_identity_is_unknown_not_invented(monkeypatch):
    for name in environment.BUILD_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(environment, "font_status_or_degraded",
                        lambda: {"status": "ready", "families": []})
    monkeypatch.setattr(environment, "glyph_coverage", lambda _: {"complete": True})
    monkeypatch.setattr(environment, "profile_status", lambda *a, **k: {
        "status": "ready", "reason": "matched_measured_profile"})
    result = environment.collect_environment({"canvas": {"source_px": [10, 10],
                                                        "target_px": [10, 10]}})
    assert result["code"]["git_commit"] is None
    assert result["code"]["sources"]["git_commit"] == "unknown"
    assert result["status"] == "ready"


def test_invalid_font_manifest_is_degraded_and_font_status_never_raises(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("VISION_FONT_MANIFEST", str(manifest))
    status = typography.font_status_or_degraded()
    assert status["status"] == "degraded"
    assert status["reason"] == "invalid_manifest"
    assert status["manifest_sha256"]
    assert status["font_count"] == 0


def test_glyph_coverage_lists_missing_characters(monkeypatch):
    monkeypatch.setattr(typography, "_candidates",
                        lambda _: [{"path": "/missing.ttf", "family": "Fixture",
                                    "bold": False, "index": 0}])
    monkeypatch.setattr(typography, "_has_glyph",
                        lambda path, index, character: character != "Ω")
    result = typography.glyph_coverage(["ABΩ"])
    assert result["complete"] is False
    assert result["status"] == "degraded"
    assert result["fonts"][0]["missing_characters"] == ["Ω"]


def test_compact_environment_has_no_paths_or_character_lists():
    compact = environment.compact_environment({
        "status": "degraded",
        "code": {"version": "0.3.0", "git_commit": None},
        "fonts": {"status": "degraded", "families": ["Fixture"],
                  "records": [{"path": "/data/fonts/a.ttf"}],
                  "manifest_sha256": "a" * 64},
        "glyph_coverage": {"required_characters": 2, "complete": False,
                           "fonts": [{"family": "Fixture", "missing_characters": ["Ω"],
                                      "missing_count": 1, "coverage": 0.5}]},
        "degraded_features": ["typography"],
    })
    text = json.dumps(compact, ensure_ascii=False)
    assert "/data/fonts" not in text
    assert "Ω" not in text
    assert "records" not in text
    assert compact["glyph_coverage"]["fonts"][0]["missing_count"] == 1


def _fixture_ir(tmp_path):
    proposal = tmp_path / "proposal"
    (proposal / "source").mkdir(parents=True)
    pixels = np.full((80, 120, 3), 255, np.uint8)
    pixels[30:60, 10:110] = [30, 90, 180]
    Image.fromarray(pixels).save(proposal / "source" / "page.png")
    source = proposal / "source" / "page.png"
    ir = {
        "schema_version": "0.1", "source_image": "source/page.png",
        "source_sha256": digest(source),
        "canvas": {"source_px": [120, 80], "target_px": [120, 80]},
        "background": {"color": "FFFFFF"}, "review_status": "needs_review",
        "review_notes": [], "evidence": [], "groups": [], "assets": [],
        "nodes": [
            {"id": "t1", "type": "text", "text": "标题", "bbox": [10, 4, 60, 18],
             "source_text_bbox": [10, 4, 60, 18], "font": {"latin": "Fixture",
             "ea": "Fixture", "size_pt": 10}, "bold": False, "color": "FFFFFF",
             "runs": [], "align": "left", "wrap": False},
            {"id": "s1", "type": "shape", "geometry": "rect", "bbox": [10, 30, 100, 30],
             "fill": "112233", "line": None},
        ],
    }
    write_json(proposal / "slideir.json", ir)
    write_json(proposal / "authoring.json",
               {"ir_sha256": digest(proposal / "slideir.json"), "model_typography": {}})
    return proposal


def test_invalid_font_manifest_degrades_without_failing_calibration(tmp_path, monkeypatch):
    manifest = tmp_path / "broken.json"
    manifest.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("VISION_FONT_MANIFEST", str(manifest))
    proposal = _fixture_ir(tmp_path)
    output = tmp_path / "calibrated"
    result = calibrate_elements(proposal / "slideir.json", output,
                                source_typography=True, calibrate_fills=True)
    assert result["typography_applied"] == 0
    assert result["solid_fills_applied"] >= 1
    report = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    assert report["environment"]["status"] == "degraded"
    assert "typography" in report["environment"]["degraded_features"]
    assert set(report["policies"]) == {"fill_surface", "fill_projection_guard",
                                       "glyph_ink_evidence", "multi_stop_gradient",
                                       "environment_inventory"}
    calibrated = json.loads((output / "slideir.json").read_text(encoding="utf-8"))
    shape = next(node for node in calibrated["nodes"] if node["id"] == "s1")
    assert shape["fill"] != "112233"
