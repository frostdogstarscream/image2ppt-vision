"""Calibration applicability guards; fixtures do not copy proprietary fonts."""
import copy
import hashlib
import json
import os
from importlib.resources import files
from pathlib import Path

import pytest

from image2ppt_vision import renderer_calibration as calibration


SOURCE = [1672, 941]
TARGET = [1280, 720]


@pytest.fixture
def calibrated_font(tmp_path, monkeypatch):
    asset = tmp_path / "synthetic.font"
    asset.write_bytes(b"test-local-font-calibration-asset")
    profile = json.loads(files("image2ppt_vision").joinpath("renderer-origin-profile.json").read_text(encoding="utf8"))
    profile["curves"][0]["font_sha256"] = hashlib.sha256(asset.read_bytes()).hexdigest()
    monkeypatch.setattr(calibration, "_profile", lambda: profile)
    monkeypatch.delenv("VISION_ARTIFACT_VERSION", raising=False)
    estimate = {"font_family": "微软雅黑", "bold": False, "font_size_px": 24,
                "diagnostics": {"font": {"font_path": str(asset)}}}
    return estimate, asset


def test_verified_profile_adjusts_only_y_without_mutating_estimate(calibrated_font):
    estimate, _ = calibrated_font
    original = copy.deepcopy(estimate)
    result = calibration.origin_correction(estimate, SOURCE, TARGET)
    assert result["applied"] is True
    assert result["dy"] == pytest.approx(1.592742)
    assert set(result) == {"dy", "applied", "profile_id", "reason"}
    assert result["profile_id"] == "artifact-tool-windows-cjk-origin-20261016-v3"
    assert estimate == original


@pytest.mark.parametrize("source,target", [([1672, 940], TARGET), (SOURCE, [1920, 1080]),
                                          (None, TARGET), (SOURCE, None)])
def test_different_canvas_is_uncalibrated(calibrated_font, source, target):
    result = calibration.origin_correction(calibrated_font[0], source, target)
    assert result["dy"] == 0 and not result["applied"]
    assert result["reason"] == "uncalibrated_canvas"


@pytest.mark.parametrize("version", ["2.8.93", "2.8.95", "", "unknown"])
def test_different_renderer_version_is_uncalibrated(calibrated_font, monkeypatch, version):
    monkeypatch.setenv("VISION_ARTIFACT_VERSION", version)
    result = calibration.origin_correction(calibrated_font[0], SOURCE, TARGET)
    assert not result["applied"] and result["dy"] == 0
    assert result["reason"] == "uncalibrated_renderer_version"


@pytest.mark.parametrize("size", [7.9, 64.01, float("nan"), float("inf"), None, "24", True])
def test_invalid_or_outside_measured_font_sizes(calibrated_font, size):
    estimate, _ = calibrated_font
    estimate["font_size_px"] = size
    result = calibration.origin_correction(estimate, SOURCE, TARGET)
    assert not result["applied"] and result["dy"] == 0
    assert result["reason"].startswith("uncalibrated_font_size")


@pytest.mark.parametrize("size", [8, 64])
def test_size_range_endpoints_are_inclusive(calibrated_font, size):
    estimate, _ = calibrated_font
    estimate["font_size_px"] = size
    assert calibration.origin_correction(estimate, SOURCE, TARGET)["applied"]


def test_font_hash_change_invalidates_cached_verification(calibrated_font):
    estimate, asset = calibrated_font
    assert calibration.origin_correction(estimate, SOURCE, TARGET)["applied"]
    before = asset.stat()
    asset.write_bytes(b"changed-local-calibration-asset!!")
    os.utime(asset, ns=(before.st_atime_ns, before.st_mtime_ns+10_000_000))
    result = calibration.origin_correction(estimate, SOURCE, TARGET)
    assert result["reason"] == "uncalibrated_font_hash"
    assert not result["applied"] and result["dy"] == 0


def test_missing_font_path_and_missing_file_are_safe(calibrated_font):
    estimate, asset = calibrated_font
    asset.unlink()
    result = calibration.origin_correction(estimate, SOURCE, TARGET)
    assert result["reason"] == "uncalibrated_font_asset_unavailable"
    estimate["diagnostics"] = {}
    result = calibration.origin_correction(estimate, SOURCE, TARGET)
    assert result["reason"] == "uncalibrated_font_asset_unavailable"
    assert result["dy"] == 0


@pytest.mark.parametrize("family,bold", [("Unknown Family", False), ("Microsoft YaHei", False),
                                        ("宋体", True), ("微软雅黑", 1)])
def test_family_and_weight_must_match_exactly(calibrated_font, family, bold):
    estimate, _ = calibrated_font
    estimate.update(font_family=family, bold=bold)
    result = calibration.origin_correction(estimate, SOURCE, TARGET)
    assert result["reason"] == "uncalibrated_font_family_or_weight"
    assert result["dy"] == 0


def test_user_environment_cannot_replace_bundled_profile(calibrated_font, monkeypatch):
    monkeypatch.setenv("VISION_RENDERER_PROFILE", "https://example.invalid/arbitrary-profile.json")
    monkeypatch.setenv("VISION_ARTIFACT_VERSION", "2.8.94")
    result = calibration.origin_correction(calibrated_font[0], SOURCE, TARGET)
    assert result["applied"] and result["profile_id"].endswith("-v3")


@pytest.mark.parametrize("filename,family,bold", [
    ("msyh.ttc", "微软雅黑", False), ("msyhbd.ttc", "微软雅黑", True),
    ("Deng.ttf", "等线", False), ("Dengb.ttf", "等线", True),
    ("simhei.ttf", "黑体", False), ("simsun.ttc", "宋体", False),
    ("arial.ttf", "Arial", False), ("arialbd.ttf", "Arial", True),
])
def test_real_measured_windows_font_matches_bundled_profile(filename, family, bold, monkeypatch):
    path = Path("C:/Windows/Fonts")/filename
    if not path.is_file():
        pytest.skip("Measured Windows font unavailable")
    monkeypatch.delenv("VISION_ARTIFACT_VERSION", raising=False)
    estimate = {"font_family": family, "bold": bold, "font_size_px": 24,
                "diagnostics": {"font": {"font_path": str(path)}}}
    result = calibration.origin_correction(estimate, SOURCE, TARGET)
    assert result["applied"], result


def test_bundled_profile_contains_only_final_measured_curves():
    profile = json.loads(files("image2ppt_vision").joinpath("renderer-origin-profile.json").read_text(encoding="utf8"))
    assert profile["validated_samples"] == 92
    assert profile["artifact_tool_version"] == calibration.FIXED_ARTIFACT_VERSION
    assert all("small_size_branch" not in curve for curve in profile["curves"])
    assert all(curve["dx_source_px"] == 0 for curve in profile["curves"])
