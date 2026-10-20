"""Measured PIL-origin correction for the fixed Artifact Tool renderer.

Only the vertical origin is adjusted. Matching a family name alone is not
enough: the measured font bytes, canvas, renderer version and size range must
all match. The bundled profile is evidence from final-PPTX reimport, not source
font identification or a PowerPoint/WPS acceptance claim.
"""
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
import hashlib
import json
import math
import os


FIXED_ARTIFACT_VERSION = "2.8.94"


@lru_cache(maxsize=1)
def _profile():
    # A deployment cannot inject an arbitrary calibration file via environment.
    return json.loads(files(__package__).joinpath("renderer-origin-profile.json").read_text(encoding="utf-8"))


def _stat_identity(stat):
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=128)
def _cached_font_hash(path, identity):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    if _stat_identity(Path(path).stat()) != identity:
        raise OSError("font changed during calibration verification")
    return digest.hexdigest()


def _font_hash(path):
    if not isinstance(path, (str, Path)) or not str(path):
        return None
    try:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            return None
        return _cached_font_hash(str(resolved), _stat_identity(resolved.stat()))
    except (OSError, RuntimeError, ValueError):
        return None


def origin_correction(estimate, source_size, target_size, *, renderer_version=None):
    """Return a verified vertical adjustment in source-image pixels.

    ``VISION_ARTIFACT_VERSION`` may explicitly declare a renderer version. When
    omitted, this package's fixed compiler contract is 2.8.94. A mismatch never
    extrapolates or falls back to a similarly named font. Missing fonts (for
    example in synthetic fixtures) simply return ``applied=False``.
    """
    result = {"dy": 0.0, "applied": False, "profile_id": None,
              "reason": "uncalibrated_profile_unavailable"}
    try:
        profile = _profile()
        result["profile_id"] = profile["profile_id"]
        expected_version = profile["artifact_tool_version"]
        curves = profile["curves"]
        expected_source, expected_target = profile["source_canvas"], profile["target_canvas"]
    except (OSError, ValueError, KeyError, TypeError):
        return result

    def uncalibrated(reason):
        result["reason"] = "uncalibrated_" + reason
        return result

    version = renderer_version if renderer_version is not None else os.environ.get("VISION_ARTIFACT_VERSION", FIXED_ARTIFACT_VERSION)
    if version != expected_version:
        return uncalibrated("renderer_version")
    if (not isinstance(source_size, (list, tuple)) or not isinstance(target_size, (list, tuple)) or
            list(source_size) != expected_source or list(target_size) != expected_target):
        return uncalibrated("canvas")
    if not isinstance(estimate, dict):
        return uncalibrated("estimate")
    size = estimate.get("font_size_px")
    if isinstance(size, bool) or not isinstance(size, (int, float)) or not math.isfinite(size):
        return uncalibrated("font_size")
    family, bold = estimate.get("font_family"), estimate.get("bold", False)
    if not isinstance(family, str) or not isinstance(bold, bool):
        return uncalibrated("font_family_or_weight")
    curve = next((item for item in curves if item["family"] == family and item["bold"] is bold), None)
    if curve is None:
        return uncalibrated("font_family_or_weight")
    minimum, maximum = curve["size_range_px"]
    if not minimum <= size <= maximum:
        return uncalibrated("font_size_range")
    diagnostics = estimate.get("diagnostics", {})
    font = diagnostics.get("font", {}) if isinstance(diagnostics, dict) else {}
    actual_hash = _font_hash(font.get("font_path") if isinstance(font, dict) else None)
    if actual_hash is None:
        return uncalibrated("font_asset_unavailable")
    if actual_hash != curve["font_sha256"]:
        return uncalibrated("font_hash")
    result.update(dy=round(float(curve["dy_intercept"]+curve["dy_slope"]*size), 6),
                  applied=True, reason="matched_measured_profile")
    return result
