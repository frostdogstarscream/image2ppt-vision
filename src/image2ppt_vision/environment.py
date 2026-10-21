"""Describe the deployment and measurement environment without failing a job.

The report separates facts the process can verify from facts only the host can
know. Missing versions are ``unknown``, an unusable font volume is ``degraded``,
and neither changes whether a reconstruction task succeeded.
"""
import os

from . import __version__
from .renderer_calibration import profile_status
from .typography import font_status_or_degraded, glyph_coverage


BUILD_VARIABLES = {
    "VISION_GIT_COMMIT": "git_commit",
    "VISION_GIT_BRANCH": "git_branch",
    "VISION_IMAGE_DIGEST": "image_digest",
    "VISION_CORE_COMMIT": "core_commit",
    "VISION_BUILD_TIME": "build_time",
}

POLICY_VARIABLES = {
    "fill_surface": "VISION_FILL_SURFACE",
    "fill_projection_guard": "VISION_FILL_PROJECTION_GUARD",
    "glyph_ink_evidence": "VISION_GLYPH_INK_EVIDENCE",
    "environment_inventory": "VISION_ENVIRONMENT_INVENTORY",
}


def _flag(name, default=True):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off", "")


def policy_flags():
    return {key: _flag(name) for key, name in POLICY_VARIABLES.items()}


def _code_status():
    code = {"version": __version__}
    sources = {}
    for name, key in BUILD_VARIABLES.items():
        value = os.environ.get(name)
        code[key] = value or None
        sources[key] = "environment" if value else "unknown"
    code["sources"] = sources
    return code


def _capabilities(renderer, fonts, coverage):
    font_status_value = fonts.get("status", "unknown")
    typography = font_status_value
    glyph_ink = "ready"
    if font_status_value == "degraded":
        typography = "degraded"
    if not coverage.get("complete", True):
        typography = "degraded"
    fills = "ready"
    result = {
        "typography": {"status": typography,
                       "reason": fonts.get("reason") or coverage.get("reason")},
        "fills": {"status": fills, "reason": None},
        "glyph_ink": {"status": glyph_ink, "reason": None},
        "renderer_origin": {"status": renderer.get("status", "unknown"),
                            "reason": renderer.get("reason")},
    }
    degraded = [name for name, item in result.items() if item["status"] in ("degraded", "unavailable")]
    return result, degraded


def collect_environment(ir, *, renderer_version=None, texts=()):
    """Collect deployment evidence. Never raises; unknown stays unknown."""
    policies = policy_flags()
    if not policies["environment_inventory"]:
        return {"status": "disabled", "policies": policies}
    code = _code_status()
    canvas = ir.get("canvas") if isinstance(ir, dict) else {}
    canvas = canvas if isinstance(canvas, dict) else {}
    source_px = canvas.get("source_px")
    target_px = canvas.get("target_px")
    renderer = profile_status(renderer_version, source_px, target_px)
    fonts = font_status_or_degraded()
    coverage = glyph_coverage([text for text in texts if isinstance(text, str)])
    capabilities, degraded = _capabilities(renderer, fonts, coverage)
    return {
        "status": "degraded" if degraded else "ready",
        "code": code,
        "source": {"source_sha256": ir.get("source_sha256") if isinstance(ir, dict) else None,
                   "source_px": source_px, "target_px": target_px},
        "renderer": renderer,
        "fonts": fonts,
        "glyph_coverage": coverage,
        "capabilities": capabilities,
        "degraded_features": degraded,
        "policies": policies,
    }


def compact_environment(environment):
    """Task-report subset: no paths and no per-character evidence."""
    if not isinstance(environment, dict):
        return {"status": "unknown"}
    fonts = environment.get("fonts") if isinstance(environment.get("fonts"), dict) else {}
    coverage = environment.get("glyph_coverage") if isinstance(
        environment.get("glyph_coverage"), dict) else {}
    fonts_compact = {key: fonts.get(key) for key in
                     ("status", "reason", "source", "font_count", "families",
                      "hash_verification", "sha256_verified", "manifest_sha256")}
    coverage_compact = {
        "status": coverage.get("status"), "reason": coverage.get("reason"),
        "required_characters": coverage.get("required_characters"),
        "complete": coverage.get("complete"),
        "fonts": [{"family": item.get("family"), "bold": item.get("bold"),
                   "coverage": item.get("coverage"), "missing_count": item.get("missing_count"),
                   "complete": item.get("complete")}
                  for item in coverage.get("fonts", []) if isinstance(item, dict)],
    }
    return {
        "status": environment.get("status", "unknown"),
        "code": environment.get("code"),
        "renderer": environment.get("renderer"),
        "fonts": fonts_compact,
        "glyph_coverage": coverage_compact,
        "capabilities": environment.get("capabilities"),
        "degraded_features": environment.get("degraded_features", []),
        "policies": environment.get("policies"),
    }
