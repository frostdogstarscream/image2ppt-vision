"""Conservative, image-derived typography and text removal.

OCR boxes are search windows, never font-size measurements.  This module fits a
local constant/linear background, locates the actual ink, and compares rasterized
installed fonts at their natural aspect ratio.  It does not read an original PPT.
Uncertain backgrounds, fonts or multi-colour alignments leave the image intact.
All lengths returned here are source-image pixels, not PowerPoint points.
"""
from functools import lru_cache
from pathlib import Path, PureWindowsPath
import hashlib
import json
import os
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _manifest_fonts(manifest_name):
    """Read only explicitly configured local assets; never acquire fonts."""
    if not isinstance(manifest_name, str) or not manifest_name.strip():
        raise ValueError("configured font manifest path is empty")
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", manifest_name):
        raise ValueError("font manifest must be a local file")
    try:
        manifest = Path(manifest_name).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        raise ValueError("configured font manifest path cannot be resolved") from None
    try:
        raw = manifest.read_bytes()
        if len(raw) > 256_000:
            raise ValueError()
        document = json.loads(raw)
    except (OSError, ValueError, UnicodeError):
        raise ValueError("configured font manifest is unreadable or invalid JSON") from None
    entries = document.get("fonts") if isinstance(document, dict) else None
    if not isinstance(entries, list) or not 1 <= len(entries) <= 32:
        raise ValueError("font manifest must contain 1 to 32 fonts")
    base = manifest.parent
    records = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("font manifest entries must be objects")
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative or "\x00" in relative or ":" in relative:
            raise ValueError("font manifest entries require a local relative path")
        # Reject Windows drive/UNC forms on Linux too. Canonical resolution also
        # rejects a symlink whose target escapes the manifest's asset directory.
        portable = PureWindowsPath(relative)
        if Path(relative).is_absolute() or portable.drive or portable.root:
            raise ValueError("font manifest paths must be relative and confined")
        if ".." in portable.parts or ".." in Path(relative).parts:
            raise ValueError("font manifest paths must not traverse parent directories")
        try:
            target = (base / relative).resolve()
        except (OSError, RuntimeError, ValueError):
            raise ValueError("configured font asset path cannot be resolved") from None
        if not target.is_relative_to(base):
            raise ValueError("font manifest path escapes its directory")
        family, bold, index, expected = (entry.get("family"), entry.get("bold"),
                                         entry.get("index", 0), entry.get("sha256"))
        if (not isinstance(family, str) or not family.strip() or len(family) > 128 or
                any(ord(character) < 32 for character in family)):
            raise ValueError("font manifest requires a valid family name")
        if not isinstance(bold, bool) or type(index) is not int or not 0 <= index <= 64:
            raise ValueError("font manifest requires boolean bold and a valid collection index")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            raise ValueError("font manifest requires a SHA256 for every font")
        try:
            stat = target.stat()
            if not target.is_file():
                raise OSError()
        except OSError:
            raise ValueError("configured font asset is missing or unreadable") from None
        records.append((str(target), family.strip(), bold, index, expected.lower(),
                        stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    # Re-read manifest content each time; cache expensive font hashes only while
    # every asset's size and change timestamps still match the verified version.
    verified = _verify_manifest(hashlib.sha256(raw).hexdigest(), tuple(records))
    return [dict(item) for item in verified]


@lru_cache(maxsize=8)
def _verify_manifest(manifest_digest, records):
    candidates = []
    for path, family, bold, index, expected, _, _, _ in records:
        try:
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for block in iter(lambda: handle.read(1024*1024), b""):
                    digest.update(block)
        except OSError:
            raise ValueError("configured font asset is missing or unreadable") from None
        if digest.hexdigest() != expected:
            raise ValueError("configured font asset SHA256 mismatch")
        try:
            ImageFont.truetype(path, 32, index=index)
        except (OSError, ValueError):
            raise ValueError("configured font asset cannot be loaded at its collection index") from None
        candidates.append({"path": path, "family": family, "bold": bold,
                           "index": index, "sha256": expected})
    # A valid manifest revision may replace the font bytes at the same path.
    _font.cache_clear()
    _has_glyph.cache_clear()
    return tuple(candidates)


def font_status():
    """Sanitized deployment status: no private manifest or font-file paths."""
    candidates = _candidates(None)
    configured = "VISION_FONT_MANIFEST" in os.environ
    return {"source": "configured_manifest" if configured else "installed_fonts",
            "font_count": len(candidates),
            "families": sorted({item["family"] for item in candidates}),
            "hash_verification": "passed" if configured else "not_requested",
            "sha256_verified": configured}


def _candidates(candidates):
    if candidates is None:
        configured = os.environ.get("VISION_FONT_MANIFEST")
        if configured is not None:
            return _manifest_fonts(configured)
        win = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        candidates = [
            {"path": str(win / name), "family": family, "bold": bold}
            for name, family, bold in [
                # Use the installed family names exposed by Windows/Skia.
                # The English aliases are accepted by Office but not by the
                # actual-file preview rasterizer on Chinese Windows; silently
                # falling back there breaks the evidence for a fitted font.
                ("msyh.ttc", "微软雅黑", False),
                ("msyhbd.ttc", "微软雅黑", True),
                ("Deng.ttf", "等线", False),
                ("Dengb.ttf", "等线", True),
                ("simhei.ttf", "黑体", False),
                ("simsun.ttc", "宋体", False),
                ("arial.ttf", "Arial", False),
                ("arialbd.ttf", "Arial", True),
            ]
        ]
        candidates += [{"path": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                        "family": "DejaVu Sans", "bold": False}]
        # Linux deployments may supply matching redistributable CJK fonts.
        candidates += [{"path": str(p), "family": "Noto Sans CJK SC", "bold": False}
                       for p in Path("/usr/share/fonts/opentype/noto").glob("NotoSansCJK-Regular.ttc")]
    result = []
    for c in candidates:
        c = {"path": str(c)} if isinstance(c, (str, Path)) else dict(c)
        if not Path(c.get("path", "")).is_file():
            continue
        try:
            font = _font(c["path"], 32, c.get("index", 0))
            c.setdefault("family", font.getname()[0])
            c.setdefault("bold", "bold" in font.getname()[1].lower())
            c.setdefault("index", 0)
            result.append(c)
        except (OSError, ValueError):
            continue
    return result


@lru_cache(maxsize=256)
def _font(path, size, index=0):
    return ImageFont.truetype(path, int(size), index=int(index))


@lru_cache(maxsize=8192)
def _has_glyph(path, index, character):
    font = _font(path, 24, index)
    glyph = font.getmask(character)
    missing = font.getmask("\U0010FFFF")
    return glyph.size != missing.size or bytes(glyph) != bytes(missing)


def _raster(text, font):
    bounds = font.getbbox(text)
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    if width <= 0 or height <= 0 or width * height > 4_000_000:
        return None
    im = Image.new("L", (width, height))
    ImageDraw.Draw(im).text((-bounds[0], -bounds[1]), text, font=font, fill=255)
    arr = np.asarray(im)
    yy, xx = np.where(arr > 12)
    if not len(xx):
        return None
    box = (int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1)
    return arr[box[1]:box[3], box[0]:box[2]], (
        bounds[0] + box[0], bounds[1] + box[1],
        bounds[0] + box[2], bounds[1] + box[3])


def _background(crop, sample_mask=None):
    """Fit a plane using robust border samples, retaining a measured uncertainty."""
    h, w = crop.shape[:2]
    yy, xx = np.indices((h, w), dtype=float)
    features = np.stack([np.ones_like(xx), xx / max(w - 1, 1), yy / max(h - 1, 1)], axis=-1)
    ring = np.zeros((h, w), bool)
    # Two pixels mitigate a single antialiased detector edge without fitting text.
    ring[:2] = ring[-2:] = True
    ring[:, :2] = ring[:, -2:] = True
    if sample_mask is not None:
        ring = sample_mask.astype(bool)
    a = features[ring]
    b = crop[ring].astype(float)
    if len(a) > 1600:
        step = max(1, len(a) // 1600)
        a, b = a[::step], b[::step]
    coef = np.zeros((3, 3)); coef[0] = np.median(b, axis=0)
    for _ in range(6):
        error = np.max(np.abs(b - a @ coef), axis=1)
        threshold = max(3.0, float(np.percentile(error, 65)) * 1.8)
        weights = np.minimum(1., threshold / np.maximum(error, 0.01)) ** 2
        coef = np.linalg.lstsq(a * weights[:, None], b * weights[:, None], rcond=None)[0]
    error = np.max(np.abs(b - a @ coef), axis=1)
    inliers = error <= 8
    if inliers.mean() >= .7:
        coef = np.linalg.lstsq(a[inliers], b[inliers], rcond=None)[0]
        error = np.max(np.abs(b - a @ coef), axis=1)
    surface = np.clip(features @ coef, 0, 255)
    p80 = float(np.percentile(error, 80))
    return surface, {
        "model": "linear_rgb_plane", "border_p80_error": round(p80, 4),
        "sample_region": "border" if sample_mask is None else "outside_glyph_neighborhood",
        "sample_pixels": int(ring.sum()),
        "border_inlier_fraction": round(float((error <= 8).mean()), 4),
        "gradient_range": round(float(np.max(np.ptp(surface.reshape(-1, 3), axis=0))), 4),
        "reliable": bool(p80 <= 8 and (error <= 8).mean() >= .8),
    }


def _place(mask, shape, dx, dy):
    out = np.zeros(shape, np.float32)
    h, w = mask.shape
    x0, y0 = max(0, dx), max(0, dy)
    x1, y1 = min(shape[1], dx + w), min(shape[0], dy + h)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = mask[y0-dy:y1-dy, x0-dx:x1-dx]
    return out


def _fit_font(text, ink, candidates):
    """Use shape overlap as well as width/height; never stretch a candidate."""
    h, w = ink.shape
    observed = ink.astype(np.float32)
    # Antialiasing from a screenshot is not identical to the local rasterizer.
    soft_observed = cv2.GaussianBlur(observed, (3, 3), .55)
    fits = []
    for c in candidates:
        if any(not _has_glyph(c["path"], c["index"], ch)
               for ch in set(text) if ord(ch) > 127 and not ch.isspace()):
            continue
        reference = _raster(text, _font(c["path"], 40, c["index"]))
        if reference is None:
            continue
        ref = reference[0]
        height_size = 40 * h / ref.shape[0]
        width_size = 40 * w / ref.shape[1]
        # Width is informative for long OCR lines but cannot hide a bad height.
        estimate = (.35 * height_size + .65 * width_size)
        sizes = sorted({max(5, int(round(estimate + d))) for d in (-1, 0, 1)})
        for size in sizes:
            if size > 256:
                continue
            raster = _raster(text, _font(c["path"], size, c["index"]))
            if raster is None:
                continue
            pixels, glyph_box = raster
            rh, rw = pixels.shape
            dim_error = abs(rw - w) / max(w, 1) + abs(rh - h) / max(h, 1)
            if dim_error > .6:
                continue
            pixels = pixels.astype(np.float32) / 255
            best = None
            for dx in {0, int(round((w-rw)/2)), w-rw}:
                for dy in {0, int(round((h-rh)/2)), h-rh}:
                    predicted = _place(pixels, (h, w), dx, dy)
                    soft = cv2.GaussianBlur(predicted, (3, 3), .55)
                    dice = float(2 * np.minimum(soft, soft_observed).sum() /
                                 max(soft.sum() + soft_observed.sum(), .001))
                    mass_error = abs(float(np.log(max(soft.sum(), .001) / max(soft_observed.sum(), .001))))
                    loss = 1 - dice + .28 * dim_error + .16 * mass_error
                    if best is None or loss < best[0]:
                        best = (loss, dice, dx, dy, predicted)
            fits.append({**c, "size": size, "loss": best[0], "dice": best[1],
                         "dx": best[2], "dy": best[3], "predicted": best[4],
                         "glyph_bbox": glyph_box, "dimension_error": dim_error,
                         "ink_mass_ratio": float(best[4].sum()/max(observed.sum(), .001))})
    if not fits:
        return None
    fits.sort(key=lambda f: (f["loss"], f["path"], f["size"]))
    best = fits[0]
    different_family = next((f for f in fits if f["family"] != best["family"]), None)
    best["family_margin"] = (different_family["loss"] - best["loss"]) if different_family else None
    return best


def _component_colors(crop, delta, mask):
    """Estimate opaque colours per connected ink component, excluding fringe."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    samples = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 3:
            continue
        member = labels == i
        values = delta[member]
        core = member & (delta >= max(12, float(np.percentile(values, 72))))
        if core.any():
            samples.append((np.median(crop[core], axis=0), int(core.sum())))
    groups = []
    for color, weight in sorted(samples, key=lambda p: -p[1]):
        nearest = min(groups, key=lambda g: np.linalg.norm(color-g["color"]), default=None)
        if nearest is not None and np.max(np.abs(color-nearest["color"])) < 36:
            total = nearest["weight"] + weight
            nearest["color"] = (nearest["color"]*nearest["weight"]+color*weight)/total
            nearest["weight"] = total
        else:
            groups.append({"color": color, "weight": weight})
    groups.sort(key=lambda g: -g["weight"])
    total = sum(g["weight"] for g in groups)
    groups = [g for g in groups if g["weight"] >= max(4, total*.08)]
    if groups:
        # Small thin strokes may never reach full opacity; averaging their
        # component medians makes the estimated ink too pale, which in turn
        # falsely favours bold font masks. Anchor each colour to the darkest /
        # brightest well-supported cores across the whole line instead.
        values = crop[mask].astype(float)
        strength = delta[mask]
        assignments = np.array([np.linalg.norm(values-g["color"], axis=1) for g in groups]).argmin(axis=0)
        for i, group in enumerate(groups):
            selected = assignments == i
            if selected.sum() >= 5:
                strongest = selected & (strength >= np.percentile(strength[selected], 90))
                group["color"] = np.median(values[strongest], axis=0)
    return groups


def _hex(color):
    return "#" + "".join(f"{int(v):02X}" for v in np.clip(np.rint(color), 0, 255))


def estimate_text(rgb, ocr, font_candidates=None, *, isolate_ocr_window=False):
    """Estimate one horizontal OCR line and safely remove its source ink.

    Input: uint8 RGB image; OCR dictionary with ``text`` and ``bbox`` (xywh).
    Optional font candidates are paths or ``{path, family, bold, index}`` dicts.
    Output ``bbox`` is the measured visible-ink rectangle. ``cleaned_patch`` is
    an RGB ndarray at ``patch_bbox``, or None on refusal. Only ``native_eligible``
    results may replace source text; a proposed style on a refusal is diagnostic.
    """
    result = {"eligible": False, "native_eligible": False, "bbox": None,
              "font_family": None, "font_size_px": None, "bold": False,
              "color": None, "runs": None, "cleaned_patch": None,
              "patch_bbox": None, "diagnostics": {"status": "REVIEW_REQUIRED"}}
    diag = result["diagnostics"]

    def refuse(reason):
        diag["reason"] = reason
        return result

    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be an HxWx3 uint8 array")
    text = str(ocr.get("text", ""))
    if not text.strip() or "\n" in text or len(text) > 250:
        return refuse("empty_multiline_or_oversized_text")
    box = ocr.get("bbox")
    if not isinstance(box, (list, tuple)) or len(box) != 4 or not np.isfinite(box).all():
        return refuse("invalid_bbox")
    x, y, w, h = map(float, box)
    if w < 3 or h < 4 or w*h > 2_000_000:
        return refuse("invalid_bbox_size")
    height, width = rgb.shape[:2]
    # OCR often already includes substantial padding. The small added margin is
    # for antialias fringe and background samples, not the font-size estimate.
    pad = max(3, min(7, int(round(h*.12))))
    x0, y0 = max(0, int(np.floor(x))-pad), max(0, int(np.floor(y))-pad)
    x1, y1 = min(width, int(np.ceil(x+w))+pad), min(height, int(np.ceil(y+h))+pad)
    if x1-x0 < 4 or y1-y0 < 4:
        return refuse("bbox_outside_image")
    crop = rgb[y0:y1, x0:x1].copy()
    result["patch_bbox"] = [x0, y0, x1-x0, y1-y0]
    surface, background = _background(crop)
    diag["background"] = background
    if not background["reliable"]:
        return refuse("non_smooth_or_crossing_background")
    delta = np.max(np.abs(crop.astype(float)-surface), axis=2)
    threshold = max(10., background["border_p80_error"]*2.5 + 3.)
    mask = delta > threshold
    raw_foreground = mask.copy()
    protected_foreground = np.zeros(mask.shape, bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    for i in range(1, count):
        sx, sy, sw, sh, area = stats[i]
        if isolate_ocr_window:
            # A neighboring label/decoration may enter the padded search area.
            # Discard only components outside the detector window. Components
            # crossing it remain evidence and can still force a refusal.
            overlap = max(0,min(sx+sw,x+w-x0)-max(sx,x-x0))*max(0,min(sy+sh,y+h-y0)-max(sy,y-y0))
            if overlap == 0:
                protected_foreground[labels == i] = True
                mask[labels == i] = False
                continue
        if area < 2:
            mask[labels == i] = False
        elif ((sy == 0 and sy+sh < y-y0+2) or
              (sy+sh == crop.shape[0] and sy > y+h-y0-2)):
            # A neighbouring text line can enter the background sampling
            # margin. Keep those pixels untouched, outside this OCR line.
            protected_foreground[labels == i] = True
            mask[labels == i] = False
        elif sw > max(20, w*.65) and sh <= 3:
            # A separator crossing the line is not a glyph. Preserve it and
            # refuse rather than infer whether it was an underline or artwork.
            return refuse("graphic_line_intersects_text_region")
    yy, xx = np.where(mask)
    if len(xx) < 8:
        return refuse("insufficient_foreground")
    ix0, iy0, ix1, iy1 = int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1
    if ix0 <= 0 or iy0 <= 0 or ix1 >= crop.shape[1] or iy1 >= crop.shape[0]:
        return refuse("foreground_touches_search_boundary")
    ink_mask = mask[iy0:iy1, ix0:ix1]
    if ink_mask.mean() > .90:
        return refuse("foreground_is_not_sparse_text")
    result["bbox"] = [x0+ix0, y0+iy0, ix1-ix0, iy1-iy0]
    diag["ink_bbox"] = result["bbox"]
    diag["detector_bbox"] = list(box)
    colors = _component_colors(crop, delta, mask)
    if not colors:
        return refuse("foreground_color_uncertain")
    diag["foreground_color_count"] = len(colors)
    color = colors[0]["color"]
    result["color"] = _hex(color)
    # Normalize each pixel by local contrast, so a red word and a black word
    # have equal influence on font shape fitting.
    norm = np.zeros_like(delta, dtype=np.float32)
    for group in colors:
        vector = group["color"] - surface
        alpha = np.clip(np.sum((crop-surface)*vector, axis=2) /
                        np.maximum(np.sum(vector*vector, axis=2), 1), 0, 1)
        reconstructed = surface + alpha[..., None]*vector
        residual = np.max(np.abs(crop-reconstructed), axis=2)
        norm = np.maximum(norm, np.where(residual < 30, alpha, 0))
    norm[~mask] = 0
    fit = _fit_font(text, norm[iy0:iy1, ix0:ix1], _candidates(font_candidates))
    if fit is None:
        return refuse("no_usable_installed_font")
    result.update(font_family=fit["family"], font_size_px=fit["size"], bold=bool(fit["bold"]))
    diag["font"] = {"shape_overlap": round(fit["dice"], 4),
                    "dimension_error": round(fit["dimension_error"], 4),
                    "ink_mass_ratio": round(fit["ink_mass_ratio"], 4),
                    "family_margin": None if fit["family_margin"] is None else round(fit["family_margin"], 4),
                    "glyph_bbox": list(fit["glyph_bbox"]),
                    "font_path": fit["path"],
                    "render_origin": [x0+ix0+fit["dx"]-fit["glyph_bbox"][0],
                                      y0+iy0+fit["dy"]-fit["glyph_bbox"][1]]}
    # Unmatched ink can be a chart line, picture, wrong OCR, or missing glyph.
    nearby = cv2.dilate((fit["predicted"] > .12).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    unmatched = float((ink_mask & ~nearby).sum() / max(1, ink_mask.sum()))
    diag["unmatched_ink_fraction"] = round(unmatched, 4)
    if fit["dice"] < .50 or fit["dimension_error"] > .25 or unmatched > .22:
        return refuse("font_or_ocr_shape_mismatch")
    diag["foreground_color_count"] = len(colors)
    if len(colors) > 1:
        if len(colors) > 3 or fit["dice"] < .72 or fit["dimension_error"] > .10:
            return refuse("multicolor_glyph_alignment_uncertain")
        font = _font(fit["path"], fit["size"], fit["index"])
        glyph_left = fit["glyph_bbox"][0]
        runs = []
        for i, character in enumerate(text):
            if character.isspace():
                if runs:
                    runs[-1]["text"] += character
                continue
            left = int(round(font.getlength(text[:i])-glyph_left+fit["dx"]))
            right = int(round(font.getlength(text[:i+1])-glyph_left+fit["dx"]))
            left, right = max(0, left), min(ix1-ix0, right)
            member = mask[iy0:iy1, ix0+left:ix0+right]
            values = crop[iy0:iy1, ix0+left:ix0+right][member]
            if len(values) < 3:
                return refuse("multicolor_character_has_no_ink")
            distances = np.array([np.max(np.abs(values.astype(float)-g["color"]), axis=1) for g in colors])
            good = distances.min(axis=0) < 38
            if good.sum() < 3:
                return refuse("multicolor_character_color_uncertain")
            votes = np.bincount(distances[:, good].argmin(axis=0), minlength=len(colors))
            winner = int(votes.argmax())
            if votes[winner]/votes.sum() < .80:
                return refuse("multicolor_boundary_crosses_glyph")
            c = _hex(colors[winner]["color"])
            if runs and runs[-1]["color"] == c:
                runs[-1]["text"] += character
            else:
                runs.append({"text": character, "color": c})
        if "".join(r["text"] for r in runs) != text:
            return refuse("multicolor_whitespace_alignment_uncertain")
        result["runs"] = runs
    # The original contrast threshold locates ink, but must NOT be reused as
    # an erase threshold: low-opacity antialias pixels would survive deletion
    # as a readable pale copy. Clear the entire verified ink neighbourhood,
    # including clean-looking pixels between the core and its AA fringe.
    kernel = np.ones((7, 7), np.uint8)
    # Isolated high-contrast pixels were omitted from glyph fitting as noise.
    # Once the line is verified, recover those close to real ink as fringe
    # seeds; JPEG/raster resampling can disconnect them from the main stroke.
    ink_nearby = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    cleanup_seed = mask | (raw_foreground & ink_nearby & ~protected_foreground)
    fringe = cv2.dilate(cleanup_seed.astype(np.uint8), kernel) > 0
    protected = cv2.dilate(protected_foreground.astype(np.uint8), kernel) > 0
    erase = fringe & ~protected
    # OCR margins can contain weak glyph fringes even when robust border fitting
    # accepts them. Refit from background pixels safely away from ALL detected
    # ink, including neighbouring lines excluded from this line's ownership.
    excluded = cv2.dilate(raw_foreground.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    safe_samples = ~excluded & (delta <= max(8., background["border_p80_error"]*2))
    refined = None
    sample_y, sample_x = np.where(safe_samples)
    if (len(sample_x) >= 30 and np.ptp(sample_x) >= crop.shape[1]*.5 and
            np.ptp(sample_y) >= crop.shape[0]*.5):
        candidate_surface, candidate_background = _background(crop, safe_samples)
        if candidate_background["reliable"]:
            surface, refined = candidate_surface, candidate_background
    clean = crop.copy()
    clean[erase] = np.clip(np.rint(surface[erase]), 0, 255).astype(np.uint8)
    result.update(eligible=True, native_eligible=True, cleaned_patch=clean)
    diag.update(status="TECHNICALLY_ESTIMATED", reason="smooth_background_and_fitted_glyphs",
                erased_pixels=int(erase.sum()), preserved_pixels=int((~erase).sum()))
    diag["cleanup"] = {"method": "full_verified_ink_neighborhood", "fringe_radius_px": 3,
                       "contrast_threshold_applied_to_erase": False,
                       "protected_neighbor_pixels": int(protected.sum()),
                       "background_refinement": refined,
                       "filled_surface_max_rounding_error": round(float(np.max(
                           np.abs(clean[erase].astype(float)-surface[erase]))), 4) if erase.any() else 0.}
    return result
