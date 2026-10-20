"""Bounded typography postprocessing, kept separate from independent model output."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .font_plan import _load_bound
from .storage import digest, read_json, write_json
from .text_ledger import structure_hash, verify_ledger
from .typography import _candidates, _font


def protected(ir):
    result = deepcopy(ir)
    for key in ("review_notes", "review_status", "source_image"):
        result.pop(key, None)
    for node in result["nodes"]:
        if node["type"] == "text":
            node.pop("font")
            node.pop("bold", None)
    return result


def assert_compatible(before, after):
    if protected(before) != protected(after):
        raise ValueError("font revision changed protected content, geometry, runs or assets")


def _face(family, bold, catalog):
    aliases = {"Microsoft YaHei": "微软雅黑", "SimSun": "宋体", "SimHei": "黑体",
               "DengXian": "等线"}
    family = aliases.get(family, family)
    found = [f for f in catalog if f["family"] == family and f["bold"] == bold]
    if not found:
        raise ValueError("no exact installed face/weight for measurement")
    return found[0]


def extent(node, size_px, catalog):
    """Measure unwrapped lines, respecting Latin/CJK faces and mixed run weights.

    Local FreeType metrics are diagnostic; the final renderer is still required.
    """
    if not math.isfinite(size_px) or size_px <= 0:
        raise ValueError("invalid font size")
    # A large reference em permits fractional sizes without integer rounding.
    reference = 256
    width, max_width, height = 0.0, 0.0, 0.0
    lines = 1
    faces = set()
    for run in node.get("runs") or [{"text": node["text"]}]:
        weight = run.get("bold", node.get("bold", False))
        for char in run["text"]:
            if char == "\n":
                max_width = max(width, max_width)
                width = 0
                lines += 1
                continue
            family = node["font"]["latin" if ord(char) < 128 else "ea"]
            face = _face(family, weight, catalog)
            font = _font(face["path"], reference, face.get("index", 0))
            faces.add((face["path"], face.get("index", 0)))
            width += font.getlength(char) * size_px/reference
            box = font.getbbox(char)
            height = max(height, (box[3]-box[1]) * size_px/reference)
    return {"advance_width_px": max(width, max_width),
            "ink_height_px": height * lines, "lines": lines,
            "measurement": "sum of glyph advances, no kerning; not actual PPT rendering",
            "face_assets": [{"path": p, "index": i} for p, i in sorted(faces)]}


def _size_px(node, ir):
    return node["font"]["size_pt"] / (.75 * ir["canvas"]["target_px"][0] / ir["canvas"]["source_px"][0])


def _limits(node):
    source_box = node.get("source_text_bbox", node["bbox"])
    return min(source_box[2], node["bbox"][2]), min(source_box[3], node["bbox"][3])


def can_select(baseline_node, candidate_node, fits, baseline_loss, candidate_loss):
    # Pixel averages can reward thinner text even where source headings are bold.
    # Do not accept a weight change without a dedicated source-weight verifier.
    return (fits
            and baseline_node.get("bold", False) == candidate_node.get("bold", False)
            and candidate_loss <= baseline_loss-.20)


def fit_node(node, ir, catalog):
    before = _size_px(node, ir)
    result = deepcopy(node)
    if node.get("wrap", False) or "\n" in node["text"]:
        return result, {"id": node["id"], "status": "REVIEW_REQUIRED",
                        "reason": "wrapped_or_multiline_not_fitted"}
    try:
        measured = extent(node, before, catalog)
    except ValueError as error:
        return result, {"id": node["id"], "status": "REVIEW_REQUIRED", "reason": str(error)}
    width, height = _limits(node)
    if min(width, height, measured["advance_width_px"], measured["ink_height_px"]) <= 0:
        raise ValueError("invalid text measurement bounds")
    wr = width/measured["advance_width_px"]
    hr = height/measured["ink_height_px"]
    ratio = min(wr, hr)
    # Shrink overflows. Modest enlargement only when both source dimensions agree.
    if ratio > 1:
        ratio = min(ratio, 1.20) if max(wr, hr)/min(wr, hr) <= 1.35 else 1.
    after = math.floor(before * ratio * 100)/100
    if after < 5 or after > 400 or ratio < .20:
        return result, {"id": node["id"], "status": "REVIEW_REQUIRED",
                        "reason": "size_change_outside_bounded_policy", "before_px": before,
                        "proposed_px": after}
    result["font"]["size_pt"] = round(after * .75 * ir["canvas"]["target_px"][0]/ir["canvas"]["source_px"][0], 4)
    return result, {"id": node["id"], "status": "MEASURED_CANDIDATE",
                    "reason": "fixed_source_bounds_size_fit", "before_px": before,
                    "after_px": after, "limit_px": [width, height], "before_measurement": measured,
                    "after_measurement": extent(result, after, catalog),
                    "family_and_weight_unchanged": True}


def write_revision(irfile, new_ir, output, report):
    irfile, original, authoring, source = _load_bound(irfile)
    output = Path(output)
    if output.exists():
        raise FileExistsError("new font revision directory required")
    assert_compatible(original, new_ir)
    ledger_exists = (irfile.parent/"text-ledger.json").exists()
    if ledger_exists:
        verify_ledger(irfile.parent)
    # Validate everything before writing a new version.
    output.mkdir(parents=True)
    shutil.copytree(irfile.parent/"assets", output/"assets")
    (output/"source").mkdir()
    shutil.copyfile(source, output/"source/page.png")
    new_ir["source_image"] = "source/page.png"
    new_ir["review_status"] = "needs_review"
    new_ir["review_notes"].append("Explicit deterministic font postprocessing; not independent model performance. Content and geometry frozen; actual-file review required.")
    write_json(output/"slideir.json", new_ir)
    authoring.update(tool="bounded-font-compatibility", created_at=datetime.now(timezone.utc).isoformat(),
                     ir_sha256=digest(output/"slideir.json"), review_notes=new_ir["review_notes"],
                     typography_origin="derived_candidate_or_retained_baseline; see font-compatibility.json")
    old_typography = authoring.get("model_typography", {})
    authoring["model_typography"] = {
        n["id"]: {**old_typography.get(n["id"], {}), "font": n["font"],
                  "bold": n.get("bold", False), "status": "explicit_postprocessing_not_raw_model"}
        for n in new_ir["nodes"] if n["type"] == "text"}
    write_json(output/"authoring.json", authoring)
    if ledger_exists:
        ledger = read_json(irfile.parent/"text-ledger.json")
        ledger["font_revision_parent_ir_sha256"] = ledger["output_ir_sha256"]
        ledger["output_ir_sha256"] = digest(output/"slideir.json")
        ledger["structure_sha256"] = structure_hash(new_ir)
        write_json(output/"text-ledger.json", ledger)
        for directory in ("crops", "evidence-inputs"):
            if (irfile.parent/directory).exists():
                shutil.copytree(irfile.parent/directory, output/directory)
        if (irfile.parent/"text-ledger.html").exists():
            shutil.copyfile(irfile.parent/"text-ledger.html", output/"text-ledger.html")
        verify_ledger(output)
    report.update(schema="FontCompatibility/1", input_ir_sha256=digest(irfile),
                  output_ir_sha256=digest(output/"slideir.json"),
                  source_sha256=new_ir["source_sha256"], human_acceptance=False,
                  independent_model_result=False)
    write_json(output/"font-compatibility.json", report)
    return {k: v for k, v in report.items() if k != "decisions"}


def constrain(irfile, output):
    irfile, ir, _, _ = _load_bound(irfile)
    new_ir, rows = deepcopy(ir), []
    catalog = _candidates(None)
    for index, node in enumerate(ir["nodes"]):
        if node["type"] == "text":
            new_ir["nodes"][index], row = fit_node(node, ir, catalog)
            rows.append(row)
    return write_revision(irfile, new_ir, output, {
        "stage": "size_constraints", "policy": {"minimum_em_px": 5, "maximum_growth": 1.20,
        "maximum_shrink_ratio": .20, "family_or_weight_changes": False},
        "counts": dict(Counter(r["status"] for r in rows)),
        "changed": sum(a != b for a, b in zip(ir["nodes"], new_ir["nodes"])), "decisions": rows})


def _pixels(preview, ir):
    pixels = Image.open(preview).convert("RGB")
    if pixels.size != tuple(ir["canvas"]["target_px"]):
        raise ValueError("preview size mismatch")
    return np.asarray(pixels, dtype=float)


def bound_preview(irfile, pptx, preview, slide=1):
    """Only accept the actual-file preview declared by this compiler's receipt."""
    from .font_plan import verify_font_export
    quality_path = Path(preview).resolve().parent.parent/"quality.json"
    quality = read_json(quality_path)
    if quality.get("pptx_sha256") != digest(pptx):
        raise ValueError("preview receipt PPTX binding mismatch")
    if quality.get("preview", {}).get("origin") != "reimported_final_pptx":
        raise ValueError("actual PPTX preview required")
    page = quality["pages"][slide-1]
    if page["ir_sha256"] != digest(irfile):
        raise ValueError("preview receipt IR binding mismatch")
    if (quality_path.parent/page["preview"]).resolve() != Path(preview).resolve():
        raise ValueError("preview receipt path mismatch")
    verify_font_export(irfile, pptx, slide=slide)
    return {"pptx_sha256": digest(pptx), "preview_sha256": digest(preview),
            "quality_sha256": digest(quality_path), "ir_sha256": digest(irfile)}


def region(node, ir, catalog, other=None):
    """Include source evidence and measured text extent, so spill is not rewarded."""
    boxes = [node["bbox"], node.get("source_text_bbox", node["bbox"])]
    for n in [node] + ([other] if other else []):
        try:
            m = extent(n, _size_px(n, ir), catalog)
            boxes.append([n["bbox"][0], n["bbox"][1], m["advance_width_px"],
                          max(n["bbox"][3], m["ink_height_px"])])
        except ValueError:
            pass
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[0]+b[2] for b in boxes); y1 = max(b[1]+b[3] for b in boxes)
    sw, sh = ir["canvas"]["source_px"]; tw, th = ir["canvas"]["target_px"]
    return [max(0, math.floor(x0*tw/sw)-4), max(0, math.floor(y0*th/sh)-4),
            min(tw, math.ceil(x1*tw/sw)+4), min(th, math.ceil(y1*th/sh)+4)]


def select(baseline, candidate, baseline_pptx, baseline_preview, candidate_pptx,
           candidate_preview, output, baseline_slide=1):
    baseline, original, _, source = _load_bound(baseline)
    candidate, proposed, _, _ = _load_bound(candidate)
    assert_compatible(original, proposed)
    receipts = {
        "baseline": bound_preview(baseline, baseline_pptx, baseline_preview, baseline_slide),
        "candidate": bound_preview(candidate, candidate_pptx, candidate_preview)}
    src = np.asarray(Image.open(source).convert("RGB").resize(
        tuple(original["canvas"]["target_px"]), Image.Resampling.LANCZOS), dtype=float)
    old = _pixels(baseline_preview, original); new = _pixels(candidate_preview, proposed)
    catalog = _candidates(None); revised = deepcopy(original); rows = []
    for index, (a, b) in enumerate(zip(original["nodes"], proposed["nodes"])):
        if a["type"] != "text":
            continue
        bbox = region(a, original, catalog, b)
        x0, y0, x1, y1 = bbox
        old_loss = float(np.abs(src[y0:y1, x0:x1]-old[y0:y1, x0:x1]).mean())
        new_loss = float(np.abs(src[y0:y1, x0:x1]-new[y0:y1, x0:x1]).mean())
        try:
            metrics = extent(b, _size_px(b, proposed), catalog)
            width, height = _limits(b)
            fits = metrics["advance_width_px"] <= width+1 and metrics["ink_height_px"] <= height+1
        except ValueError:
            fits = False
        # This only nominates candidates; a final mixed-version render must pass again.
        same_weight = a.get("bold", False) == b.get("bold", False)
        choose = can_select(a, b, fits, old_loss, new_loss)
        if choose:
            revised["nodes"][index] = deepcopy(b)
        rows.append({"id": a["id"], "selected": "candidate" if choose else "baseline",
                     "reason": "measured_fit_and_region_improvement" if choose else
                               "weight_change_requires_separate_evidence" if not same_weight else
                               "candidate_not_demonstrably_better",
                     "bbox_target_xyxy": bbox, "baseline_mae": old_loss, "candidate_mae": new_loss,
                     "measured_fits": fits, "weight_unchanged": same_weight})
    return write_revision(baseline, revised, output, {
        "stage": "render_evidence_selection", "preview_receipts": receipts,
        "candidate_ir_sha256": digest(candidate), "minimum_region_mae_gain": .20,
        "weight_policy": "preserve_baseline_until_separate_source_weight_evidence",
        "counts": dict(Counter(r["selected"] for r in rows)), "decisions": rows,
        "final_mixed_render_required": True})


def verify_selection(baseline, selected, baseline_pptx, baseline_preview,
                     selected_pptx, selected_preview, baseline_slide=1):
    baseline, before, _, source = _load_bound(baseline)
    selected, after, _, _ = _load_bound(selected)
    assert_compatible(before, after)
    report = read_json(selected.parent/"font-compatibility.json")
    if (report.get("stage") != "render_evidence_selection"
            or report["input_ir_sha256"] != digest(baseline)
            or report["output_ir_sha256"] != digest(selected)):
        raise ValueError("selection revision binding mismatch")
    old_receipt = bound_preview(baseline, baseline_pptx, baseline_preview, baseline_slide)
    if old_receipt != report["preview_receipts"]["baseline"]:
        raise ValueError("baseline preview changed after selection")
    new_receipt = bound_preview(selected, selected_pptx, selected_preview)
    verify_ledger(selected.parent, selected_pptx)
    src = np.asarray(Image.open(source).convert("RGB").resize(
        tuple(before["canvas"]["target_px"]), Image.Resampling.LANCZOS), dtype=float)
    old = _pixels(baseline_preview, before); new = _pixels(selected_preview, after)
    old_delta = np.abs(src-old); new_delta = np.abs(src-new)
    catalog = _candidates(None); mask = np.zeros(old.shape[:2], dtype=bool); regions = []
    decisions = {r["id"]: r for r in report["decisions"]}
    for a, b in zip(before["nodes"], after["nodes"]):
        if a["type"] != "text":
            continue
        decision = decisions[a["id"]]
        if a.get("bold", False) != b.get("bold", False):
            raise ValueError("selected version changed protected base weight")
        if decision["selected"] == "baseline":
            if a != b:
                raise ValueError("retained baseline object changed")
            continue
        x0, y0, x1, y1 = region(a, before, catalog, b)
        mask[y0:y1, x0:x1] = True
        old_loss = float(old_delta[y0:y1, x0:x1].mean())
        new_loss = float(new_delta[y0:y1, x0:x1].mean())
        regions.append({"id": a["id"], "bbox_target_xyxy": [x0,y0,x1,y1],
                        "baseline_mae": old_loss, "final_mae": new_loss,
                        "regressed": new_loss > old_loss+.05})
    changed_pixels = np.any(old != new, axis=2)
    outside_changes = int((changed_pixels & ~mask).sum())
    regression_ids = [r["id"] for r in regions if r["regressed"]]
    improvement = float(old_delta.mean()-new_delta.mean())
    passed = bool(regions) and improvement > 0 and not regression_ids and outside_changes == 0
    return {
        "schema": "FontSelectionVerification/1",
        "status": "PASS_COMPUTATIONAL_GATES" if passed else "REVIEW_REQUIRED",
        "baseline_mae": float(old_delta.mean()), "final_mae": float(new_delta.mean()),
        "global_mae_improvement": improvement, "changed_text_objects": len(regions),
        "selected_region_regressions": regression_ids,
        "outside_changed_text_envelopes_changed_pixels": outside_changes,
        "outside_envelopes_pixels_checked": int((~mask).sum()),
        "regions": regions, "final_receipt": new_receipt,
        "text_and_layout_unchanged": True, "human_acceptance": False,
        "source_font_identity_verified": False, "independent_model_result": False,
        "meaning": "Same-sample computational comparison; source identity, unseen pages and human acceptance not established."}
