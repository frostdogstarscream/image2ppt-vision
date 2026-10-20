"""Conservative source-pixel diagnostics and native neutral-bar completion.

This is local postprocessing, not a repaired model response. It neither estimates
text nor moves existing objects. Every candidate still needs actual-file review.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image

from .font_plan import _load_bound, verify_font_export
from .module_plan import assign_nodes, edge_support, _save_review, verify_module_page
from .storage import digest, read_json, write_json
from .supervised_elements import _intersection
from .text_ledger import structure_hash, verify_ledger


def refine_modules(ir, plan, rgb):
    """Expand only edges implicated by pending nodes, with direct pixel support.

    Do not shrink boundaries, relabel modules, silently repair model anchors or
    collect page-sized backgrounds into a semantic module.
    """
    modules = deepcopy(plan["modules"])
    observations = []
    by_id = {m["id"]: m for m in modules}
    width, height = ir["canvas"]["source_px"]
    for row in plan["node_assignments"]:
        box = row["reference_bbox"]
        if box[2]*box[3] >= .8*width*height:
            observations.append({"node_id": row["node_id"], "status": "PAGE_SCOPE",
                                 "reason": "page_sized_extent_not_a_semantic_module"})
            continue
        if row["status"] == "assigned":
            if row.get("current_bbox_overflow_px", 0) <= 2:
                continue
            mid = row["module_id"]
        else:
            candidates = row.get("candidates", [])
            if not candidates or candidates[0]["coverage"] < .55:
                continue
            if len(candidates) > 1 and candidates[0]["coverage"]-candidates[1]["coverage"] < .10:
                continue
            mid = candidates[0]["module_id"]
        module = by_id[mid]
        # Each candidate is bounded relative to the ORIGINAL model edges, not
        # an accumulating search window that could grow with successive nodes.
        original = next(m for m in plan["modules"] if m["id"] == mid)["bbox"]
        for axis, side in ((0, 0), (1, 0), (0, 1), (1, 1)):
            current = module["bbox"]
            edge = current[axis] + side*current[axis+2]
            need = box[axis] + side*box[axis+2]
            # One-pixel antialiasing/rounding differences are not a reason to
            # jump from a matched edge to another nearby card's edge.
            if abs(need-edge) <= 2:
                continue
            if (side == 0 and need >= edge) or (side == 1 and need <= edge):
                continue
            origin = original[axis]+side*original[axis+2]
            if abs(need-origin) > 24:
                continue
            gradient = np.max(np.abs(np.diff(rgb.astype(np.int16), axis=1-axis)), axis=2)
            lo = max(0, int(original[1-axis])+3)
            hi = min(rgb.shape[axis], int(original[1-axis]+original[3-axis])-3)
            scores = []
            for position in range(max(1, int(origin)-24), min(rgb.shape[1-axis], int(origin)+25)):
                if (side == 0 and not position < edge) or (side == 1 and not position > edge):
                    continue
                if (side == 0 and position > need+1) or (side == 1 and position < need-1):
                    continue
                pixels = gradient[lo:hi, position-1] if axis == 0 else gradient[position-1, lo:hi]
                support = float((pixels >= 6).mean()) if pixels.size else 0.
                if support >= .40:
                    scores.append((support, -abs(position-origin), position))
            if not scores:
                continue
            support, _, position = max(scores)
            before = list(current)
            if side == 0:
                current[axis+2] += current[axis]-position
                current[axis] = position
            else:
                current[axis+2] = position-current[axis]
            observations.append({"module_id": mid, "trigger_node_id": row["node_id"],
                                 "axis": "xy"[axis], "side": "start" if side == 0 else "end",
                                 "before": before, "after": list(current),
                                 "source_edge_support": round(support, 6),
                                 "origin": "local_source_pixel_analysis",
                                 "applied_to_ir": False})
    assignments = assign_nodes(ir, modules)
    assigned = {a["node_id"]: a["module_id"] for a in assignments}
    for module in modules:
        module["member_ids"] = [a["node_id"] for a in assignments if a["module_id"] == module["id"]]
        module["pixel_edge_support"] = edge_support(rgb, module["bbox"])
    overlaps = [{"module_ids": [a["id"], b["id"]],
                 "intersection_px2": _intersection(a["bbox"], b["bbox"]),
                 "status": "REVIEW_REQUIRED", "auto_resolved": False}
                for i, a in enumerate(modules) for b in modules[i+1:]
                if _intersection(a["bbox"], b["bbox"]) > 0]
    return {"schema": "DerivedModuleReview/1", "source_sha256": ir["source_sha256"],
            "modules": modules, "node_assignments": assignments,
            "counts": dict(Counter(a["status"] for a in assignments)),
            "observations": observations,
            "module_overlaps": overlaps,
            "anchor_issues": [{"module_id": m["id"], "anchor_node_id": anchor,
                              "geometric_module_id": assigned[anchor],
                              "status": "REVIEW_REQUIRED", "model_anchor_preserved": True}
                             for m in modules for anchor in m["anchor_node_ids"]
                             if assigned[anchor] != m["id"]],
            "containment_issues": [a for a in assignments if a.get("current_bbox_overflow_px", 0) > 2],
            "origin": "deterministic_local_postprocessing_not_model_output",
            "geometry_changes": 0, "human_acceptance": False}


def neutral_bars(ir, rgb, modules):
    """Find long, solid, neutral light-gray rectangles; exclude text and assets.

    Conservative deliberately: faint/antialiased/colored bars can remain missing.
    Existing containers do not count as existing bars, but foreground images,
    text envelopes and any overlapping foreground shape prevent additions.
    """
    rgb = np.asarray(rgb, dtype=np.int16)
    mask = ((rgb.max(2)-rgb.min(2) <= 10) &
            (rgb.min(2) >= 205) & (rgb.max(2) <= 238)).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    accepted, rejected = [], []
    for component, (x, y, w, h, area) in enumerate(stats[1:], 1):
        x, y, w, h, area = map(int, (x, y, w, h, area))
        if not (45 <= w <= 300 and 3 <= h <= 10 and w/h >= 8 and area/(w*h) >= .90):
            continue
        box = [x, y, w, h]
        owners = [m["id"] for m in modules if m["kind"] in ("panel", "workflow")
                  and _intersection(box, m["bbox"])/(w*h) >= .98]
        blockers = []
        for node in ir["nodes"]:
            candidates = [node["bbox"]]
            if "source_text_bbox" in node:
                candidates.append(node["source_text_bbox"])
            overlap = max(_intersection(box, b) for b in candidates)
            if overlap <= 0:
                continue
            if node["type"] == "text" or node["type"] == "image":
                blockers.append(node["id"])
            elif node["bbox"][2]*node["bbox"][3] <= 10*w*h:
                blockers.append(node["id"])
        row = {"bbox": box, "pixel_occupancy": area/(w*h), "module_ids": owners}
        if len(owners) != 1 or blockers:
            rejected.append({**row, "blockers": blockers, "status": "REVIEW_REQUIRED"})
            continue
        pixels = rgb[y:y+h, x:x+w][labels[y:y+h, x:x+w] == component]
        color = np.rint(np.median(pixels, axis=0)).astype(int)
        accepted.append({**row, "fill": "".join(f"{v:02X}" for v in color),
                         "origin": "connected_source_pixels_not_model_output"})
    return accepted, rejected


def complete_bars(ir, bars):
    result = deepcopy(ir)
    z = max(n["z"] for n in result["nodes"]) + 1
    for i, bar in enumerate(bars):
        identifier = f"source-bar-{i:03d}"
        evidence = f"ev-{identifier}"
        if any(n["id"] == identifier for n in result["nodes"]):
            raise ValueError("source-bar ID collision")
        result["evidence"].append({"id": evidence, "kind": "shape", "bbox": bar["bbox"]})
        result["nodes"].append({"id": identifier, "type": "shape", "geometry": "rect",
                                "bbox": bar["bbox"], "fill": bar["fill"], "z": z+i,
                                "evidence_ids": [evidence]})
    return result


def materialize(folder, output):
    folder, output = Path(folder), Path(output)
    if output.exists():
        raise FileExistsError("a new graphics revision is required")
    verify_module_page(folder)
    irfile, ir, authoring, source = _load_bound(folder/"slideir.json")
    rgb = np.asarray(Image.open(source).convert("RGB"))
    refinement = refine_modules(ir, read_json(folder/"module-plan.json"), rgb)
    bars, rejected = neutral_bars(ir, rgb, refinement["modules"])
    updated = complete_bars(ir, bars)
    output.mkdir(parents=True)
    for name in ("assets", "source", "crops", "evidence-inputs"):
        if (folder/name).is_dir():
            shutil.copytree(folder/name, output/name)
    parents = output/"graphics-parent"
    parents.mkdir()
    for name in ("slideir.json", "module-plan.json", "module-proposal.json", "authoring.json",
                 "text-ledger.json"):
        shutil.copyfile(folder/name, parents/name)
    if (folder/"text-ledger.html").exists():
        shutil.copyfile(folder/"text-ledger.html", output/"text-ledger.html")
    write_json(output/"slideir.json", updated)
    write_json(output/"derived-module-review.json", refinement)
    _save_review(output, refinement)
    # The generic renderer's caption refers to the model; make the provenance
    # explicit without changing the retained original model proposal.
    review = output/"module-review.html"
    review.write_text(review.read_text("utf8").replace(
        "模型边界待核对；本次未移动对象，未生成真实 PPT 分组。",
        "本地源图边缘派生边界，非模型原始输出；未移动现有对象，未生成 PPT 分组。"), "utf8")
    ledger = read_json(folder/"text-ledger.json")
    ledger.update(graphics_revision_parent_ir_sha256=digest(irfile),
                  output_ir_sha256=digest(output/"slideir.json"), structure_sha256=structure_hash(updated))
    write_json(output/"text-ledger.json", ledger)
    for key in list(authoring):
        if key.startswith("module_plan_"):
            authoring.pop(key)
    authoring.update(tool="source-pixel-neutral-bar-completion", ir_sha256=digest(output/"slideir.json"),
                     created_at=datetime.now(timezone.utc).isoformat(),
                     derived_module_review_sha256=digest(output/"derived-module-review.json"))
    write_json(output/"authoring.json", authoring)
    report = {"schema": "SourceGraphics/1", "parent_ir_sha256": digest(irfile),
              "parent_module_plan_sha256": digest(parents/"module-plan.json"),
              "output_ir_sha256": digest(output/"slideir.json"), "source_sha256": ir["source_sha256"],
              "bars": bars, "rejected": rejected, "existing_node_changes": 0,
              "added_native_shapes": len(bars), "independent_model_result": False,
              "human_acceptance": False}
    write_json(output/"source-graphics.json", report)
    return verify(folder=output)


def verify(folder, pptx=None):
    folder = Path(folder)
    irfile, ir, authoring, source = _load_bound(folder/"slideir.json")
    report = read_json(folder/"source-graphics.json")
    parent = folder/"graphics-parent"
    base = read_json(parent/"slideir.json")
    if (digest(parent/"slideir.json") != report["parent_ir_sha256"] or
            digest(parent/"module-plan.json") != report["parent_module_plan_sha256"] or
            digest(irfile) != report["output_ir_sha256"]):
        raise ValueError("graphics provenance binding mismatch")
    if base["source_sha256"] != digest(source):
        raise ValueError("graphics parent source mismatch")
    rgb = np.asarray(Image.open(source).convert("RGB"))
    refined = refine_modules(base, read_json(parent/"module-plan.json"), rgb)
    if (refined != read_json(folder/"derived-module-review.json") or
            digest(folder/"derived-module-review.json") != authoring["derived_module_review_sha256"]):
        raise ValueError("derived module review changed")
    bars, rejected = neutral_bars(base, rgb, refined["modules"])
    if bars != report["bars"] or rejected != report["rejected"] or complete_bars(base, bars) != ir:
        raise ValueError("graphics revision differs from deterministic source evidence")
    text = verify_ledger(folder, pptx)
    fonts = verify_font_export(irfile, pptx) if pptx else None
    if pptx:
        ns = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
              "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
        with ZipFile(pptx) as package:
            root = ET.fromstring(package.read("ppt/slides/slide1.xml"))
        shapes = {e.find("p:nvSpPr/p:cNvPr", ns).get("name"): e
                  for e in root.findall(".//p:sp", ns)}
        for node in ir["nodes"][len(base["nodes"]):]:
            shape = shapes.get(node["id"])
            if shape is None:
                raise ValueError("new bar is not a native PPT shape")
            geom = shape.find("p:spPr/a:prstGeom", ns)
            fill = shape.find("p:spPr/a:solidFill/a:srgbClr", ns)
            if geom is None or geom.get("prst") != "rect" or fill is None or fill.get("val") != node["fill"]:
                raise ValueError("bar geometry/fill changed during export")
            transform = shape.find("p:spPr/a:xfrm", ns)
            actual = [int(transform.find("a:off", ns).get(k)) for k in ("x", "y")]
            actual += [int(transform.find("a:ext", ns).get(k)) for k in ("cx", "cy")]
            expected = [node["bbox"][i]*ir["canvas"]["target_px"][i % 2] /
                        ir["canvas"]["source_px"][i % 2]*9525 for i in range(4)]
            if any(abs(a-b) > 2 for a, b in zip(actual, expected)):
                raise ValueError("bar position changed during export")
    return {"status": "PASS_SOURCE_BOUND_STRUCTURE", "existing_node_changes": 0,
            "added_native_shapes": len(bars), "nodes": len(ir["nodes"]),
            "module_counts_for_existing_nodes": refined["counts"],
            "anchor_issues": len(refined["anchor_issues"]),
            "containment_issues": len(refined["containment_issues"]),
            "text": text, "fonts": fonts, "visual_acceptance": False, "human_acceptance": False}


def compare_pixels(source, before, after, bars, source_size):
    """Measure each changed region and demand exact equality elsewhere."""
    source, before, after = [np.asarray(a, dtype=np.float64) for a in (source, before, after)]
    if source.shape != before.shape or before.shape != after.shape:
        raise ValueError("comparison canvas mismatch")
    height, width = before.shape[:2]
    mask = np.zeros((height, width), bool)
    rows = []
    for bar in bars:
        x, y, w, h = bar["bbox"]
        a = max(0, int(np.floor(x*width/source_size[0]))-2)
        b = max(0, int(np.floor(y*height/source_size[1]))-2)
        c = min(width, int(np.ceil((x+w)*width/source_size[0]))+2)
        d = min(height, int(np.ceil((y+h)*height/source_size[1]))+2)
        mask[b:d, a:c] = True
        old = float(np.abs(before[b:d, a:c]-source[b:d, a:c]).mean())
        new = float(np.abs(after[b:d, a:c]-source[b:d, a:c]).mean())
        rows.append({"source_bbox": bar["bbox"], "target_roi": [a, b, c-a, d-b],
                     "before_mae": old, "after_mae": new, "improvement": old-new})
    changed = np.any(before != after, axis=2)
    outside_changes = int((changed & ~mask).sum())
    old_global = float(np.abs(before-source).mean())
    new_global = float(np.abs(after-source).mean())
    passed = (bool(rows) and outside_changes == 0 and new_global < old_global and
              all(r["improvement"] > 0 for r in rows))
    return {"status": "PASS_COMPUTATIONAL_GATES" if passed else "REVIEW_REQUIRED",
            "regions": rows, "outside_changed_pixels": outside_changes,
            "outside_checked_pixels": int((~mask).sum()), "changed_pixels": int(changed.sum()),
            "global_before_mae": old_global, "global_after_mae": new_global,
            "human_acceptance": False}
