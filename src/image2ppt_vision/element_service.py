"""Source-bound HTTP element reconstruction, with no model/network calls."""
import json
from .storage import digest, read_json, write_json
from .supervised_elements import validate_element_plan, materialize_elements


def _keys(value, allowed, required=()):
    if not isinstance(value, dict) or set(value)-set(allowed) or set(required)-set(value):
        raise ValueError("invalid element plan fields")


def validate_payload(plan):
    """Bound and validate externally supplied fields before queue admission."""
    try:
        if len(json.dumps(plan, allow_nan=False).encode()) > 1_000_000:
            raise ValueError("element plan exceeds one MB")
        _keys(plan, ("schema", "source_sha256", "image_size", "background_color",
                     "objects", "text_decisions", "limitations"),
              ("schema", "source_sha256", "image_size", "background_color", "objects", "text_decisions"))
        if not isinstance(plan["objects"], list) or not isinstance(plan["text_decisions"], list):
            raise ValueError("element lists required")
        if not 1 <= len(plan["objects"]) <= 400 or len(plan["objects"])+len(plan["text_decisions"]) > 500:
            raise ValueError("element count exceeds limit")
        for obj in plan["objects"]:
            _keys(obj, ("id", "kind", "bbox", "z", "candidate_id", "label", "geometry",
                        "fill", "line", "corner_radius_px", "fill_gradient", "mask", "contains_text_ids", "rationale"),
                  ("id", "kind", "bbox", "z"))
            for field in ("label", "rationale"):
                if field in obj and (not isinstance(obj[field],str) or len(obj[field])>4000):
                    raise ValueError("invalid element annotation")
            if obj.get("line") is not None:
                _keys(obj["line"], ("color", "width_pt", "dash"), ("color", "width_pt"))
                if obj["line"].get("dash", "solid") not in ("solid", "dash", "dot"):
                    raise ValueError("invalid line dash")
            if obj.get("fill_gradient") is not None:
                grad = obj["fill_gradient"]
                _keys(grad, ("angle_deg", "stops"), ("angle_deg", "stops"))
                if type(grad["angle_deg"]) not in (int, float) or not 0 <= grad["angle_deg"] <= 360:
                    raise ValueError("invalid gradient")
                if not isinstance(grad["stops"], list) or not 2 <= len(grad["stops"]) <= 10:
                    raise ValueError("invalid gradient stops")
                previous = -1
                from .supervised_elements import _color
                for stop in grad["stops"]:
                    _keys(stop, ("offset", "color"), ("offset", "color"))
                    if type(stop["offset"]) not in (int, float) or not 0 <= stop["offset"] <= 100000 or stop["offset"] < previous:
                        raise ValueError("invalid gradient offset")
                    previous = stop["offset"]; _color(stop["color"])
            if "corner_radius_px" in obj and (type(obj["corner_radius_px"]) not in (int, float) or not 0 <= obj["corner_radius_px"] <= 1000):
                raise ValueError("invalid corner radius")
        for text in plan["text_decisions"]:
            _keys(text, ("id", "action", "text", "bbox", "font_size_px", "bold", "color",
                         "z", "align", "runs", "font_family", "font_latin", "font_confidence", "font_evidence"),
                  ("id", "action", "text", "bbox", "font_size_px", "bold", "color", "z"))
            if type(text["z"]) is not int or text.get("align", "left") not in ("left", "center", "right"):
                raise ValueError("invalid text placement")
            if "runs" in text:
                if not isinstance(text["runs"], list) or len(text["runs"]) > 100:
                    raise ValueError("invalid runs")
                from .supervised_elements import _color
                for run in text["runs"]:
                    _keys(run, ("text", "color", "bold"), ("text",))
                    if not isinstance(run["text"], str):
                        raise ValueError("invalid run text")
                    if "color" in run: _color(run["color"])
                    if "bold" in run and type(run["bold"]) is not bool:
                        raise ValueError("invalid run weight")
        limitations = plan.get("limitations", [])
        if not isinstance(limitations, list) or len(limitations) > 100 or any(not isinstance(s, str) or len(s) > 4000 for s in limitations):
            raise ValueError("invalid limitations")
    except (KeyError, TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("malformed element plan") from exc
    return plan


def _verified(directory, manifest, name):
    path = directory/name
    if name not in manifest or not path.is_file() or digest(path) != manifest[name]["sha256"]:
        raise ValueError("analysis artifact integrity mismatch")
    return read_json(path)


def prepare(store, request):
    """Validate the same sealed analysis both at submission and worker time."""
    validate_payload(request.element_plan)
    asset_dir = store.path("assets", request.asset_id)
    asset = read_json(asset_dir/"asset.json")
    if asset.get("exif_orientation", 1) != 1:
        raise ValueError("element reconstruction requires an orientation-normalized input")
    if digest(asset_dir/"original") != asset["original_sha256"]:
        raise ValueError("source integrity mismatch")
    directory = store.path("jobs", request.analysis_job_id)
    if store.status(request.analysis_job_id)["status"] != "SUCCEEDED":
        raise ValueError("element reconstruction requires completed successful analysis")
    manifest = read_json(directory/"manifest.json")
    analysis_request = _verified(directory, manifest, "request.json")
    if analysis_request["operation"] != "analyze" or not analysis_request.get("element_inventory"):
        raise ValueError("analysis must include element_inventory")
    analysis = _verified(directory, manifest, "result.json")
    if not any(s["name"] == "ocr" and s["status"] == "PASS" for s in analysis["stages"]):
        raise ValueError("successful OCR evidence required")
    analyzed_asset = analysis["inputs"].get(analysis_request["asset_id"], {})
    if analyzed_asset.get("original_sha256") != asset["original_sha256"] or analysis["image_size"] != asset["size"]:
        raise ValueError("analysis source binding mismatch")
    inventory = _verified(directory, manifest, "elements/inventory.json")
    if inventory["source_sha256"] != asset["original_sha256"] or inventory["image_size"] != asset["size"]:
        raise ValueError("inventory source binding mismatch")
    try:
        plan = validate_element_plan(request.element_plan, inventory)
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("malformed element plan") from exc
    return asset_dir/"original", inventory, plan


def reconstruct(store, request, output):
    source, inventory, plan = prepare(store, request)
    output.mkdir()
    # Keep the submitted proposal separate from derived/calibrated output.
    write_json(output/"submitted-plan.json", request.element_plan)
    materialize_elements(source, inventory, plan, output/"proposal",
                         model_id="external-element-plan",
                         response_sha256=digest(output/"submitted-plan.json"))
    from .element_calibration import calibrate_elements
    calibration = calibrate_elements(output/"proposal/slideir.json", output/"calibrated",
                                    source_typography=True,
                                    renderer_version=request.renderer_version,
                                    calibrate_fills=False)
    details = read_json(output/"calibrated/calibration.json")
    rows = details["diagnostics"]
    issues = []
    for row in rows:
        if row["kind"] != "typography": continue
        if (row.get("output_contrast") or {}).get("low_contrast_risk"):
            issues.append({"id":row["id"], "code":"low_contrast_against_source_background"})
        if not row["applied"]:
            issues.append({"id":row["id"], "code":"source_typography_refused", "reason":row.get("reason")})
        elif not row["renderer_origin_calibration"]["applied"]:
            issues.append({"id":row["id"], "code":"renderer_origin_unverified",
                           "reason":row["renderer_origin_calibration"]["reason"]})
    quality = {"schema":"ElementServiceQuality/1", "status":"REVIEW_REQUIRED",
               "source_sha256":inventory["source_sha256"], "analysis_job_id":request.analysis_job_id,
               "renderer_version":request.renderer_version, "issues":issues,
               "typography_applied":calibration["typography_applied"],
               "source_color_only_applied":calibration["source_color_only_applied"],
               "typography_total":calibration["typography_total"],
               "pptx_compiled":False, "actual_render_verified":False, "human_acceptance":False}
    write_json(output/"quality.json", quality)
    return {"status":"REVIEW_REQUIRED", "slideir":"reconstruction/calibrated/slideir.json",
            "quality":"reconstruction/quality.json", "typography_applied":calibration["typography_applied"],
            "unresolved_issues":len(issues), "human_acceptance":False}
