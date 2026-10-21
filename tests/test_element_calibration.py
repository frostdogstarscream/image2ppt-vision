import json
from copy import deepcopy
import numpy as np
from PIL import Image, ImageDraw
from image2ppt_vision.element_calibration import (
    _fill, _fill_decision, _fill_measured, _ink, _ink_measured, _tighten,
    calibrate_elements,
)
from image2ppt_vision.storage import digest, write_json

def test_fill_measurement_excludes_child_objects_and_never_mutates_pixels():
    p=np.full((100,200,3),245,np.uint8);p[30:70,50:150]=[0,90,180];before=p.copy()
    parent={'id':'p','type':'shape','geometry':'roundRect','bbox':[0,0,200,100],'fill':'000000'}
    child={'id':'c','type':'shape','geometry':'rect','bbox':[50,30,100,40],'fill':'005AB4'}
    result=_fill(p,parent,[parent,child]);assert result['color']=='F5F5F5';np.testing.assert_array_equal(p,before)

def test_unsupported_geometry_and_nonuniform_fill_refused():
    p=np.full((100,200,3),255,np.uint8);node={'id':'p','type':'shape','geometry':'rightArrow','bbox':[0,0,200,100],'fill':'000000'}
    assert _fill(p,node,[node]) is None
    node['geometry']='rect';p[:,:100]=0
    assert _fill(p,node,[node]) is None

def test_gradient_bar_declared_dark_is_measured_light():
    """Page 05's panel header: a light gradient bar the plan calls dark blue."""
    p=np.empty((60,400,3),np.uint8)
    for x in range(400):
        p[:,x]=[int(235+20*x/399)]*3
    node={'id':'bar','type':'shape','geometry':'rect','bbox':[0,0,400,60],'fill':'1a5fb4',
          'fill_gradient':{'angle_deg':0,'stops':[{'offset':0,'color':'1a5fb4'},{'offset':100000,'color':'0d4a9e'}]}}
    result=_fill(p,node,[node])
    assert result is not None and int(result['color'][:2],16)>=228

def test_declared_gradient_survives_when_source_is_a_gradient():
    p=np.empty((60,400,3),np.uint8)
    for x in range(400):
        p[:,x]=[int(240-120*x/399)]*3
    node={'id':'bar','type':'shape','geometry':'rect','bbox':[0,0,400,60],'fill':'f0f0f0',
          'fill_gradient':{'angle_deg':0,'stops':[{'offset':0,'color':'f0f0f0'},{'offset':100000,'color':'787878'}]}}
    result=_fill(p,node,[node])
    assert result is not None and 'gradient' in result

def test_text_ink_separates_dark_glyphs_from_a_light_bar():
    p=np.full((30,80,3),240,np.uint8);p[8:22,10:50]=[20,74,195]
    ink=_ink(p,[0,0,80,30])
    assert ink is not None
    assert abs(int(ink['color'][:2],16)-20)<12 and ink['separation']>100

def test_text_ink_refuses_white_on_white():
    """A proposal that is invisible in the source has nothing to measure."""
    p=np.full((30,80,3),255,np.uint8)
    assert _ink(p,[0,0,80,30]) is None

def _banner_pixels():
    """A bordered banner inside a box that also covers the page around it."""
    p=np.full((120,400,3),245,np.uint8)
    p[30:80,60:300]=[232,244,253]
    p[30:32,60:300]=[200,216,240];p[78:80,60:300]=[200,216,240]
    p[30:80,60:62]=[200,216,240];p[30:80,298:300]=[200,216,240]
    return p

def test_overrunning_box_is_tightened_to_its_declared_border():
    p=_banner_pixels()
    node={'id':'b','type':'shape','geometry':'roundRect','bbox':[0,0,400,120],
          'fill':'e8f4fd','line':{'color':'c8d8f0','width_pt':1}}
    box=_tighten(p,node,[node])
    assert box is not None
    x,y,w,h=box
    assert abs(x-60)<=3 and abs(y-30)<=3 and abs(w-240)<=5 and abs(h-50)<=5

def test_box_whose_border_is_already_at_its_edge_is_kept():
    p=_banner_pixels()
    node={'id':'b','type':'shape','geometry':'roundRect','bbox':[60,30,240,50],
          'fill':'e8f4fd','line':{'color':'c8d8f0','width_pt':1}}
    assert _tighten(p,node,[node]) is None

def test_box_without_a_border_is_never_tightened():
    p=_banner_pixels()
    node={'id':'b','type':'shape','geometry':'rect','bbox':[0,0,400,120],'fill':'e8f4fd'}
    assert _tighten(p,node,[node]) is None


def _gradient_pixels(start, end, width=400, height=60):
    pixels = np.empty((height, width, 3), np.uint8)
    for x in range(width):
        fraction = x / float(width - 1)
        pixels[:, x] = [int(start[c] + (end[c] - start[c]) * fraction) for c in range(3)]
    return pixels


def _decide(pixels, node):
    fitted, measurement = _fill_measured(pixels, node, [node])
    return fitted, (None if fitted is None else _fill_decision(node, fitted, measurement))


def _gradient_node(start, end, angle=0):
    return {"id": "bar", "type": "shape", "geometry": "rect", "bbox": [0, 0, 400, 60],
            "fill": start, "fill_gradient": {"angle_deg": angle, "stops": [
                {"offset": 0, "color": start}, {"offset": 100000, "color": end}]}}


def test_declared_gradient_wrong_endpoint_is_replaced():
    source = _gradient_pixels((240, 240, 240), (120, 120, 120))
    node = _gradient_node("F0F0F0", "000000")
    fitted, decision = _decide(source, node)
    assert decision["apply"] and decision["decision"] == "applied"
    last = fitted["gradient"]["stops"][-1]["color"]
    assert 0x70 <= int(last[:2], 16) <= 0x85
    assert decision["declared_error"]["mean"] > decision["candidate_error"]["mean"]


def test_declared_gradient_wrong_angle_is_replaced():
    source = _gradient_pixels((240, 240, 240), (120, 120, 120))
    node = _gradient_node("F0F0F0", "787878", angle=90)
    fitted, decision = _decide(source, node)
    assert decision["apply"]
    assert fitted["gradient"]["angle_deg"] == 0


def test_declared_gradient_matching_source_is_kept():
    source = _gradient_pixels((240, 240, 240), (120, 120, 120))
    node = _gradient_node("F0F0F0", "787878")
    fitted, decision = _decide(source, node)
    assert fitted is not None and not decision["apply"]
    assert decision["decision"] == "declared_matches_source"


def test_declared_solid_against_source_gradient_becomes_gradient():
    source = _gradient_pixels((240, 240, 240), (60, 60, 60))
    node = {"id": "bar", "type": "shape", "geometry": "rect", "bbox": [0, 0, 400, 60],
            "fill": "808080"}
    fitted, decision = _decide(source, node)
    assert decision["apply"] and decision["candidate_kind"] == "gradient"
    assert "gradient" in fitted and fitted["gradient"]["angle_deg"] == 0


def test_declared_gradient_against_flat_source_becomes_solid():
    source = np.full((60, 400, 3), 205, np.uint8)
    node = _gradient_node("F0F0F0", "303030")
    fitted, decision = _decide(source, node)
    assert decision["apply"] and decision["candidate_kind"] != "gradient"
    assert "gradient" not in fitted


def test_complex_texture_leaves_declaration_unchanged():
    grid = np.indices((60, 400)).sum(axis=0) % 40
    source = np.repeat(grid[..., None], 3, axis=2).astype(np.uint8) * 6
    node = {"id": "tex", "type": "shape", "geometry": "rect", "bbox": [0, 0, 400, 60],
            "fill": "112233"}
    fitted, decision = _decide(source, node)
    assert fitted is None or "gradient" not in fitted


def test_non_axis_angle_ambiguous_is_not_rewritten():
    yy, xx = np.mgrid[:60, :400]
    projected = .5 * (xx / 399.0) + .5 * (yy / 59.0)
    level = np.rint(255 * projected).astype(np.uint8)
    source = np.repeat(level[..., None], 3, axis=2)
    node = _gradient_node("000000", "FFFFFF", angle=45)
    fitted, measurement = _fill_measured(source, node, [node])
    # The single-axis projection guard rejects this diagonal source; the
    # declaration check still runs and must refuse to rewrite an ambiguous angle.
    decision = _fill_decision(node, fitted, measurement)
    assert not decision["apply"] and decision["decision"] == "angle_semantics_ambiguous"


def test_diagonal_plane_is_not_projected_to_axis_gradient():
    yy, xx = np.mgrid[:120, :200]
    level = np.rint(220 - 70 * xx / 199 - 70 * yy / 119).astype(np.uint8)
    source = np.repeat(level[..., None], 3, axis=2)
    node = {"id": "diag", "type": "shape", "geometry": "rect", "bbox": [0, 0, 200, 120],
            "fill": "000000"}
    fitted = _fill(source, node, [node])
    assert fitted is None or "gradient" not in fitted


def test_single_axis_gradient_survives_projection_guard():
    source = np.empty((200, 120, 3), np.uint8)
    for y in range(200):
        fraction = y / 199.0
        source[y, :] = [int(244 + (180 - 244) * fraction),
                        int(245 + (181 - 245) * fraction),
                        int(246 + (182 - 246) * fraction)]
    node = {"id": "vertical", "type": "shape", "geometry": "rect", "bbox": [0, 0, 120, 200],
            "fill": "F4F5F6"}
    fitted = _fill(source, node, [node])
    assert fitted is not None and fitted["gradient"]["angle_deg"] == 90


def test_ink_refuses_nonlinear_gradient_without_glyphs():
    x = np.linspace(0, 1, 400)
    level = np.rint(80 + 175 * x * x).astype(np.uint8)
    source = np.repeat(np.repeat(level[None, :, None], 60, axis=0), 3, axis=2)
    ink, diagnostics = _ink_measured(source, [0, 0, 400, 60])
    assert ink is None and diagnostics["reason"] == "no_glyph_evidence"


def test_ink_refuses_separator_line():
    source = np.full((60, 400, 3), 245, np.uint8)
    source[28:31, 10:390] = 40
    ink, diagnostics = _ink_measured(source, [0, 0, 400, 60])
    assert ink is None and diagnostics["reason"] == "separator_line"


def test_ink_ignores_a_rule_when_glyphs_are_present():
    source = np.full((60, 200, 3), 245, np.uint8)
    source[10:13, 10:190] = 40
    source[25:42, 60:90] = [20, 74, 195]
    source[25:42, 110:140] = [20, 74, 195]
    ink, diagnostics = _ink_measured(source, [0, 0, 200, 60])
    assert ink is not None
    assert abs(int(ink["color"][:2], 16) - 20) < 15


def test_ink_excludes_neighbour_text_outside_ocr_box():
    source = np.full((60, 160, 3), 245, np.uint8)
    source[18:30, 60:90] = [20, 74, 195]
    source[37:47, 60:90] = [200, 30, 30]
    ink, diagnostics = _ink_measured(source, [50, 15, 60, 20])
    assert ink is not None
    assert abs(int(ink["color"][:2], 16) - 20) < 15


def test_ink_refuses_mixed_colour_glyphs():
    source = np.full((60, 160, 3), 245, np.uint8)
    source[18:33, 60:75] = [20, 74, 195]
    source[18:33, 90:105] = [200, 30, 30]
    ink, diagnostics = _ink_measured(source, [50, 15, 80, 30])
    assert ink is None and diagnostics["reason"] == "mixed_colour_glyphs"


def test_ink_refuses_mostly_filled_box():
    source = np.full((60, 160, 3), 245, np.uint8)
    source[10:50, 30:140] = [20, 40, 80]
    ink, diagnostics = _ink_measured(source, [10, 5, 140, 50])
    assert ink is None and diagnostics["reason"] == "foreground_not_sparse_text"


def test_ink_refuses_one_pixel_specks():
    source = np.full((60, 160, 3), 245, np.uint8)
    source[20, 60] = 0
    source[30, 90] = 0
    ink, diagnostics = _ink_measured(source, [50, 15, 80, 30])
    assert ink is None


def test_ink_accepts_small_antialiased_cjk_components():
    source = np.full((30, 60, 3), 245, np.uint8)
    for left in (12, 22, 32):
        source[8:13, left:left + 4] = [35, 45, 60]
        source[8:11, left + 1:left + 3] = [35, 45, 60]
    ink, diagnostics = _ink_measured(source, [6, 4, 38, 16])
    assert ink is not None and diagnostics["accepted_components"] >= 2
    assert int(ink["color"][:2], 16) < 80


def test_ink_accepts_white_glyphs_on_dark_bar():
    source = np.full((60, 300, 3), 40, np.uint8)
    source[20:40, 60:90] = 245
    source[20:40, 110:140] = 245
    ink = _ink(source, [50, 15, 110, 30])
    assert ink is not None and int(ink["color"][:2], 16) > 220


def test_ink_accepts_dark_glyphs_on_light_gradient_bar():
    """Page 05's panel header must keep its glyph-ink repair."""
    source = np.empty((60, 400, 3), np.uint8)
    for x in range(400):
        source[:, x] = [int(235 + 15 * x / 399)] * 3
    source[20:40, 60:90] = [20, 40, 80]
    source[20:40, 110:140] = [20, 40, 80]
    ink = _ink(source, [50, 15, 110, 30])
    assert ink is not None and int(ink["color"][:2], 16) < 60


def _write_ir(tmp_path, pixels, nodes):
    proposal = tmp_path / "proposal"
    (proposal / "source").mkdir(parents=True)
    Image.fromarray(pixels).save(proposal / "source" / "page.png")
    source = proposal / "source" / "page.png"
    height, width = pixels.shape[:2]
    ir = {"schema_version": "0.1", "source_image": "source/page.png",
          "source_sha256": digest(source),
          "canvas": {"source_px": [width, height], "target_px": [width, height]},
          "background": {"color": "FFFFFF"}, "review_status": "needs_review",
          "review_notes": [], "evidence": [], "groups": [], "assets": [], "nodes": nodes}
    write_json(proposal / "slideir.json", ir)
    write_json(proposal / "authoring.json",
               {"ir_sha256": digest(proposal / "slideir.json"), "model_typography": {}})
    return proposal


def test_calibration_records_surface_errors_for_fill_decision(tmp_path):
    pixels = _gradient_pixels((240, 240, 240), (120, 120, 120))
    node = _gradient_node("F0F0F0", "000000")
    proposal = _write_ir(tmp_path, pixels, [node])
    output = tmp_path / "calibrated"
    calibrate_elements(proposal / "slideir.json", output, source_typography=False,
                       calibrate_fills=True)
    report = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    row = next(item for item in report["diagnostics"] if item["kind"] == "fill")
    assert row["applied"] and row["basis"] == "source_pixel_surface_comparison"
    assert row["declared_error"]["mean"] > row["candidate_error"]["mean"]
    assert row["before"]["fill_gradient"]["stops"][-1]["color"] == "000000"
    assert row["after"]["fill_gradient"]["stops"][-1]["color"] != "000000"
    calibrated = json.loads((output / "slideir.json").read_text(encoding="utf-8"))
    assert calibrated["nodes"][0]["fill_gradient"]["stops"][-1]["color"] != "000000"


def test_glyph_ink_refusal_is_recorded_and_colour_unchanged(tmp_path, monkeypatch):
    pixels = np.full((80, 120, 3), 255, np.uint8)
    node = {"id": "t1", "type": "text", "text": "标题", "bbox": [10, 10, 60, 20],
            "source_text_bbox": [10, 10, 60, 20],
            "font": {"latin": "Fixture", "ea": "Fixture", "size_pt": 10},
            "bold": False, "color": "FFFFFF", "runs": [], "align": "left", "wrap": False}
    proposal = _write_ir(tmp_path, pixels, [node])
    monkeypatch.setattr("image2ppt_vision.element_calibration.estimate_text",
                        lambda *a, **k: {"native_eligible": False, "diagnostics": {
                            "reason": "non_smooth_or_crossing_background"}})
    output = tmp_path / "calibrated"
    calibrate_elements(proposal / "slideir.json", output, source_typography=True,
                       calibrate_fills=True)
    report = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    row = next(item for item in report["diagnostics"] if item["kind"] == "typography")
    assert row["source_color_refused_reason"] == "no_glyph_evidence"
    calibrated = json.loads((output / "slideir.json").read_text(encoding="utf-8"))
    assert calibrated["nodes"][0]["color"] == "FFFFFF"


def test_glyph_ink_repairs_dark_text_when_font_refused(tmp_path, monkeypatch):
    pixels = np.full((80, 120, 3), 245, np.uint8)
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    draw.rectangle((14, 14, 30, 24), fill=(20, 74, 195))
    draw.rectangle((38, 14, 54, 24), fill=(20, 74, 195))
    pixels = np.array(image)
    node = {"id": "t1", "type": "text", "text": "标题", "bbox": [10, 10, 60, 20],
            "source_text_bbox": [10, 10, 60, 20],
            "font": {"latin": "Fixture", "ea": "Fixture", "size_pt": 10},
            "bold": False, "color": "FFFFFF", "runs": [], "align": "left", "wrap": False}
    proposal = _write_ir(tmp_path, pixels, [node])
    monkeypatch.setattr("image2ppt_vision.element_calibration.estimate_text",
                        lambda *a, **k: {"native_eligible": False, "diagnostics": {
                            "reason": "non_smooth_or_crossing_background"}})
    output = tmp_path / "calibrated"
    calibrate_elements(proposal / "slideir.json", output, source_typography=True,
                       calibrate_fills=True)
    report = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    row = next(item for item in report["diagnostics"] if item["kind"] == "typography")
    assert row["source_color_applied"] and row["source_color_basis"] == "glyph_ink"
    calibrated = json.loads((output / "slideir.json").read_text(encoding="utf-8"))
    assert calibrated["nodes"][0]["color"] != "FFFFFF"


def _policy_node(text="标题", runs=None):
    return {"id": "t1", "type": "text", "text": text, "bbox": [10, 10, 60, 20],
            "source_text_bbox": [10, 10, 60, 20],
            "font": {"latin": "ProposalLatin", "ea": "ProposalEA", "size_pt": 10},
            "bold": False, "color": "FFFFFF", "runs": runs or [], "align": "center",
            "wrap": False}


def _eligible_estimate(**overrides):
    estimate = {"native_eligible": True, "bbox": [12, 12, 56, 18],
                "font_family": "MeasuredFont", "font_size_px": 14.25, "bold": True,
                "color": "#112233", "runs": None,
                "diagnostics": {"reason": "fixture", "foreground_color_count": 1,
                                "background": {"reliable": True},
                                "font": {"render_origin": [11, 11], "glyph_bbox": [1, 1, 55, 17],
                                         "shape_overlap": .9, "dimension_error": .05,
                                         "font_path": "missing-font-fixture"}}}
    estimate.update(overrides)
    return estimate


def _run_policy(tmp_path, monkeypatch, node, estimate, **kwargs):
    pixels = np.full((80, 160, 3), 245, np.uint8)
    proposal = _write_ir(tmp_path, pixels, [node])
    monkeypatch.setattr("image2ppt_vision.element_calibration.estimate_text",
                        lambda *a, **k: deepcopy(estimate))
    output = tmp_path / "calibrated"
    calibrate_elements(proposal / "slideir.json", output, calibrate_fills=True, **kwargs)
    report = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    calibrated = json.loads((output / "slideir.json").read_text(encoding="utf-8"))
    row = next(item for item in report["diagnostics"] if item["kind"] == "typography")
    return report, row, calibrated["nodes"][0], calibrated


def test_policy_diagnostic_measures_without_applying(tmp_path, monkeypatch):
    node = _policy_node()
    report, row, calibrated, authoring_ir = _run_policy(
        tmp_path, monkeypatch, node, _eligible_estimate(),
        source_typography=True, typography_policy="diagnostic")
    assert report["typography_policy"] == "diagnostic"
    assert row["policy"] == "diagnostic" and not row["applied"]
    assert row["proposal"]["color"] == "FFFFFF" and row["source_color"] == "#112233"
    assert calibrated["color"] == "FFFFFF" and calibrated["bbox"] == [10, 10, 60, 20]
    assert calibrated["font"]["ea"] == "ProposalEA"
    assert report["typography_applied"] == 0


def test_policy_color_preserves_font_geometry_and_runs(tmp_path, monkeypatch):
    runs = [{"text": "标", "color": "FFFFFF"}, {"text": "题", "color": "FFFFFF"}]
    node = _policy_node(runs=runs)
    report, row, calibrated, _ = _run_policy(
        tmp_path, monkeypatch, node, _eligible_estimate(),
        source_typography=True, typography_policy="color")
    assert calibrated["color"] == "112233"
    assert calibrated["font"]["ea"] == "ProposalEA"
    assert calibrated["bbox"] == [10, 10, 60, 20]
    assert [run["color"] for run in calibrated["runs"]] == ["112233", "112233"]
    assert row["applied_fields"] == ["color"] and row["source_color_basis"] == "typography_estimate"
    assert report["typography_applied"] == 0 and report["source_color_only_applied"] == 1


def test_policy_typography_protects_mixed_colour_runs(tmp_path, monkeypatch):
    runs = [{"text": "标", "color": "FF0000"}, {"text": "题", "color": "0000FF"}]
    node = _policy_node(runs=runs)
    report, row, calibrated, _ = _run_policy(
        tmp_path, monkeypatch, node, _eligible_estimate(),
        source_typography=True, typography_policy="typography")
    assert row["reason"] == "mixed_runs_protected"
    assert calibrated["font"]["ea"] == "ProposalEA" and calibrated["color"] == "FFFFFF"
    assert calibrated["runs"] == runs
    assert report["typography_applied"] == 0


def test_policy_typography_keeps_latin_font_for_mixed_script(tmp_path, monkeypatch):
    node = _policy_node(text="AI赋能")
    report, row, calibrated, _ = _run_policy(
        tmp_path, monkeypatch, node, _eligible_estimate(),
        source_typography=True, typography_policy="typography")
    assert row["applied"] and row["font_scope"] == "ea_only_mixed_script"
    assert calibrated["font"]["ea"] == "MeasuredFont"
    assert calibrated["font"]["latin"] == "ProposalLatin"
    assert calibrated["bold"] is True and calibrated["color"] == "112233"
    assert calibrated["font"]["size_pt"] == 14.25 * .75


def test_geometry_policy_diagnostic_keeps_proposal_bbox(tmp_path, monkeypatch):
    node = _policy_node()
    report, row, calibrated, _ = _run_policy(
        tmp_path, monkeypatch, node, _eligible_estimate(),
        source_typography=True, typography_policy="typography",
        geometry_policy="diagnostic")
    assert row["applied"] and row["geometry_policy"] == "kept_proposal"
    assert calibrated["bbox"] == [10, 10, 60, 20]
    assert "geometry" not in row["applied_fields"]
    assert calibrated["font"]["ea"] == "MeasuredFont"


def test_legacy_boolean_still_maps_to_full_policy(tmp_path, monkeypatch):
    report, row, calibrated, _ = _run_policy(
        tmp_path, monkeypatch, _policy_node(), _eligible_estimate(),
        source_typography=True)
    assert report["typography_policy"] == "typography"
    assert row["applied"] and report["typography_applied"] == 1
    assert calibrated["font"]["ea"] == "MeasuredFont"
