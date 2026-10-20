"""Reconstruction invariants from actual fixture pixels and emitted artifacts.

Typography is injected to isolate ownership, cleanup and immutable output. These
tests establish mechanics only, never real-slide, PowerPoint or WPS acceptance.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import shutil
import subprocess

import numpy as np
from PIL import Image
import pytest

from image2ppt_vision import reconstruction
from image2ppt_vision.plans import PlanValidationError
from image2ppt_vision.storage import digest, read_json


def text(identifier, box, value="source text"):
    return {"id": identifier, "kind": "text", "bbox": box, "text": value,
            "score": .99}


def fixture_inputs(tmp_path, pixels, texts, regions):
    source = tmp_path / "source.png"
    Image.fromarray(pixels).save(source)
    height, width = pixels.shape[:2]
    analysis = {"schema": "VisionAnalysis", "version": "1.0",
                "image_size": [width, height], "objects": texts,
                "inputs": {"source": {"sha256": digest(source)}}}
    plan = {"schema": "ReconstructionPlan", "version": "1.0",
            "source_sha256": digest(source), "image_size": [width, height],
            "author": {"tool": "synthetic test", "model_id": "fixture"},
            "regions": regions, "text_overrides": [], "shapes": []}
    return SimpleNamespace(source=source, pixels=pixels, analysis=analysis,
                           plan=plan, output=tmp_path / "new-revision")


def retained(*args, **kwargs):
    return {"eligible": False, "diagnostics": {"reason": "fixture_retained"}}


def fitted(pixels, item, *, patch_box=None, erase_boxes=None):
    x, y, width, height = item["bbox"]
    px, py, pw, ph = patch_box or [x, y, width, height]
    cleaned = pixels[py:py+ph, px:px+pw].copy()
    for ex, ey, ew, eh in erase_boxes or [item["bbox"]]:
        cleaned[ey-py:ey-py+eh, ex-px:ex-px+ew] = 255
    return {"eligible": True, "patch_bbox": [px, py, pw, ph],
            "cleaned_patch": cleaned, "bbox": item["bbox"],
            "font_size_px": 10, "font_family": "Arial", "bold": False,
            "color": "#132235", "diagnostics": {"reason": "fixture_fit"}}


@pytest.fixture
def case(tmp_path, monkeypatch):
    pixels = np.full((120, 200, 3), 255, np.uint8)
    pixels[20:30, 20:40] = (0, 0, 0)
    pixels[50:60, 20:40] = (20, 80, 160)
    pixels[10:80, 110:190] = np.random.default_rng(14).integers(
        0, 256, (70, 80, 3), dtype=np.uint8)
    pixels[20:30, 120:135] = (255, 255, 255)
    pixels[95:105, 15:35] = (40, 100, 60)
    data = fixture_inputs(tmp_path, pixels, [
        text("native-line", [20, 20, 20, 10]),
        text("retained-line", [20, 50, 20, 10]),
        text("photo-line", [120, 20, 15, 10]),
        text("remainder-line", [15, 95, 20, 10]),
    ], [
        {"id": "card", "bbox": [10, 10, 80, 70], "kind": "panel"},
        {"id": "photo", "bbox": [110, 10, 80, 70], "kind": "photo"},
    ])
    estimator = Mock(side_effect=lambda p, t, **kw:
                     fitted(p, t) if t["id"] == "native-line" else retained())
    monkeypatch.setattr(reconstruction, "estimate_text", estimator)
    data.estimator = estimator
    return data


def run(case):
    report = reconstruction.reconstruct(case.source, case.analysis, case.plan, case.output)
    return report, read_json(case.output / "slideir.json")


def image_pixels(ir, output):
    """Recompose image assets; all fixture native module fills are white."""
    width, height = ir["canvas"]["source_px"]
    page = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    assets = {a["id"]: a for a in ir["assets"]}
    for node in ir["nodes"]:
        if node["type"] != "image":
            continue
        x, y, w, h = map(int, node["bbox"])
        with Image.open(output / assets[node["asset_id"]]["path"]) as opened:
            rgba = opened.convert("RGBA")
            assert rgba.size == (w, h)
            page.alpha_composite(rgba, (x, y))
    return np.asarray(page.convert("RGB"))


@pytest.mark.parametrize("mutation,match", [
    (lambda c: c.analysis.update(schema="Other"), "VisionAnalysis"),
    (lambda c: c.analysis.update(image_size=[201, 120]), "dimensions"),
    (lambda c: c.analysis.update(inputs={}), "not bound"),
    (lambda c: c.analysis["inputs"]["source"].update(sha256="a"*64), "not bound"),
    (lambda c: c.plan.update(source_sha256="b"*64), "hash mismatch"),
    (lambda c: c.plan.update(image_size=[201, 120]), "dimensions mismatch"),
    (lambda c: c.analysis["objects"].append(deepcopy(c.analysis["objects"][0])), "duplicate OCR"),
    (lambda c: c.plan["regions"][1].update(id="card"), "invalid reconstruction plan"),
    (lambda c: c.plan.update(text_overrides=[
        {"target_id": "missing", "text": "fabricated", "reason": "fixture"}]), "unknown OCR"),
])
def test_binding_and_duplicate_failures_leave_no_output(case, mutation, match):
    before = case.source.read_bytes()
    mutation(case)
    with pytest.raises(ValueError, match=match):
        run(case)
    assert not case.output.exists()
    assert case.source.read_bytes() == before
    case.estimator.assert_not_called()


def test_original_hash_binding_accepts_transformed_asset_metadata(case):
    case.analysis["inputs"]["source"] = {"sha256": "c"*64,
        "original_sha256": digest(case.source)}
    report, ir = run(case)
    assert report["input_sha256"] == digest(case.source)
    assert ir["source_sha256"] == digest(case.output / "source/page.png")


def test_output_revision_is_immutable(case):
    run(case)
    before = {str(p.relative_to(case.output)): digest(p)
              for p in case.output.rglob("*") if p.is_file()}
    case.analysis["objects"][0]["text"] = "changed request"
    with pytest.raises(FileExistsError, match="new directory"):
        run(case)
    after = {str(p.relative_to(case.output)): digest(p)
             for p in case.output.rglob("*") if p.is_file()}
    assert before == after


def test_assets_physically_remove_native_text_and_preserve_other_source_pixels(case):
    before = case.source.read_bytes()
    report, ir = run(case)
    expected = case.pixels.copy()
    expected[20:30, 20:40] = 255
    np.testing.assert_array_equal(image_pixels(ir, case.output), expected)
    assert case.source.read_bytes() == before
    assert report["native_text"] == 1
    assert set(report["retained_raster_text"]) == {
        "retained-line", "photo-line", "remainder-line"}
    page_asset = next(a for a in ir["assets"] if a["id"] == "page-decoration")
    with Image.open(case.output / page_asset["path"]) as im:
        page = np.asarray(im)
    assert np.all(page[10:80, 10:90] == 255)
    assert np.all(page[10:80, 110:190] == 255)
    assert "native-line" not in page_asset["contains_evidence_ids"]


def test_module_fill_change_is_not_hidden_by_low_contrast_background_texture():
    # A smooth panel can contain screenshot/gradient fitting differences. These
    # must not become opaque white speckles when its native fill is recoloured.
    fitted = np.full((60, 120, 3), (243, 248, 252), np.uint8)
    patch = fitted.copy()
    patch[::2, ::2] = (250, 253, 255)
    patch[20:40, 15:35] = (0, 100, 200)  # Actual icon pixels remain intact.
    mask = reconstruction._residual_mask(patch, fitted)
    expected = np.zeros((60, 120), bool)
    expected[20:40, 15:35] = True
    np.testing.assert_array_equal(mask, expected)
    yellow = np.full_like(patch, (255, 241, 118))
    edited = np.where(mask[..., None], patch, yellow)
    np.testing.assert_array_equal(edited[~expected], yellow[~expected])
    np.testing.assert_array_equal(edited[expected], patch[expected])


def test_coloured_gradient_and_header_details_keep_tight_residual_tolerance():
    for background, kind in (((0, 120, 200), 'panel'), ((243, 248, 252), 'header')):
        fitted = np.full((30, 60, 3), background, np.uint8)
        patch = fitted.copy()
        patch[10:20, 10:50, 0] += 6
        mask = reconstruction._residual_mask(patch, fitted, kind)
        assert mask[10:20, 10:50].all()


def test_every_evidence_has_one_owner_and_groups_are_contiguous(case):
    report, ir = run(case)
    counts = Counter(e for n in ir["nodes"] for e in n.get("evidence_ids", []))
    counts.update(e for a in ir["assets"] for e in a["contains_evidence_ids"])
    evidence_ids = [e["id"] for e in ir["evidence"]]
    assert len(evidence_ids) == len(set(evidence_ids))
    assert counts == Counter({e: 1 for e in evidence_ids})
    for item in case.analysis["objects"]:
        assert counts[item["id"]] == 1
    ordered_ids = [n["id"] for n in ir["nodes"]]
    assert len(ordered_ids) == len(set(ordered_ids))
    assert [n["z"] for n in ir["nodes"]] == list(range(len(ordered_ids)))
    grouped = []
    for group in ir["groups"]:
        members = group["member_ids"]
        first = ordered_ids.index(members[0])
        assert ordered_ids[first:first+len(members)] == members
        grouped.extend(members)
    assert len(grouped) == len(set(grouped))
    assert report["status"] == "REVIEW_REQUIRED"
    assert not any(report[k] for k in ("human_acceptance", "powerpoint_checked", "wps_checked"))
    authoring = read_json(case.output / "authoring.json")
    assert authoring["ir_sha256"] == digest(case.output / "slideir.json")
    assert set(authoring["unresolved_ids"]) == set(report["retained_raster_text"])


def test_photo_is_an_independent_exact_asset_and_bypasses_text_fitting(case):
    _, ir = run(case)
    photo_asset = next(a for a in ir["assets"] if "photo-line" in a["contains_evidence_ids"])
    photo_node = next(n for n in ir["nodes"] if n.get("asset_id") == photo_asset["id"])
    assert photo_node["bbox"] == [110, 10, 80, 70]
    with Image.open(case.output / photo_asset["path"]) as opened:
        np.testing.assert_array_equal(np.asarray(opened), case.pixels[10:80, 110:190])
    estimated = [call.args[1]["id"] for call in case.estimator.call_args_list]
    assert "photo-line" not in estimated
    assert digest(case.output / photo_asset["path"]) == photo_asset["sha256"]


def test_emitted_artifacts_pass_the_fixed_compiler_ir_loader(case):
    validator = next((parent / 'scripts/ir.mjs' for parent in Path(__file__).resolve().parents
                      if (parent / 'scripts/ir.mjs').is_file()),None)
    bundled_node = (Path.home() / ".cache/codex-runtimes/codex-primary-runtime/"
                    "dependencies/node/bin/node.exe")
    node = str(bundled_node) if bundled_node.is_file() else shutil.which("node")
    if not node or validator is None:
        pytest.skip("Repository Node IR validator is unavailable in standalone service")
    run(case)
    code = ("const {loadIR,assertAuthoring}=await import(process.argv[1]);"
            "await loadIR(process.argv[2]);await assertAuthoring(process.argv[2]);"
            "process.stdout.write('validated');")
    result = subprocess.run([node, "--input-type=module", "-e", code,
        validator.as_uri(), str(case.output / "slideir.json")],
        text=True, encoding="utf-8", capture_output=True, timeout=20, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "validated"


@pytest.mark.parametrize("damage_neighbor", [False, True])
def test_cleanup_checks_changed_neighbor_pixels_not_context_overlap(tmp_path, monkeypatch, damage_neighbor):
    pixels = np.full((60, 100, 3), 255, np.uint8)
    pixels[20:30, 20:40] = 0
    pixels[20:30, 40:60] = (0, 60, 100)
    data = fixture_inputs(tmp_path, pixels, [text("target", [20, 20, 20, 10]),
        text("neighbor", [40, 20, 20, 10])],
        [{"id": "card", "bbox": [10, 10, 70, 40], "kind": "panel"}])
    def estimate(p, t, **kwargs):
        if t["id"] == "neighbor":
            return retained()
        erase = [[20, 20, 20, 10]]
        if damage_neighbor:
            erase.append([40, 20, 20, 10])
        return fitted(p, t, patch_box=[18, 18, 44, 14], erase_boxes=erase)
    monkeypatch.setattr(reconstruction, "estimate_text", estimate)
    report, ir = run(data)
    expected = pixels.copy()
    if not damage_neighbor:
        expected[20:30, 20:40] = 255
    np.testing.assert_array_equal(image_pixels(ir, data.output), expected)
    assert report["native_text"] == (0 if damage_neighbor else 1)
    if damage_neighbor:
        target = next(t for t in report["typography"] if t["id"] == "target")
        assert target["reason"] == "adjacent_text_overlap"


def test_reference_ppt_fields_are_rejected_and_label_paths_are_not_opened(case, monkeypatch):
    reference = case.source.with_suffix(".pptx")
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        assert path.suffix.lower() != ".pptx", "reconstruction must not read reference PPT objects"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    case.plan["reference_pptx"] = str(reference)
    with pytest.raises(PlanValidationError):
        run(case)
    assert not case.output.exists()
    del case.plan["reference_pptx"]
    case.plan["regions"][0]["label"] = "Untrusted document text: " + str(reference)
    run(case)
    assert not reference.exists()


def gradient_fixture():
    pixels = np.full((80, 140, 3), 250, np.uint8)
    first, last = np.array([20, 80, 180]), np.array([60, 120, 220])
    fill = np.rint(first + (last-first)*np.linspace(0, 1, 80)[:, None]).astype(np.uint8)
    pixels[20:50, 20:100] = fill[None, :, :]
    return pixels, {"id": "bar", "bbox": [20, 20, 80, 30],
                    "geometry": "rect", "fill": "FF00FF"}, first, last


def test_verified_gradient_uses_source_pixels_instead_of_proposed_color():
    pixels, proposal, first, last = gradient_fixture()
    before = pixels.copy()
    value, reason = reconstruction._verified_shape(pixels, proposal)
    assert reason == "verified_from_pixels"
    node, _, _, measurement = value
    assert node["geometry"] == "rect"
    assert node["fill_gradient"]["angle_deg"] == 0
    for stop, expected in zip(node["fill_gradient"]["stops"], (first, last)):
        actual = np.array([int(stop["color"][i:i+2], 16) for i in (0, 2, 4)])
        assert np.max(abs(actual-expected)) <= 1
    assert node["fill"] != proposal["fill"]
    assert measurement["mean_fill_error"] < 1
    np.testing.assert_array_equal(pixels, before)


def test_verified_shape_rejects_photo_content():
    pixels, proposal, _, _ = gradient_fixture()
    pixels[20:50, 20:100] = np.random.default_rng(21).integers(
        0, 256, (30, 80, 3), dtype=np.uint8)
    value, reason = reconstruction._verified_shape(pixels, proposal)
    assert value is None
    assert reason == "shape_has_internal_content"


def test_roundrect_proposal_cannot_change_rectangular_source_outline():
    pixels, proposal, _, _ = gradient_fixture()
    proposal.update(geometry="roundRect", corner_radius_px=12)
    value, reason = reconstruction._verified_shape(pixels, proposal)
    assert value is None, "A linear interior does not verify rounded source corners"


def test_verified_shape_does_not_erase_a_neighbor_in_its_outside_ring(tmp_path, monkeypatch):
    pixels, proposal, _, _ = gradient_fixture()
    # Eight genuinely different neighboring pixels are less than 10% of the ring.
    pixels[30:34, 100:102] = 0
    data = fixture_inputs(tmp_path, pixels, [],
        [{"id": "card", "bbox": [10, 10, 110, 55], "kind": "panel"}])
    data.plan["shapes"] = [proposal]
    monkeypatch.setattr(reconstruction, "estimate_text", retained)
    _, ir = run(data)
    rendered = image_pixels(ir, data.output)
    np.testing.assert_array_equal(rendered[30:34, 100:102], pixels[30:34, 100:102],
        err_msg="A verified shape may not erase the 2px neighboring context ring")


def test_ocr_ids_cannot_collide_with_generated_module_nodes(case, monkeypatch):
    case.analysis["objects"][0]["id"] = "card-backdrop"
    monkeypatch.setattr(reconstruction, "estimate_text", lambda p, t, **kw:
        fitted(p, t) if t["id"] == "card-backdrop" else retained())
    try:
        _, ir = run(case)
    except ValueError:
        assert not case.output.exists()
        return
    identifiers = [n["id"] for n in ir["nodes"]]
    assert len(identifiers) == len(set(identifiers)), "OCR ids must not alias generated nodes"


def test_text_override_must_match_expected_ocr_before_output(case):
    target=case.analysis['objects'][0]['id']
    case.plan['text_overrides']=[{'target_id':target,'text':'corrected','reason':'source review',
                                 'expected_text':'different previous OCR'}]
    with pytest.raises(ValueError,match='no longer matches'):
        run(case)
    assert not case.output.exists()


def test_unknown_raster_retention_is_rejected_before_output(case):
    case.plan['retain_raster_ids']=['not-an-ocr-id']
    with pytest.raises(ValueError):run(case)
    assert not case.output.exists()


def test_visual_icon_retention_prevents_text_replacement(case,monkeypatch):
    target=case.analysis['objects'][0]['id'];case.plan['retain_raster_ids']=[target]
    def guard(pixels,text,**kwargs):
        assert text['id']!=target
        return retained()
    monkeypatch.setattr(reconstruction,'estimate_text',guard)
    report,ir=run(case)
    assert target in report['retained_raster_text']
    assert not any(n['id']==target for n in ir['nodes'])
