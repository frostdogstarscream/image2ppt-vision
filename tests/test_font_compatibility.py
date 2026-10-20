from copy import deepcopy
import pytest
from image2ppt_vision.font_compatibility import (
    assert_compatible, fit_node, write_revision, bound_preview, verify_selection, can_select)
from image2ppt_vision.typography import _candidates
from image2ppt_vision.storage import read_json, write_json
from test_font_plan import setup


def node():
    return {"id": "a", "type": "text", "text": "Wide sample text",
            "font": {"ea": "Arial", "latin": "Arial", "size_pt": 30},
            "bbox": [10, 10, 100, 24], "source_text_bbox": [10, 10, 100, 24],
            "bold": False, "wrap": False, "color": "000000"}


def canvas():
    return {"canvas": {"source_px": [400, 300], "target_px": [400, 300]}}


def test_size_fit_reduces_overflow_without_changing_words_or_boxes():
    catalog = _candidates(None)
    if not any(c["family"] == "Arial" for c in catalog):
        pytest.skip("Arial fixture unavailable")
    original = node()
    adjusted, report = fit_node(original, canvas(), catalog)
    assert report["after_px"] < report["before_px"]
    assert report["after_measurement"]["advance_width_px"] <= 100
    changed = deepcopy(adjusted)
    changed["font"] = original["font"]
    assert changed == original


@pytest.mark.parametrize("field,value", [("text", "Changed"), ("bbox", [0, 0, 5, 5]),
                                        ("color", "FF0000"), ("runs", [{"text": "a", "bold": True}])])
def test_font_revision_cannot_change_protected_fields(field, value):
    before = {"nodes": [node()]}
    after = deepcopy(before)
    after["nodes"][0][field] = value
    with pytest.raises(ValueError, match="protected"):
        assert_compatible(before, after)


def test_missing_bold_face_is_not_silently_measured_as_regular():
    original = node()
    original["bold"] = True
    result, report = fit_node(original, canvas(), [])
    assert result == original and report["status"] == "REVIEW_REQUIRED"


def test_pixel_gain_does_not_authorize_losing_bold():
    before = node(); before["bold"] = True
    after = deepcopy(before); after["bold"] = False
    assert not can_select(before, after, True, 20., 1.)
    after["bold"] = True
    assert can_select(before, after, True, 20., 1.)


def test_wrapping_is_not_silently_removed_to_make_text_fit():
    original = node()
    original["wrap"] = True
    result, report = fit_node(original, canvas(), [])
    assert result == original and report["reason"] == "wrapped_or_multiline_not_fitted"


def test_invalid_revision_fails_before_creating_output(tmp_path):
    irfile, _ = setup(tmp_path)
    revised = read_json(irfile)
    revised["nodes"][-1]["text"] = "unapproved rewrite"
    output = tmp_path/"output"
    with pytest.raises(ValueError, match="protected"):
        write_revision(irfile, revised, output, {})
    assert not output.exists()


def test_preview_receipt_rejects_replaced_pptx(tmp_path):
    (tmp_path/"preview").mkdir()
    preview = tmp_path/"preview/page-001.png"
    preview.write_bytes(b"not needed before receipt rejection")
    pptx = tmp_path/"deck.pptx"
    pptx.write_bytes(b"replaced")
    write_json(tmp_path/"quality.json", {"pptx_sha256": "old"})
    with pytest.raises(ValueError, match="binding"):
        bound_preview(tmp_path/"slideir.json", pptx, preview)


@pytest.mark.parametrize("case", ["improvement", "outside_change", "region_regression"])
def test_final_render_gate_requires_local_gain_and_unchanged_surroundings(tmp_path, monkeypatch, case):
    """Synthetic pixels test the gate, not screenshot fidelity."""
    from PIL import Image
    import numpy as np
    import image2ppt_vision.font_compatibility as module
    from image2ppt_vision.storage import digest
    irfile, _ = setup(tmp_path)
    original = read_json(irfile)
    revised = deepcopy(original)
    revised["nodes"][-1]["font"]["size_pt"] += 1
    output = tmp_path/"selected"
    receipt = {"test_receipt": True}
    write_revision(irfile, revised, output, {
        "stage": "render_evidence_selection",
        "preview_receipts": {"baseline": receipt},
        "decisions": [{"id": revised["nodes"][-1]["id"], "selected": "candidate"}]})
    source = irfile.parent/original["source_image"]
    source_pixels = np.asarray(Image.open(source).convert("RGB").resize(
        tuple(original["canvas"]["target_px"])), dtype=float)
    old, new = source_pixels.copy(), source_pixels.copy()
    # Use a white source region and an artificial baseline error.
    old[2:8, 2:8] = 0
    new[2:8, 2:8] = 100 if case != "region_regression" else 0
    if case == "region_regression":
        old[2:8, 2:8] = 100
    if case == "outside_change":
        new[-1, -1] = 0
    monkeypatch.setattr(module, "bound_preview", lambda *a: receipt)
    monkeypatch.setattr(module, "verify_ledger", lambda *a: {"status": "fixture"})
    monkeypatch.setattr(module, "region", lambda *a: [2, 2, 8, 8])
    monkeypatch.setattr(module, "_pixels", lambda path, ir: old if path == "old" else new)
    result = verify_selection(irfile, output/"slideir.json", "unused", "old", "unused", "new")
    assert result["human_acceptance"] is False
    assert result["status"] == ("PASS_COMPUTATIONAL_GATES" if case == "improvement" else "REVIEW_REQUIRED")
