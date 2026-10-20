from copy import deepcopy
import json
import numpy as np
import pytest
from image2ppt_vision.module_plan import (
    prepare_module_job, validate_proposal, assign_nodes, build_module_plan,
    apply_module_job, verify_module_page)
from image2ppt_vision.storage import digest, read_json, write_json
from test_font_plan import setup


def fixture(tmp_path):
    irfile, _ = setup(tmp_path)
    ir = read_json(irfile)
    proposal = {
        "schema": "ModuleProposal/1", "source_sha256": ir["source_sha256"],
        "input_ir_sha256": digest(irfile), "image_size": ir["canvas"]["source_px"],
        "modules": [{"id": "panel", "kind": "panel", "label": "Synthetic panel",
                     "bbox": [20, 20, 300, 130], "confidence": .7,
                     "evidence": "synthetic boundary", "anchor_node_ids": ["t1"]}],
        "alignments": [], "limitations": ["Synthetic fixture; no real source accuracy claim"]}
    return irfile, ir, proposal


@pytest.mark.parametrize("tamper", ["source", "ir", "nan", "outside", "duplicate",
                                   "unknown_anchor", "whole_page", "extra_field"])
def test_invalid_proposals_rejected(tmp_path, tamper):
    f, ir, p = fixture(tmp_path)
    if tamper == "source": p["source_sha256"] = "0"*64
    if tamper == "ir": p["input_ir_sha256"] = "0"*64
    if tamper == "nan": p["modules"][0]["confidence"] = float("nan")
    if tamper == "outside": p["modules"][0]["bbox"] = [390, 0, 100, 20]
    if tamper == "duplicate": p["modules"].append(deepcopy(p["modules"][0]))
    if tamper == "unknown_anchor": p["modules"][0]["anchor_node_ids"] = ["unknown"]
    if tamper == "whole_page": p["modules"][0]["bbox"] = [0, 0, 400, 200]
    if tamper == "extra_field": p["modules"][0]["replace_text"] = "No"
    with pytest.raises(ValueError):
        validate_proposal(p, ir, digest(f))


def test_ambiguous_membership_preserves_every_node(tmp_path):
    _, ir, p = fixture(tmp_path)
    p["modules"].append({**deepcopy(p["modules"][0]), "id": "overlapping"})
    rows = assign_nodes(ir, p["modules"])
    assert len(rows) == len(ir["nodes"])
    assert all(row["status"] == "ambiguous" and row["module_id"] is None for row in rows)


def test_unassigned_nodes_are_not_deleted_or_forced_to_nearest_module(tmp_path):
    _, ir, p = fixture(tmp_path)
    p["modules"][0]["bbox"] = [350, 160, 40, 30]
    rows = assign_nodes(ir, p["modules"])
    assert {r["node_id"] for r in rows} == {n["id"] for n in ir["nodes"]}
    assert all(r["status"] == "unassigned" for r in rows)


def test_containment_audit_uses_source_box_but_reports_current_overflow(tmp_path):
    _, ir, p = fixture(tmp_path)
    n = ir["nodes"][-1]
    n["source_text_bbox"] = [100, 50, 100, 25]
    n["bbox"] = [310, 50, 100, 25]
    row = next(r for r in assign_nodes(ir, p["modules"]) if r["node_id"] == "t1")
    assert row["status"] == "assigned" and row["current_bbox_overflow_px"] == 90


def test_model_alignment_without_pixel_edges_is_not_verified(tmp_path):
    f, ir, p = fixture(tmp_path)
    p["modules"].append({**deepcopy(p["modules"][0]), "id": "other", "bbox": [20, 155, 300, 30]})
    p["alignments"] = [{"id": "a1", "module_ids": ["panel", "other"], "axis": "x",
                        "edge": "start", "tolerance_px": 2, "confidence": .9,
                        "evidence": "model asserts alignment"}]
    plan = build_module_plan(ir, p, digest(f), np.full((200, 400, 3), 255, dtype=np.uint8))
    assert plan["alignments"][0]["within_proposed_tolerance"]
    assert plan["alignments"][0]["status"] == "REVIEW_REQUIRED"
    assert not plan["alignments"][0]["applied"]


def test_preparation_sends_only_previously_approved_text_categories(tmp_path):
    f, ir, _ = fixture(tmp_path)
    result = prepare_module_job(f, tmp_path/"job", model="fixture", base_url="https://example.invalid/v1")
    content = read_json(tmp_path/"job/request.json")["messages"][1]["content"]
    evidence = json.loads(content[0]["text"])
    assert set(evidence) == {"source_sha256", "input_ir_sha256", "image_size", "fixed_texts"}
    assert set(evidence["fixed_texts"][0]) == {"id", "text", "bbox"}
    assert len(evidence["fixed_texts"]) == 1 and result["local_nodes"] == 3
    assert not result["api_called"] and not result["nontext_object_metadata_sent"]


def test_materialization_is_identity_ir_and_tamper_is_rejected(tmp_path):
    f, ir, p = fixture(tmp_path)
    job = tmp_path/"job"; output = tmp_path/"output"
    prepare_module_job(f, job, model="fixture", base_url="https://example.invalid/v1")
    write_json(job/"response.body", {"choices": [{"finish_reason": "stop",
                                                  "message": {"content": json.dumps(p)}}]})
    result = apply_module_job(job, output)
    assert digest(output/"slideir.json") == digest(f)
    assert result["geometry_changes"] == result["font_changes"] == 0
    assert verify_module_page(output)["node_count"] == 3
    plan = read_json(output/"module-plan.json")
    plan["modules"][0]["member_ids"] = []
    write_json(output/"module-plan.json", plan)
    with pytest.raises(ValueError, match="changed"):
        verify_module_page(output)


def test_changed_request_never_materializes(tmp_path):
    f, _, _ = fixture(tmp_path)
    prepare_module_job(f, tmp_path/"job", model="fixture", base_url="https://example.invalid/v1")
    write_json(tmp_path/"job/request.json", {})
    with pytest.raises(ValueError, match="request binding"):
        apply_module_job(tmp_path/"job", tmp_path/"output")
    assert not (tmp_path/"output").exists()


def test_explicit_edges_are_converted_only_for_declared_v2(tmp_path):
    f, ir, p = fixture(tmp_path)
    p["schema"] = "ModuleProposal/2"
    p["modules"][0].pop("bbox")
    p["modules"][0]["bounds_px"] = {"left": 20, "top": 20, "right": 320, "bottom": 150}
    result = validate_proposal(p, ir, digest(f))
    assert result["modules"][0]["bbox"] == [20, 20, 300, 130]
    assert "bounds_px" in p["modules"][0]  # Original model response is never rewritten.
    p["schema"] = "ModuleProposal/1"
    with pytest.raises(ValueError, match="fields"):
        validate_proposal(p, ir, digest(f))


@pytest.mark.parametrize("value", [float("nan"), True, -1, 500])
def test_explicit_edges_still_reject_invalid_coordinates(tmp_path, value):
    f, ir, p = fixture(tmp_path)
    p["schema"] = "ModuleProposal/2"
    p["modules"][0].pop("bbox")
    p["modules"][0]["bounds_px"] = {"left": 20, "top": 20, "right": value, "bottom": 150}
    with pytest.raises(ValueError):
        validate_proposal(p, ir, digest(f))


def test_observed_boundary_alignment_is_separate_from_model_claims(tmp_path):
    f, ir, p = fixture(tmp_path)
    p["modules"][0]["bbox"] = [20, 20, 150, 130]
    p["modules"].append({**deepcopy(p["modules"][0]), "id": "other", "bbox": [200, 20, 150, 130]})
    pixels = np.full((200, 400, 3), 255, dtype=np.uint8)
    pixels[20:22, 20:170] = 0
    pixels[20:22, 200:350] = 0
    plan = build_module_plan(ir, p, digest(f), pixels)
    assert plan["alignments"] == []
    assert any(r["axis"] == "y" and r["edge"] == "start" for r in plan["derived_alignments"])
    assert all(r["origin"] == "deterministic_source_edge_observation" and not r["applied"]
               for r in plan["derived_alignments"])
