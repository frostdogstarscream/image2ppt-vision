from copy import deepcopy

import numpy as np
import pytest

from image2ppt_vision.module_plan import assign_nodes
from image2ppt_vision.source_graphics import compare_pixels, complete_bars, neutral_bars, refine_modules


def fixture():
    ir = {"source_sha256": "fixture", "canvas": {"source_px": [400, 200]},
          "evidence": [], "nodes": [
              {"id": "background", "type": "shape", "bbox": [0, 0, 400, 200],
               "z": 0, "fill": "FFFFFF", "geometry": "rect"}]}
    modules = [{"id": "panel", "kind": "panel", "bbox": [0, 0, 400, 200],
                "anchor_node_ids": []}]
    rgb = np.full((200, 400, 3), 255, np.uint8)
    rgb[60:65, 80:200] = (220, 222, 225)
    return ir, modules, rgb


def test_source_bar_has_measured_bounds_and_color():
    ir, modules, rgb = fixture()
    bars, rejected = neutral_bars(ir, rgb, modules)
    assert not rejected
    assert len(bars) == 1
    assert bars[0]["bbox"] == [80, 60, 120, 5]
    assert bars[0]["fill"] == "DCDEE1"


@pytest.mark.parametrize("kind", ["text", "image", "shape"])
def test_existing_foreground_blocks_new_overlay(kind):
    ir, modules, rgb = fixture()
    ir["nodes"].append({"id": "foreground", "type": kind, "bbox": [100, 60, 20, 10], "z": 1})
    bars, rejected = neutral_bars(ir, rgb, modules)
    assert not bars
    assert rejected[0]["blockers"] == ["foreground"]


def test_source_text_extent_blocks_even_when_current_text_has_moved():
    ir, modules, rgb = fixture()
    ir["nodes"].append({"id": "text", "type": "text", "bbox": [0, 0, 20, 10],
                        "source_text_bbox": [80, 60, 120, 10], "z": 1})
    assert not neutral_bars(ir, rgb, modules)[0]


@pytest.mark.parametrize("value", [(220, 195, 210), (150, 150, 150), (250, 250, 250)])
def test_reject_colored_dark_and_near_white_regions(value):
    ir, modules, rgb = fixture()
    rgb[60:65, 80:200] = value
    assert not neutral_bars(ir, rgb, modules)[0]


def test_multiple_modules_are_not_silently_disambiguated():
    ir, modules, rgb = fixture()
    modules.append({**modules[0], "id": "other"})
    assert not neutral_bars(ir, rgb, modules)[0]


def test_completion_preserves_existing_ir_and_owns_new_evidence():
    ir, modules, rgb = fixture()
    original = deepcopy(ir)
    bars, _ = neutral_bars(ir, rgb, modules)
    result = complete_bars(ir, bars)
    assert ir == original
    assert result["nodes"][:-1] == ir["nodes"]
    assert result["nodes"][-1]["evidence_ids"] == [result["evidence"][-1]["id"]]
    with pytest.raises(ValueError, match="collision"):
        complete_bars(result, bars)


def refinement_fixture():
    ir = {"source_sha256": "fixture", "canvas": {"source_px": [400, 200]},
          "nodes": [{"id": str(i), "type": "shape", "bbox": [20, 77, 300, h]}
                    for i, h in enumerate((90, 30))]}
    modules = [{"id": "panel", "kind": "panel", "bbox": [0, 83, 400, 100],
                "member_ids": [], "anchor_node_ids": []}]
    plan = {"modules": modules, "node_assignments": assign_nodes(ir, modules)}
    rgb = np.full((200, 400, 3), 255, np.uint8)
    rgb[62:78] = 235
    return ir, plan, rgb


def test_refinement_does_not_jump_again_to_neighbor_edge_for_one_pixel():
    ir, plan, rgb = refinement_fixture()
    refined = refine_modules(ir, plan, rgb)
    assert refined["modules"][0]["bbox"][1] == 78
    assert plan["modules"][0]["bbox"][1] == 83
    assert refined["geometry_changes"] == 0


def test_refinement_requires_source_edge_and_bounded_distance():
    ir, plan, rgb = refinement_fixture()
    assert refine_modules(ir, plan, np.full_like(rgb, 255))["modules"][0]["bbox"][1] == 83
    ir["nodes"][0]["bbox"][1] = 10
    plan["node_assignments"] = assign_nodes(ir, plan["modules"])
    refined = refine_modules(ir, plan, rgb)
    assert refined["modules"][0]["bbox"][1] >= 59


def test_page_extent_is_recorded_without_forced_module_membership():
    ir, modules, rgb = fixture()
    plan = {"modules": [{**modules[0], "bbox": [0, 50, 400, 150]}]}
    plan["node_assignments"] = assign_nodes(ir, plan["modules"])
    result = refine_modules(ir, plan, rgb)
    assert result["observations"][0]["status"] == "PAGE_SCOPE"
    assert result["counts"] == {"unassigned": 1}


def test_pixel_gate_requires_target_gain_and_unchanged_outside_region():
    ir, modules, source = fixture()
    before = np.full_like(source, 255)
    bars, _ = neutral_bars(ir, source, modules)
    result = compare_pixels(source, before, source, bars, [400, 200])
    assert result["status"] == "PASS_COMPUTATIONAL_GATES"
    damaged = source.copy()
    damaged[0, 0] = 0
    result = compare_pixels(source, before, damaged, bars, [400, 200])
    assert result["status"] == "REVIEW_REQUIRED"
    assert result["outside_changed_pixels"] == 1
    assert compare_pixels(source, before, before, bars, [400, 200])["status"] == "REVIEW_REQUIRED"
