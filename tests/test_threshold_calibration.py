"""The sweep is advisory; these tests use synthetic labels only."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from image2ppt_vision.threshold_calibration import evaluate, load_samples, sweep
from image2ppt_vision.typography import ESTIMATE_THRESHOLDS


def _font_path():
    for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file():
            return path
    pytest.skip("No test font installed")


def _samples(tmp_path, count=1, label_source="synthetic"):
    path = _font_path()
    font = ImageFont.truetype(path, 30)
    text = "Threshold sample"
    width = int(font.getlength(text)) + 90
    pixels = np.full((80, width, 3), 245, np.uint8)
    image = Image.fromarray(pixels)
    ImageDraw.Draw(image).text((34, 18), text, font=font, fill=(20, 40, 80))
    image.save(tmp_path / "page.png")
    bounds = font.getbbox(text)
    bbox = [27, 9, bounds[2] - bounds[0] + 14, bounds[3] - bounds[1] + 18]
    samples = [{"id": "positive-%03d" % index, "image": "page.png", "text": text,
                "bbox": bbox, "expected_native": True}
               for index in range(count)]
    manifest = tmp_path / "samples.json"
    manifest.write_text(json.dumps({"schema": "TypographyThresholdSamples/1",
                                    "label_source": label_source,
                                    "samples": samples}), encoding="utf-8")
    return manifest


def test_sweep_reports_current_point_and_refuses_synthetic_recommendation(tmp_path):
    manifest = _samples(tmp_path)
    samples = load_samples(manifest)
    report = sweep(samples, dice_values=[.48, .52], dimension_values=[.20, .24])
    assert report["schema"] == "TypographyThresholdCalibration/1"
    assert report["sample_count"] == 1
    assert report["current"]["metrics"]["samples"] == 1
    assert len(report["grid"]) == 4
    assert report["recommended"] is None
    assert "not human-labelled" in report["recommendation_refused_reason"]
    assert report["current_defaults"] == ESTIMATE_THRESHOLDS


def test_human_labelled_set_can_produce_an_advisory_point(tmp_path):
    manifest = _samples(tmp_path, count=20, label_source="human")
    report = sweep(load_samples(manifest), dice_values=[.48, .52],
                   dimension_values=[.20, .24])
    assert report["recommended"] is not None
    assert report["recommended"]["thresholds"]["min_dice"] in (.48, .52)
    assert report["human_acceptance"] is False


def test_explicit_thresholds_are_honoured_without_mutating_defaults(tmp_path):
    manifest = _samples(tmp_path)
    samples = load_samples(manifest)
    before = dict(ESTIMATE_THRESHOLDS)
    result = evaluate(samples, {**ESTIMATE_THRESHOLDS, "min_dice": .49})
    assert result["metrics"]["samples"] == 1
    assert ESTIMATE_THRESHOLDS == before


def test_manifest_rejects_unlabelled_or_missing_fields(tmp_path):
    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps({"schema": "TypographyThresholdSamples/1",
                                    "samples": [{"image": "x.png", "text": "x",
                                                 "bbox": [0, 0, 10, 10]}]}),
                        encoding="utf-8")
    with pytest.raises(ValueError):
        load_samples(manifest)
