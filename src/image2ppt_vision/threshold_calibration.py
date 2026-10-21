"""Sweep typography acceptance thresholds over labelled samples.

This module never changes defaults. It reports the operating points measured on
a manifest so a human can review the confusion matrix and decide whether a
threshold revision is justified. Synthetic labels are useful for regression
tests but are not a substitute for a labelled real-slide set.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .typography import ESTIMATE_THRESHOLDS, estimate_text

SCHEMA = "TypographyThresholdSamples/1"
REPORT_SCHEMA = "TypographyThresholdCalibration/1"
HUMAN_LABEL_SOURCES = ("human", "human_labelled", "human_labeled")
MIN_RECOMMENDATION_SAMPLES = 20


def load_samples(path):
    """Read a sample manifest and resolve image paths relative to it."""
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError("sample manifest requires schema " + SCHEMA)
    samples = document.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("sample manifest requires a non-empty samples list")
    resolved = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError("sample %d must be an object" % index)
        image = sample.get("image")
        bbox = sample.get("bbox")
        if (not isinstance(image, str) or not image or
                not isinstance(sample.get("text"), str) or not sample["text"].strip()):
            raise ValueError("sample %d requires image and text" % index)
        if (not isinstance(bbox, list) or len(bbox) != 4 or
                not all(isinstance(value, (int, float)) and np.isfinite(value) for value in bbox)):
            raise ValueError("sample %d requires a finite bbox" % index)
        if type(sample.get("expected_native")) is not bool:
            raise ValueError("sample %d requires boolean expected_native" % index)
        image_path = (path.parent / image).resolve()
        if not image_path.is_file():
            raise ValueError("sample %d image is missing" % index)
        resolved.append({"id": sample.get("id", "sample-%03d" % index),
                         "image": image_path, "text": sample["text"], "bbox": list(bbox),
                         "expected_native": sample["expected_native"],
                         "expected_color": sample.get("expected_color")})
    return {"schema": SCHEMA, "label_source": document.get("label_source", "unspecified"),
            "notes": document.get("notes"), "samples": resolved}


def _metrics(rows):
    tp = sum(row["predicted"] and row["expected"] for row in rows)
    fp = sum(row["predicted"] and not row["expected"] for row in rows)
    fn = sum(not row["predicted"] and row["expected"] for row in rows)
    tn = sum(not row["predicted"] and not row["expected"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else 0.0)
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "true_negative": tn, "precision": precision, "recall": recall,
            "f1": f1, "samples": len(rows)}


def evaluate(samples, thresholds=None, font_candidates=None):
    """Apply one threshold set and return per-sample rows and aggregate metrics."""
    rows = []
    for sample in samples["samples"]:
        with Image.open(sample["image"]) as image:
            pixels = np.asarray(image.convert("RGB"))
        result = estimate_text(pixels, {"text": sample["text"], "bbox": sample["bbox"]},
                               font_candidates, thresholds=thresholds)
        predicted = bool(result.get("native_eligible"))
        row = {"id": sample["id"], "expected": sample["expected_native"],
               "predicted": predicted,
               "reason": (result.get("diagnostics") or {}).get("reason")}
        expected_color = sample.get("expected_color")
        if expected_color and result.get("color"):
            expected = [int(expected_color.lstrip("#")[index:index + 2], 16) for index in (0, 2, 4)]
            actual = [int(result["color"].lstrip("#")[index:index + 2], 16) for index in (0, 2, 4)]
            row["color_max_channel_error"] = int(np.max(np.abs(np.array(expected) - np.array(actual))))
        rows.append(row)
    return {"metrics": _metrics(rows), "rows": rows}


def sweep(samples, *, dice_values=None, dimension_values=None, font_candidates=None):
    """Evaluate a grid; never writes or adopts a threshold."""
    dice_values = dice_values or [round(.44 + .02 * index, 2) for index in range(12)]
    dimension_values = dimension_values or [round(.10 + .02 * index, 2) for index in range(11)]
    current = evaluate(samples, ESTIMATE_THRESHOLDS, font_candidates)
    grid = []
    for dice in dice_values:
        for dimension in dimension_values:
            thresholds = {**ESTIMATE_THRESHOLDS, "min_dice": dice,
                          "max_dimension_error": dimension}
            grid.append({"thresholds": thresholds,
                         **evaluate(samples, thresholds, font_candidates)["metrics"]})
    best = sorted(grid, key=lambda item: (-item["f1"], -(item["precision"] or 0),
                                          -item["thresholds"]["min_dice"],
                                          item["thresholds"]["max_dimension_error"]))[0]
    label_source = samples.get("label_source")
    recommend = (label_source in HUMAN_LABEL_SOURCES and
                 len(samples["samples"]) >= MIN_RECOMMENDATION_SAMPLES)
    return {"schema": REPORT_SCHEMA,
            "label_source": label_source,
            "sample_count": len(samples["samples"]),
            "current_defaults": dict(ESTIMATE_THRESHOLDS),
            "current": current,
            "grid": grid,
            "best_observed": best,
            "recommended": best if recommend else None,
            "recommendation_refused_reason": None if recommend else
                ("labels are not human-labelled; synthetic samples cannot justify a default change"
                 if label_source not in HUMAN_LABEL_SOURCES else
                 "fewer than %d labelled samples" % MIN_RECOMMENDATION_SAMPLES),
            "human_acceptance": False,
            "note": "This report is advisory. Review it before editing defaults."}
