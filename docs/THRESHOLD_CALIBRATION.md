# Typography Threshold Calibration

The acceptance thresholds in `typography.estimate_text` and the strict gate in
`element_calibration` are heuristics. They were not fitted on a labelled set of
Chinese small text, anti-aliased screenshots and mixed-script lines, so a higher
number is not automatically better.

This tooling makes the current values visible and lets a reviewer measure an
alternative operating point. It never edits defaults.

## Current defaults

`typography.ESTIMATE_THRESHOLDS`:

| key | value | meaning |
|---|---|---|
| `min_dice` | 0.50 | minimum glyph-shape overlap |
| `max_dimension_error` | 0.25 | maximum relative width+height error |
| `max_unmatched_ink` | 0.22 | maximum unmatched ink fraction |
| `multicolor_min_dice` | 0.72 | stricter overlap for multi-colour glyphs |
| `multicolor_max_dimension_error` | 0.10 | stricter size error for multi-colour glyphs |

`element_calibration.TYPOGRAPHY_THRESHOLDS` (the strict source-font gate):

| key | value |
|---|---|
| `shape_overlap` | 0.58 |
| `dimension_error` | 0.12 |

## Sample manifest

`TypographyThresholdSamples/1`:

```json
{
  "schema": "TypographyThresholdSamples/1",
  "label_source": "human",
  "notes": "who labelled what, which pages and sizes",
  "samples": [
    {
      "id": "page05-title-01",
      "image": "relative/path/page.png",
      "text": "示例标题",
      "bbox": [120, 80, 300, 36],
      "expected_native": true,
      "expected_color": "#112233"
    }
  ]
}
```

`expected_native` is the human decision: may this measurement replace the
proposal? It is not "is there text in the crop". A refusal is a valid label.

## Run

```bash
python tools/calibrate_thresholds.py --samples samples.json --output report.json
```

The report contains:

- the current operating point and its confusion matrix;
- a grid of `min_dice` × `max_dimension_error` operating points;
- the best observed point by F1 (ties prefer higher precision, then a stricter
  `min_dice`, then a looser dimension bound);
- `recommended: null` unless the manifest is human-labelled and has at least
  20 samples.

## Decision rule

1. Synthetic labels are regression evidence only; they cannot justify changing a
   default. `label_source` must be `human` / `human_labelled`.
2. Do not adopt a point that improves F1 only by accepting clearly wrong glyphs.
   Inspect the false positives.
3. Record the sample count, label source and confusion matrix in the commit that
   changes a default.
4. No default is changed by this milestone. The tool is advisory.

## Known gap

The current grid covers the two global acceptance thresholds. Per-size,
per-script or per-profile thresholds should only be added after the labelled set
shows a real difference between those strata.
