# ElementPlan structure extension

`ElementPlan/1` and `/2` now accept native groups, tables and charts in addition
to shapes, raster modules and text decisions. The plan remains an author
proposal: materialisation does not prove fidelity and no source pixels are
invented.

## Groups

```json
{
  "groups": [
    {"id": "kpi-cards", "member_ids": ["card-a", "card-b"], "label": "KPI cards"}
  ]
}
```

- A group has at least two members.
- Members must exist, be unique and belong to one group only.
- Members must be **contiguous in `z` order** after sorting; a group that skips
  an ungrouped node is rejected because it cannot preserve the original
  occlusion order.
- Group ids cannot collide with node ids.

## Tables

Object fields:

```json
{
  "id": "table-1", "kind": "table", "bbox": [40, 160, 560, 210], "z": 3,
  "rows": [["项目", "数量"], ["A", "12"]],
  "font": {"latin": "Arial", "ea": "微软雅黑", "size_px": 14},
  "column_tracks": [{"mode": "fr", "value": 3}, {"mode": "fr", "value": 2}],
  "header_row": true, "banded_rows": true,
  "row_heights": [48, 54],
  "merges": [{"row": 0, "column": 0, "row_span": 2, "column_span": 2}],
  "cell_styles": [{"row": 0, "column": 0, "fill": "E8F1FA", "bold": true}]
}
```

- 1–2000 rows, 1–100 columns, equal row lengths.
- Cells are strings (≤2048) or finite numbers.
- `font` is required; family names resolve against the verified font catalog.
- `row_heights` are **target-canvas pixels** and must sum to the scaled bbox
  height. `merges` must not overlap and covered non-anchor cells must be empty.
- `column_tracks` length equals the column count; `mode` is `fr` or `fixed`.

## Charts

```json
{
  "id": "chart-1", "kind": "chart", "bbox": [640, 160, 520, 300], "z": 4,
  "chart_type": "bar", "title": "数量",
  "categories": ["A", "B"], "series": [{"name": "数量", "values": [1, 2]}],
  "font": {"latin": "Arial", "ea": "微软雅黑", "size_px": 12},
  "has_legend": true, "legend_position": "bottom"
}
```

- `chart_type` is one of `bar`, `line`, `area`, `pie`, `doughnut`.
- Categories 1–1000; series 1–128; every series length matches categories.
- Values are finite numbers or `null`.
- A chart is only appropriate when the data is traceable from the source. Do
  not invent values from a low-resolution picture; keep that region as a raster
  module instead.

## Scope

`materialize_elements` emits native nodes and the declared groups; it does not
classify tables/charts or infer their data from pixels. Those remain explicit
author decisions. The compiler's own schema and layout gates are the final
authority on whether the emitted node is deliverable.
