# Freeform curves, icon vectorisation and `dx` — evaluation

Status: **evaluated, not implemented**. This note records what the current
evidence supports and the conditions a future implementation must meet. It does
not change any runtime behaviour.

Inputs: renderer profile candidate `candidate-2.8.99.json` / report
`report-2.8.99.json` (84 samples, 8 family/weight curves), the 14-page strict
deck (`outputs/14页可编辑deck-严格`, compiler 2.8.99), the `page05` element plan,
and the core IR geometry list at commit `4f9c119`.

## 1. Freeform / custom geometry

Current contract:

- `IR_SHAPE_GEOMETRIES` in core accepts 13 presets (`rect`, `roundRect`,
  `ellipse`, `line`, `chevron`, `pentagon`, `rightArrow`, `leftArrow`,
  `trapezoid`, `parallelogram`, `hexagon`, `diamond`, `triangle`). The engine
  probe in `doctor` exercises each one, so a dropped preset fails at probe time.
- `ElementPlan` object kinds are `native_shape`, `raster_asset`, `table`,
  `chart`. There is no freeform kind and no path point list.
- `geometry_scoring.py` explicitly leaves curves unresolved: it accepts literal
  polygon/line paths only and returns `None` for `cubicBezTo`, guide lists,
  theme paints and complex strokes.
- The 14-page strict deck contains **zero `custGeom`** elements; every preset is
  `rect` (887) or `roundRect` (243).

Conclusion: adding freeform would require four coordinated changes (IR schema,
compiler emission, engine capability probe, scoring/calibration), and the
current decks do not contain a measured case that the preset list cannot
express. Implementing it now would produce untestable output.

Conditions before implementation:

1. A real page region whose boundary is a curve that no preset approximates
   better than the current masked raster (compare ink MAE both ways).
2. A compiler probe proving the installed Artifact Tool round-trips
   `custGeom` paths at target resolution.
3. A scoring path for Bézier segments, or an explicit "not scored" report so
   fidelity cannot silently pass.

## 2. Icon vectorisation

The 14-page strict deck embeds 326 images (10.8 MB). 237 of them are at most
120 px on their longest side and 228 are square; those small images cover
962,801 px² against 9,013,071 px² for all embedded images (about 11 %). They
are icons, badges and avatars, not content modules.

Most of them are already `raster_asset` with a mask (`ellipse` / `rect`), which
is the current fidelity-preserving answer. Vectorising them would need a shape
classifier per icon and would lose photographic icons (avatars); the source
screenshots are low resolution, so a fitted vector outline is not guaranteed to
be more faithful than the masked crop.

Conclusion: do not vectorise by default. A future pilot is only justified for
flat-colour, non-photographic icons where (a) source resolution is at least 2×
the target size and (b) a per-icon ink-MAE gate shows the vector is no worse
than the masked raster. Otherwise keep masked rasters; this does not violate the
"no page/region bitmap fallback" rule because each asset stays a bounded,
evidence-linked module.

## 3. Renderer `dx`

The origin profile records `measured_dx_mean_px` per curve but keeps
`dx_source_px = 0`; `origin_correction()` only returns `dy`. The 2.8.99
candidate measures these horizontal offsets (source px):

| family | weight | latin mean | CJK mean | CJK per size (8/12/16/24/32/40) |
|---|---|---|---|---|
| Arial | regular | −1.0062 | — | — |
| Arial | bold | −0.0062 | — | — |
| 微软雅黑 | regular | −0.0062 | −0.5229 | 0 / 0 / 0 / 0 / −1.39 / −1.39 |
| 微软雅黑 | bold | −0.0062 | −0.0875 | 0 throughout |
| 宋体 | regular | −0.0062 | −1.8291 | −1.39 ×4 / −2.70 ×2 |
| 等线 | regular | −0.0062 | −1.1760 | 0 / 0 / −1.39 ×3 / −2.70 |
| 等线 | bold | −0.0062 | −0.5229 | 0 ×4 / −1.39 ×2 |
| 黑体 | regular | −0.0062 | −1.6114 | −1.39 ×5 / −2.70 |

The pattern is not a font-level bias:

- Latin offsets sit at one renderer subpixel (−0.0062 source px) for every
  family except Arial regular, which is offset by one target pixel (−1.0062).
  Bold vs regular of the same family flipping by 1 px points at the probe
  string's first glyph / hinting, not at a family-wide translation.
- CJK offsets are **step functions of size**, quantised to roughly 1.31 source
  px (one target pixel) and switching at family-specific sizes. A linear `dx`
  curve would smooth a staircase and only appear to fit; a constant per family
  leaves up to 2.11 source px residual, above the 2.0 px gate used for `dy`.
- The probe uses one string per script (`纪检智能分析`, `Ag09Bypx`). The left
  edge depends on the first glyph's side bearing, so a curve fitted to these
  strings is not evidence for arbitrary slide text.

Decision: keep `dx_source_px = 0`. `dy` remains a valid correction because it
is a baseline property (same for any glyph at a size, max curve residual
1.8586 ≤ 2.0). `dx` is not yet a stable property.

Conditions before enabling `dx`:

1. Probe at least three strings per family/script with different first glyphs
   and both scripts, and record per-size buckets.
2. Require the offset to agree across strings within 0.5 source px; model it as
   a per-size bucket (or leave it unapplied) rather than a linear curve.
3. Run the page05 A/B and a 14-page regression with `dx` on and off, and accept
   only if the per-node ink MAE improves without any page regressing beyond the
   guard.

## 4. Half-pixel correction

The sub-pixel font-size search (PR-C2) already accepts quarter-pixel candidates
and the `dy` curves are continuous in size. No separate half-pixel `dx` term is
justified by the data above; the remaining horizontal error is quantisation,
which a sub-pixel translation would chase rather than remove.
