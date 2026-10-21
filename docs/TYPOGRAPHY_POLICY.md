# Source Typography Policy

`reconstruct_elements` no longer has one all-or-nothing typography switch.
`source_typography` remains as a compatibility alias, but new callers should
send `typography_policy` and optionally `geometry_policy`.

## Policies

| `typography_policy` | Applies | Never applies |
|---|---|---|
| `diagnostic` | nothing; records proposal, measurement and contrast | font, size, weight, colour, bbox |
| `color` | colour only, from typography or glyph evidence | font, size, weight, bbox, runs |
| `typography` | font, size, weight, colour; bbox only when `geometry_policy=measured` | mixed-colour runs |

The default for HTTP is `typography`; the legacy boolean `source_typography=True`
maps to the same policy. `source_typography=False` with no policy keeps the
pre-policy behaviour (fills only).

## Guards

- **Authored runs are preserved.** A node with two or more distinct run colours
  refuses typography and colour replacement (`mixed_runs_protected`). A node with
  runs is never stripped of them.
- **Mixed scripts.** A Latin share of at most 25% treats the fitted family as
  evidence for the whole line. A Latin share of at least 75% applies it to the
  Latin side and keeps the proposal's East Asian family. A balanced line keeps
  the proposal's Latin family and applies the measured family to `ea`.
- **Geometry is separate.** `geometry_policy=diagnostic` keeps the proposal bbox
  while still applying measured font/size/colour. `measured` applies the ink-edge
  origin and measured box.
- **Colour floor.** Colour-only repair requires a mean per-channel drift above
  12/255, so raster noise does not rewrite a matching proposal.
- **Refusals are explicit.** `calibration.json` records `policy`, `applied_fields`,
  `proposal`, `source_color`, `reason`, `guards`, `origin_policy` and the
  renderer-origin correction per node.

## Reporting

`quality.json` includes `typography_policy`, `geometry_policy`,
`typography_applied`, `source_color_only_applied` and `typography_total`.
`calibration.json` keeps the model's original typography for every node and only
removes entries for nodes where the guarded typography path actually applied.

## Risk boundary

The guarded policy does not prove that the measured font/size lands correctly in
PowerPoint. `renderer_origin_calibration.applied=false` still means the origin
correction was not verified for that renderer, and `human_acceptance` remains
false. The rendered closed loop (PR-D) is the next verification layer.

## Rollback

- Per request: `typography_policy=diagnostic` or `color`.
- Process-wide: `VISION_GLYPH_INK_EVIDENCE=0` keeps colour measurement on the
  legacy median path; it does not re-enable the pre-policy typography behaviour.
- Code rollback: revert the policy commit; the legacy boolean path is unchanged.
