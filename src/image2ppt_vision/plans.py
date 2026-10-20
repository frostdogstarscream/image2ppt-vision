"""Bounded, declarative model output; never executable instructions or assets.

The plan is an authoring proposal, not SlideIR and not a quality acceptance.
Source and OCR binding must be checked again when materializing a plan.
"""
import json
import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, ValidationError, field_validator, model_serializer, model_validator


MAX_PLAN_BYTES = 2_000_000
MAX_PIXELS = 40_000_000
Identifier = Annotated[StrictStr, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")]
Coordinate = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0)]
BBox = Annotated[list[Coordinate], Field(min_length=4, max_length=4)]
Color = Annotated[StrictStr, Field(pattern=r"^[A-Fa-f0-9]{6}$")]
Geometry = Literal[
    "rect", "roundRect", "ellipse", "line", "chevron", "pentagon", "rightArrow", "leftArrow",
    "trapezoid", "parallelogram", "hexagon", "diamond", "triangle",
]


class PlanValidationError(ValueError):
    """Validation failure whose message does not include source/model text."""


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class Author(_Record):
    tool: Annotated[StrictStr, Field(min_length=1, max_length=160)]
    model_id: Annotated[StrictStr, Field(min_length=1, max_length=256)]


class Region(_Record):
    id: Identifier
    bbox: BBox
    kind: Literal["panel", "photo", "diagram", "header", "footer"]
    label: Annotated[StrictStr, Field(max_length=500)] | None = None


class TextOverride(_Record):
    target_id: Identifier
    text: Annotated[StrictStr, Field(min_length=1, max_length=4000)]
    reason: Annotated[StrictStr, Field(min_length=1, max_length=1000)]
    expected_text: Annotated[StrictStr, Field(min_length=1, max_length=4000)] | None = None


class LineStyle(_Record):
    color: Color
    width_px: Annotated[float, Field(strict=True, allow_inf_nan=False, gt=0, le=100)]


class Shape(_Record):
    id: Identifier
    bbox: BBox
    geometry: Geometry
    fill: Annotated[StrictStr, Field(pattern=r"^(?:[A-Fa-f0-9]{6}|none)$")]
    line: LineStyle | None = None
    corner_radius_px: Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0)] | None = None
    # A model may propose geometry but cannot waive pixel verification.
    native_only_if_verified: Literal[True] = True

    @field_validator("native_only_if_verified", mode="before")
    @classmethod
    def verification_cannot_be_waived(cls, value):
        if value is not True:
            raise ValueError("native shape verification is required")
        return value

    @model_validator(mode="after")
    def geometry_style(self):
        if self.corner_radius_px is not None:
            if self.geometry != "roundRect" or self.corner_radius_px > min(self.bbox[2:]) / 2:
                raise ValueError("corner radius requires a bounded roundRect")
        return self


class TableMerge(_Record):
    row: Annotated[StrictInt, Field(ge=0)]
    column: Annotated[StrictInt, Field(ge=0)]
    row_span: Annotated[StrictInt, Field(ge=1)] = 1
    column_span: Annotated[StrictInt, Field(ge=1)] = 1


class TableProposal(_Record):
    id: Annotated[StrictStr, Field(pattern=r'^[A-Za-z0-9_-]{1,40}$')]
    region_id: Identifier
    bbox: BBox
    rows: Annotated[list[list[Annotated[StrictStr, Field(max_length=2048)]]], Field(min_length=1,max_length=200)]
    column_widths: Annotated[list[Coordinate], Field(min_length=1,max_length=100)]
    row_heights: Annotated[list[Coordinate], Field(min_length=1,max_length=200)]
    ocr_ids: Annotated[list[Identifier], Field(max_length=2000)]
    merges: Annotated[list[TableMerge], Field(max_length=2000)] = Field(default_factory=list)

    @model_validator(mode='after')
    def table_integrity(self):
        columns=len(self.rows[0])
        if columns<1 or columns>100 or len(self.rows)*columns>2000 or any(len(row)!=columns for row in self.rows):
            raise ValueError('table must be a bounded rectangular matrix')
        if len(self.column_widths)!=columns or len(self.row_heights)!=len(self.rows):raise ValueError('table tracks mismatch')
        if any(not isinstance(v,(int,float)) or v<=0 for v in self.column_widths+self.row_heights):raise ValueError('invalid table tracks')
        if abs(sum(self.column_widths)-self.bbox[2])>.01 or abs(sum(self.row_heights)-self.bbox[3])>.01:raise ValueError('table tracks must sum to bbox')
        if len(self.ocr_ids)!=len(set(self.ocr_ids)):raise ValueError('duplicate table OCR id')
        covered=set()
        for merge in self.merges:
            if merge.row+merge.row_span>len(self.rows) or merge.column+merge.column_span>columns or merge.row_span*merge.column_span<2:
                raise ValueError('invalid table merge')
            for row in range(merge.row,merge.row+merge.row_span):
                for col in range(merge.column,merge.column+merge.column_span):
                    if (row,col) in covered:raise ValueError('overlapping table merges')
                    covered.add((row,col))
                    if (row,col)!=(merge.row,merge.column) and self.rows[row][col]!='':raise ValueError('merged covered cells must be empty')
        return self


class ReconstructionPlan(_Record):
    schema_name: Literal["ReconstructionPlan"] = Field(default="ReconstructionPlan", alias="schema")
    version: Literal["1.0", "1.1"] = "1.0"
    source_sha256: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    image_size: Annotated[list[StrictInt], Field(min_length=2, max_length=2)]
    author: Author
    regions: Annotated[list[Region], Field(min_length=1, max_length=256)]
    text_overrides: Annotated[list[TextOverride], Field(max_length=2000)] = Field(default_factory=list)
    retain_raster_ids: Annotated[list[Identifier], Field(max_length=2000)] = Field(default_factory=list)
    shapes: Annotated[list[Shape], Field(max_length=1000)] = Field(default_factory=list)
    tables: Annotated[list[TableProposal], Field(max_length=100)] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def serialize_schema_alias(self, handler):
        # Store/Request use model_dump without by_alias; nested plans must still
        # round-trip as the public schema instead of exposing schema_name.
        value = handler(self)
        if "schema_name" in value:
            value["schema"] = value.pop("schema_name")
        if self.version=='1.0' and not self.tables:value.pop('tables',None)
        return value

    @model_validator(mode="after")
    def spatial_integrity(self):
        width, height = self.image_size
        if min(width, height) <= 0 or width * height > MAX_PIXELS:
            raise ValueError("invalid image dimensions")
        if self.version=='1.0' and self.tables:raise ValueError('tables require plan version 1.1')
        all_ids = [item.id for item in (*self.regions, *self.shapes, *self.tables)]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("duplicate region or shape id")
        target_ids = [item.target_id for item in self.text_overrides]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("duplicate text override")
        if len(self.retain_raster_ids)!=len(set(self.retain_raster_ids)):
            raise ValueError("duplicate retained raster id")
        if set(target_ids)&set(self.retain_raster_ids):
            raise ValueError("retained raster item cannot also be text override")
        for item in (*self.regions, *self.shapes, *self.tables):
            x, y, w, h = item.bbox
            if min(w, h) <= 0 or x + w > width or y + h > height:
                raise ValueError("bbox outside image or empty")
        for i, a in enumerate(self.regions):
            ax, ay, aw, ah = a.bbox
            for b in self.regions[i + 1:]:
                bx, by, bw, bh = b.bbox
                if min(ax + aw, bx + bw) > max(ax, bx) and min(ay + ah, by + bh) > max(ay, by):
                    raise ValueError("regions overlap")
        table_regions=[];table_text=[]
        for table in self.tables:
            region=next((r for r in self.regions if r.id==table.region_id),None)
            if region is None or region.bbox!=table.bbox:raise ValueError('table requires an exact dedicated region')
            table_regions.append(region.id);table_text.extend(table.ocr_ids)
            if any(min(s.bbox[0]+s.bbox[2],table.bbox[0]+table.bbox[2])>max(s.bbox[0],table.bbox[0]) and
                   min(s.bbox[1]+s.bbox[3],table.bbox[1]+table.bbox[3])>max(s.bbox[1],table.bbox[1]) for s in self.shapes):
                raise ValueError('table overlaps shape proposal')
        if len(table_regions)!=len(set(table_regions)) or len(table_text)!=len(set(table_text)):raise ValueError('duplicate table ownership')
        if set(table_text)&set(self.retain_raster_ids):raise ValueError('table text cannot also be retained raster')
        return self

    def to_dict(self):
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)


def validate_plan(data, *, source_sha256=None, image_size=None, ocr_ids=None):
    """Validate structure and, when supplied, exact source/OCR associations.

    Always pass all three association arguments at materialization time. Merely
    loading a plan without OCR IDs does not approve its proposed text changes.
    """
    try:
        # Revalidate model instances as well; nested mutable lists may have changed.
        plan = ReconstructionPlan.model_validate(data.to_dict() if isinstance(data, ReconstructionPlan) else data)
    except (ValidationError, TypeError, ValueError) as exc:
        if isinstance(exc, ValidationError):
            details = "; ".join(
                ".".join(str(part) for part in issue["loc"]) + " (" + issue["type"] + ")"
                for issue in exc.errors(include_input=False, include_context=False, include_url=False)[:8]
            )
            raise PlanValidationError("invalid reconstruction plan: " + details) from None
        raise PlanValidationError("invalid reconstruction plan") from None
    if source_sha256 is not None:
        if not isinstance(source_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", source_sha256):
            raise PlanValidationError("invalid expected source hash")
        if plan.source_sha256 != source_sha256:
            raise PlanValidationError("plan source hash mismatch")
    if image_size is not None and list(image_size) != plan.image_size:
        raise PlanValidationError("plan image dimensions mismatch")
    if ocr_ids is not None:
        allowed = set(ocr_ids)
        if any(override.target_id not in allowed for override in plan.text_overrides):
            raise PlanValidationError("text override refers to unknown OCR id")
        if any(identifier not in allowed for identifier in plan.retain_raster_ids):
            raise PlanValidationError("retained raster item refers to unknown OCR id")
        if any(identifier not in allowed for table in plan.tables for identifier in table.ocr_ids):
            raise PlanValidationError('table refers to unknown OCR id')
    return plan


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PlanValidationError("duplicate JSON key")
        result[key] = value
    return result


def parse_plan_json(content, **binding):
    """Accept one JSON object only; Markdown/code fences are deliberately invalid."""
    if not isinstance(content, (str, bytes)):
        raise PlanValidationError("plan must be JSON text")
    if len(content.encode("utf-8") if isinstance(content, str) else content) > MAX_PLAN_BYTES:
        raise PlanValidationError("plan exceeds size limit")
    try:
        data = json.loads(content, object_pairs_hook=_unique_keys)
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise PlanValidationError("invalid plan JSON") from None
    return validate_plan(data, **binding)


def load_plan(path, **binding):
    """Read a local JSON plan, bounded before parsing. No paths inside it are read."""
    with Path(path).open("rb") as handle:
        content = handle.read(MAX_PLAN_BYTES + 1)
    return parse_plan_json(content, **binding)
