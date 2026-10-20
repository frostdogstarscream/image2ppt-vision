from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
import math
from .plans import ReconstructionPlan


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    operation: Literal["analyze", "segment", "extract", "compare", "reconstruct", "reconstruct_elements"]
    asset_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    reference_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    mask_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    bbox: tuple[int, int, int, int] | None = None
    points: list[tuple[float, float]] = Field(default_factory=list, max_length=64)
    labels: list[Literal[0, 1]] = Field(default_factory=list, max_length=64)
    mode: Literal["automatic", "prompt"] = "automatic"
    crop_prompt: bool = False
    ocr: bool = True
    segment: bool = False
    resize_reference: bool = False
    register_translation: bool = False
    timeout_seconds: int = Field(default=300, ge=1, le=1800)
    plan: ReconstructionPlan | None = None
    element_inventory: bool = False
    analysis_job_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    element_plan: dict | None = None
    renderer_version: str | None = Field(default=None, pattern=r"^\d+\.\d+\.\d+$", max_length=40)

    @model_validator(mode="after")
    def consistent(self):
        if self.element_inventory and (self.operation != "analyze" or not self.ocr):
            raise ValueError("element_inventory requires analyze with OCR")
        if self.operation == "reconstruct_elements":
            if not self.analysis_job_id or self.element_plan is None:
                raise ValueError("reconstruct_elements requires analysis_job_id and element_plan")
            if self.segment or self.bbox or self.points or self.labels or self.mode != "automatic":
                raise ValueError("element reconstruction does not accept segmentation or crop prompts")
            from .element_service import validate_payload
            validate_payload(self.element_plan)
        elif self.analysis_job_id or self.element_plan is not None or self.renderer_version is not None:
            raise ValueError("element reconstruction options only valid for reconstruct_elements")
        if self.operation == "reconstruct":
            if self.plan is None:
                raise ValueError("reconstruct requires a validated plan")
            if self.segment or self.bbox or self.points or self.labels or self.mode != "automatic":
                raise ValueError("reconstruct does not accept segmentation or crop prompts")
        elif self.plan is not None:
            raise ValueError("plan only valid for reconstruct")
        if self.bbox and (min(self.bbox) < 0 or min(self.bbox[2:]) <= 0):
            raise ValueError("bbox must be x,y,positive width,height")
        if len(self.points) != len(self.labels):
            raise ValueError("each point requires a label")
        if any(not math.isfinite(v) for p in self.points for v in p):
            raise ValueError("nonfinite point")
        if self.operation == "compare" and not self.reference_id:
            raise ValueError("compare requires reference_id")
        if self.operation == "extract" and not (self.bbox or self.mask_id):
            raise ValueError("extract requires bbox or mask_id")
        segmentation = self.operation == 'segment' or (self.operation == 'analyze' and self.segment)
        if self.crop_prompt and not (segmentation and self.mode=='prompt' and self.bbox):
            raise ValueError('crop_prompt requires prompt segmentation and bbox')
        if segmentation and self.mode == "prompt" and not (self.bbox or self.points):
            raise ValueError("prompt segmentation requires box or points")
        if segmentation and self.mode == 'automatic' and (self.bbox or self.points):
            raise ValueError('automatic mode does not accept prompts; select prompt mode')
        if self.reference_id and self.operation != 'compare':raise ValueError('reference only valid for compare')
        if self.mask_id and self.operation != 'extract':raise ValueError('mask only valid for extract')
        if (self.resize_reference or self.register_translation) and self.operation != 'compare':raise ValueError('comparison options only valid for compare')
        return self
