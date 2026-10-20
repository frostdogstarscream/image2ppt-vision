import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from .storage import read_json, digest
from . import __version__


def model_root():
    return Path(os.environ.get("VISION_MODELS", "models")).resolve()


def verified_models():
    root = model_root()
    manifest = read_json(root / "manifest.json")
    for name, record in manifest["files"].items():
        p = (root / name).resolve()
        if not p.is_relative_to(root) or digest(p) != record["sha256"]:
            raise ValueError("model integrity failure")
    return manifest


def capabilities():
    result = {"version": __version__, "geometry": True, "ocr": False, "segmentation": False,
              "reconstruction": True, "reconstruction_requires_plan": True, "pptx_compilation": False,
              "element_reconstruction": {"operation":"reconstruct_elements",
                  "requires":"successful analyze with element_inventory; source-bound ElementPlan/1 or /2",
                  "source_typography_calibration":True,"quality_status":"REVIEW_REQUIRED"},
              "network_inference": False, "models": None,
              "model_plan_provider": {"available": True, "automatic": False, "invocation": "explicit_cli_plan",
                  "configured": all(bool(os.environ.get(k)) for k in ("VISION_LLM_BASE_URL","VISION_LLM_MODEL","VISION_LLM_API_KEY"))}}
    from .typography import font_status
    try:result['fonts']=font_status()
    except ValueError:result['fonts']={'status':'INVALID_MANIFEST'}
    try:
        m = verified_models()
        result["models"] = m
        result["ocr"] = importlib.util.find_spec("rapidocr") is not None and all(k in m["files"] for k in ("det.onnx", "rec.onnx", "cls.onnx"))
        if importlib.util.find_spec("torch") and importlib.util.find_spec("sam2"):
            import torch
            result["segmentation"] = torch.cuda.is_available() and "sam2.1_hiera_small.pt" in m["files"]
            result["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
    except (OSError, ValueError, KeyError):
        result["model_error"] = "missing or invalid local model manifest"
    return result


@lru_cache(maxsize=1)
def ocr_engine(root):
    verified_models()
    from rapidocr import RapidOCR
    from rapidocr.utils.typings import OCRVersion
    # The model adapter never permits an implicit network download.
    from rapidocr.utils.download_file import DownloadFile
    def offline(*args, **kwargs):
        raise RuntimeError("runtime model download disabled")
    DownloadFile.run = offline
    r = Path(root)
    engine = RapidOCR(params={"Det.model_path":str(r / "det.onnx"), "Rec.model_path":str(r / "rec.onnx"),
                              "Cls.model_path":str(r / "cls.onnx"), "EngineConfig.onnxruntime.intra_op_num_threads":4,
                              "EngineConfig.onnxruntime.inter_op_num_threads":1,
                              "Det.ocr_version":OCRVersion.PPOCRV4,"Rec.ocr_version":OCRVersion.PPOCRV4,
                              "Global.log_level":"error"})
    return engine


def recognize(rgb):
    result = ocr_engine(str(model_root()))(rgb[:, :, ::-1].copy())
    objects = []
    rows = zip(result.boxes, result.txts, result.scores) if result.boxes is not None else []
    for i, (poly, text, score) in enumerate(rows):
        poly = poly.tolist() if hasattr(poly, "tolist") else poly
        xs, ys = zip(*poly)
        objects.append({"id": f"text-{i:04}", "kind": "text", "polygon": poly,
                        "bbox": [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)],
                        "text": text, "score": float(score), "score_type": "ocr_confidence", "method": "RapidOCR"})
    return objects


@lru_cache(maxsize=1)
def sam_model(root):
    import torch
    from sam2.build_sam import build_sam2
    verified_models()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.5, 0)
    return build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", str(Path(root)/"sam2.1_hiera_small.pt"), device="cuda:0")


def segment(rgb, request):
    import cv2
    import numpy as np
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    model = sam_model(str(model_root()))
    h, w = rgb.shape[:2]
    with torch.inference_mode():
        if request.mode == "automatic":
            scale = min(1.0, 1536/max(h,w))
            work = cv2.resize(rgb, (round(w*scale), round(h*scale))) if scale < 1 else rgb
            results = SAM2AutomaticMaskGenerator(model, points_per_side=32, points_per_batch=16, crop_n_layers=0).generate(work)
            return [(cv2.resize(r["segmentation"].astype("uint8"), (w,h), interpolation=cv2.INTER_NEAREST)>0,
                     {"predicted_iou":float(r["predicted_iou"]), "stability_score":float(r["stability_score"])}) for r in results]
        predictor = SAM2ImagePredictor(model)
        ox,oy=0,0
        work=rgb
        if request.crop_prompt:
            ox,oy,cw,ch=request.bbox
            work=rgb[oy:oy+ch,ox:ox+cw]
        predictor.set_image(work)
        box = None
        if request.bbox:
            x,y,bw,bh=request.bbox
            box=np.array([x-ox,y-oy,x+bw-ox,y+bh-oy])
        masks,scores,_=predictor.predict(point_coords=np.array(request.points)-[ox,oy] if request.points else None,
                                        point_labels=np.array(request.labels) if request.labels else None,
                                        box=box,multimask_output=True)
        output=[]
        for m,s in zip(masks,scores):
            if request.crop_prompt:
                full=np.zeros((h,w),dtype=bool);full[oy:oy+m.shape[0],ox:ox+m.shape[1]]=m;m=full
            output.append((m, {"predicted_iou":float(s), "score_type":"sam_predicted_iou"}))
        return output
