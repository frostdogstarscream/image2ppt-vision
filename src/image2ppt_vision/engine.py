import os
import time
import traceback
import importlib.metadata
from pathlib import Path
import numpy as np
from PIL import Image
from . import algorithms, models, __version__
from .contracts import Request
from .plans import validate_plan
from .storage import Store, read_json, write_json, digest


def validate_bounds(request,w,h):
    if request.bbox:
        x,y,bw,bh=request.bbox
        if x+bw>w or y+bh>h:raise ValueError("bbox outside image")
    if any(x<0 or y<0 or x>=w or y>=h for x,y in request.points):
        raise ValueError("point outside image")
    if request.crop_prompt:
        x,y,bw,bh=request.bbox
        if any(not (x<=px<x+bw and y<=py<y+bh) for px,py in request.points):
            raise ValueError('point outside prompt crop')


def execute(root,id):
    store=Store(root);out=store.path("jobs",id);r=Request.model_validate(read_json(out/"request.json"))
    image=algorithms.rgb(store.path("assets",r.asset_id)/"image.png")
    h,w=image.shape[:2];validate_bounds(r,w,h)
    result={"schema":"VisionAnalysis", "version":"1.0", "image_size":[w,h],"coordinate_space":"exif_transposed_source_pixels",
            "objects":[],"stages":[],"quality_status":"REVIEW_REQUIRED","human_acceptance":False}
    # Source association is needed by reconstruction before any assets are made.
    result["inputs"]={a:read_json(store.path("assets",a)/"asset.json") for a in (r.asset_id,r.reference_id,r.mask_id) if a}
    if r.operation == "reconstruct":
        validate_plan(r.plan, source_sha256=result["inputs"][r.asset_id]["original_sha256"], image_size=[w,h])
    result['actual_config']={'request':r.model_dump(),'geometry_max_side':1536,'geometry_scale':min(1.,1536/max(h,w)),
                             'sam_max_side_automatic':1536,'sam_points_per_side':32,'sam_crop_n_layers':0,'sam_dtype':'float32','sam_cuda_allocator_fraction':.5}
    def stage(name,fn):
        start=time.monotonic()
        store.update(id,'RUNNING',stage=name)
        try:
            value=fn();result["stages"].append({"name":name,"status":"PASS","elapsed_seconds":time.monotonic()-start})
            return value
        except Exception as e:
            # Do not expose paths, OCR contents or raw model request in general logs.
            result["stages"].append({"name":name,"status":"FAIL","error_type":type(e).__name__,"elapsed_seconds":time.monotonic()-start})
            return None
        finally:
            write_json(out/"result.json",result)
    if r.operation in {"analyze", "reconstruct"}:
        result["objects"].extend(stage("geometry",lambda:algorithms.geometry(image)) or [])
        if r.ocr: result["objects"].extend(stage("ocr",lambda:models.recognize(image)) or [])
        result["relations"]=stage("relations",lambda:algorithms.relations(result["objects"]))
        stage("overlay",lambda:algorithms.overlay(image,result["objects"],out/"overlay.png"))
        if r.element_inventory:
            def inventory():
                from .element_inventory import build_inventory
                if result["inputs"][r.asset_id].get("exif_orientation",1)!=1:
                    raise ValueError("element inventory requires orientation-normalized input")
                if not any(s["name"]=="ocr" and s["status"]=="PASS" for s in result["stages"]):
                    raise ValueError("successful OCR required")
                bundle=build_inventory(store.path("assets",r.asset_id)/"original",result,out/"elements")
                return {"path":"elements/inventory.json","source_sha256":bundle["source_sha256"],
                        "candidate_count":bundle["candidate_count"],"text_count":len(bundle["texts"])}
            result["element_inventory"]=stage("element_inventory",inventory)
    if r.operation == "reconstruct_elements":
        def do_elements():
            from .element_service import reconstruct
            return reconstruct(store,r,out/"reconstruction")
        result["reconstruction"]=stage("reconstruct_elements",do_elements)
    if r.operation == "reconstruct":
        def do_reconstruct():
            from .reconstruction import reconstruct
            return reconstruct(store.path("assets", r.asset_id)/"original", result, r.plan.to_dict(), out/"reconstruction")
        result["reconstruction"] = stage("reconstruction", do_reconstruct)
    if r.operation=="segment" or (r.operation=="analyze" and r.segment):
        predictions=stage("segmentation",lambda:models.segment(image,r))
        result["masks"]=[]
        for i,(mask,score) in enumerate(predictions or []):
            p=out/f"mask-{i:04}";p.mkdir()
            yy,xx=np.where(mask)
            if not len(xx):continue
            box=(int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1))
            Image.fromarray(mask.astype("uint8")*255).save(p/"mask-full.png")
            record=algorithms.extract(image,mask,box,p);record.update(score=score,artifact_prefix=p.name)
            record['geometry_candidates']=[{'id':n['id'],'bbox_iou':algorithms.iou(box,n['bbox'])}
                                           for n in result['objects'] if n['kind']!='text' and algorithms.iou(box,n['bbox'])>.3]
            texts=[n for n in result["objects"] if n["kind"]=="text"]
            if texts:
                record["text_overlap"]=[n["id"] for n in texts if mask[int(n["bbox"][1]):int(n["bbox"][1]+n["bbox"][3]),int(n["bbox"][0]):int(n["bbox"][0]+n["bbox"][2])].any()]
            result["masks"].append(record)
    if r.operation=="extract":
        def do_extract():
            mask=np.ones((h,w),dtype=bool)
            if r.mask_id:
                with Image.open(store.path("assets",r.mask_id)/"image.png") as im:
                    if im.size!=(w,h):raise ValueError("mask dimensions differ")
                    gray=np.array(im.convert("L"))
                    if not np.isin(gray,[0,255]).all():raise ValueError("mask must be binary")
                    mask=gray>0
            return algorithms.extract(image,mask,r.bbox,out)
        result["extraction"]=stage("extract",do_extract)
    if r.operation=="compare":
        def do_compare():
            reference=algorithms.rgb(store.path("assets",r.reference_id)/"image.png")
            original_reference_size=list(reference.shape[1::-1])
            if r.resize_reference and reference.shape!=image.shape:
                import cv2
                reference=cv2.resize(reference,(w,h))
            if r.bbox:
                if reference.shape!=image.shape:raise ValueError("coordinate spaces differ")
                x,y,bw,bh=r.bbox
                a=image[y:y+bh,x:x+bw];b=reference[y:y+bh,x:x+bw]
            else:a,b=image,reference
            measured=algorithms.compare(a,b,out,r.resize_reference,r.register_translation)
            measured['original_reference_size']=original_reference_size
            measured['reference_to_source_scale']=[w/original_reference_size[0],h/original_reference_size[1]] if r.resize_reference else [1.,1.]
            measured['region']=list(r.bbox) if r.bbox else [0,0,w,h]
            return measured
        result["comparison"]=stage("compare",do_compare)
    packages={}
    for name in ("opencv-python-headless","numpy","Pillow","rapidocr","onnxruntime","torch","torchvision","fastapi","pydantic"):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    result["environment"]={"code_version":__version__,"packages":packages,"source_hashes":{p.name:digest(p) for p in Path(__file__).parent.glob("*.py")},"models":models.capabilities()}
    write_json(out/"result.json",result)
    store.seal(id)
    failed=sum(s["status"]=="FAIL" for s in result["stages"])
    status = "PARTIAL" if failed and failed<len(result["stages"]) else "FAILED" if failed else "SUCCEEDED"
    if r.operation in {"reconstruct","reconstruct_elements"} and result.get("reconstruction") is None:
        status = "FAILED"
    store.update(id,status,result="result.json")
