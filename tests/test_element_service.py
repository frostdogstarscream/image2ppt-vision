"""Formal HTTP contracts; synthetic fixtures do not establish visual fidelity."""
from copy import deepcopy
from unittest.mock import Mock
import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient
from image2ppt_vision import engine, models, element_calibration, model_provider, client
from image2ppt_vision.api import create_app
from image2ppt_vision.contracts import Request
from image2ppt_vision.storage import Store, read_json, write_json, digest
from image2ppt_vision.supervised_elements import materialize_elements

AUTH={"Authorization":"Bearer fixture"}

@pytest.fixture
def case(tmp_path,monkeypatch):
    source=tmp_path/"s.png";image=Image.new("RGB",(400,200),"white")
    ImageDraw.Draw(image).text((100,60),"02",fill="black");image.save(source)
    store=Store(tmp_path/"store");asset=store.upload(source)
    text={"id":"t1","kind":"text","text":"02","bbox":[100,60,28,18],"score":.99}
    monkeypatch.setattr(models,"recognize",lambda _: [deepcopy(text)])
    monkeypatch.setattr(element_calibration,"estimate_text",lambda *_,**kwargs: {
        "native_eligible":True,"bbox":[100,60,28,18],"font_size_px":20,"font_family":"fixture",
        "bold":False,"color":"#000000",
        "diagnostics":{"font":{"render_origin":[99,57]},"reason":"fixture"}})
    forbidden=Mock(side_effect=AssertionError("no model calls allowed"))
    monkeypatch.setattr(model_provider,"generate_plan",forbidden)
    job=store.submit(Request(operation="analyze",asset_id=asset["id"],element_inventory=True))
    engine.execute(store.root,job["id"])
    plan={"schema":"ElementPlan/1","source_sha256":digest(source),"image_size":[400,200],
          "background_color":"FFFFFF","objects":[{"id":"panel","kind":"native_shape",
              "bbox":[80,40,100,70],"geometry":"rect","fill":"EEEEEE","z":0}],
          "text_decisions":[{"id":"t1","action":"native","text":"02","bbox":[110,60,28,18],
              "font_size_px":14,"bold":True,"color":"FFFFFF","z":1}]}
    request={"operation":"reconstruct_elements","asset_id":asset["id"],
             "analysis_job_id":job["id"],"element_plan":plan,"renderer_version":"2.8.99"}
    return store,source,request,forbidden


def test_http_uses_sealed_analysis_and_returns_review_required(case,monkeypatch):
    store,source,request,forbidden=case
    monkeypatch.setattr(models,"recognize",Mock(side_effect=AssertionError("must not rerun OCR")))
    with TestClient(create_app(store.root,"fixture",run_worker=False)) as api:
        response=api.post("/v1/jobs",headers=AUTH,json=request)
        assert response.status_code==202
        job=response.json()["id"];engine.execute(store.root,job)
        status=api.get("/v1/jobs/"+job,headers=AUTH).json()
        assert status["status"]=="SUCCEEDED"
        assert "reconstruction/calibrated/slideir.json" in status["artifacts"]
        ir=api.get(f"/v1/jobs/{job}/artifacts/reconstruction/calibrated/slideir.json",headers=AUTH).json()
        text=next(n for n in ir["nodes"] if n["type"]=="text")
        assert text["color"]=="000000" and text["bold"] is False and text["bbox"][:2]==[99,57]
        quality=read_json(store.path("jobs",job)/"reconstruction/quality.json")
        assert quality["status"]=="REVIEW_REQUIRED" and quality["actual_render_verified"] is False
        assert any(i["code"]=="renderer_origin_unverified" for i in quality["issues"])
        assert read_json(store.path("jobs",job)/"reconstruction/submitted-plan.json")==request["element_plan"]
    forbidden.assert_not_called()


@pytest.mark.parametrize("mutate",[
    lambda r:r.update(analysis_job_id="f"*32),
    lambda r:r["element_plan"].update(source_sha256="a"*64),
    lambda r:r["element_plan"].update(code="untrusted"),
    lambda r:r["element_plan"]["text_decisions"][0].update(path="../../x"),
    lambda r:r["element_plan"]["text_decisions"][0].update(id="unknown"),
    lambda r:r["element_plan"]["text_decisions"][0].update(runs=[{"text":"02","color":"bad"}]),
    lambda r:r["element_plan"]["objects"][0].update(fill_gradient={"angle_deg":0,"stops":[]}),
])
def test_reject_before_queue(case,mutate):
    store,_,request,_=case;mutate(request)
    with TestClient(create_app(store.root,"fixture",run_worker=False)) as api:
        response=api.post("/v1/jobs",headers=AUTH,json=request)
        assert response.status_code in (404,422)
    with store.db() as db:assert db.execute("SELECT count(*) FROM jobs").fetchone()[0]==1


def test_tampered_analysis_rejected_at_worker_execution(case):
    store,_,request,_=case
    with TestClient(create_app(store.root,"fixture",run_worker=False)) as api:
        job=api.post("/v1/jobs",headers=AUTH,json=request).json()["id"]
    inventory=store.path("jobs",request["analysis_job_id"])/"elements/inventory.json"
    data=read_json(inventory);data["texts"][0]["text"]="tampered";write_json(inventory,data)
    engine.execute(store.root,job)
    assert store.status(job)["status"]=="FAILED"
    assert not (store.path("jobs",job)/"reconstruction/calibrated/slideir.json").exists()


def test_refused_white_text_is_reported_never_accepted(case,monkeypatch):
    store,_,request,_=case
    monkeypatch.setattr(element_calibration,"estimate_text",lambda *_,**kwargs:{
        "native_eligible":False,"diagnostics":{"reason":"uncertain_glyph"}})
    job=store.submit(Request(**request))["id"];engine.execute(store.root,job)
    quality=read_json(store.path("jobs",job)/"reconstruction/quality.json")
    assert {i["code"] for i in quality["issues"]}=={
        "source_typography_refused","low_contrast_against_source_background"}
    assert quality["typography_applied"]==0 and not quality["human_acceptance"]


def test_model_font_is_measured_but_original_proposal_retained(case):
    store,source,request,_=case
    inv=read_json(store.path("jobs",request["analysis_job_id"])/"elements/inventory.json")
    base=store.root/"proposal"
    materialize_elements(source,inv,request["element_plan"],base,model_id="fixture",response_sha256="0"*64)
    ir=read_json(base/"slideir.json");text=next(n for n in ir["nodes"] if n["type"]=="text")
    author=read_json(base/"authoring.json")
    author["model_typography"]={"t1":{"font":text["font"],"bold":True}}
    write_json(base/"authoring.json",author)
    output=store.root/"calibrated"
    element_calibration.calibrate_elements(base/"slideir.json",output,source_typography=True,calibrate_fills=False)
    calibrated=read_json(output/"authoring.json")
    assert calibrated["original_model_typography"]==author["model_typography"]
    assert calibrated["model_typography"]=={}
    assert calibrated["applied_source_typography"]["t1"]["color"]=="000000"
    report=read_json(output/"calibration.json")["diagnostics"][0]
    assert report["proposal_contrast"]["low_contrast_risk"]
    assert not report["output_contrast"]["low_contrast_risk"]
    assert report["renderer_origin_calibration"]["reason"]=="uncalibrated_renderer_version"


def test_wrong_source_or_failed_analysis_rejected(case,tmp_path):
    store,_,request,_=case
    other=tmp_path/"other.png";Image.new("RGB",(400,200),"red").save(other)
    wrong=deepcopy(request);wrong["asset_id"]=store.upload(other)["id"]
    with TestClient(create_app(store.root,"fixture",run_worker=False)) as api:
        assert api.post("/v1/jobs",headers=AUTH,json=wrong).status_code==422
        store.update(request["analysis_job_id"],"PARTIAL")
        assert api.post("/v1/jobs",headers=AUTH,json=request).status_code==422


def test_inventory_requires_successful_ocr(case,monkeypatch):
    store,_,request,_=case
    monkeypatch.setattr(models,"recognize",Mock(side_effect=RuntimeError("offline unavailable")))
    job=store.submit(Request(operation="analyze",asset_id=request["asset_id"],element_inventory=True))["id"]
    engine.execute(store.root,job)
    assert store.status(job)["status"]!="SUCCEEDED"
    assert not (store.path("jobs",job)/"elements/inventory.json").exists()


def test_fitting_search_translation_is_not_exported_as_layout(case,monkeypatch):
    store,_,request,_=case
    monkeypatch.setattr(element_calibration,"estimate_text",lambda *_,**kwargs:{
        "native_eligible":True,"bbox":[100,60,28,18],"font_size_px":20,"font_family":"fixture",
        "bold":False,"color":"#000000","diagnostics":{"font":{
            "render_origin":[113,58],"glyph_bbox":[0,4,28,22],
            "shape_overlap":.85,"dimension_error":.03}}})
    job=store.submit(Request(**request))["id"];engine.execute(store.root,job)
    ir=read_json(store.path("jobs",job)/"reconstruction/calibrated/slideir.json")
    text=next(n for n in ir["nodes"] if n["type"]=="text")
    assert text["bbox"][:2]==[100,56]
    assert text["align"]=="left"


def test_single_source_color_can_repair_visibility_without_claiming_font_fit(case,monkeypatch):
    store,_,request,_=case
    monkeypatch.setattr(element_calibration,"estimate_text",lambda *_,**kwargs:{
        "native_eligible":False,"bbox":[100,60,28,18],"color":"#0B9A8E",
        "diagnostics":{"reason":"font_or_ocr_shape_mismatch",
                       "foreground_color_count":1,"background":{"reliable":True}}})
    job=store.submit(Request(**request))["id"];engine.execute(store.root,job)
    directory=store.path("jobs",job)/"reconstruction"
    ir=read_json(directory/"calibrated/slideir.json")
    text=next(n for n in ir["nodes"] if n["type"]=="text")
    assert text["color"]=="0B9A8E"
    assert text["bbox"]==request["element_plan"]["text_decisions"][0]["bbox"]
    report=read_json(directory/"quality.json")
    assert report["typography_applied"]==0 and report["source_color_only_applied"]==1
    assert report["issues"]==[{"id":"t1","code":"source_typography_refused","reason":"font_or_ocr_shape_mismatch"}]


def test_remote_client_downloads_formal_element_bundle(case,monkeypatch,tmp_path):
    from argparse import Namespace
    store,source,request,_=case
    plan=tmp_path/"plan.json";write_json(plan,request["element_plan"])
    with TestClient(create_app(store.root,"fixture",run_worker=False)) as api:
        def transport(base,path,payload=None,content_type="application/json"):
            response=api.request("POST" if payload is not None else "GET",path,content=payload,
                headers={**AUTH,"Content-Type":content_type})
            response.raise_for_status()
            if path=="/v1/jobs":engine.execute(store.root,response.json()["id"])
            return response.content
        monkeypatch.setattr(client,"request",transport)
        monkeypatch.setattr(client.time,"sleep",lambda _:None)
        output=tmp_path/"download"
        result=client.run(Namespace(command="reconstruct_elements",url="http://local",options=None,
            output=output,plan=plan,input=source,analysis_job_id=request["analysis_job_id"],
            renderer_version="2.8.99",reference=None,mask=None,data=tmp_path))
        assert result["status"]=="SUCCEEDED"
        assert read_json(output/"reconstruction/quality.json")["status"]=="REVIEW_REQUIRED"
