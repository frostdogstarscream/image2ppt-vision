"""Exercise real OCR and reconstruction through the local ASGI HTTP application.

Uses the installed service and real local model/font assets. No injected OCR,
model calls or handwritten SlideIR. This is local integration evidence, not a
remote deployment, actual PPTX rendering, or visual acceptance.
"""
import argparse
import json
import time
from pathlib import Path
from fastapi.testclient import TestClient
from image2ppt_vision.api import create_app
from image2ppt_vision.storage import read_json, write_json, digest


def check(source, plan_file, output, renderer_version=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    data=output/"service-data"
    headers={"Authorization":"Bearer local-integration-fixture"}
    report={"scope":"local ASGI HTTP + spawned service worker + real OCR + service reconstruction",
            "remote_deployment_verified":False,"human_acceptance":False,
            "input_sha256":digest(source),"submitted_plan_sha256":digest(plan_file)}
    write_json(output/"check.json",report)
    with TestClient(create_app(data,"local-integration-fixture",run_worker=True)) as api:
        with open(source,"rb") as handle:
            response=api.post("/v1/assets",headers=headers,files={"file":("source.png",handle)})
        response.raise_for_status();asset=response.json()
        def run(request):
            response=api.post("/v1/jobs",headers=headers,json=request)
            if response.status_code != 202:
                write_json(output/"admission-error.json",response.json())
                response.raise_for_status()
            job=response.json()["id"];deadline=time.monotonic()+420
            while True:
                response=api.get("/v1/jobs/"+job,headers=headers);response.raise_for_status()
                result=response.json()
                if result["status"] not in ("QUEUED","RUNNING"):break
                if time.monotonic()>deadline:raise TimeoutError("integration worker timeout")
                time.sleep(.3)
            if result["status"]!="SUCCEEDED":
                write_json(output/"failed-job.json",result)
                raise RuntimeError("service integration job failed: "+job)
            return result
        analysis=run({"operation":"analyze","asset_id":asset["id"],"element_inventory":True})
        report["analysis_job_id"]=analysis["id"];write_json(output/"check.json",report)
        reconstruction=run({"operation":"reconstruct_elements","asset_id":asset["id"],
            "analysis_job_id":analysis["id"],"element_plan":read_json(plan_file),
            "renderer_version":renderer_version})
        bundle=output/"downloaded";bundle.mkdir()
        for name,meta in reconstruction["artifacts"].items():
            response=api.get("/v1/jobs/"+reconstruction["id"]+"/artifacts/"+name,headers=headers)
            response.raise_for_status();target=(bundle/name).resolve()
            if not target.is_relative_to(bundle):raise ValueError("unsafe artifact")
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(response.content)
            if digest(target)!=meta["sha256"]:raise ValueError("artifact integrity mismatch")
        quality=read_json(bundle/"reconstruction/quality.json")
        report.update(execution_status="SUCCEEDED",quality_status=quality["status"],
            reconstruction_job_id=reconstruction["id"],verified_artifacts=len(reconstruction["artifacts"]),
            typography_applied=quality["typography_applied"],typography_total=quality["typography_total"],
            source_color_only_applied=quality["source_color_only_applied"],
            unresolved_issues=len(quality["issues"]),
            slideir="downloaded/reconstruction/calibrated/slideir.json")
        write_json(output/"check.json",report)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for key in ("input","plan","output"):parser.add_argument("--"+key,required=True)
    parser.add_argument("--renderer-version")
    args=parser.parse_args()
    print(json.dumps(check(args.input,args.plan,args.output,args.renderer_version),indent=2))
