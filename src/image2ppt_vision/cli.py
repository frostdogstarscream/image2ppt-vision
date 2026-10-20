import argparse
import json
import os
import shutil
from pathlib import Path
from .storage import Store, read_json
from .contracts import Request
from .plans import load_plan


def _new_output(path):
    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError("output must be new; preserve previous revisions")
    return output


def _summary(report, output):
    return {"status": "REVIEW_REQUIRED", "local_output": str(output),
            **{k: report[k] for k in ("input_sha256", "ocr_lines", "native_text", "semantic_regions",
                                      "native_module_backdrops", "independent_images", "groups") if k in report},
            "human_acceptance": False}


def main(argv=None):
    p=argparse.ArgumentParser(prog="image2ppt-vision")
    p.add_argument("--data",default=os.environ.get("VISION_DATA","vision-data"))
    p.add_argument("--url",help="HTTP service URL, normally forwarded through SSH")
    sub=p.add_subparsers(dest="command",required=True)
    calibration=sub.add_parser('calibrate-ocr',help='explicit two-pass API text calibration on source crops')
    calibration.add_argument('--input',required=True);calibration.add_argument('--analysis',required=True)
    calibration.add_argument('--config',required=True);calibration.add_argument('--output',required=True)
    calibration.add_argument('--prepare-only',action='store_true')
    sub.add_parser("doctor")
    serve=sub.add_parser("serve");serve.add_argument("--host",default="127.0.0.1");serve.add_argument("--port",type=int,default=18761)
    bench=sub.add_parser("benchmark");bench.add_argument("--suite",required=True);bench.add_argument("--output",required=True)
    evaluation=sub.add_parser('evaluate-api',help='frozen one-request-per-page API generation, without a reference PPTX')
    evaluation.add_argument('--config',required=True);evaluation.add_argument('--output',required=True)
    evaluation.add_argument('--compiler-root',required=True)
    evaluation.add_argument('--round',type=int,default=1)
    evaluation.add_argument('--resume',action='store_true');evaluation.add_argument('--prepare-only',action='store_true')
    reference=sub.add_parser('prepare-api-reference',help='independent evaluation-only reference mapping and rendering')
    reference.add_argument('--reference-config',required=True);reference.add_argument('--generation-config',required=True)
    reference.add_argument('--compiler-root',required=True);reference.add_argument('--output',required=True)
    score=sub.add_parser('score-api',help='score a sealed generation; never sends reference contents to a model')
    score.add_argument('--run',required=True);score.add_argument('--reference-config',required=True)
    score.add_argument('--output',required=True);score.add_argument('--compiler-root',required=True)
    comparison=sub.add_parser('compare-api-rounds',help='compare complete reports, without cherry-picking pages')
    comparison.add_argument('--first',required=True);comparison.add_argument('--second',required=True);comparison.add_argument('--output',required=True)
    convert_parser=sub.add_parser("convert",help="reconstruct PNG pages from saved evidence and compile one deck")
    for field in ("input-dir","analysis-dir","plan-dir","compiler-root","output"):
        convert_parser.add_argument("--"+field,required=True)
    convert_parser.add_argument('--layers-dir',help='directory of source-stem.json reviewed layer manifests')
    decomp=sub.add_parser('decompose',help='explicit one-shot generation or offline import of layer candidates')
    decomp.add_argument('--input',required=True);decomp.add_argument('--output',required=True)
    decomp.add_argument('--provider',required=True,choices=['import','ming'])
    decomp.add_argument('--manifest',help='source-bound LayerManifest for offline import')
    plan_parser=sub.add_parser("plan",help="explicitly invoke the configured vision model; never automatic")
    plan_parser.add_argument("--input",required=True);plan_parser.add_argument("--analysis",required=True)
    plan_parser.add_argument("--output",required=True,help="new JSON plan file; hash-only provenance is written beside it")
    for name in ("analyze","segment","extract","compare","reconstruct","reconstruct_elements"):
        c=sub.add_parser(name);c.add_argument("--input",required=True);c.add_argument("--reference");c.add_argument("--mask");c.add_argument("--options",help="JSON options file")
        if name == "reconstruct":
            c.add_argument('--layers',help='reviewed LayerManifest; requires local --analysis')
            c.add_argument("--plan",required=True,help="source-bound ReconstructionPlan JSON")
            c.add_argument("--analysis",help="saved VisionAnalysis; omit to run local/remote OCR")
            c.add_argument("--output",required=True,help="new reconstruction directory, or remote job bundle directory")
        if name == "reconstruct_elements":
            c.add_argument("--plan",required=True,help="source-bound ElementPlan JSON")
            c.add_argument("--analysis-job-id",required=True,help="successful analysis job with element_inventory")
            c.add_argument("--renderer-version",help="actual downstream Artifact Tool version")
            c.add_argument("--output",required=True,help="new reconstruction/job bundle directory")
    a=p.parse_args(argv)
    if a.command=='calibrate-ocr':
        if a.url:p.error('calibrate-ocr uses the model API configured in --config, not --url')
        from .ocr_calibration import prepare_calibration,run_calibrations
        result=prepare_calibration(a.input,read_json(a.analysis),a.output,read_json(a.config))
        if not a.prepare_only:result=run_calibrations([a.output])
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    if a.command in ('prepare-api-reference','score-api','compare-api-rounds'):
        if a.url:p.error('API evaluation is local; --url is unsupported')
        from .api_scoring import prepare_reference,score_api,compare_rounds
        if a.command=='prepare-api-reference':
            report=prepare_reference(a.reference_config,a.generation_config,a.output,a.compiler_root)
            result={'status':report['status'],'output':a.output,'pages':len(report['pages'])}
        elif a.command=='score-api':result=score_api(a.run,a.reference_config,a.output,a.compiler_root)
        else:result=compare_rounds(a.first,a.second,a.output)
        print(json.dumps(result,indent=2));return
    if a.command=='evaluate-api':
        if a.url:p.error('evaluate-api is local; --url is unsupported')
        from .api_evaluation import evaluate_api
        report=evaluate_api(a.config,a.output,a.compiler_root,a.round,resume=a.resume,prepare_only=a.prepare_only)
        # The report on disk contains detailed metadata; stdout has no model text.
        print(json.dumps({k:report[k] for k in ('status','round','requests_reserved','succeeded_pages','expected_pages','network_calls') if k in report},indent=2));return
    if a.command == 'decompose':
        if a.url: p.error('decompose is explicit and local; --url is unsupported')
        if (a.provider=='import') != bool(a.manifest): p.error('--manifest is required only for import')
        from .decomposition import decompose
        print(json.dumps(decompose(a.input,a.output,provider=a.provider,manifest=a.manifest),indent=2));return
    if a.command == "convert":
        if a.url:
            p.error("convert is local and invokes the fixed compiler under --compiler-root")
        from .conversion import convert
        options={'layers_dir':a.layers_dir} if a.layers_dir else {}
        result=convert(a.input_dir,a.analysis_dir,a.plan_dir,a.compiler_root,a.output,**options)
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    if a.command == "plan":
        if a.url:
            p.error("plan uses the explicitly configured model provider, not --url")
        from .model_provider import generate_plan
        output=_new_output(a.output); provenance=_new_output(str(output)+".provenance.json")
        output.parent.mkdir(parents=True,exist_ok=True)
        plan=generate_plan(a.input,read_json(a.analysis),provenance_path=provenance)
        with output.open("x",encoding="utf-8") as handle:
            json.dump(plan.to_dict(),handle,ensure_ascii=False,indent=2)
        print(json.dumps({"status":"REVIEW_REQUIRED","plan":str(output),"provenance":str(provenance),
                          "regions":len(plan.regions),"shapes":len(plan.shapes),"human_acceptance":False},ensure_ascii=False))
        return
    if a.command == "reconstruct":
        if a.layers and (not a.analysis or a.url): p.error('--layers requires local saved --analysis')
        output=_new_output(a.output)
        if a.analysis and a.url:
            p.error("saved --analysis is local only; remote reconstruction runs its own analysis")
        if a.analysis:
            if a.options or a.reference or a.mask:
                p.error("saved-analysis reconstruction accepts only input, analysis, plan and output")
            from .reconstruction import reconstruct
            options={'layers':a.layers} if a.layers else {}
            report=reconstruct(a.input,read_json(a.analysis),load_plan(a.plan),output,**options)
            print(json.dumps(_summary(report,output),ensure_ascii=False,indent=2));return
    if a.command == "reconstruct_elements":
        output=_new_output(a.output)
    if a.url:
        from .client import run
        result=run(a);print(json.dumps(result,ensure_ascii=False,indent=2))
        if result.get('status') in {'FAILED','PARTIAL','INTERRUPTED'}:raise SystemExit(2)
        return
    if a.command=="doctor":
        from .models import capabilities
        print(json.dumps(capabilities(),ensure_ascii=False,indent=2));return
    if a.command=="serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(a.data),host=a.host,port=a.port,access_log=False);return
    if a.command=="benchmark":
        from .benchmark import benchmark
        benchmark(a.suite,a.output,a.data);return
    store=Store(a.data);opts=read_json(a.options) if a.options else {}
    opts.update(operation=a.command,asset_id=store.upload(a.input)["id"])
    if a.command == "reconstruct":opts["plan"]=load_plan(a.plan).to_dict()
    if a.command == "reconstruct_elements":
        opts.update(element_plan=read_json(a.plan),analysis_job_id=a.analysis_job_id,renderer_version=a.renderer_version)
    if a.reference:opts["reference_id"]=store.upload(a.reference)["id"]
    if a.mask:opts["mask_id"]=store.upload(a.mask)["id"]
    job=store.submit(Request.model_validate(opts))
    from .worker import Worker
    worker=Worker(store)
    try:worker.run(job["id"])
    finally:worker.stop()
    status=store.status(job["id"])
    if a.command in {"reconstruct","reconstruct_elements"}:
        reconstruction=store.path("jobs",job["id"])/"reconstruction"
        if (reconstruction/("calibrated/slideir.json" if a.command=="reconstruct_elements" else "slideir.json")).is_file():
            shutil.copytree(reconstruction,output)
            status["local_output"]=str(output)
            status["quality_status"]="REVIEW_REQUIRED"
    print(json.dumps(status,ensure_ascii=False,indent=2))
    if status["status"]!="SUCCEEDED":raise SystemExit(2)


if __name__=="__main__":main()
