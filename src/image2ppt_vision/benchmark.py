"""Frozen, separately annotated references. Missing human labels are NOT_RUN."""
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
from .algorithms import iou
from .contracts import Request
from .storage import Store, read_json, write_json, digest
from .worker import Worker


def match(pred,truth,threshold):
    pairs=sorted([(iou(p["bbox"],t["bbox"]),i,j) for i,p in enumerate(pred) for j,t in enumerate(truth) if p.get('kind')==t.get('kind')],reverse=True)
    pi=set();ti=set();matched=[]
    for score,i,j in pairs:
        if score>=threshold and i not in pi and j not in ti:
            pi.add(i);ti.add(j);matched.append((i,j))
    return matched


def edit_distance(a,b):
    row=list(range(len(b)+1))
    for i,c in enumerate(a,1):
        new=[i]
        for j,d in enumerate(b,1):new.append(min(new[-1]+1,row[j]+1,row[j-1]+(c!=d)))
        row=new
    return row[-1]


def mask_metrics(predictions, references, prompted):
    def overlap(a,b):
        if a.shape!=b.shape:raise ValueError('mask dimensions differ')
        return float(np.logical_and(a,b).sum()/max(1,np.logical_or(a,b).sum()))
    if prompted:
        if len(predictions)!=1 or len(references)!=1:raise ValueError('prompted case requires one selected mask and one reference')
        value=overlap(predictions[0],references[0])
        return {'iou':value,'status':'FAIL' if value<.70 else 'PASS','mode':'prompt'}
    pairs=sorted([(overlap(a,b),i,j) for i,a in enumerate(predictions) for j,b in enumerate(references)],reverse=True)
    pi=set();ti=set()
    for score,i,j in pairs:
        if score>=.5 and i not in pi and j not in ti:pi.add(i);ti.add(j)
    duplicates=sum(overlap(a,b)>.9 for i,a in enumerate(predictions) for b in predictions[i+1:])
    merges=sum(sum(np.logical_and(p,t).sum()/max(1,t.sum())>.5 for t in references)>1 for p in predictions)
    return {'mode':'automatic','recall':len(ti)/max(1,len(references)),'duplicate_pairs':duplicates,'possible_merged_masks':merges,'status':'REVIEW_REQUIRED'}


def benchmark(suite,output,data):
    suite=Path(suite).resolve();spec=read_json(suite)
    if not spec.get("cases"):raise ValueError("empty benchmark")
    out=Path(output);out.mkdir(parents=True,exist_ok=False);store=Store(data);reports=[];worker=Worker(store);seen=set()
    for case in spec["cases"]:
        try:
            source=suite.parent/case["image"]
            if digest(source)!=case["sha256"]:raise ValueError("frozen input hash mismatch")
            opts=case.get("options",{}).copy();opts.update(operation=case.get("operation","analyze"),asset_id=store.upload(source)["id"])
            signature=(opts['operation'],opts.get('ocr',True),opts.get('segment',False),opts.get('mode','automatic'))
            warm_case=signature in seen
            job=store.submit(Request.model_validate(opts));start=time.monotonic();worker.run(job["id"])
            elapsed=time.monotonic()-start;directory=store.path("jobs",job["id"])
            result=read_json(directory/"result.json") if (directory/"result.json").exists() else {}
            report={"case":case["id"],"job":job["id"],"execution":store.status(job["id"]),"elapsed_seconds":elapsed,
                    "timing_scope":"worker_reused" if warm_case else "cold_start","algorithm_status":"NOT_RUN"}
            if report['execution']['status']=='SUCCEEDED':seen.add(signature)
            if case.get("labels"):
                labels_path=suite.parent/case["labels"]
                if digest(labels_path)!=case["labels_sha256"]:raise ValueError("label hash mismatch")
                labels=read_json(labels_path)
                if labels.get("human_verified"):
                    metrics={}
                    for kind,threshold,target in [("text",.5,.90),("geometry",.7,.85)]:
                        allowed={"text"} if kind=="text" else {"rectangle","circle"}
                        pred=[x for x in result.get("objects",[]) if x["kind"] in allowed];truth=[x for x in labels["objects"] if x["kind"] in allowed]
                        if not truth:metrics[kind]={"status":"NOT_RUN"};continue
                        matches=match(pred,truth,threshold);n=len(matches);precision=n/max(1,len(pred));recall=n/len(truth);f1=2*precision*recall/max(1e-12,precision+recall)
                        metrics[kind]={"precision":precision,"recall":recall,"f1":f1,"status":"PASS" if (f1>=target if kind=="text" else min(precision,recall)>=target) else "FAIL"}
                        if kind=="text":
                            pi={i for i,j in matches};ti={j for i,j in matches}
                            errors=sum(edit_distance(pred[i]["text"],truth[j]["text"]) for i,j in matches)+sum(len(t["text"]) for j,t in enumerate(truth) if j not in ti)+sum(len(p["text"]) for i,p in enumerate(pred) if i not in pi)
                            cer=errors/max(1,sum(len(t["text"]) for t in truth));metrics[kind]["cer"]=cer
                            if cer>.05:metrics[kind]["status"]="FAIL"
                    if labels.get('masks'):
                        refs=[]
                        for m in labels['masks']:
                            path=labels_path.parent/m['file']
                            if digest(path)!=m['sha256']:raise ValueError('mask reference hash mismatch')
                            refs.append(np.array(Image.open(path).convert('L'))>0)
                        masks=result.get('masks',[])
                        prompted=opts.get('mode')=='prompt'
                        if prompted and masks:masks=[max(masks,key=lambda m:m['score']['predicted_iou'])]
                        preds=[np.array(Image.open(directory/m['artifact_prefix']/'mask-full.png'))>0 for m in masks]
                        metrics['segmentation']=mask_metrics(preds,refs,prompted) if preds else {'status':'FAIL','mode':'prompt' if prompted else 'automatic','iou':0.0}
                    report.update(metrics=metrics,algorithm_status="FAIL" if any(v["status"]=="FAIL" for v in metrics.values()) else "REVIEW_REQUIRED" if any(v['status']=='REVIEW_REQUIRED' for v in metrics.values()) else "PASS" if any(v["status"]=="PASS" for v in metrics.values()) else "NOT_RUN")
            reports.append(report)
        except Exception as e:reports.append({"case":case.get("id"),"execution":{"status":"FAILED","error_type":type(e).__name__},"algorithm_status":"NOT_RUN"})
    worker.stop()
    times=[r["elapsed_seconds"] for r in reports if r.get("timing_scope")=="cold_start"]
    warm=[r["elapsed_seconds"] for r in reports if r.get("timing_scope")=="worker_reused" and r['execution']['status']=='SUCCEEDED']
    prompted=[r['metrics']['segmentation']['iou'] for r in reports if r.get('metrics',{}).get('segmentation',{}).get('mode')=='prompt']
    prompt_gate={'count':len(prompted),'mean_iou':float(np.mean(prompted)) if prompted else None,'status':'NOT_RUN' if len(prompted)<30 else 'PASS' if min(prompted)>=.7 and np.mean(prompted)>=.85 else 'FAIL'}
    execution_ok=all(r['execution']['status']=='SUCCEEDED' for r in reports)
    full_analysis=all(c.get('operation','analyze')=='analyze' and c.get('options',{}).get('segment',False) and c.get('options',{}).get('ocr',True) for c in spec['cases'])
    write_json(out/"report.json",{"suite_sha256":digest(suite),"cases":reports,"execution_status":"PASS" if execution_ok else "FAIL","cold_end_to_end_p95":float(np.percentile(times,95)) if times else None,
                                  "prompt_segmentation_gate":prompt_gate,"warm_p95":float(np.percentile(warm,95)) if warm else None,
                                  "warm_p95_status":"FAIL" if not execution_ok else "NOT_RUN" if not warm or not full_analysis else "PASS" if np.percentile(warm,95)<=120 else "FAIL","human_acceptance":False})
