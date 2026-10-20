"""Real upload/submit/poll/download/hash checks through the configured HTTP endpoint."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from image2ppt_vision.client import run

def main():
    p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--image',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    prompt=out/'prompt.json';prompt.write_text(json.dumps({'mode':'prompt','bbox':[585,392,105,105]}))
    reports=[]
    for operation in ['analyze','segment','extract','compare']:
        mask=None
        if operation=='extract':
            segmented=Path(reports[1]['local_output']);result=json.loads((segmented/'result.json').read_text())
            candidate=max(result['masks'],key=lambda m:m['score']['predicted_iou'])
            mask=str(segmented/candidate['artifact_prefix']/'mask-full.png')
        args=SimpleNamespace(command=operation,url=a.url,data=str(out),input=a.image,
                             reference=a.image if operation=='compare' else None,mask=mask,
                             options=str(prompt) if operation=='segment' else None)
        result=run(args);result['operation']=operation;reports.append(result)
        (out/'report.json').write_text(json.dumps(reports,indent=2))
        if result['status']!='SUCCEEDED':raise RuntimeError(operation+' failed')
    compared=json.loads((Path(reports[-1]['local_output'])/'result.json').read_text())
    assert compared['comparison']['before']['mae']==0
    print(json.dumps({'status':'PASS','operations':len(reports),'verified_files':sum(r['verified_files'] for r in reports)}))

if __name__=='__main__':main()
