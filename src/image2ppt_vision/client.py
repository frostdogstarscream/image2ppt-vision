"""HTTP client; shared contracts, no second inference implementation."""
import hashlib
import json
import os
import time
import urllib.request
import uuid
from pathlib import Path
from .contracts import Request
from .storage import read_json,write_json,digest
from .plans import load_plan


def token():
    if os.environ.get('VISION_TOKEN'):return os.environ['VISION_TOKEN']
    if os.environ.get('VISION_TOKEN_FILE'):return Path(os.environ['VISION_TOKEN_FILE']).read_text().strip()
    raise ValueError('VISION_TOKEN or VISION_TOKEN_FILE required')


def request(base,path,payload=None,content_type='application/json'):
    req=urllib.request.Request(base.rstrip('/')+path,data=payload,headers={'Authorization':'Bearer '+token(),'Content-Type':content_type})
    with urllib.request.urlopen(req,timeout=60) as r:return r.read()


def upload(base,path):
    boundary='vision-'+uuid.uuid4().hex
    body=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="image"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode()
          +Path(path).read_bytes()+f'\r\n--{boundary}--\r\n'.encode())
    return json.loads(request(base,'/v1/assets',body,'multipart/form-data; boundary='+boundary))['id']


def run(args):
    if args.command=='doctor':return json.loads(request(args.url,'/v1/capabilities'))
    if args.command not in {'analyze','segment','extract','compare','reconstruct','reconstruct_elements'}:raise ValueError('remote command not supported')
    opts=read_json(args.options) if args.options else {}
    output=None
    if args.command == 'reconstruct':
        if getattr(args,'analysis',None):raise ValueError('remote reconstruction requires fresh server analysis')
        output=Path(args.output).resolve()
        if output.exists():raise FileExistsError('output must be a new directory')
        opts['plan']=load_plan(args.plan,source_sha256=digest(args.input)).to_dict()
    if args.command == 'reconstruct_elements':
        output=Path(args.output).resolve()
        if output.exists():raise FileExistsError('output must be a new directory')
        opts.update(element_plan=read_json(args.plan),analysis_job_id=args.analysis_job_id,
                    renderer_version=args.renderer_version)
    opts.update(operation=args.command,asset_id=upload(args.url,args.input))
    for field,source in [('reference_id',args.reference),('mask_id',args.mask)]:
        if source:opts[field]=upload(args.url,source)
    validated=Request.model_validate(opts)
    job=json.loads(request(args.url,'/v1/jobs',validated.model_dump_json().encode()))
    deadline=time.monotonic()+validated.timeout_seconds+120
    while job['status'] in {'QUEUED','RUNNING'}:
        if time.monotonic()>deadline:raise TimeoutError('client wait expired; remote task ID '+job['id'])
        time.sleep(.5);job=json.loads(request(args.url,'/v1/jobs/'+job['id']))
    out=output or Path(args.data).resolve()/'remote-results'/job['id'];out.mkdir(parents=True,exist_ok=False)
    write_json(out/'status.json',job)
    for name,meta in job.get('artifacts',{}).items():
        target=(out/name).resolve()
        if not target.is_relative_to(out):raise ValueError('unsafe artifact path')
        data=request(args.url,'/v1/jobs/'+job['id']+'/artifacts/'+name)
        if hashlib.sha256(data).hexdigest()!=meta['sha256']:raise ValueError('download hash mismatch')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    return {'id':job['id'],'status':job['status'],'local_output':str(out),'verified_files':len(job.get('artifacts',{}))}
