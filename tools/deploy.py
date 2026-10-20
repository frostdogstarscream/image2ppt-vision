"""Explicit deployment action for the authorized host; rollback on any failure."""
import argparse
import json
import os
import secrets
import subprocess
import time
import urllib.request
from pathlib import Path

from gpu_services import NAMES

ROOT=Path('/data/image2ppt-vision')
PROJECT='image2ppt-vision-v1'
GPU='GPU-452202ca-1679-e799-d19d-e8cca8bcfb12'

def call(*args,output=None):
    return subprocess.run(args,check=True,stdout=output,text=True)

def inventory():
    ids=subprocess.check_output(['docker','ps','-aq'],text=True).split()
    data=json.loads(subprocess.check_output(['docker','inspect',*ids],text=True))
    return {x['Name'].lstrip('/'):{'id':x['Id'],'running':x['State']['Running']} for x in data if (x['Config'].get('Labels') or {}).get('com.docker.compose.project')!=PROJECT}

def main():
    p=argparse.ArgumentParser();p.add_argument('--release',required=True);a=p.parse_args()
    release=Path(a.release).resolve()
    if not release.is_relative_to(ROOT/'releases'):raise ValueError('release outside deployment root')
    evidence=ROOT/'evidence'/time.strftime('deploy-%Y%m%d-%H%M%S');evidence.mkdir()
    before=inventory();(evidence/'containers-before.json').write_text(json.dumps(before,indent=2))
    # Validate CPU package and engineering tests before interrupting existing services.
    with (evidence/'tests.log').open('w') as f:
        call('docker','run','--rm','--network=none','--cpus=8','--memory=16g','image2ppt-vision:0.2.1','python','-m','pytest','tests','-q',output=f)
    script=str(release/'tools/gpu_services.py');record=str(evidence/'gpu-services.json')
    call('python3',script,'snapshot','--record',record)
    compose=['docker','compose','--project-name',PROJECT,'--file',str(release/'compose.yaml')]
    try:
        call('python3',script,'stop','--record',record)
        time.sleep(3)
        if any(inventory()[name]['running'] for name in NAMES):raise RuntimeError('external restart detected')
        with (evidence/'gpu-smoke.json').open('w') as f:
            call('docker','run','--rm','--network=none','--gpus','device='+GPU,'--cpus=8','--memory=16g',
                 '-v',str(ROOT/'models-ready')+':/models:ro','image2ppt-vision:0.2.1','python','tools/gpu_smoke.py',output=f)
        env=release/'runtime.env'
        if not env.exists():
            token=secrets.token_urlsafe(48)
            fd=os.open(env,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'w') as f:f.write('VISION_TOKEN='+token+'\n')
            tokenfile=ROOT/'client-token'
            fd=os.open(tokenfile,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'w') as f:f.write(token)
        call(*compose,'up','-d')
        alive=False
        for _ in range(30):
            try:
                with urllib.request.urlopen('http://127.0.0.1:18761/health',timeout=2) as r:
                    alive=json.load(r).get('status')=='alive'
                if alive:break
            except OSError:time.sleep(1)
        if not alive:raise RuntimeError('health timeout')
        after=inventory();(evidence/'containers-after.json').write_text(json.dumps(after,indent=2))
        changed=[n for n,v in before.items() if n not in NAMES and after.get(n)!=v]
        if changed:raise RuntimeError('unrelated container state changed')
        if any(after[name]['running'] for name in NAMES):raise RuntimeError('external restart detected')
        (evidence/'deployment.json').write_text(json.dumps({'status':'PASS','unchanged_other_containers':len(before)-9,'stopped':NAMES,'gpu_uuid':GPU,'endpoint':'127.0.0.1:18761'},indent=2))
        print('deployed; evidence:',evidence)
    except BaseException:
        subprocess.run([*compose,'stop'],check=False)
        call('python3',script,'restore','--record',record)
        (evidence/'deployment.json').write_text(json.dumps({'status':'FAILED','rollback':'restored recorded GPU services'}))
        raise

if __name__=='__main__':main()
