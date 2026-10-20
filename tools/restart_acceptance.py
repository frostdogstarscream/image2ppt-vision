"""Authorized deployment acceptance: restart only this service's API container."""
import json
import subprocess
import time
import urllib.request
from pathlib import Path

def main():
    root=Path('/data/image2ppt-vision');token=(root/'client-token').read_text().strip()
    base='http://127.0.0.1:18761'
    def call(path,data=None,kind='application/json'):
        req=urllib.request.Request(base+path,data=data,headers={'Authorization':'Bearer '+token,'Content-Type':kind})
        with urllib.request.urlopen(req,timeout=15) as r:return json.load(r)
    image=(root/'data/reference/page-13.png').read_bytes();boundary='acceptance-restart'
    data=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="page.png"\r\nContent-Type: image/png\r\n\r\n'.encode()+image+f'\r\n--{boundary}--\r\n'.encode())
    asset=call('/v1/assets',data,'multipart/form-data; boundary='+boundary)['id']
    def submit(**kwargs):return call('/v1/jobs',json.dumps({'operation':'analyze','asset_id':asset,**kwargs}).encode())['id']
    running=submit(segment=True)
    deadline=time.monotonic()+30
    while call('/v1/jobs/'+running)['status']!='RUNNING':
        if time.monotonic()>deadline:raise TimeoutError('job did not start')
        time.sleep(.05)
    queued=submit(ocr=False)
    subprocess.run(['docker','restart','-t','15','image2ppt-vision-v1-api-1'],check=True,capture_output=True)
    deadline=time.monotonic()+60
    while True:
        try:
            first=call('/v1/jobs/'+running);second=call('/v1/jobs/'+queued)
            if second['status'] not in {'QUEUED','RUNNING'}:break
        except OSError:pass
        if time.monotonic()>deadline:raise TimeoutError('restart recovery timeout')
        time.sleep(.5)
    report={'running_after_restart':first,'queued_after_restart':second,
            'status':'PASS' if first['status']=='INTERRUPTED' and second['status']=='SUCCEEDED' else 'FAIL'}
    (root/'evidence/restart-acceptance.json').write_text(json.dumps(report,indent=2))
    assert report['status']=='PASS',report
    print(json.dumps({'status':'PASS','interrupted':running,'retained_queue':queued}))

if __name__=='__main__':main()
