"""One network attempt in a disposable process; its parent enforces wall time.

Only image/OCR-derived request JSON enters this process. No reference document,
manual plan, or retry logic is available. Credentials are environment-only.
"""
import hashlib
import json
import os
from pathlib import Path
import time
from urllib import error, request

from .model_provider import _NoRedirect, _endpoint, _read_response, ModelProviderError
from .storage import read_json, write_json


def exchange(request_file, settings_file, output):
    output=Path(output)
    settings=read_json(settings_file)
    key=os.environ.get('VISION_LLM_API_KEY','')
    if not key.strip() or any(ord(c)<32 or ord(c)>126 for c in key):
        write_json(output/'transport.json',{'status':'FAILED','error_code':'model_configuration_required'})
        return
    started=time.monotonic(); raw=None; status=None;quota_headers={}
    def retain_quota_headers(headers):
        if headers is None:return
        for name,value in headers.items():
            name=name.lower()
            if name=='retry-after' or 'ratelimit' in name or 'rate-limit' in name:
                quota_headers[name]=str(value).replace(key,'[REDACTED]')[:256]
    try:
        body=Path(request_file).read_bytes()
        req=request.Request(_endpoint(settings['base_url']),data=body,method='POST',headers={
            'Authorization':'Bearer '+key,'Content-Type':'application/json','Accept':'application/json'})
        opener=request.build_opener(_NoRedirect)
        # No urllib retries. Parent kills only this process at the absolute deadline.
        with opener.open(req,timeout=settings['timeout_seconds']) as response:
            retain_quota_headers(getattr(response,'headers',None))
            raw=_read_response(response,started+settings['timeout_seconds'])
        status={'status':'RECEIVED','error_code':None}
    except error.HTTPError as exc:
        code=exc.code
        retain_quota_headers(exc.headers)
        try:
            raw=_read_response(exc,started+settings['timeout_seconds'])
        except Exception:
            pass
        finally:
            exc.close()
        status={'status':'FAILED','error_code':'model_http_'+str(code),'http_status':code}
    except Exception as exc:
        status={'status':'FAILED','error_code':str(exc) if isinstance(exc,ModelProviderError) else 'model_transport_failed'}
    if raw is not None:
        # Keep exact response unless a credential is echoed; hashes still identify
        # the received bytes. Do not print private image/model text in logs.
        redacted=key.encode() in raw
        saved=raw.replace(key.encode(),b'[REDACTED]') if redacted else raw
        with (output/'response.body').open('xb') as handle:
            handle.write(saved);handle.flush();os.fsync(handle.fileno())
        status.update(response_sha256=hashlib.sha256(raw).hexdigest(),
                      saved_response_sha256=hashlib.sha256(saved).hexdigest(),
                      response_redacted=redacted,response_bytes=len(raw))
        if redacted:
            status.update(status='FAILED',error_code='credential_echo_rejected')
    status['elapsed_seconds']=round(time.monotonic()-started,4)
    if quota_headers:status['quota_headers']=quota_headers
    write_json(output/'transport.json',status)


if __name__=='__main__':
    import sys
    exchange(*sys.argv[1:])
