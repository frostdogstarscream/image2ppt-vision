"""Explicit one-shot layer generation. Never invoked by reconstruct/convert."""
import base64
import io
import json
import os
from pathlib import Path
import shutil
import urllib.request
import urllib.error
import urllib.parse
import uuid

from PIL import Image

from .layers import snapshot_layers, preview_layers, LayerError
from .storage import digest, write_json

MAX_IMAGE=20*1024*1024
MAX_RESPONSE=96*1024*1024
PROMPT=('Decompose the supplied slide into aligned RGBA PNG layers at the exact original canvas size. '
        'Separate background, visual modules, decorations and text. Preserve all original words, numbers, '
        'colors and geometry. Remove foreground ghosts from background and complete occluded background. '
        'Do not redesign or add content. Return layers in bottom-to-top composition order.')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise LayerError('provider_redirect_rejected')


def _fetch(request, limit, timeout):
    with urllib.request.build_opener(NoRedirect).open(request,timeout=timeout) as response:
        data=response.read(limit+1)
    if len(data)>limit: raise LayerError('provider_response_too_large')
    return data


def decompose(source, output, *, provider, manifest=None):
    source=Path(source); output=Path(output)
    if output.exists(): raise FileExistsError('output must be new')
    with Image.open(source) as image:
        if image.format!='PNG': raise LayerError('decompose_requires_png')
        size=list(image.size)
        if min(size)<=0 or size[0]*size[1]>40_000_000: raise LayerError('invalid_source_size')
    sha=digest(source)
    if provider=='import':
        if manifest is None: raise LayerError('import_manifest_required')
        result=snapshot_layers(manifest,output,source_sha256=sha,image_size=size,require_review=False)
        preview_layers(result)
        shutil.copyfile(source,output/'source.png')
        write_json(output/'decomposition.json',{'status':'REVIEW_REQUIRED','provider':'import','source_sha256':sha,'human_acceptance':False})
        return {'status':'REVIEW_REQUIRED','manifest':str(result),'human_acceptance':False}
    if provider!='ming': raise LayerError('unsupported_layer_provider')
    output.mkdir(parents=True)
    status={'status':'RUNNING','provider':'ming','source_sha256':sha,'attempts':0,'human_acceptance':False}
    try:
        key=os.environ.get('VISION_LAYER_API_KEY','').strip()
        if not key: raise LayerError('missing_layer_api_key')
        base=os.environ.get('VISION_LAYER_BASE_URL','https://maas-api.antdigital.com/v1').rstrip('/')
        url=urllib.parse.urlparse(base)
        if url.scheme!='https' or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise LayerError('invalid_layer_base_url')
        model=os.environ.get('VISION_LAYER_MODEL','ming-image-0.1-design-layer')
        if any(c in model for c in '\r\n') or len(model)>256: raise LayerError('invalid_layer_model')
        blob=source.read_bytes()
        if len(blob)>MAX_IMAGE: raise LayerError('source_too_large')
        shutil.copyfile(source,output/'source.png')
        boundary='image2ppt'+uuid.uuid4().hex
        chunks=[]
        for name,value in [('model',model),('prompt',PROMPT)]:
            chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="input.png"\r\nContent-Type: image/png\r\n\r\n'.encode()+blob+b'\r\n')
        chunks.append(f'--{boundary}--\r\n'.encode())
        req=urllib.request.Request(base+'/images/edits',data=b''.join(chunks),
                headers={'Authorization':'Bearer '+key,'Content-Type':'multipart/form-data; boundary='+boundary},method='POST')
        status.update(attempts=1,model_id=model); write_json(output/'decomposition.json',status)
        raw=_fetch(req,MAX_RESPONSE,600)
        result=json.loads(raw)
        items=result.get('data',[])
        if not isinstance(items,list) or not 1<=len(items)<=256: raise LayerError('invalid_layer_response')
        # Preserve original image bytes, never signed URLs or arbitrary server metadata.
        (output/'raw').mkdir(); (output/'assets').mkdir(); layers=[]; total_bytes=0
        for i,item in enumerate(items):
            if item.get('b64_json'): data=base64.b64decode(item['b64_json'],validate=True)
            elif item.get('url'):
                parsed=urllib.parse.urlparse(item['url'])
                if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password:
                    raise LayerError('invalid_layer_download_url')
                data=_fetch(urllib.request.Request(item['url']),MAX_IMAGE,60)
            else: raise LayerError('layer_image_missing')
            if len(data)>MAX_IMAGE: raise LayerError('layer_image_too_large')
            total_bytes+=len(data)
            if total_bytes>MAX_RESPONSE: raise LayerError('layer_bundle_too_large')
            (output/'raw'/f'{i:03}.bin').write_bytes(data)
            with Image.open(io.BytesIO(data)) as image:
                if list(image.size)!=size: raise LayerError('provider_layer_canvas_mismatch')
                image.load(); rgba=image.convert('RGBA')
            dest=output/'assets'/f'{i:03}.png'; rgba.save(dest)
            layers.append({'id':f'{i:03}','path':f'assets/{i:03}.png','sha256':digest(dest),
                           'bbox':[0,0,*size],'z':i,'module':None,'retained_text_ids':[]})
        write_json(output/'layers.json',{'schema':'LayerManifest','version':'1.0','source_sha256':sha,
                   'image_size':size,'provider':'ming','model_id':model,'reviewed':False,
                   'layers':layers,'native_text':[],'repairs':[]})
        preview_layers(output/'layers.json')
        status.update(status='REVIEW_REQUIRED',raw_response_sha256=__import__('hashlib').sha256(raw).hexdigest(),
                      layer_count=len(layers),ownership_review_required=True)
        write_json(output/'decomposition.json',status)
        return {'status':'REVIEW_REQUIRED','manifest':str(output/'layers.json'),'human_acceptance':False}
    except Exception as exc:
        code=str(exc) if isinstance(exc,LayerError) else ('provider_http_'+str(exc.code) if isinstance(exc,urllib.error.HTTPError) else 'layer_provider_failed')
        status.update(status='FAILED',error_code=code)
        write_json(output/'decomposition.json',status)
        raise LayerError(code) from None
