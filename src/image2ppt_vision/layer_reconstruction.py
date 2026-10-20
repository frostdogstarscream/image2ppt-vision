"""Materialize reviewed layers through existing SlideIR; no model calls."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import numpy as np

from PIL import Image

from .layers import load_layers, snapshot_layers, preview_layers, LayerError
from .plans import validate_plan
from .storage import digest, read_json, write_json


def reconstruct_layers(source, analysis, plan, output, layers, **kwargs):
    from .reconstruction import reconstruct
    source=Path(source); output=Path(output)
    if output.exists(): raise FileExistsError('output must be new')
    texts=[t for t in analysis['objects'] if t['kind']=='text']
    with Image.open(source) as im: size=list(im.size)
    sha=digest(source)
    plan=validate_plan(plan,source_sha256=sha,image_size=size,ocr_ids=[t['id'] for t in texts]).to_dict()
    manifest,_=load_layers(layers,source_sha256=sha,image_size=size,texts=texts,shapes=plan['shapes'])
    output.mkdir(parents=True)
    snapshot=snapshot_layers(layers,output/'layers',source_sha256=sha,image_size=size,texts=texts,shapes=plan['shapes'])
    manifest,assets=load_layers(snapshot,source_sha256=sha,image_size=size,texts=texts)
    preview_layers(snapshot)
    # Estimate typography from original evidence, never model-generated letters.
    candidate=deepcopy(plan)
    candidate['regions']=[{'id':'layer-typography-page','bbox':[0,0,*size],'kind':'panel'}]
    candidate['shapes']=[]
    wanted={t.text_id for t in manifest.native_text}
    candidate['retain_raster_ids']=[t['id'] for t in texts if t['id'] not in wanted]
    candidate['text_overrides']=[t for t in candidate['text_overrides'] if t['target_id'] in wanted]
    report=reconstruct(source,analysis,candidate,output/'typography-evidence',**kwargs)
    estimated=read_json(output/'typography-evidence/slideir.json')
    native={n['id']:n for n in estimated['nodes'] if n['type']=='text'}
    if not wanted.issubset(native): raise LayerError('requested_native_typography_unavailable')
    (output/'source').mkdir(); shutil.copyfile(source,output/'source/page.png')
    evidence=[{'id':t['id'],'kind':'text','bbox':t['bbox'],'text':t['text']} for t in texts]
    nodes=[]; ir_assets=[]; membership={}
    for layer in manifest.layers:
        identifier='layer-image-'+layer.id; eid=identifier+'-pixels'
        if eid in {e['id'] for e in evidence} or identifier in native: raise LayerError('layer_namespace_collision')
        evidence.append({'id':eid,'kind':'image','bbox':layer.bbox})
        ir_assets.append({'id':identifier,'path':assets[layer.id].relative_to(output).as_posix(),
                          'sha256':layer.sha256,'purpose':'module',
                          'contains_evidence_ids':[eid]+layer.retained_text_ids})
        nodes.append({'id':identifier,'type':'image','asset_id':identifier,'bbox':layer.bbox,'z':layer.z,'evidence_ids':[]})
        membership[identifier]=layer.module
    for item in manifest.native_text:
        node=deepcopy(native[item.text_id]); node['z']=item.z; nodes.append(node)
        membership[node['id']]=item.module
    from .reconstruction import _verified_shape
    with Image.open(source) as original: pixels=np.array(original.convert('RGB'))
    for item in manifest.native_shapes:
        proposal=next(s for s in plan['shapes'] if s['id']==item.shape_id)
        verified,reason=_verified_shape(pixels,proposal)
        if verified is None: raise LayerError('requested_native_shape_unverified')
        node=verified[0]
        if any(node['id']==n['id'] for n in nodes): raise LayerError('layer_namespace_collision')
        eid=node['evidence_ids'][0]
        if any(eid==e['id'] for e in evidence): raise LayerError('layer_namespace_collision')
        evidence.append({'id':eid,'kind':'shape','bbox':node['bbox']})
        node['z']=item.z;nodes.append(node);membership[node['id']]=item.module
    nodes.sort(key=lambda n:n['z'])
    groups=[]
    for module in sorted({m for m in membership.values() if m is not None}):
        members=[n['id'] for n in nodes if membership[n['id']]==module]
        if len(members)>1: groups.append({'id':'layer-group-'+module,'member_ids':members})
    notes=['Reviewed layered reconstruction; background completion is inferred, not original evidence.',
           'Complex assets are movable raster layers; only declared native text is text-editable.',
           'Layer review does not establish PowerPoint/WPS or human acceptance.']
    ir={**estimated,'source_sha256':sha,'evidence':evidence,'assets':ir_assets,'nodes':nodes,'groups':groups,'review_notes':notes}
    write_json(output/'plan.json',plan); write_json(output/'slideir.json',ir)
    write_json(output/'authoring.json',{'format_version':'1','tool':'image2ppt-vision layered reconstruct',
               'model_id':manifest.model_id,'created_at':datetime.now(timezone.utc).isoformat(),
               'ir_sha256':digest(output/'slideir.json'),'node_decisions':{n['id']:'reviewed layer ownership' for n in nodes},
               'review_notes':notes,'unresolved_ids':[t['id'] for t in texts if t['id'] not in wanted]})
    report.update(native_text=len(wanted),native_text_ratio=len(wanted)/max(1,len(texts)),
                  retained_raster_text=[t['id'] for t in texts if t['id'] not in wanted],
                  groups=len(groups),independent_images=len(ir_assets),native_module_backdrops=0,
                  native_verified_shapes=len(manifest.native_shapes),shape_diagnostics=[],semantic_regions=len(plan['regions']),
                  layer_manifest_sha256=digest(snapshot),repairs=[r.model_dump() for r in manifest.repairs],
                  plan_sha256=digest(output/'plan.json'),mode='hybrid_layers',
                  region_residual_fraction=None)
    write_json(output/'reconstruction.json',report)
    return report
