"""Experimental object plans: explicit ownership, no page-sized residual escape.

The model is an author, never an acceptance authority. Native shape parameters
are source-referenced proposals; only actual PPTX inspection and independent
review can assess fidelity. This module does not call a model or alter baselines.
"""
from copy import deepcopy
from datetime import datetime,timezone
import json,math,re,shutil
from pathlib import Path
from PIL import Image,ImageDraw
from .storage import digest,write_json
from .font_contract import validate_font_proposal, resolve_font

GEOMETRIES={'rect','roundRect','ellipse','line','chevron','pentagon','rightArrow','leftArrow','trapezoid','parallelogram','hexagon','diamond','triangle'}

def _box(box,size):
    if not isinstance(box,list) or len(box)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in box):raise ValueError('invalid bbox')
    x,y,w,h=box
    if min(x,y)<0 or min(w,h)<=0 or x+w>size[0]+.01 or y+h>size[1]+.01:raise ValueError('bbox outside source')
    return box

def _intersection(a,b):
    return max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))*max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))

def _color(value,none=False):
    if not (none and value=='none') and (not isinstance(value,str) or not re.fullmatch('[0-9a-fA-F]{6}',value)):raise ValueError('invalid color')

def parse_response(path):
    response=json.loads(Path(path).read_text(encoding='utf8'))
    choice=response['choices'][0]
    if choice.get('finish_reason') not in ('stop',None):raise ValueError('model response incomplete')
    content=choice['message']['content'].strip()
    if content.startswith('```'):
        content=re.sub(r'^```(?:json)?\s*','',content);content=re.sub(r'\s*```$','',content)
    return json.loads(content)

def raster_text_conflicts(plan,inventory):
    ocr={t['id']:t for t in inventory['texts']};conflicts=[]
    for t in plan['text_decisions']:
        if t['action']!='native':continue
        original=ocr[t['id']]['bbox']
        for obj in plan['objects']:
            if obj['kind']=='raster_asset' and _intersection(original,obj['bbox'])>original[2]*original[3]*.12:
                conflicts.append({'text_id':t['id'],'ocr_text':ocr[t['id']]['text'],'text_bbox':original,'asset_id':obj['id'],'asset_bbox':obj['bbox']})
    return conflicts

def apply_element_patch(plan,patch):
    if patch.get('schema')!='ElementPatch/1' or patch.get('source_sha256')!=plan['source_sha256']:raise ValueError('patch source mismatch')
    result=deepcopy(plan)
    remove=patch.get('remove_object_ids',[])
    if len(remove)!=len(set(remove)) or any(i not in {o['id'] for o in plan['objects']} for i in remove):raise ValueError('invalid patch deletion')
    result['objects']=[o for o in result['objects'] if o['id'] not in remove]
    for key in ('objects','text_decisions'):
        updates=patch.get(key,[])
        if len(updates)!=len({o['id'] for o in updates}):raise ValueError('duplicate patch id')
        mapping={o['id']:o for o in result[key]}
        for update in updates:
            if key=='text_decisions' and update['id'] not in mapping:raise ValueError('unknown patched text id')
            # A field patch retains unspecified source-bound properties. New
            # objects still need every required field during full validation.
            merged={**mapping.get(update['id'],{}),**deepcopy(update)}
            mapping[update['id']]={k:v for k,v in merged.items() if v is not None}
        result[key]=list(mapping.values())
    result['limitations']=[*result.get('limitations',[]),*patch.get('limitations',[])]
    return result

def validate_element_plan(plan,inventory):
    plan=deepcopy(plan)
    if plan.get('schema') not in ('ElementPlan/1','ElementPlan/2') or plan.get('source_sha256')!=inventory['source_sha256'] or plan.get('image_size')!=inventory['image_size']:raise ValueError('plan source binding mismatch')
    size=inventory['image_size'];_color(plan['background_color'])
    objects=plan['objects'];texts=plan['text_decisions'];ocr={t['id']:t for t in inventory['texts']}
    if not isinstance(objects,list) or not 1<=len(objects)<=400 or len(objects)+len(texts)>500:raise ValueError('object bound exceeded')
    if len(texts)!=len(ocr) or {t['id'] for t in texts}!=set(ocr):raise ValueError('OCR decisions must cover exactly once')
    ids=[o['id'] for o in objects]+[t['id'] for t in texts]
    if len(ids)!=len(set(ids)) or any(not re.fullmatch('[A-Za-z0-9_-]{1,70}',s) for s in ids):raise ValueError('invalid or duplicate object id')
    retained={};candidates={c['id'] for c in inventory.get('candidates',[])}
    for obj in objects:
        _box(obj['bbox'],size)
        if isinstance(obj['z'],bool) or not isinstance(obj['z'],int):raise ValueError('z must be integer')
        if obj.get('candidate_id') and obj['candidate_id'] not in candidates:raise ValueError('unknown CV candidate')
        if obj['kind']=='native_shape':
            if obj['geometry'] not in GEOMETRIES:raise ValueError('unsupported native geometry')
            if obj.get('fill') is None and obj.get('fill_gradient'):
                obj['fill']=obj['fill_gradient']['stops'][0]['color']
            if obj.get('fill') is None and obj['geometry']=='line':obj['fill']='none'
            _color(obj['fill'],True)
            if obj.get('line') is None:obj.pop('line',None)
            if 'line' in obj:
                _color(obj['line']['color'])
                if not 0<=obj['line']['width_pt']<=20:raise ValueError('invalid line width')
        elif obj['kind']=='raster_asset':
            if obj['bbox'][2]*obj['bbox'][3]>size[0]*size[1]*.18:raise ValueError('large raster fallback rejected')
            if obj.get('mask','rect') not in ('rect','ellipse'):raise ValueError('unsupported asset mask')
            for tid in obj.get('contains_text_ids',[]):
                if tid not in ocr or tid in retained:raise ValueError('invalid raster text owner')
                if _intersection(obj['bbox'],ocr[tid]['bbox'])<ocr[tid]['bbox'][2]*ocr[tid]['bbox'][3]*.95:raise ValueError('raster does not contain declared text')
                retained[tid]=obj['id']
        else:raise ValueError('invalid object kind')
    for t in texts:
        _box(t['bbox'],size);_color(t['color'])
        if not isinstance(t['text'],str) or not t['text'].strip() or len(t['text'])>5000:raise ValueError('invalid text')
        if t['action'] not in ('native','retained_in_asset') or (t['action']=='retained_in_asset')!=(t['id'] in retained):raise ValueError('text retention mismatch')
        if not isinstance(t['bold'],bool) or not isinstance(t['z'],int) or not 1<=t['font_size_px']<=400:raise ValueError('invalid typography')
        if type(t['font_size_px']) not in (int,float) or not math.isfinite(t['font_size_px']):raise ValueError('invalid typography')
        if t['action']=='native':validate_font_proposal(t,required=plan['schema']=='ElementPlan/2')
        if t.get('runs') and ''.join(r['text'] for r in t['runs'])!=t['text']:raise ValueError('run text mismatch')
    conflicts=raster_text_conflicts(plan,inventory)
    if conflicts:raise ValueError('native text duplicates raster pixels: '+conflicts[0]['text_id']+' / '+conflicts[0]['asset_id'])
    return deepcopy(plan)

def materialize_elements(source,inventory,plan,output,*,model_id,response_sha256):
    source=Path(source);output=Path(output)
    if output.exists():raise FileExistsError('new output required')
    if digest(source)!=inventory['source_sha256']:raise ValueError('source changed')
    plan=validate_element_plan(plan,inventory)
    resolved_fonts={t['id']:resolve_font(t) for t in plan['text_decisions'] if t['action']=='native' and 'font_family' in t}
    with Image.open(source) as original:image=original.convert('RGBA')
    if list(image.size)!=inventory['image_size']:raise ValueError('source size mismatch')
    output.mkdir(parents=True);(output/'assets').mkdir();(output/'source').mkdir();shutil.copyfile(source,output/'source/page.png')
    scale=1280/image.width;nodes=[];evidence=[];assets=[];decisions={};corrections=[];font_proposals={}
    for obj in plan['objects']:
        eid='ev-'+obj['id'];evidence.append({'id':eid,'kind':'shape' if obj['kind']=='native_shape' else 'image','bbox':obj['bbox'],'description':obj.get('label',obj['id'])})
        node={'id':obj['id'],'type':'shape' if obj['kind']=='native_shape' else 'image','bbox':obj['bbox'],'z':obj['z'],'evidence_ids':[eid]}
        if obj['kind']=='native_shape':
            for k in ('geometry','fill','line','corner_radius_px','fill_gradient'):
                if k in obj:node[k]=obj[k]
        else:
            x,y,w,h=obj['bbox'];bounds=(math.floor(x),math.floor(y),math.ceil(x+w),math.ceil(y+h));crop=image.crop(bounds)
            # Preserve interior white glyphs. No global color key / text erasure.
            if obj.get('mask')=='ellipse':
                mask=Image.new('L',crop.size,0);ImageDraw.Draw(mask).ellipse((0,0,crop.width-1,crop.height-1),fill=255);crop.putalpha(mask)
            relative='assets/'+obj['id']+'.png';crop.save(output/relative)
            node['asset_id']=obj['id'];node['evidence_ids']=[]
            assets.append({'id':obj['id'],'path':relative,'purpose':'module','sha256':digest(output/relative),'contains_evidence_ids':[eid]+obj.get('contains_text_ids',[]),'description':'Independent raster crop; internal pixels are not native editable objects.'})
        nodes.append(node);decisions[node['id']]='Model object proposal; source fidelity requires independent rendered review.'
    ocr={t['id']:t for t in inventory['texts']}
    for t in plan['text_decisions']:
        old=ocr[t['id']];evidence.append({'id':t['id'],'kind':'text','bbox':old['bbox'],'text':old['text']})
        if t['action']=='retained_in_asset':continue
        node={'id':t['id'],'type':'text','bbox':t['bbox'],'z':t['z'],'evidence_ids':[t['id']],'text':t['text'],'font':{'latin':'Microsoft YaHei','ea':'Microsoft YaHei','size_pt':round(t['font_size_px']*scale*.75,4)},'bold':t['bold'],'color':t['color'],'align':t.get('align','left'),'wrap':False,'source_text_bbox':old['bbox']}
        if t['id'] in resolved_fonts:
            node['font'].update(resolved_fonts[t['id']])
            font_proposals[t['id']]={'requested_family':t['font_family'],'requested_latin':t.get('font_latin',t['font_family']),'confidence':t['font_confidence'],'evidence':t['font_evidence'],'font':deepcopy(node['font']),'bold':t['bold'],'status':'visual_candidate_not_verified_original'}
        if t['text']!=old['text']:
            node['text_override_reason']='Vision planner correction against original; independent supervisor must verify.'
            corrections.append({'id':t['id'],'ocr':old['text'],'proposed':t['text']})
        if t.get('runs'):node['runs']=t['runs']
        nodes.append(node);decisions[node['id']]='Source OCR with explicit vision-planner typography and correction proposal.'
    nodes.sort(key=lambda n:n['z'])
    for z,node in enumerate(nodes):node['z']=z
    limitations=plan.get('limitations',[])
    ir={'schema_version':'0.1','source_image':'source/page.png','source_sha256':digest(source),'canvas':{'source_px':list(image.size),'target_px':[1280,round(image.height*scale)]},'background':{'color':plan['background_color']},'review_status':'needs_review','review_notes':['Experimental supervised element reconstruction; no page residual bitmap.','Native emission does not prove geometric accuracy. Raster icons remain raster.',*limitations],'evidence':evidence,'nodes':nodes,'assets':assets,'groups':[]}
    write_json(output/'slideir.json',ir);write_json(output/'element-plan.json',plan)
    report={'schema':'ElementReconstruction/1','model_id':model_id,'response_sha256':response_sha256,'source_sha256':digest(source),'native_shapes':sum(n['type']=='shape' for n in nodes),'native_text':sum(n['type']=='text' for n in nodes),'raster_assets':len(assets),'whole_page_residual':False,'text_corrections':corrections,'fidelity_verified':False,'human_acceptance':False,'limitations':limitations}
    write_json(output/'reconstruction.json',report)
    write_json(output/'authoring.json',{'format_version':'1','tool':'supervised-element-experiment','model_id':model_id,'created_at':datetime.now(timezone.utc).isoformat(),'ir_sha256':digest(output/'slideir.json'),'node_decisions':decisions,'review_notes':ir['review_notes'],'unresolved_ids':[],'model_typography':font_proposals})
    return report
