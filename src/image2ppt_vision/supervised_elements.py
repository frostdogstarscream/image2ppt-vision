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
from .font_contract import validate_font_proposal, resolve_font, family_name
from .typography import font_status

GEOMETRIES={'rect','roundRect','ellipse','line','chevron','pentagon','rightArrow','leftArrow','trapezoid','parallelogram','hexagon','diamond','triangle'}
CHART_TYPES={'bar','line','area','pie','doughnut'}
LEGEND_POSITIONS={'left','top','topRight','right','bottom'}


def _finite_number(value,minimum=None,maximum=None):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        return False
    return (minimum is None or value>=minimum) and (maximum is None or value<=maximum)


def _native_font(obj,scale):
    spec=obj.get('font')
    if not isinstance(spec,dict) or set(spec)-{'latin','ea','size_px'}:
        raise ValueError('native table/chart requires font latin/ea/size_px')
    family=family_name(spec.get('ea'));latin=family_name(spec.get('latin'))
    if family not in font_status()['families'] or latin not in font_status()['families']:
        raise ValueError('font unavailable in verified catalog')
    if not _finite_number(spec.get('size_px'),.1,400):
        raise ValueError('invalid native font size')
    return {'latin':latin,'ea':family,'size_pt':round(spec['size_px']*scale*.75,4)}

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
        elif obj['kind']=='table':
            rows=obj.get('rows')
            if not isinstance(rows,list) or not 1<=len(rows)<=2000:raise ValueError('table rows must be 1-2000')
            if not isinstance(rows[0],list):raise ValueError('invalid table shape')
            cols=len(rows[0])
            if not 1<=cols<=100 or any(not isinstance(row,list) or len(row)!=cols for row in rows):raise ValueError('invalid table shape')
            for row in rows:
                for cell in row:
                    if isinstance(cell,bool) or not ((isinstance(cell,str) and len(cell)<=2048) or _finite_number(cell)):raise ValueError('invalid table cell')
            _native_font(obj,1280/size[0])
            if obj.get('column_tracks') is not None:
                tracks=obj['column_tracks']
                if not isinstance(tracks,list) or len(tracks)!=cols:raise ValueError('invalid column tracks')
                for track in tracks:
                    if not isinstance(track,dict) or track.get('mode') not in ('fr','fixed') or not _finite_number(track.get('value'),.0001):raise ValueError('invalid column track')
            for key in ('header_row','banded_rows'):
                if key in obj and not isinstance(obj[key],bool):raise ValueError('invalid table style flag')
            if obj.get('row_heights') is not None:
                heights=obj['row_heights']
                if (not isinstance(heights,list) or len(heights)!=len(rows) or
                        any(not _finite_number(value,.0001) for value in heights)):raise ValueError('invalid row heights')
                target_height=round(size[1]*1280/size[0])
                if abs(sum(heights)-obj['bbox'][3]*target_height/size[1])>.01:raise ValueError('row heights sum mismatch')
            occupied=set()
            for merge in obj.get('merges',[]):
                if not isinstance(merge,dict) or any(not isinstance(merge.get(k),int) or isinstance(merge.get(k),bool) for k in ('row','column','row_span','column_span')):raise ValueError('invalid table merge')
                if not (merge['row']>=0 and merge['column']>=0 and merge['row_span']>=1 and merge['column_span']>=1 and merge['row_span']*merge['column_span']>1 and merge['row']+merge['row_span']<=len(rows) and merge['column']+merge['column_span']<=cols):raise ValueError('table merge outside bounds')
                for row_index in range(merge['row'],merge['row']+merge['row_span']):
                    for column_index in range(merge['column'],merge['column']+merge['column_span']):
                        key=(row_index,column_index)
                        if key in occupied:raise ValueError('overlapping table merges')
                        occupied.add(key)
                        if (row_index,column_index)!=(merge['row'],merge['column']) and rows[row_index][column_index]!='':raise ValueError('merge covers non-empty cell')
            styled=set()
            for style in obj.get('cell_styles',[]):
                if not isinstance(style,dict) or any(not isinstance(style.get(k),int) or isinstance(style.get(k),bool) for k in ('row','column')):raise ValueError('invalid cell style')
                row_index,column_index=style['row'],style['column']
                if not (0<=row_index<len(rows) and 0<=column_index<cols) or (row_index,column_index) in styled:raise ValueError('cell style outside bounds or duplicate')
                styled.add((row_index,column_index))
                if 'fill' in style:_color(style['fill'])
                if 'color' in style:_color(style['color'])
                if 'bold' in style and not isinstance(style['bold'],bool):raise ValueError('invalid cell bold')
        elif obj['kind']=='chart':
            if obj.get('chart_type') not in CHART_TYPES:raise ValueError('unsupported chart type')
            title=obj.get('title')
            if title is not None and (not isinstance(title,str) or not title.strip() or title.strip().lower().replace(' ','')=='charttitle'):raise ValueError('invalid chart title')
            categories=obj.get('categories')
            if not isinstance(categories,list) or not 1<=len(categories)<=1000 or any(not isinstance(c,str) or len(c)>2000 for c in categories):raise ValueError('invalid chart categories')
            series=obj.get('series')
            if not isinstance(series,list) or not 1<=len(series)<=128:raise ValueError('invalid chart series')
            points=0
            for item in series:
                if not isinstance(item,dict) or not isinstance(item.get('name'),str) or not item['name'].strip() or len(item['name'])>2000:raise ValueError('invalid chart series name')
                values=item.get('values')
                if not isinstance(values,list) or len(values)!=len(categories) or any(not (v is None or _finite_number(v)) for v in values):raise ValueError('invalid chart values')
                points+=len(values)
            if points>100000:raise ValueError('chart has too many points')
            _native_font(obj,1280/size[0])
            if 'has_legend' in obj and not isinstance(obj['has_legend'],bool):raise ValueError('invalid legend flag')
            if 'legend_position' in obj and obj['legend_position'] not in LEGEND_POSITIONS:raise ValueError('invalid legend position')
        else:raise ValueError('invalid object kind')
    for t in texts:
        _box(t['bbox'],size);_color(t['color'])
        if not isinstance(t['text'],str) or not t['text'].strip() or len(t['text'])>5000:raise ValueError('invalid text')
        if t['action'] not in ('native','retained_in_asset') or (t['action']=='retained_in_asset')!=(t['id'] in retained):raise ValueError('text retention mismatch')
        if not isinstance(t['bold'],bool) or not isinstance(t['z'],int) or not 1<=t['font_size_px']<=400:raise ValueError('invalid typography')
        if type(t['font_size_px']) not in (int,float) or not math.isfinite(t['font_size_px']):raise ValueError('invalid typography')
        if t['action']=='native':validate_font_proposal(t,required=plan['schema']=='ElementPlan/2')
        if t.get('runs') and ''.join(r['text'] for r in t['runs'])!=t['text']:raise ValueError('run text mismatch')
    groups=plan.get('groups',[])
    if not isinstance(groups,list) or len(groups)>100:raise ValueError('invalid groups')
    order_ids=[item['id'] for item in sorted(
        objects+[t for t in texts if t['action']=='native'],key=lambda item:item['z'])]
    allowed=set(order_ids);group_ids=set();grouped=set()
    for group in groups:
        if not isinstance(group,dict) or set(group)-{'id','member_ids','label'}:raise ValueError('invalid group fields')
        group_id=group.get('id');members=group.get('member_ids')
        if not isinstance(group_id,str) or not re.fullmatch('[A-Za-z0-9_-]{1,70}',group_id) or group_id in ids or group_id in group_ids:raise ValueError('invalid or duplicate group id')
        if not isinstance(members,list) or not 2<=len(members)<=500 or len(members)!=len(set(members)):raise ValueError('invalid group members')
        if any(member not in allowed for member in members):raise ValueError('group member missing')
        if any(member in grouped for member in members):raise ValueError('member belongs to multiple groups')
        indexes=sorted(order_ids.index(member) for member in members)
        if indexes[-1]-indexes[0]+1!=len(indexes):raise ValueError('group members are not contiguous in z order')
        grouped.update(members);group_ids.add(group_id)
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
        eid='ev-'+obj['id'];kind='image' if obj['kind']=='raster_asset' else 'shape'
        evidence.append({'id':eid,'kind':kind,'bbox':obj['bbox'],'description':obj.get('label',obj['id'])})
        node_type={'native_shape':'shape','raster_asset':'image','table':'table','chart':'chart'}[obj['kind']]
        node={'id':obj['id'],'type':node_type,'bbox':obj['bbox'],'z':obj['z'],'evidence_ids':[eid]}
        if obj['kind']=='native_shape':
            for k in ('geometry','fill','line','corner_radius_px','fill_gradient'):
                if k in obj:node[k]=obj[k]
        elif obj['kind']=='table':
            node.update(rows=deepcopy(obj['rows']),font=_native_font(obj,scale))
            for k in ('column_tracks','header_row','banded_rows','row_heights','merges','cell_styles'):
                if k in obj:node[k]=deepcopy(obj[k])
        elif obj['kind']=='chart':
            node.update(chart_type=obj['chart_type'],font=_native_font(obj,scale),
                        categories=deepcopy(obj['categories']),series=deepcopy(obj['series']))
            for k in ('title','has_legend','legend_position'):
                if k in obj:node[k]=deepcopy(obj[k])
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
    ir={'schema_version':'0.1','source_image':'source/page.png','source_sha256':digest(source),'canvas':{'source_px':list(image.size),'target_px':[1280,round(image.height*scale)]},'background':{'color':plan['background_color']},'review_status':'needs_review','review_notes':['Experimental supervised element reconstruction; no page residual bitmap.','Native emission does not prove geometric accuracy. Raster icons remain raster.',*limitations],'evidence':evidence,'nodes':nodes,'assets':assets,'groups':deepcopy(plan.get('groups',[]))}
    write_json(output/'slideir.json',ir);write_json(output/'element-plan.json',plan)
    report={'schema':'ElementReconstruction/1','model_id':model_id,'response_sha256':response_sha256,'source_sha256':digest(source),'native_shapes':sum(n['type']=='shape' for n in nodes),'native_text':sum(n['type']=='text' for n in nodes),'native_tables':sum(n['type']=='table' for n in nodes),'native_charts':sum(n['type']=='chart' for n in nodes),'groups':len(ir['groups']),'raster_assets':len(assets),'whole_page_residual':False,'text_corrections':corrections,'fidelity_verified':False,'human_acceptance':False,'limitations':limitations}
    write_json(output/'reconstruction.json',report)
    write_json(output/'authoring.json',{'format_version':'1','tool':'supervised-element-experiment','model_id':model_id,'created_at':datetime.now(timezone.utc).isoformat(),'ir_sha256':digest(output/'slideir.json'),'node_decisions':decisions,'review_notes':ir['review_notes'],'unresolved_ids':[],'model_typography':font_proposals})
    return report
