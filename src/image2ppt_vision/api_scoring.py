"""Independent post-generation OOXML/visual scoring. Never calls a model API."""
from collections import Counter
from copy import copy
from difflib import SequenceMatcher
import json
import hashlib
import math
from pathlib import Path
import posixpath
import re
import shutil
import subprocess
import sys
import unicodedata
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image
from pydantic import Field, StrictInt, StrictStr, ValidationError
from typing import Literal

from .plans import _Record
from .storage import digest,read_json,write_json
from .api_evaluation import verify_generation, EvaluationError
from . import geometry_scoring


NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
for prefix,uri in NS.items():ET.register_namespace(prefix,uri)


class ReferenceSuite(_Record):
    schema_name:Literal['APIReferenceSuite']=Field(default='APIReferenceSuite',alias='schema')
    version:Literal['1.0']='1.0'
    reference_pptx:StrictStr
    reference_sha256:StrictStr
    mapping:dict[StrictStr,StrictInt]


def normalize(text):
    return re.sub(r'\s+','',unicodedata.normalize('NFKC',text or ''))


def _text(element):
    return '' if element is None else '\n'.join(
        ''.join(t.text or '' for t in paragraph.findall('.//a:t',NS)) for paragraph in element.findall('a:p',NS))


def _rels(z,owner):
    path=posixpath.join(posixpath.dirname(owner),'_rels',posixpath.basename(owner)+'.rels')
    if path not in z.namelist():return {}
    return {e.get('Id'):posixpath.normpath(posixpath.join(posixpath.dirname(owner),e.get('Target'))).lstrip('/')
            for e in ET.fromstring(z.read(path)) if e.get('TargetMode')!='External'}


def _transform(el,path,parent):
    t=el.find(path,NS)
    if t is None:return None,parent
    off=t.find('a:off',NS);ext=t.find('a:ext',NS)
    if off is None or ext is None:return None,parent
    x,y=float(off.get('x')),float(off.get('y'));w,h=float(ext.get('cx')),float(ext.get('cy'))
    local=np.array([[1.,0,x],[0,1.,y],[0,0,1.]])
    angle=float(t.get('rot','0'))/60000*math.pi/180
    cosine,sine=math.cos(angle),math.sin(angle)
    rotate=np.array([[cosine,-sine,0],[sine,cosine,0],[0,0,1.]])
    center=np.array([[1.,0,w/2],[0,1.,h/2],[0,0,1.]])
    flip=np.diag([-1. if t.get('flipH') in ('1','true') else 1.,-1. if t.get('flipV') in ('1','true') else 1.,1.])
    local=local@center@rotate@flip@np.linalg.inv(center)
    full=parent@local
    corners=full@np.array([[0,w,w,0],[0,0,h,h],[1,1,1,1]])
    bbox=[float(corners[0].min()),float(corners[1].min()),float(np.ptp(corners[0])),float(np.ptp(corners[1]))]
    ch_off=t.find('a:chOff',NS);ch_ext=t.find('a:chExt',NS)
    if ch_off is not None and ch_ext is not None:
        cw,ch=float(ch_ext.get('cx')),float(ch_ext.get('cy'))
        if cw<=0 or ch<=0:return bbox,full
        children=full@np.array([[w/cw,0,-float(ch_off.get('x'))*w/cw],[0,h/ch,-float(ch_off.get('y'))*h/ch],[0,0,1.]])
    else:children=full
    return bbox,children


def inspect_objects(file):
    pages=[]
    with ZipFile(file) as z:
        if len(z.namelist())>10000 or sum(i.file_size for i in z.infolist())>512_000_000:raise EvaluationError('pptx_too_large')
        if z.testzip() is not None:raise EvaluationError('pptx_crc_failed')
        presentation=ET.fromstring(z.read('ppt/presentation.xml'));size=presentation.find('p:sldSz',NS)
        width,height=float(size.get('cx')),float(size.get('cy'))
        rels=_rels(z,'ppt/presentation.xml')
        parts=[rels[e.get('{'+NS['r']+'}id')] for e in presentation.findall('p:sldIdLst/p:sldId',NS)]
        for index,part in enumerate(parts,1):
            tree=ET.fromstring(z.read(part));objects=[]
            def walk(parent,matrix,group=None):
                for e in parent:
                    kind=e.tag.rsplit('}',1)[-1]
                    if kind not in ('sp','pic','grpSp','graphicFrame','cxnSp'):continue
                    nv=e.find('.//p:cNvPr',NS)
                    if nv is None:continue
                    path='p:grpSpPr/a:xfrm' if kind=='grpSp' else 'p:xfrm' if kind=='graphicFrame' else 'p:spPr/a:xfrm'
                    box,child=_transform(e,path,matrix)
                    body=e.find('p:txBody',NS);text=_text(body)
                    table=e.find('.//a:tbl',NS);geometry=e.find('p:spPr/a:prstGeom',NS)
                    custom=e.find('p:spPr/a:custGeom',NS)
                    typ='group' if kind=='grpSp' else 'image' if kind=='pic' else 'table' if table is not None else 'text' if normalize(text) else 'shape'
                    cells=[];merges=[];cell_objects=[]
                    if table is not None:
                        tracks=[float(c.get('w')) for c in table.findall('a:tblGrid/a:gridCol',NS)]
                        heights=[float(r.get('h','0')) for r in table.findall('a:tr',NS)]
                        for r,row in enumerate(table.findall('a:tr',NS)):
                            values=[]
                            for c,cell in enumerate(row.findall('a:tc',NS)):
                                value=_text(cell.find('a:txBody',NS));values.append(value)
                                attributes={k:cell.get(k) for k in ('gridSpan','rowSpan') if int(cell.get(k,'1'))>1}
                                attributes.update({k:'1' for k in ('hMerge','vMerge') if cell.get(k) in ('1','true')})
                                if attributes:merges.append({'row':r,'column':c,**attributes})
                                if normalize(value) and box is not None and tracks and sum(tracks)>0 and sum(heights)>0 and not any(attributes.get(k)=='1' for k in ('hMerge','vMerge')):
                                    cs=int(cell.get('gridSpan','1'));rs=int(cell.get('rowSpan','1'))
                                    cell_objects.append({'id':nv.get('id')+f':cell:{r}:{c}','type':'text','text':value,
                                        'bbox':[(box[0]+box[2]*sum(tracks[:c])/sum(tracks))/width,
                                                (box[1]+box[3]*sum(heights[:r])/sum(heights))/height,
                                                box[2]*sum(tracks[c:c+cs])/sum(tracks)/width,
                                                box[3]*sum(heights[r:r+rs])/sum(heights)/height]})
                            cells.append(values)
                    # Transparent empty primitives have no semantic drawing role.
                    props=e.find('p:spPr',NS)
                    invisible=typ=='shape' and props is not None and props.find('a:noFill',NS) is not None and (
                        props.find('a:ln',NS) is None or props.find('a:ln/a:noFill',NS) is not None)
                    objects.append({'id':nv.get('id'),'name':nv.get('name'),'type':typ,'parent_group':group,
                        'bbox':None if box is None else [box[0]/width,box[1]/height,box[2]/width,box[3]/height],
                        'text':text,'geometry':'custom' if custom is not None else geometry.get('prst') if geometry is not None else None,
                        'cells':cells,'merges':merges,'cell_objects':cell_objects,
                        'custom_geometry_sha256':hashlib.sha256(ET.tostring(custom)).hexdigest() if custom is not None else None,
                        'custom_geometry_signature':geometry_scoring.custom_signature(props,child,[width,height]) if custom is not None else None,
                        'ignored_nonvisual':invisible,'z':len(objects)})
                    if kind=='grpSp':walk(e,child,nv.get('id'))
            walk(tree.find('p:cSld/p:spTree',NS),np.eye(3))
            pages.append({'slide':index,'part':part,'objects':objects})
    return {'sha256':digest(file),'size_emu':[width,height],'slides':pages}


def _box_near(a,b,tolerance=.01):
    return a is not None and b is not None and all(abs(x-y)<=tolerance for x,y in zip(a,b))


def match_objects(reference,actual,kind):
    expected=[o for o in reference if o['type']==kind and not o['ignored_nonvisual']]
    observed=[o for o in actual if o['type']==kind and not o['ignored_nonvisual']]
    edges={}
    for i,r in enumerate(expected):
        candidates=[]
        for j,a in enumerate(observed):
            if not _box_near(r['bbox'],a['bbox']):continue
            if kind=='shape' and r['geometry']!=a['geometry']:continue
            if kind=='shape' and r['geometry']=='custom' and not geometry_scoring.equivalent(r.get('custom_geometry_signature'),a.get('custom_geometry_signature')):continue
            similarity=SequenceMatcher(None,normalize(r['text']),normalize(a['text']),autojunk=False).ratio() if kind=='text' else 1
            if similarity>=.95:candidates.append((similarity,j))
        edges[i]=[j for _,j in sorted(candidates,reverse=True)]
    owners={}
    def assign(i,seen):
        for j in edges[i]:
            if j in seen:continue
            seen.add(j)
            if j not in owners or assign(owners[j],seen):owners[j]=i;return True
        return False
    for i in sorted(edges,key=lambda i:(len(edges[i]),i)):assign(i,set())
    pairs=[{'reference':expected[i]['id'],'actual':observed[j]['id']} for j,i in sorted(owners.items())]
    precision=len(pairs)/len(observed) if observed else (1. if not expected else 0.)
    recall=len(pairs)/len(expected) if expected else 1.
    threshold=.95 if kind=='text' else .90 if kind=='shape' else 1.
    unresolved=sum(o['bbox'] is None or (kind=='shape' and o['geometry']=='custom' and not o.get('custom_geometry_signature')) for o in expected+observed)
    return {'reference_count':len(expected),'actual_count':len(observed),'matched':len(pairs),
            'precision':precision,'recall':recall,'pairs':pairs,
            'unmatched_reference':[r['id'] for i,r in enumerate(expected) if i not in set(owners.values())],
            'unmatched_actual':[a['id'] for j,a in enumerate(observed) if j not in owners],
            'status':'NOT_RUN_UNRESOLVED_GEOMETRY' if unresolved else 'PASS' if precision>=threshold and recall>=threshold else 'FAIL'}


def _overlap(a,b):
    return max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))*max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))


def split_merge_diagnostics(reference,actual):
    result={'possible_splits':[],'possible_merges':[]}
    for left,right,key in ((reference,actual,'possible_splits'),(actual,reference,'possible_merges')):
        for obj in left:
            if obj['type']!='text' or obj['bbox'] is None:continue
            parts=[o for o in right if o['type']=='text' and o['bbox'] is not None and _overlap(obj['bbox'],o['bbox'])>0]
            parts.sort(key=lambda o:(o['bbox'][1],o['bbox'][0]))
            if len(parts)>1 and normalize(obj['text'])==normalize(''.join(o['text'] for o in parts)):
                result[key].append({'id':obj['id'],'parts':[o['id'] for o in parts]})
    return result


def image_difference(source,preview,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with Image.open(source) as a,Image.open(preview) as b:
        if abs(a.width/a.height*b.height-b.width)>1:raise EvaluationError('preview_aspect_mismatch')
        aligned=a.convert('RGB').resize(b.size,Image.Resampling.LANCZOS);final=b.convert('RGB')
        difference=np.abs(np.array(aligned,dtype=float)-np.array(final,dtype=float))
        Image.fromarray(difference.astype('uint8')).save(output/'difference.png')
        aligned.save(output/'source-aligned.png')
    h,w=difference.shape[:2];tiles=[]
    for row in range(4):
        for col in range(4):
            x0,x1=col*w//4,(col+1)*w//4;y0,y1=row*h//4,(row+1)*h//4
            tiles.append({'bbox':[x0,y0,x1-x0,y1-y0],'mean_absolute_rgb':float(difference[y0:y1,x0:x1].mean())})
    mean=float(difference.mean())
    return {'mean_absolute_rgb':mean,'threshold':10.,'status':'PASS' if mean<=10 else 'FAIL',
            'tiles':tiles,'source_resampling':'Lanczos to actual-file preview size','semantic_acceptance':False}


def _render(compiler_root,pptx,output,timeout=120):
    from .conversion import _compiler,_final_json
    root,node=_compiler(compiler_root)
    js="import {pathToFileURL} from 'node:url';const {renderPptx}=await import(pathToFileURL(process.argv[1]));console.log(JSON.stringify(await renderPptx(process.argv[2],process.argv[3])));"
    command=[str(node),'--input-type=module','-e',js,str(root/'scripts/render.mjs'),str(pptx),str(output)]
    try:response=subprocess.run(command,cwd=root,capture_output=True,timeout=timeout)
    except subprocess.TimeoutExpired:raise EvaluationError('evaluation_render_timeout') from None
    output=Path(output)
    (output.parent/(output.name+'.stderr.log')).write_bytes(response.stderr)
    if response.returncode:raise EvaluationError('evaluation_render_failed')
    return _final_json(response.stdout)


def render_reference_pages(compiler_root,pptx,output,slides):
    """Evaluation-only copies isolate slow reference slides, retaining their dependencies.

    Unselected slide parts are removed from evaluation copies. Other parts and
    namespaces are preserved. These bytes stay outside generation inputs.
    """
    from lxml import etree as xml
    output=Path(output);output.mkdir(parents=True,exist_ok=False);before=digest(pptx);pages={}
    with ZipFile(pptx) as source:
        for index in sorted(set(slides)):
            folder=output/f'slide-{index:03}';folder.mkdir()
            tree=xml.fromstring(source.read('ppt/presentation.xml'));listing=tree.find('p:sldIdLst',NS)
            if not 1<=index<=len(listing):raise EvaluationError('reference_slide_missing')
            for i,item in enumerate(list(listing),1):
                if i!=index:listing.remove(item)
            rel_part='ppt/_rels/presentation.xml.rels';rels=xml.fromstring(source.read(rel_part))
            selected_id=listing[0].get('{'+NS['r']+'}id');removed=set()
            for rel in list(rels):
                if rel.get('Type','').endswith('/slide') and rel.get('Id')!=selected_id:
                    part=posixpath.normpath(posixpath.join('ppt',rel.get('Target'))).lstrip('/')
                    removed.update((part,posixpath.join(posixpath.dirname(part),'_rels',posixpath.basename(part)+'.rels')))
                    rels.remove(rel)
            types=xml.fromstring(source.read('[Content_Types].xml'))
            for entry in list(types):
                if entry.get('PartName','').lstrip('/') in removed:types.remove(entry)
            replacements={'ppt/presentation.xml':tree,rel_part:rels,'[Content_Types].xml':types}
            selected=folder/'evaluation-only.pptx'
            with ZipFile(selected,'x',compression=ZIP_DEFLATED) as target:
                for entry in source.infolist():
                    if entry.filename not in removed:target.writestr(copy(entry),xml.tostring(replacements[entry.filename],encoding='utf-8',xml_declaration=True) if entry.filename in replacements else source.read(entry.filename))
            try:
                result=_render(compiler_root,selected,folder/'render')
                pages[index]={'status':'RENDERED','path':result['pages'][0]['path'],'reference_slide':index,'method':'isolated_slide_list_copy'}
            except Exception as exc:
                pages[index]={'status':'NOT_RUN_REFERENCE_RENDER','error_code':str(exc) if isinstance(exc,EvaluationError) else 'reference_render_failed','reference_slide':index}
            write_json(folder/'result.json',pages[index])
    if digest(pptx)!=before:raise EvaluationError('reference_changed')
    return pages


def _reference(config):
    config=Path(config).resolve()
    try:spec=ReferenceSuite.model_validate(read_json(config))
    except (ValidationError,ValueError):raise EvaluationError('invalid_reference_suite') from None
    file=(config.parent/spec.reference_pptx).resolve()
    if digest(file)!=spec.reference_sha256:raise EvaluationError('reference_hash_mismatch')
    if len(set(spec.mapping.values()))!=len(spec.mapping) or any(v<1 for v in spec.mapping.values()):raise EvaluationError('invalid_reference_mapping')
    return spec,file


def prepare_reference(config,generation_config,output,compiler_root):
    """Read-only reference intake, with its own output; no data fed to generation."""
    from .api_evaluation import _suite
    spec,file=_reference(config);suite=_suite(generation_config)
    if set(spec.mapping)!={p.id for p in suite.pages}:raise EvaluationError('reference_mapping_incomplete')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    objects=inspect_objects(file);write_json(out/'objects.json',objects)
    render=render_reference_pages(compiler_root,file,out/'render',spec.mapping.values());write_json(out/'render.json',render)
    rows=[]
    for page in suite.pages:
        index=spec.mapping[page.id]
        if index>len(objects['slides']):raise EvaluationError('reference_slide_missing')
        source=(Path(generation_config).resolve().parent/page.source).resolve()
        if digest(source)!=page.source_sha256:raise EvaluationError('input_hash_mismatch')
        difference=image_difference(source,render[index]['path'],out/page.id) if render[index]['status']=='RENDERED' else render[index]
        rows.append({'id':page.id,'reference_slide':index,'source_sha256':page.source_sha256,
                     'visual_reference_difference':difference,
                     'types':dict(Counter(o['type'] for o in objects['slides'][index-1]['objects'] if not o['ignored_nonvisual']))})
    report={'status':'REVIEW_REQUIRED' if all(r['status']=='RENDERED' for r in render.values()) else 'INCOMPLETE_REFERENCE_RENDER','reference_sha256':spec.reference_sha256,'mapping':spec.mapping,
            'pages':rows,'reference_only_for_evaluation':True,'human_acceptance':False}
    write_json(out/'reference-report.json',report);return report


def _distance(a,b):
    if len(a)<len(b):a,b=b,a
    row=list(range(len(b)+1))
    for i,c in enumerate(a,1):
        nxt=[i]
        for j,d in enumerate(b,1):nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(c!=d)))
        row=nxt
    return row[-1]


def content_metrics(reference,actual,analysis,preview):
    """Only source-OCR-confirmed reference strings enter the content denominator.

    Native-text correctness is reported independently from rendered OCR. Missing
    OCR weights never become a content PASS or an implicit network download.
    """
    w,h=analysis['image_size'];source=[{**t,'bbox':[t['bbox'][0]/w,t['bbox'][1]/h,t['bbox'][2]/w,t['bbox'][3]/h]}
                                      for t in analysis['objects'] if t['kind']=='text']
    expected=[];uncertain=[]
    reference_text=[o for o in reference if o['type']=='text']+[c for o in reference if o['type']=='table' for c in o.get('cell_objects',[])]
    for obj in reference_text:
        if obj['type']!='text' or not normalize(obj['text']) or obj['bbox'] is None:continue
        nearby=sorted([s for s in source if _overlap(obj['bbox'],s['bbox'])>0],key=lambda s:(s['bbox'][1],s['bbox'][0]))
        if normalize(obj['text'])==normalize(''.join(s['text'] for s in nearby)):expected.append(obj)
        else:uncertain.append(obj['id'])
    def compare(observed):
        errors=0;total=0;numeric=[];rows=[]
        for obj in expected:
            candidates=sorted([s for s in observed if s['bbox'] is not None and _overlap(obj['bbox'],s['bbox'])>0],key=lambda s:(s['bbox'][1],s['bbox'][0]))
            a=normalize(obj['text']);b=normalize(''.join(s['text'] for s in candidates))
            # Extra words or numbers in a confirmed region are content errors too.
            exact=a==b
            error=0 if exact else min(len(a),_distance(a,b))
            errors+=error;total+=len(a)
            numbers=re.findall(r'\d+(?:[.,]\d+)*',a)
            actual_numbers=re.findall(r'\d+(?:[.,]\d+)*',b)
            numeric.append(numbers==actual_numbers)
            rows.append({'reference':obj['id'],'character_errors':error,'characters':len(a),'numeric_exact':numeric[-1]})
        accuracy=1-errors/total if total else None
        return {'character_accuracy':accuracy,'numeric_exact':all(numeric) if total else None,'characters':total,
                'status':'NOT_RUN_NO_SOURCE_CONFIRMED_TEXT' if not total else 'PASS' if accuracy>=.995 and all(numeric) else 'FAIL','objects':rows}
    native=compare([a for a in actual if a['type']=='text']+[c for o in actual if o['type']=='table' for c in o.get('cell_objects',[])])
    try:
        from .models import recognize,verified_models
        manifest=verified_models()
        with Image.open(preview) as image:
            pw,ph=image.size;ocr=recognize(np.array(image.convert('RGB')))
        rendered=compare([{**t,'bbox':[t['bbox'][0]/pw,t['bbox'][1]/ph,t['bbox'][2]/pw,t['bbox'][3]/ph]} for t in ocr])
        rendered['method']='local OCR on final-PPTX render, an OCR proxy not human acceptance'
        rendered['ocr_manifest_hash']=__import__('hashlib').sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    except Exception:
        rendered={'status':'NOT_RUN_OCR_UNAVAILABLE','character_accuracy':None,'numeric_exact':None}
    return {'source_confirmed_reference_texts':len(expected),'unconfirmed_reference_ids':uncertain,
            'native_content':native,'rendered_content':rendered}


def editing_probes(pptx,objects,output,compiler_root):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    original_hash=digest(pptx);results=[]
    # Core helper operates on diagnostic copies and checks OOXML invariants.
    core=Path(compiler_root)/'src'
    sys.path.insert(0,str(core))
    try:
        from image2ppt.quality import edit_copy
        from image2ppt.contracts import ProbeCase
        for kind,action in (('text','delete_text'),('group','move_group')):
            candidates=[o for o in objects if o['type']==kind and o['bbox'] is not None]
            if not candidates:
                results.append({'action':action,'status':'NOT_RUN_NO_NATIVE_OBJECT'});continue
            chosen=sorted(candidates,key=lambda o:o['bbox'][2]*o['bbox'][3])[len(candidates)//2]
            dest=output/(action+'.pptx')
            case=ProbeCase(id=action,action=action,targets=[chosen['name']],**({'delta_px':[24.,16.]} if kind=='group' else {}))
            try:
                edit_copy(pptx,dest,case)
                rendered=_render(compiler_root,dest,output/(action+'-render'))
                results.append({'action':action,'status':'PASS','target':chosen['name'],'render':rendered,
                                'semantic_visual':'REVIEW_REQUIRED'})
            except Exception:results.append({'action':action,'status':'FAIL','error_code':'edit_probe_failed'})
        tables=[o for o in objects if o['type']=='table']
        for n,table in enumerate(tables):
            dest=output/f'table-{n:03}.pptx'
            try:
                with ZipFile(pptx) as original:
                    part='ppt/slides/slide1.xml';tree=ET.fromstring(original.read(part))
                    element=next(e for e in tree.findall('.//p:graphicFrame',NS) if e.find('.//p:cNvPr',NS).get('name')==table['name'])
                    first=element.find('.//a:tbl/a:tr/a:tc/a:txBody/a:p/a:r/a:t',NS)
                    if first is None:raise EvaluationError('table_cell_missing')
                    first.text='EDIT-PROBE'
                    with ZipFile(dest,'x',compression=ZIP_DEFLATED) as target:
                        for entry in original.infolist():target.writestr(entry,ET.tostring(tree,encoding='utf-8',xml_declaration=True) if entry.filename==part else original.read(entry.filename))
                actual=inspect_objects(dest)['slides'][0]['objects']
                changed=next(o for o in actual if o['name']==table['name'])
                if not any('EDIT-PROBE' in cell for row in changed['cells'] for cell in row):raise EvaluationError('cell_edit_failed')
                rendered=_render(compiler_root,dest,output/f'table-{n:03}-render')
                results.append({'action':'edit_table_cell','target':table['name'],'status':'PASS','render':rendered,'semantic_visual':'REVIEW_REQUIRED'})
            except Exception:results.append({'action':'edit_table_cell','status':'FAIL','target':table['name']})
    finally:sys.path.remove(str(core))
    if digest(pptx)!=original_hash:raise EvaluationError('probe_modified_original')
    return {'original_preserved':True,'method':'direct_final_pptx_copy','cases':results,'human_acceptance':False}


def _render_source_resolution(compiler_root,pptx,source,preview,output):
    from .conversion import _compiler,_final_json
    with Image.open(source) as image:source_size=image.size
    with Image.open(preview) as image:preview_size=image.size
    if source_size==preview_size:return preview
    root,node=_compiler(compiler_root);before=digest(pptx)
    try:
        result=subprocess.run([str(node),str(root/'scripts/render_for_ocr.mjs'),str(pptx),str(output),
                               str(source_size[0]/preview_size[0])],cwd=root,capture_output=True,timeout=120,
                               creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
    except subprocess.TimeoutExpired:raise EvaluationError('content_render_timeout') from None
    if result.returncode or digest(pptx)!=before:raise EvaluationError('content_render_failed')
    path=Path(_final_json(result.stdout)['pages'][0]['path'])
    with Image.open(path) as image:
        if any(abs(a-b)>1 for a,b in zip(image.size,source_size)):raise EvaluationError('content_render_resolution_mismatch')
    return path


def score_api(run,reference_config,output,compiler_root,*,probes=True):
    run=Path(run).resolve();verify_generation(run)
    spec,file=_reference(reference_config);generation=read_json(run/'generation.json')
    if set(spec.mapping)!=set(generation['source_order']):raise EvaluationError('reference_mapping_incomplete')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    reference=inspect_objects(file);write_json(out/'reference-objects.json',reference)
    reference_render=render_reference_pages(compiler_root,file,out/'reference-render',spec.mapping.values())
    source_hashes={p['id']:p['source_sha256'] for p in read_json(run/'round.json')['pages']}
    pages=[]
    for row in generation['pages']:
        id=row['id'];dest=out/id;dest.mkdir()
        if row['status']!='SUCCEEDED':pages.append({'id':id,'status':'GENERATION_FAILED','stage':row['stage'],'error_code':row.get('error_code')});continue
        page=spec.mapping[id]
        if page>len(reference['slides']):raise EvaluationError('reference_slide_missing')
        pptx=run/row['artifacts']['pptx'];preview=run/row['artifacts']['preview']/'page-001.png'
        actual=inspect_objects(pptx)['slides'][0]['objects'];expected=reference['slides'][page-1]['objects']
        write_json(dest/'actual-objects.json',actual)
        structure={kind:match_objects(expected,actual,kind) for kind in ('text','shape','table','image')}
        expected_tables={o['id']:o for o in expected if o['type']=='table'}
        actual_tables={o['id']:o for o in actual if o['type']=='table'}
        table_details=[]
        for pair in structure['table']['pairs']:
            a,b=expected_tables[pair['reference']],actual_tables[pair['actual']]
            cells_equal=[[normalize(c) for c in r] for r in a['cells']]==[[normalize(c) for c in r] for r in b['cells']]
            table_details.append({**pair,'cell_structure_equal':cells_equal,'merges_equal':a['merges']==b['merges']})
        if any(not t['cell_structure_equal'] or not t['merges_equal'] for t in table_details):structure['table']['status']='FAIL'
        customs=[o for o in expected if o['geometry']=='custom']
        custom_matched={p['reference'] for p in structure['shape']['pairs']}&{o['id'] for o in customs}
        unresolved_custom=[o['id'] for o in customs if not o.get('custom_geometry_signature')]
        custom={'expected':len(customs),'matched':len(custom_matched),'unresolved_ids':unresolved_custom,
                'status':'NOT_RUN_UNRESOLVED_GEOMETRY' if unresolved_custom else 'PASS' if len(custom_matched)==len(customs) else 'FAIL'}
        source=run/'inputs'/id/'source.png'
        visual=image_difference(source,preview,dest/'visual')
        reference_diff=image_difference(source,reference_render[page]['path'],dest/'reference-consistency') if reference_render[page]['status']=='RENDERED' else reference_render[page]
        content_preview=_render_source_resolution(compiler_root,pptx,source,preview,dest/'content-render')
        content=content_metrics(expected,actual,read_json(run/'inputs'/id/'analysis.json'),content_preview)
        content['render_resolution_policy']='actual PPTX rendered at source image width; no bitmap upsampling'
        edits=editing_probes(pptx,actual,dest/'probes',compiler_root) if probes else {'status':'NOT_RUN'}
        checks=[structure[k]['status'] for k in ('text','shape','table')]+[custom['status'],visual['status'],content['rendered_content']['status']]
        checks += [c['status'] for c in edits.get('cases',[])] or ['NOT_RUN']
        if reference_render[page]['status']!='RENDERED':checks.append('NOT_RUN_REFERENCE_RENDER')
        status='FAIL' if 'FAIL' in checks else 'METRIC_PASS_REVIEW_REQUIRED' if all(c=='PASS' for c in checks) else 'INCOMPLETE'
        pages.append({'id':id,'reference_slide':page,'status':status,'structure':structure,'table_details':table_details,
             'custom_geometry':custom,'visual':visual,'reference_consistency':reference_diff,'content':content,
             'split_merge':split_merge_diagnostics(expected,actual),'editing':edits,'usage':row.get('usage'),
             'model_returned':row.get('model_returned'),'transport':row.get('transport')})
        write_json(dest/'score.json',pages[-1])
    report={'schema':'IndependentAPIScore','version':'1.0','round':generation['round'],
            'evaluator_source_sha256':hashlib.sha256((digest(__file__)+digest(geometry_scoring.__file__)+digest(Path(compiler_root)/'scripts/render_for_ocr.mjs')).encode()).hexdigest(),
            'generation_seal_sha256':digest(run/'generation.seal.json'),'reference_sha256':spec.reference_sha256,
            'source_sha256':source_hashes,'reference_mapping':spec.mapping,
            'status':'FAIL' if any(p['status'] in ('FAIL','GENERATION_FAILED') for p in pages) else 'REVIEW_REQUIRED',
            'pages':pages,'successful_generation':generation['succeeded_pages'],'expected_pages':generation['expected_pages'],
            'requests_reserved':generation['requests_reserved'],'cost':None,'cost_reason':'No verified unit pricing supplied; report usage only',
            'powerpoint_checked':False,'wps_checked':False,'human_acceptance':False}
    write_json(out/'score.json',report)
    lines=['# 独立 API 评测结果','','| 图片 | 生成/评测状态 | 原生文本 P/R | 原生图形 P/R | 图像平均差 |','|---|---|---|---|---|']
    for p in pages:
        if 'structure' not in p:lines.append(f"| {p['id']} | {p['status']} | — | — | — |");continue
        t,s=p['structure']['text'],p['structure']['shape']
        lines.append(f"| {p['id']} | {p['status']} | {t['precision']:.1%}/{t['recall']:.1%} | {s['precision']:.1%}/{s['recall']:.1%} | {p['visual']['mean_absolute_rgb']:.2f} |")
    lines+=['','指标通过不等于人工验收。未配置本地 OCR 时，渲染内容准确率明确标为未执行。生成失败页不从分母中移除。',
            '对象类型、图片文字保留、表格与自定义几何分别计分，不以视觉近似替代原生编辑能力。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    verify_generation(run)
    return {'status':report['status'],'report':str(out/'score.json'),'human_acceptance':False}


def compare_rounds(first,second,output):
    a,b=read_json(first),read_json(second)
    if a['reference_sha256']!=b['reference_sha256']:raise EvaluationError('comparison_reference_mismatch')
    if a.get('evaluator_source_sha256')!=b.get('evaluator_source_sha256'):raise EvaluationError('comparison_evaluator_mismatch')
    if a.get('source_sha256')!=b.get('source_sha256') or a.get('reference_mapping')!=b.get('reference_mapping'):raise EvaluationError('comparison_inputs_mismatch')
    aa={p['id']:p for p in a['pages']};bb={p['id']:p for p in b['pages']}
    if set(aa)!=set(bb):raise EvaluationError('comparison_pages_mismatch')
    rows=[]
    for id in aa:
        row={'id':id,'round_one_status':aa[id]['status'],'round_two_status':bb[id]['status']}
        if 'structure' in aa[id] and 'structure' in bb[id]:
            row['delta']={kind:{key:bb[id]['structure'][kind][key]-aa[id]['structure'][kind][key] for key in ('precision','recall')}
                          for kind in ('text','shape','table')}
            row['visual_mean_delta']=bb[id]['visual']['mean_absolute_rgb']-aa[id]['visual']['mean_absolute_rgb']
        rows.append(row)
    target=Path(output)
    if target.exists():raise FileExistsError('comparison output must be new')
    write_json(target,{'schema':'APIRoundComparison','pages':rows,'no_page_cherry_picking':True,'human_acceptance':False})
    return {'comparison':str(target)}
