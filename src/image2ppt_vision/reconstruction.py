"""Source-bound, declarative page reconstruction; never executes model code.

The plan partitions semantic modules, while pixel measurements select typography.
Complex pixels remain explicitly named local assets. A successful build is still
a review candidate, not a visual or application acceptance result.
"""
from datetime import datetime, timezone
from pathlib import Path
import math
import shutil
import hashlib
import json

import cv2
import numpy as np
from PIL import Image, ImageOps

from .storage import digest, read_json, write_json
from .plans import validate_plan
from .typography import estimate_text
from .renderer_calibration import origin_correction


def _hex(rgb):
    return ''.join(f'{int(v):02X}' for v in np.clip(np.rint(rgb), 0, 255))


def _bounds(box, width, height):
    x, y, w, h = box
    x0, y0 = max(0, math.floor(x)), max(0, math.floor(y))
    x1, y1 = min(width, math.ceil(x+w)), min(height, math.ceil(y+h))
    return x0, y0, x1, y1


def _background(pixels):
    """Estimate the dominant module backdrop; residuals preserve every exception."""
    h, w = pixels.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    x = xx / max(1, w-1); y = yy / max(1, h-1)
    design = np.stack([np.ones_like(x), x, y], -1)
    flat = pixels.reshape(-1, 3).astype(float)
    # Dominant colour neighbourhood prevents photos from biasing a white panel.
    codes = (pixels//16).astype(int)
    bins = (codes[:,:,0]*256+codes[:,:,1]*16+codes[:,:,2]).ravel()
    mode = np.bincount(bins, minlength=4096).argmax()
    selected = bins == mode
    if selected.sum() < flat.shape[0]*.12:
        return None
    a = design.reshape(-1, 3)
    centre = np.median(flat[selected], axis=0)
    selected = np.max(abs(flat-centre), axis=1) < 24
    if selected.sum() < 20:
        return None
    coef = np.linalg.lstsq(a[selected], flat[selected], rcond=None)[0]
    for _ in range(3):
        pred = a@coef
        selected = np.max(abs(flat-pred), axis=1) < 10
        if selected.sum() < flat.shape[0]*.20:
            return None
        coef = np.linalg.lstsq(a[selected], flat[selected], rcond=None)[0]
    # A linear PPT fill has one axis. Keep the stronger measured direction and
    # preserve the other direction's error as source pixels in the residual.
    axis = 0 if np.linalg.norm(coef[1]) >= np.linalg.norm(coef[2]) else 1
    if axis == 0:
        first, last = coef[0]+coef[2]*.5, coef[0]+coef[1]+coef[2]*.5
        fraction = x
    else:
        first, last = coef[0]+coef[1]*.5, coef[0]+coef[2]+coef[1]*.5
        fraction = y
    fitted = np.clip(np.rint(first+(last-first)*fraction[...,None]), 0, 255).astype('uint8')
    return fitted, first, last, axis


def _components(mask):
    joined = cv2.dilate(mask.astype('uint8'), np.ones((5,5), 'uint8'))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
    parts=[]; small=np.zeros_like(mask,dtype=bool)
    for k in range(1, count):
        x,y,w,h,area=stats[k]
        actual=(labels[y:y+h,x:x+w]==k) & mask[y:y+h,x:x+w]
        if not actual.any():continue
        if actual.sum()<48 or min(w,h)<7:
            small[y:y+h,x:x+w] |= actual
        else:
            parts.append((int(x),int(y),int(w),int(h),actual))
    # Keep tiny antialias/decorative pixels exactly, but do not turn each fleck
    # into a separate PowerPoint object. Five large islands + one residual is a
    # useful editing unit and stays within the fixed compiler's page budget.
    parts.sort(key=lambda p:int(p[4].sum()),reverse=True)
    for x,y,w,h,actual in parts[5:]:small[y:y+h,x:x+w] |= actual
    parts=parts[:5]
    yy,xx=np.where(small)
    if len(xx):
        x,y=int(xx.min()),int(yy.min());w,h=int(xx.max()-x+1),int(yy.max()-y+1)
        parts.append((x,y,w,h,small[y:y+h,x:x+w]))
    return parts


def _residual_mask(pixels, fitted, kind='panel'):
    """Keep visible local content, not opaque near-background fitting noise.

    A binary cutout of every >3-level discrepancy retained almost-white
    background texture and antialias fringes as fully opaque pixels. Those
    islands hid any later native fill change. Normalize differences within
    12/255 only on pale, smooth panel fills. Coloured/strong gradients and
    header/footer artwork keep the tighter 3/255 fidelity tolerance.
    Photographs never use this path.
    """
    error = np.max(np.abs(pixels.astype(float)-fitted), axis=2)
    flat = fitted.reshape(-1, 3).astype(float)
    pale_panel = kind == 'panel' and flat.min() >= 225 and np.ptp(flat, axis=0).max() <= 28
    return error > (12 if pale_panel else 3)


def _verified_shape(pixels, proposal):
    """Materialize simple unobscured fills only after independent pixel fitting."""
    h,w=pixels.shape[:2];x0,y0,x1,y1=_bounds(proposal['bbox'],w,h)
    geometry=proposal['geometry']
    if geometry not in ('rect','roundRect') or proposal.get('line'):
        return None,'geometry_verifier_not_implemented'
    if min(x1-x0,y1-y0)<6:return None,'shape_too_small'
    # An outside ring must be smooth, so erasure cannot destroy a neighbouring
    # object or photo. The interior must be an unobscured linear fill.
    a,b,c,d=max(0,x0-2),max(0,y0-2),min(w,x1+2),min(h,y1+2)
    context=pixels[b:d,a:c];ring=np.ones(context.shape[:2],bool)
    ring[y0-b:y1-b,x0-a:x1-a]=False
    if not ring.any():return None,'no_surrounding_background'
    border=context[ring].astype(float);background=np.median(border,axis=0)
    if np.percentile(np.max(abs(border-background),axis=1),90)>12:
        return None,'shape_background_not_smooth'
    patch=pixels[y0:y1,x0:x1];ph,pw=patch.shape[:2]
    yy,xx=np.mgrid[:ph,:pw];design=np.stack([np.ones_like(xx),xx/max(1,pw-1),yy/max(1,ph-1)],-1)
    inside=np.zeros((ph,pw),bool);inside[2:-2,2:-2]=True
    coefficients=np.linalg.lstsq(design[inside],patch[inside].astype(float),rcond=None)[0]
    axis=0 if np.linalg.norm(coefficients[1])>=np.linalg.norm(coefficients[2]) else 1
    first=coefficients[0]+coefficients[2 if axis==0 else 1]*.5
    last=first+coefficients[1 if axis==0 else 2]
    fill=first+(last-first)*(xx/max(1,pw-1) if axis==0 else yy/max(1,ph-1))[...,None]
    error=np.max(abs(patch.astype(float)-fill),axis=2)[inside]
    if np.percentile(error,95)>12 or error.mean()>5:return None,'shape_has_internal_content'
    if np.linalg.norm((first+last)/2-background)<25:return None,'insufficient_shape_contrast'
    if geometry=='roundRect':
        radius=float(proposal.get('corner_radius_px',min(pw,ph)/4))
        # Check the corners themselves, not just the interior fill. Otherwise a
        # model could change a rectangular source into any rounded shape.
        corners=np.zeros((ph,pw),bool)
        for cx,cy in ((radius,radius),(pw-1-radius,radius),(radius,ph-1-radius),(pw-1-radius,ph-1-radius)):
            corner=((xx<radius) if cx==radius else (xx>pw-1-radius)) & ((yy<radius) if cy==radius else (yy>ph-1-radius))
            corners |= corner & ((xx-cx)**2+(yy-cy)**2>(radius+.5)**2)
        if corners.any() and np.percentile(np.max(abs(patch.astype(float)-background),axis=2)[corners],75)>20:
            return None,'roundrect_outline_not_verified'
    node={'id':proposal['id'],'type':'shape','geometry':geometry,'bbox':[x0,y0,pw,ph],
          'fill':_hex(first),'evidence_ids':[proposal['id']+'-evidence'],'z':0}
    if np.max(abs(first-last))>2:
        node['fill_gradient']={'angle_deg':0 if axis==0 else 90,
            'stops':[{'offset':0,'color':_hex(first)},{'offset':100000,'color':_hex(last)}]}
    if proposal.get('corner_radius_px') is not None:node['corner_radius_px']=proposal['corner_radius_px']
    return (node,[x0,y0,x1,y1],background,{'mean_fill_error':float(error.mean()),'p95_fill_error':float(np.percentile(error,95))}),'verified_from_pixels'


def reconstruct(source, analysis, plan, output, *, font_candidates=None, target_width=1280, layers=None):
    proposed=plan.to_dict() if hasattr(plan,'to_dict') else plan
    if proposed.get('tables'):
        if layers is not None:raise ValueError('table planning and reviewed layers are separate reconstruction modes')
        from .table_reconstruction import reconstruct_tables
        return reconstruct_tables(source,analysis,plan,output,font_candidates=font_candidates,target_width=target_width)
    if layers is not None:
        from .layer_reconstruction import reconstruct_layers
        return reconstruct_layers(source, analysis, plan, output, layers,
                                  font_candidates=font_candidates, target_width=target_width)
    source=Path(source).resolve(); output=Path(output).resolve()
    if output.exists(): raise FileExistsError('reconstruction output must be a new directory')
    analysis=read_json(analysis) if isinstance(analysis,(str,Path)) else analysis
    if analysis.get('schema')!='VisionAnalysis': raise ValueError('VisionAnalysis required')
    with Image.open(source) as opened:
        pixels=np.array(ImageOps.exif_transpose(opened).convert('RGB'))
    h,w=pixels.shape[:2]; original_sha=digest(source)
    if analysis.get('image_size') != [w,h]: raise ValueError('analysis dimensions differ from source')
    source_hashes={v.get(k) for v in analysis.get('inputs',{}).values() for k in ('original_sha256','sha256')}
    if original_sha not in source_hashes: raise ValueError('analysis is not bound to this source image')
    texts=[dict(o) for o in analysis.get('objects',[]) if o.get('kind')=='text']
    ids=[t['id'] for t in texts]
    if len(ids)!=len(set(ids)): raise ValueError('duplicate OCR evidence ids')
    plan=validate_plan(plan.to_dict() if hasattr(plan,'to_dict') else plan,
                       source_sha256=original_sha,image_size=[w,h],ocr_ids=set(ids)).to_dict()
    # Reserve the complete deterministic namespace before creating output. OCR
    # identifiers are external evidence and must never alias generated objects.
    reserved={'page-decoration','page-decoration-pixels'}
    for r in plan['regions']:
        rid=r['id']
        reserved.update((rid+'-backdrop',rid+'-backdrop-evidence',rid+'-group'))
        for suffix in ['-visual','-residual']+[f'-asset-{i:03}' for i in range(6)]:
            reserved.update((rid+suffix,rid+suffix+'-pixels'))
    for s in plan.get('shapes',[]):reserved.update((s['id'],s['id']+'-evidence'))
    if set(ids)&reserved:raise ValueError('OCR id collides with reconstruction namespace')
    overrides={t['target_id']:t for t in plan.get('text_overrides',[])}
    for text in texts:
        edit=overrides.get(text['id'])
        if edit and edit.get('expected_text') is not None and edit['expected_text']!=text['text']:
            raise ValueError('text override no longer matches the OCR evidence')
    output.mkdir(parents=True); (output/'assets').mkdir(); (output/'source').mkdir()
    Image.fromarray(pixels).save(output/'source/page.png')
    write_json(output/'plan.json',plan)
    scale=target_width/w
    evidence=[{'id':t['id'],'kind':'text','bbox':t['bbox'],'text':t['text']} for t in texts]
    notes=['Screenshot reconstruction with a source-bound declarative visual plan.',
           'Text style is estimated from pixels; font identification and human acceptance remain unverified.',
           'Raster assets support movement and resizing; their internal objects remain non-native.']
    clean=pixels.copy(); text_nodes=[]; retained=[]; diagnostics=[]
    regions=plan['regions']; owners={r['id']:[] for r in regions}; owners['page-remainder']=[]
    def region_for(t):
        x,y,bw,bh=t['bbox']
        return next((r['id'] for r in regions if r['bbox'][0]<=x and r['bbox'][1]<=y
                     and x+bw<=r['bbox'][0]+r['bbox'][2]
                     and y+bh<=r['bbox'][1]+r['bbox'][3]),'page-remainder')
    for text in texts:
        owner=region_for(text); original_text=text['text']
        if text['id'] in overrides: text['text']=overrides[text['id']]['text']
        # Photo/diagram regions explicitly retain their internal type. Small OCR
        # regions outside these modules are still reconstructed independently.
        region=next((r for r in regions if r['id']==owner),None)
        if text['id'] in plan.get('retain_raster_ids',[]):
            estimate={'eligible':False,'diagnostics':{'reason':'visual_review_retains_source_pixels'}}
        elif region and region['kind'] in ('photo','diagram'):
            estimate={'eligible':False,'diagnostics':{'reason':'explicit_complex_region'}}
        else:
            estimate=estimate_text(pixels,text,font_candidates=font_candidates)
        diagnostic={'id':text['id'],'region':owner,**estimate.get('diagnostics',{})}
        if estimate.get('eligible',estimate.get('native_eligible',False)):
            x0,y0,pw,ph=estimate['patch_bbox']; x1=x0+pw; y1=y0+ph
            # Do not let one line's repair erase another OCR line.
            patch=estimate['cleaned_patch']
            changed=np.any(patch!=pixels[y0:y1,x0:x1],axis=2)
            overlap=False
            # Repair may use surrounding context but cannot erase remote borders.
            tx,ty,tw,th=text['bbox']
            allowed=np.zeros(changed.shape,bool)
            ax=max(x0,math.floor(tx)-2);ay=max(y0,math.floor(ty)-2)
            bx=min(x1,math.ceil(tx+tw)+2);by=min(y1,math.ceil(ty+th)+2)
            if bx>ax and by>ay: allowed[ay-y0:by-y0,ax-x0:bx-x0]=True
            # Ignore the existing <=12-channel background normalization tolerance;
            # only structural changes outside the text envelope block lifting.
            structural_change=np.max(abs(patch.astype(float)-pixels[y0:y1,x0:x1].astype(float)),axis=2)>12
            unsafe_cleanup=int((structural_change & ~allowed).sum())>3
            # A cleanup reaching outside its module must remain page-owned.
            if region_for({'bbox': estimate['patch_bbox']}) != owner:
                owner='page-remainder'
                diagnostic['region']=owner
            for other in texts:
                if other['id']==text['id']: continue
                ox,oy,ow,oh=other['bbox']
                # OCR detector padding can intersect adjacent lines even when
                # their glyphs are separate. Check modified pixels in the other
                # line's interior rather than intersection of context boxes.
                ax=max(x0,math.ceil(ox+1));ay=max(y0,math.ceil(oy+min(2,oh*.1)))
                bx=min(x1,math.floor(ox+ow-1));by=min(y1,math.floor(oy+oh-min(2,oh*.1)))
                if bx>ax and by>ay and changed[ay-y0:by-y0,ax-x0:bx-x0].sum()>3:
                    overlap=True;break
            if overlap: diagnostic['reason']='adjacent_text_overlap'
            elif unsafe_cleanup: diagnostic['reason']='cleanup_outside_text_bounds'
            else:
                clean[y0:y1,x0:x1][changed]=patch[changed]
                x,y,bw,bh=estimate['bbox']; size=estimate['font_size_px']
                # The fixed compiler's textbox uses a top anchor. The fit exposes
                # the font's Pillow ascent bearing; calibration is recorded.
                font_diagnostics=estimate.get('diagnostics',{}).get('font',{})
                bearing=font_diagnostics.get('glyph_bbox',[0,0,0,0])
                origin=font_diagnostics.get('render_origin',[x,y-bearing[1]])
                calibration=origin_correction(estimate,[w,h],[target_width,round(target_width*h/w)])
                diagnostic['renderer_origin_calibration']=calibration
                tx=max(0,float(origin[0])); ty=max(0,float(origin[1])+calibration['dy'])
                box=[tx,ty,min(w-tx,max(bw+3,size*len(text['text'])*.45)),min(h-ty,max(size*1.5,bh+4))]
                node={'id':text['id'],'type':'text','bbox':box,'z':0,'evidence_ids':[text['id']],
                      'text':text['text'],'font':{'latin':estimate['font_family'],'ea':estimate['font_family'],
                      'size_pt':round(size*scale*.75,4)},'bold':estimate['bold'],
                      'color':estimate['color'].lstrip('#'),'align':'left','wrap':False}
                if estimate.get('runs'): node['runs']=[{**r,**({'color':r['color'].lstrip('#')} if 'color' in r else {})} for r in estimate['runs']]
                if text['id'] in overrides: node['text_override_reason']=overrides[text['id']]['reason']
                text_nodes.append((owner,node));diagnostic['native']=True
                diagnostics.append(diagnostic);continue
        retained.append(text['id']);owners[owner].append(text['id']); diagnostic['native']=False;diagnostics.append(diagnostic)
    nodes=[]; assets=[]; groups=[]; residual_area=0; native_backgrounds=0
    shape_nodes=[];shape_diagnostics=[]
    for proposal in plan.get('shapes',[]):
        value,reason=_verified_shape(pixels,proposal)
        owner=region_for(proposal)
        if owner=='page-remainder':value=None;reason='shape_requires_semantic_region'
        if any(max(proposal['bbox'][0],t['bbox'][0])<min(proposal['bbox'][0]+proposal['bbox'][2],t['bbox'][0]+t['bbox'][2])
               and max(proposal['bbox'][1],t['bbox'][1])<min(proposal['bbox'][1]+proposal['bbox'][3],t['bbox'][1]+t['bbox'][3]) for t in texts):
            value=None;reason='shape_overlaps_text'
        if value:
            node,box,bg,measurement=value;a,b,c,d=box
            clean[b:d,a:c]=np.rint(bg).astype('uint8');shape_nodes.append((owner,node))
            evidence.append({'id':node['evidence_ids'][0],'kind':'shape','bbox':node['bbox']})
            shape_diagnostics.append({'id':proposal['id'],'native':True,'reason':reason,**measurement})
        else:shape_diagnostics.append({'id':proposal['id'],'native':False,'reason':reason})
    # Retained cross-boundary text stays a single page-level pixel object.
    # Punch its rectangle out of every lower image so it is never split/duplicated.
    crossing=[];excluded=np.zeros((h,w),bool)
    for tid in owners['page-remainder']:
        t=next(t for t in texts if t['id']==tid)
        a,b,c,d=_bounds(t['bbox'],w,h)
        if any(max(a,r['bbox'][0])<min(c,r['bbox'][0]+r['bbox'][2]) and
               max(b,r['bbox'][1])<min(d,r['bbox'][1]+r['bbox'][3]) for r in regions):
            crossing.append(t);excluded[b:d,a:c]=True
    owners['page-remainder']=[tid for tid in owners['page-remainder'] if tid not in {t['id'] for t in crossing}]
    def add_asset(identifier,array,bbox,owned,exclude=True):
        if exclude:
            a,b,c,d=_bounds(bbox,w,h);mask=excluded[b:d,a:c]
            if mask.any():
                array=array.copy() if array.shape[2]==4 else np.dstack([array,np.full(array.shape[:2],255,np.uint8)])
                array[:,:,3][mask]=0
        path=output/'assets'/f'{identifier}.png';Image.fromarray(array).save(path)
        eid=identifier+'-pixels'
        evidence.append({'id':eid,'kind':'image','bbox':bbox})
        assets.append({'id':identifier,'path':'assets/'+path.name,'purpose':'module',
                       'sha256':digest(path),'contains_evidence_ids':[eid]+owned})
        node={'id':identifier,'type':'image','asset_id':identifier,'bbox':bbox,'z':len(nodes),'evidence_ids':[]}
        nodes.append(node);return identifier
    # The page remainder contains only content outside planned modules. Every
    # module is physically removed before the independent region is added.
    remainder=clean.copy()
    for r in regions:
        x0,y0,x1,y1=_bounds(r['bbox'],w,h); remainder[y0:y1,x0:x1]=255
    add_asset('page-decoration',remainder,[0,0,w,h],owners['page-remainder'])
    for region in regions:
        rid=region['id']; x0,y0,x1,y1=_bounds(region['bbox'],w,h)
        patch=clean[y0:y1,x0:x1]; bbox=[x0,y0,x1-x0,y1-y0]; members=[]
        fit=None if region['kind'] in ('photo','diagram') or excluded[y0:y1,x0:x1].any() else _background(patch)
        if fit is None:
            members.append(add_asset(rid+'-visual',patch,bbox,owners[rid]));residual_area+=patch.shape[0]*patch.shape[1]
        else:
            fitted,first,last,axis=fit
            shapeid=rid+'-backdrop'; eid=shapeid+'-evidence'
            evidence.append({'id':eid,'kind':'shape','bbox':bbox})
            node={'id':shapeid,'type':'shape','geometry':'rect','bbox':bbox,'z':len(nodes),
                  'evidence_ids':[eid],'fill':_hex(first)}
            if np.max(abs(first-last))>2:
                node['fill_gradient']={'angle_deg':0 if axis==0 else 90,
                    'stops':[{'offset':0,'color':_hex(first)},{'offset':100000,'color':_hex(last)}]}
            nodes.append(node);members.append(shapeid);native_backgrounds+=1
            mask=_residual_mask(patch,fitted,region['kind'])
            residual_area+=int(mask.sum())
            parts=_components(mask)
            # Each unresolved OCR item belongs to exactly one residual component;
            # when it crosses components, retain the whole module residual.
            assigned={}; valid=True
            for tid in owners[rid]:
                t=next(t for t in texts if t['id']==tid);cx=t['bbox'][0]+t['bbox'][2]/2-x0;cy=t['bbox'][1]+t['bbox'][3]/2-y0
                hit=next((i for i,(px,py,pw,ph,m) in enumerate(parts) if px<=cx<px+pw and py<=cy<py+ph),None)
                if hit is None:valid=False;break
                assigned.setdefault(hit,[]).append(tid)
            if not valid or len(parts)>100:
                rgba=np.dstack([patch,mask.astype('uint8')*255])
                members.append(add_asset(rid+'-residual',rgba,bbox,owners[rid]))
            else:
                for i,(px,py,pw,ph,alpha) in enumerate(parts):
                    rgba=np.dstack([patch[py:py+ph,px:px+pw],alpha.astype('uint8')*255])
                    members.append(add_asset(f'{rid}-asset-{i:03}',rgba,[x0+px,y0+py,pw,ph],assigned.get(i,[])))
        for owner,node in shape_nodes:
            if owner==rid:node['z']=len(nodes);nodes.append(node);members.append(node['id'])
        for owner,node in text_nodes:
            if owner==rid:node['z']=len(nodes);nodes.append(node);members.append(node['id'])
        if len(members)>1:groups.append({'id':rid+'-group','member_ids':members})
    # Cross-boundary native text belongs above all module backgrounds.
    for index,t in enumerate(crossing):
        a,b,c,d=_bounds(t['bbox'],w,h)
        add_asset(f'crossing-text-{index:03}',pixels[b:d,a:c],[a,b,c-a,d-b],[t['id']],exclude=False)
    for owner,node in text_nodes:
        if owner=='page-remainder': node['z']=len(nodes);nodes.append(node)
    ir={'schema_version':'0.1','source_image':'source/page.png','source_sha256':digest(output/'source/page.png'),
        'canvas':{'source_px':[w,h],'target_px':[target_width,round(target_width*h/w)]},
        'background':{'color':'FFFFFF'},'review_status':'needs_review','review_notes':notes,
        'evidence':evidence,'assets':assets,'groups':groups,'nodes':nodes}
    write_json(output/'slideir.json',ir)
    write_json(output/'authoring.json',{'format_version':'1','tool':'image2ppt-vision reconstruct','model_id':plan['author']['model_id'],
        'created_at':datetime.now(timezone.utc).isoformat(),'ir_sha256':digest(output/'slideir.json'),
        'node_decisions':{n['id']:'source-bound pixel fit and declarative module plan' for n in nodes},
        'review_notes':notes,'unresolved_ids':retained})
    report={'schema':'ReconstructionReport','version':'1.0','status':'REVIEW_REQUIRED','input_sha256':original_sha,
        'analysis_sha256':hashlib.sha256(json.dumps(analysis,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        'plan_sha256':digest(output/'plan.json'),'ocr_lines':len(texts),'native_text':len(text_nodes),
        'native_text_ratio':len(text_nodes)/max(1,len(texts)),'retained_raster_text':retained,
        'semantic_regions':len(regions),'native_module_backdrops':native_backgrounds,'independent_images':len(assets),
        'groups':len(groups),'native_verified_shapes':len(shape_nodes),'shape_diagnostics':shape_diagnostics,
        'region_residual_fraction':residual_area/(w*h),
        'module_background_normalization_max_channel_delta':12,
        'typography':diagnostics,
        'human_acceptance':False,'powerpoint_checked':False,'wps_checked':False}
    write_json(output/'reconstruction.json',report)
    return report
