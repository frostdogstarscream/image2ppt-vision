"""Measure model-proposed typography and simple fills from the original pixels.

This is source-only calibration, not semantic correction or acceptance. Refused
measurements leave the model proposal unchanged and remain explicit in reports.
"""
from copy import deepcopy
from pathlib import Path
import math,shutil
import numpy as np
from PIL import Image
from .storage import read_json,write_json,digest
from .typography import estimate_text, _background
from .renderer_calibration import origin_correction

def _contains(a,b):
    return a[0]<=b[0] and a[1]<=b[1] and a[0]+a[2]>=b[0]+b[2] and a[1]+a[3]>=b[1]+b[3]

def _source_contrast(pixels,node):
    """Diagnostic only: proposed text against a reliable local source surface."""
    x,y,w,h=node.get('source_text_bbox',node['bbox'])
    l,t=max(0,math.floor(x)-3),max(0,math.floor(y)-3)
    r,b=min(pixels.shape[1],math.ceil(x+w)+3),min(pixels.shape[0],math.ceil(y+h)+3)
    if min(r-l,b-t)<4:return None
    surface,diagnostic=_background(pixels[t:b,l:r])
    if not diagnostic['reliable']:return None
    background=np.median(surface.reshape(-1,3),axis=0)
    colors=[run.get('color',node['color']) for run in node.get('runs',[])] or [node['color']]
    distances=[float(np.max(np.abs(np.array([int(c[i:i+2],16) for i in (0,2,4)])-background))) for c in colors]
    return {'minimum_rgb_distance':round(min(distances),3),'low_contrast_risk':min(distances)<18,
            'basis':'source_background_estimate_not_actual_pptx_render'}

def _fill(pixels,node,nodes):
    if node['geometry'] not in ('rect','roundRect','ellipse') or node['fill']=='none':return None
    h,w=pixels.shape[:2];x,y,bw,bh=node['bbox'];l,t,r,b=math.floor(x),math.floor(y),math.ceil(x+bw),math.ceil(y+bh)
    crop=pixels[t:b,l:r];mask=np.ones(crop.shape[:2],bool)
    margin=max(2,min(5,int(min(bw,bh)*.08)))
    mask[:margin]=False;mask[-margin:]=False;mask[:,:margin]=False;mask[:,-margin:]=False
    if node['geometry']=='ellipse':
        yy,xx=np.mgrid[:b-t,:r-l];mask &= ((xx-(r-l)/2)/max(1,(r-l)/2-margin))**2+((yy-(b-t)/2)/max(1,(b-t)/2-margin))**2<1
    for other in nodes:
        if other['id']==node['id']:continue
        box=other.get('source_text_bbox',other['bbox']) if other['type']=='text' else other['bbox']
        if other['type']=='shape' and (not _contains(node['bbox'],box) or box[2]*box[3]>=bw*bh*.94):continue
        a,c=max(l,math.floor(box[0])-2),min(r,math.ceil(box[0]+box[2])+2);d,e=max(t,math.floor(box[1])-2),min(b,math.ceil(box[1]+box[3])+2)
        if c>a and e>d:mask[d-t:e-t,a-l:c-l]=False
    values=crop[mask]
    if len(values)<max(30,bw*bh*.08):return None
    q=(values//16).astype(int);bins=q[:,0]*256+q[:,1]*16+q[:,2];mode=np.bincount(bins,minlength=4096).argmax();selected=bins==mode
    if selected.mean()<.55:return None
    color=np.median(values[selected],axis=0);near=np.max(abs(values.astype(float)-color),axis=1)<12
    if near.mean()<.75:return None
    return {'color':''.join(f'{round(v):02X}' for v in np.median(values[near],axis=0)), 'sample_pixels':len(values),'dominant_fraction':round(float(near.mean()),4)}

def calibrate_elements(irfile,output,*,source_typography=False,renderer_version=None,calibrate_fills=True):
    irfile=Path(irfile);output=Path(output)
    if output.exists():raise FileExistsError('calibration output must be new')
    ir=read_json(irfile);src=irfile.parent/ir['source_image']
    authoring=read_json(irfile.parent/'authoring.json')
    if authoring.get('ir_sha256')!=digest(irfile):raise ValueError('authoring IR binding changed')
    model_typography=authoring.get('model_typography',{})
    if digest(src)!=ir['source_sha256']:raise ValueError('source binding changed')
    with Image.open(src) as im:pixels=np.array(im.convert('RGB'))
    w,h=ir['canvas']['source_px'];tw,th=ir['canvas']['target_px'];scale=tw/w
    diagnostics=[];measured=0;fills=0
    original_nodes=deepcopy(ir['nodes'])
    for node in ir['nodes']:
        if node['type']=='text':
            estimate=estimate_text(pixels,{'id':node['id'],'text':node['text'],'bbox':node.get('source_text_bbox',node['bbox'])},
                                   **({'isolate_ocr_window':True} if source_typography else {}))
            if source_typography and estimate.get('native_eligible'):
                fontdiag=estimate['diagnostics']['font']
                if fontdiag.get('dimension_error',0)>.12 or fontdiag.get('shape_overlap',1)<.58:
                    estimate['native_eligible']=False
                    estimate['diagnostics'].update(status='REVIEW_REQUIRED',reason='strict_source_font_fit_refused')
            row={'id':node['id'],'kind':'typography','applied':False,'diagnostics':estimate['diagnostics']}
            if source_typography:
                row['proposal']={k:deepcopy(node.get(k)) for k in ('bbox','font','bold','color','runs')}
                row['proposal_contrast']=_source_contrast(pixels,node)
            if node['id'] in model_typography and not source_typography:
                proposal=model_typography[node['id']]
                if node['font']!=proposal['font'] or node['bold']!=proposal['bold']:raise ValueError('model typography binding changed')
                row.update(reason='model_typography_preserved',suggestion={k:estimate.get(k) for k in ('font_family','font_size_px','bold','color','bbox')})
            elif estimate.get('native_eligible'):
                x,y,bw,bh=estimate['bbox'];size=estimate['font_size_px'];fontdiag=estimate['diagnostics']['font'];origin=fontdiag['render_origin']
                correction=origin_correction(estimate,[w,h],[tw,th],renderer_version=(renderer_version or "unspecified") if source_typography else renderer_version)
                if source_typography:
                    # Font matching optimizes local overlap and can shift a long
                    # line by many pixels. Restore its measured visible left/top
                    # edges; do not export that search translation as layout.
                    glyph=fontdiag.get('glyph_bbox')
                    if glyph:
                        origin=[x-glyph[0],y-glyph[1]]
                        row['origin_policy']='measured_ink_edges_minus_font_glyph_bearings'
                tx=max(0,float(origin[0]));ty=max(0,float(origin[1])+correction['dy'])
                row['before']={k:deepcopy(node.get(k)) for k in ('bbox','font','bold','color','runs')}
                node.update(bbox=[tx,ty,min(w-tx,max(bw+3,size*len(node['text'])*.45)),min(h-ty,max(size*1.5,bh+4))],font={'latin':estimate['font_family'],'ea':estimate['font_family'],'size_pt':round(size*scale*.75,4)},bold=estimate['bold'],color=estimate['color'].lstrip('#'))
                node.pop('runs',None)
                if estimate.get('runs'):node['runs']=[{**r,'color':r['color'].lstrip('#')} for r in estimate['runs']]
                if source_typography:node['align']='left'
                row.update(applied=True,renderer_origin_calibration=correction);measured+=1
            if source_typography:
                row['source_ink_bbox']=estimate.get('bbox')
                row['source_color']=estimate.get('color')
                if not row['applied']:
                    row['reason']=estimate['diagnostics'].get('reason','source_measurement_refused')
                    # A missing/ambiguous font does not invalidate an isolated,
                    # single-color foreground. Preserve typography, but repair
                    # e.g. a white "02" over a light background using source ink.
                    if (row['reason'] in ('font_or_ocr_shape_mismatch','strict_source_font_fit_refused','no_usable_installed_font')
                            and estimate.get('color') and estimate['diagnostics'].get('foreground_color_count')==1
                            and estimate['diagnostics'].get('background',{}).get('reliable')):
                        node['color']=estimate['color'].lstrip('#')
                        for run in node.get('runs',[]):run['color']=node['color']
                        row['source_color_applied']=True
                row['needs_rendered_review']=True
                row['output_contrast']=_source_contrast(pixels,node)
                row['output']={k:deepcopy(node.get(k)) for k in ('bbox','font','bold','color','runs')}
            diagnostics.append(row)
        elif node['type']=='shape' and calibrate_fills:
            fitted=_fill(pixels,node,original_nodes)
            if fitted:
                diagnostics.append({'id':node['id'],'kind':'fill','applied':True,'before':node['fill'],**fitted});node['fill']=fitted['color'];node.pop('fill_gradient',None);fills+=1
    output.mkdir(parents=True);(output/'source').mkdir();(output/'assets').mkdir();shutil.copyfile(src,output/'source/page.png');ir['source_image']='source/page.png'
    for asset in ir['assets']:
        old=irfile.parent/asset['path'];relative='assets/'+asset['id']+old.suffix;shutil.copyfile(old,output/relative);asset['path']=relative
    ir['review_notes'].append('Source-pixel typography measurements applied where eligible; refused measurements remain proposals. Font and origin estimates require actual-render review. Solid fills changed: '+str(fills)+'.')
    write_json(output/'slideir.json',ir)
    if source_typography:
        authoring['original_model_typography']=model_typography
        authoring['model_typography']={}
        authoring['applied_source_typography']={n['id']:{k:deepcopy(n[k]) for k in ('font','bold','color','bbox')} for n in ir['nodes'] if n['type']=='text' and any(r['id']==n['id'] and r['applied'] for r in diagnostics)}
    authoring['tool']='supervised-elements source-pixel calibration';authoring['ir_sha256']=digest(output/'slideir.json');authoring['review_notes']=ir['review_notes'];write_json(output/'authoring.json',authoring)
    report={'schema':'ElementCalibration/1','source_sha256':ir['source_sha256'],'input_ir_sha256':digest(irfile),'output_ir_sha256':digest(output/'slideir.json'),'typography_applied':measured,'source_color_only_applied':sum(bool(r.get('source_color_applied')) for r in diagnostics),'typography_total':sum(n['type']=='text' for n in ir['nodes']),'solid_fills_applied':fills,'diagnostics':diagnostics,'human_acceptance':False}
    write_json(output/'calibration.json',report)
    return {k:v for k,v in report.items() if k!='diagnostics'}

if __name__=='__main__':
    import json,sys
    print(json.dumps(calibrate_elements(sys.argv[1],sys.argv[2]),ensure_ascii=False))
