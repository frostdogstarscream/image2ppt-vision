"""OCR-aware OpenCV element proposals, before any destructive text cleanup.

Proposals are pixel evidence, not accepted native objects or semantic truth.
Keep hierarchy and intersecting OCR IDs; never turn their enclosing page into
one accepted visual object just because its foreground is connected.
"""
from pathlib import Path
import math
import shutil

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from .storage import digest,write_json
from .algorithms import iou


def _overlap(a,b):
    return max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))*max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))


def _contains(a,b):
    return a[0]<=b[0] and a[1]<=b[1] and a[0]+a[2]>=b[0]+b[2] and a[1]+a[3]>=b[1]+b[3]


def propose_elements(pixels,texts,*,maximum=180):
    h,w=pixels.shape[:2]
    if pixels.dtype!=np.uint8 or pixels.ndim!=3 or pixels.shape[2]!=3 or w*h>40_000_000:
        raise ValueError('bounded RGB uint8 image required')
    if not 1<=maximum<=500:raise ValueError('invalid candidate bound')
    protected=np.zeros((h,w),np.uint8)
    for t in texts:
        x,y,bw,bh=t['bbox']
        if min(bw,bh)<=0 or min(x,y)<0 or x+bw>w or y+bh>h:raise ValueError('OCR outside image')
        # This is a detection mask only. Original pixels are never repainted.
        a,b=max(0,int(x)-1),max(0,int(y)-1);c,d=min(w,math.ceil(x+bw)+1),min(h,math.ceil(y+bh)+1)
        protected[b:d,a:c]=255
    gray=cv2.cvtColor(pixels,cv2.COLOR_RGB2GRAY)
    hsv=cv2.cvtColor(pixels,cv2.COLOR_RGB2HSV)
    masks=[]
    for low,high in ((12,35),(35,90)):
        edge=cv2.Canny(gray,low,high)
        if low==12:
            masks.append(('source-edges-12-35',cv2.morphologyEx(edge,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))))
        edge[protected>0]=0
        edge=cv2.morphologyEx(edge,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        masks.append((f'edges-{low}-{high}',edge))
    saturated=((hsv[:,:,1]>75)&(hsv[:,:,2]>50)&(protected==0)).astype(np.uint8)*255
    masks.append(('saturated-components',cv2.morphologyEx(saturated,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))))
    proposals=[]
    for method,mask in masks:
        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x,y,bw,bh=cv2.boundingRect(contour);area=cv2.contourArea(contour)
            if min(bw,bh)<12 or area<120 or area>w*h*.80 or bw*bh>w*h*.90:continue
            if np.mean(protected[y:y+bh,x:x+bw]>0)>.65:continue
            perimeter=cv2.arcLength(contour,True)
            if perimeter<=0:continue
            circularity=4*np.pi*area/(perimeter*perimeter)
            solidity=area/max(1,cv2.contourArea(cv2.convexHull(contour)))
            occupancy=area/(bw*bh)
            if solidity<.65 or occupancy<.28:continue
            poly=cv2.approxPolyDP(contour,.012*perimeter,True)
            hint='ellipse' if circularity>.78 and .75<bw/bh<1.33 else 'panel' if occupancy>.80 and len(poly)<=12 else 'polygon' if len(poly)<=10 else 'visual'
            proposals.append(dict(bbox=[x,y,bw,bh],geometry_hint=hint,method=method,
                contour_area=round(float(area),2),occupancy=round(occupancy,4),circularity=round(float(circularity),4),
                contour=[[int(a),int(b)] for a,b in poly[:,0,:]]))
    # Prefer strong complete boundaries, retaining large and nested objects.
    proposals.sort(key=lambda c:(-c['occupancy'],-c['contour_area'],c['bbox']))
    selected=[]
    for c in proposals:
        if any(iou(c['bbox'],s['bbox'])>.88 for s in selected):continue
        selected.append(c)
    # Reserve space for icons as well as large modules instead of area-only top-k.
    buckets=[[],[],[]]
    for c in selected:
        fraction=c['bbox'][2]*c['bbox'][3]/(w*h)
        buckets[0 if fraction>.025 else 1 if fraction>.002 else 2].append(c)
    selected=[]
    while len(selected)<maximum and any(buckets):
        for bucket in buckets:
            if bucket and len(selected)<maximum:selected.append(bucket.pop(0))
    selected.sort(key=lambda c:(c['bbox'][1],c['bbox'][0],-c['contour_area']))
    for i,c in enumerate(selected):
        c['id']=f'cv-{i:03}'
        c['ocr_inside']=[t['id'] for t in texts if _contains(c['bbox'],t['bbox'])]
        c['ocr_crossing']=[t['id'] for t in texts if _overlap(c['bbox'],t['bbox'])>0 and not _contains(c['bbox'],t['bbox'])]
        c['native_verified']=False
    for c in selected:
        parents=[p for p in selected if p is not c and _contains(p['bbox'],c['bbox']) and p['contour_area']>c['contour_area']]
        c['parent_candidate']=min(parents,key=lambda p:p['bbox'][2]*p['bbox'][3])['id'] if parents else None
    return selected,protected


def build_inventory(source,analysis,output,*,maximum=180):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('inventory output must be new')
    with Image.open(source) as image:pixels=np.array(image.convert('RGB'))
    sha=digest(source);h,w=pixels.shape[:2]
    if analysis['image_size']!=[w,h] or sha not in {v.get(k) for v in analysis.get('inputs',{}).values() for k in ('sha256','original_sha256')}:
        raise ValueError('analysis source binding mismatch')
    texts=[t for t in analysis['objects'] if t['kind']=='text']
    if len({t['id'] for t in texts})!=len(texts):raise ValueError('duplicate OCR IDs')
    candidates,mask=propose_elements(pixels,texts,maximum=maximum)
    output.mkdir(parents=True);shutil.copyfile(source,output/'source.png')
    Image.fromarray(mask).save(output/'ocr-detection-mask.png')
    overlay=Image.fromarray(pixels);draw=ImageDraw.Draw(overlay)
    font_path=Path('C:/Windows/Fonts/arial.ttf');font=ImageFont.truetype(str(font_path),16) if font_path.exists() else ImageFont.load_default()
    columns=8;cw,ch=200,150
    contact=Image.new('RGB',(columns*cw,max(1,math.ceil(len(candidates)/columns))*ch),'#e5e7eb');cd=ImageDraw.Draw(contact)
    (output/'crops').mkdir()
    for i,c in enumerate(candidates):
        x,y,bw,bh=c['bbox'];crop=Image.fromarray(pixels[y:y+bh,x:x+bw]);crop.save(output/'crops'/f"{c['id']}.png")
        c['crop']=f"crops/{c['id']}.png";c['crop_sha256']=digest(output/c['crop'])
        draw.rectangle((x,y,x+bw-1,y+bh-1),outline='#d000d0',width=2)
        draw.text((x,y),c['id'],font=font,fill='black',stroke_fill='white',stroke_width=1)
        crop.thumbnail((cw-10,ch-30));xx=(i%columns)*cw;yy=(i//columns)*ch
        contact.paste(crop,(xx+5,yy+25));cd.text((xx+5,yy+3),c['id']+' '+c['geometry_hint'],font=font,fill='black')
    overlay.save(output/'candidate-overlay.png');contact.save(output/'candidate-contact.jpg',quality=95)
    inventory=dict(schema='CVElementInventory',version='1.0',source_sha256=sha,image_size=[w,h],
        analysis_sha256=__import__('hashlib').sha256(__import__('json').dumps(analysis,sort_keys=True).encode()).hexdigest(),
        method='OpenCV multi-threshold contours and saturated components; OCR masked for detection only',
        candidates=candidates,texts=[{k:t[k] for k in ('id','text','bbox')} for t in texts],
        candidate_count=len(candidates),native_verified_count=0,human_acceptance=False,
        limitations=['Candidate contours are not semantic objects.','A raster crop is not a native shape.',
                     'Nested candidates may share pixels; model selection and ownership verification are required.'])
    write_json(output/'inventory.json',inventory)
    return inventory
