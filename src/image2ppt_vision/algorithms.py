import cv2
import numpy as np
from PIL import Image, ImageDraw


def rgb(path):
    with Image.open(path) as im:
        rgba=im.convert("RGBA")
        base=Image.new("RGBA", im.size, "white")
        return np.array(Image.alpha_composite(base,rgba).convert("RGB"))


def iou(a,b):
    x,y,w,h=a; X,Y,W,H=b
    area=max(0,min(x+w,X+W)-max(x,X))*max(0,min(y+h,Y+H)-max(y,Y))
    return area/max(1,w*h+W*H-area)


def geometry(image):
    h,w=image.shape[:2]
    scale=min(1.,1536/max(h,w))
    work=cv2.resize(image,(round(w*scale),round(h*scale)))
    gray=cv2.cvtColor(work,cv2.COLOR_RGB2GRAY)
    edges=cv2.Canny(gray,40,120)
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    nodes=[]
    for c in sorted(contours,key=cv2.contourArea,reverse=True):
        area=cv2.contourArea(c); perimeter=cv2.arcLength(c,True)
        if area<80 or perimeter<=0:continue
        poly=cv2.approxPolyDP(c,.025*perimeter,True)
        x,y,bw,bh=cv2.boundingRect(c)
        if min(bw,bh)<8 or area>0.98*work.shape[0]*work.shape[1]:continue
        circ=4*np.pi*area/(perimeter**2)
        kind="circle" if circ>.78 and .8<bw/bh<1.25 else "rectangle" if 4<=len(poly)<=8 and area/(bw*bh)>.75 else None
        if not kind:continue
        box=[round(v/scale,2) for v in (x,y,bw,bh)]
        box[2]=min(box[2],w-box[0]);box[3]=min(box[3],h-box[1])
        if any(iou(box,n["bbox"])>.90 for n in nodes):continue
        nodes.append({"id":f"shape-{len(nodes):04}","kind":kind,"bbox":box,
                      "score":round(float(circ if kind=="circle" else area/(bw*bh)),4),
                      "score_type":"geometric_heuristic","method":"opencv_contour",
                      "median_rgb":np.median(work[y:y+bh,x:x+bw].reshape(-1,3),axis=0).astype(int).tolist()})
        if len(nodes)>=500:break
    lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=45,minLineLength=40,maxLineGap=5)
    for line in ([] if lines is None else lines[:200]):
        x,y,X,Y=(line[0]/scale).tolist()
        if abs(X-x)>3 and abs(Y-y)>3:continue
        nodes.append({"id":f"line-{len(nodes):04}","kind":"line","bbox":[min(x,X),min(y,Y),max(1,abs(X-x)),max(1,abs(Y-y))],
                      "endpoints":[[x,y],[X,Y]],"method":"opencv_hough","score_type":"not_calibrated","score":None})
    return nodes


def relations(nodes):
    result=[]
    for i,a in enumerate(nodes):
        ax,ay,aw,ah=a["bbox"]
        for b in nodes[i+1:]:
            bx,by,bw,bh=b["bbox"]
            kind=[]
            if ax<=bx and ay<=by and ax+aw>=bx+bw and ay+ah>=by+bh:kind.append("contains")
            elif bx<=ax and by<=ay and bx+bw>=ax+aw and by+bh>=ay+ah:kind.append("inside")
            elif iou(a["bbox"],b["bbox"])>0:kind.append("overlaps")
            if abs(ay-by)<=3 and abs(ah-bh)<=3:kind.append("aligned_row")
            if abs(ax-bx)<=3 and abs(aw-bw)<=3:kind.append("aligned_column")
            if kind:result.append({"a":a["id"],"b":b["id"],"relations":kind,"method":"geometry_only"})
    # Record repeated equal-size runs and their gaps; do not infer semantic groups.
    for axis in (0,1):
        candidates=sorted([n for n in nodes if n["kind"] in ("rectangle","circle")],key=lambda n:n["bbox"][axis])
        for i,a in enumerate(candidates):
            group=[a]+[b for b in candidates[i+1:] if abs(a["bbox"][1-axis]-b["bbox"][1-axis])<=3 and all(abs(a["bbox"][j]-b["bbox"][j])<=3 for j in (2,3))]
            if len(group)>=3:
                gaps=[group[j+1]["bbox"][axis]-group[j]["bbox"][axis]-group[j]["bbox"][axis+2] for j in range(len(group)-1)]
                if min(gaps)>=0 and max(gaps)-min(gaps)<=3:result.append({"members":[n["id"] for n in group],"relations":["repeated_spacing"],"axis":axis,"gaps":gaps})
    return result


def overlay(image,nodes,path):
    im=Image.fromarray(image);draw=ImageDraw.Draw(im)
    for n in nodes:
        x,y,w,h=n["bbox"];draw.rectangle((x,y,x+w,y+h),outline="red" if n["kind"]=="text" else "blue",width=2)
        draw.text((x,y),n["id"],fill="red")
    im.save(path)


def extract(image,mask,box,out):
    h,w=image.shape[:2]
    x,y,bw,bh=box or (0,0,w,h)
    crop=image[y:y+bh,x:x+bw];m=mask[y:y+bh,x:x+bw].astype(bool)
    Image.fromarray(crop).save(out/"original.png")
    Image.fromarray(m.astype("uint8")*255).save(out/"mask.png")
    rgba=np.dstack([crop,m.astype("uint8")*255]);Image.fromarray(rgba).save(out/"transparent.png")
    yy,xx=np.indices(m.shape)
    checker=np.where(((xx//12+yy//12)%2)[...,None],210,245)*np.ones((1,1,3))
    for name,bg in [("white",np.full_like(crop,255)),("dark",np.full_like(crop,32)),("checkerboard",checker.astype("uint8"))]:
        Image.fromarray(np.where(m[...,None],crop,bg)).save(out/f"{name}.png")
    return {"bbox":[x,y,bw,bh],"mask_pixels":int(m.sum()),"touches_boundary":bool(m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any()),"alpha_method":"binary_mask_not_matting","text_overlap":"NOT_RUN"}


def compare(a,b,out,resize=False,register=False):
    original_size=list(b.shape[1::-1])
    if a.shape!=b.shape:
        if not resize:raise ValueError("different image sizes; explicit resize_reference required")
        b=cv2.resize(b,(a.shape[1],a.shape[0]))
    def measure(x,y,name):
        d=cv2.absdiff(x,y);Image.fromarray(d).save(out/name)
        ea=cv2.Canny(x,40,120)>0;eb=cv2.Canny(y,40,120)>0
        return {"mae":float(d.mean()),"edge_disagreement":float(np.logical_xor(ea,eb).mean())}
    result={"before":measure(a,b,"difference.png"),"original_reference_size":original_size,"resize_reference":resize,"quality_status":"REVIEW_REQUIRED"}
    if register:
        shift,response=cv2.phaseCorrelate(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype("float32"),cv2.cvtColor(b,cv2.COLOR_RGB2GRAY).astype("float32"))
        M=np.float32([[1,0,-shift[0]],[0,1,-shift[1]]]);aligned=cv2.warpAffine(b,M,(a.shape[1],a.shape[0]))
        valid=cv2.warpAffine(np.ones(b.shape[:2],dtype="uint8"),M,(a.shape[1],a.shape[0]),flags=cv2.INTER_NEAREST)>0
        result.update(translation=list(shift),response=float(response),valid_fraction=float(valid.mean()),after=measure(a,aligned,"difference-aligned.png"))
        result["after"]["overlap_mae"]=float(cv2.absdiff(a,aligned)[valid].mean()) if valid.any() else None
        Image.fromarray(aligned).save(out/"aligned.png")
    return result
