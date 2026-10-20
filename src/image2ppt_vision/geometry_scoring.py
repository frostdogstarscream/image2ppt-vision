"""Conservative semantic comparison of literal OOXML polygon/line paths.

Coordinates are transformed into slide space. Unsupported guides, curves,
inherited/theme paints and complex stroke effects remain explicitly unresolved.
XML byte identity is never a matching criterion.
"""
import math
import numpy as np

NS={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}


def _paint(props):
    if props.find('a:noFill',NS) is not None:return 'none'
    fill=props.find('a:solidFill',NS)
    if fill is None:return None
    color=fill.find('a:srgbClr',NS)
    if color is None or list(color):return None
    value=color.get('val','').upper()
    return value if len(value)==6 and all(c in '0123456789ABCDEF' for c in value) else None


def _canonical(points,closed):
    points=[tuple(round(float(v),6) for v in p) for p in points]
    points=[p for i,p in enumerate(points) if i==0 or p!=points[i-1]]
    if closed and len(points)>1 and points[0]==points[-1]:points.pop()
    changed=True
    while changed and len(points)>(3 if closed else 2):
        changed=False
        for i in range(len(points)) if closed else range(1,len(points)-1):
            a=np.array(points[i-1]);b=np.array(points[i]);c=np.array(points[(i+1)%len(points)])
            ab=b-a;bc=c-b
            if abs(ab[0]*bc[1]-ab[1]*bc[0])<1e-10 and np.dot(ab,bc)>=0:
                points.pop(i);changed=True;break
    if closed:
        options=[tuple(p[i:]+p[:i]) for p in (points,list(reversed(points))) for i in range(len(p))]
        return min(options) if options else ()
    return min(tuple(points),tuple(reversed(points)))


def custom_signature(props,matrix,slide_size):
    if props is None:return None
    custom=props.find('a:custGeom',NS)
    if custom is None:return None
    try:
        # Literal numeric paths need no evaluation of a guide language.
        if custom.findall('a:gdLst/a:gd',NS) or custom.findall('a:avLst/a:gd',NS):return None
        if any(props.find('a:'+k,NS) is not None for k in ('effectLst','effectDag','scene3d','sp3d')):return None
        paths=custom.findall('a:pathLst/a:path',NS)
        if not paths or len(paths)>256:return None
        xfrm=props.find('a:xfrm/a:ext',NS)
        if xfrm is None:return None
        w,h=float(xfrm.get('cx')),float(xfrm.get('cy'))
        sw,sh=slide_size;records=[]
        fill=_paint(props);line=props.find('a:ln',NS)
        stroke=_paint(line) if line is not None else None
        if line is not None:
            if any(c.tag.rsplit('}',1)[-1] not in ('solidFill','noFill','prstDash') for c in line):return None
            dash=line.find('a:prstDash',NS)
            if dash is not None and dash.get('val')!='solid':return None
        line_width=float(line.get('w','12700')) if line is not None else 0
        # A nonuniform group transform gives direction-dependent stroke width.
        axes=matrix[:2,:2];sx=np.linalg.norm(axes[:,0]);sy=np.linalg.norm(axes[:,1])
        if abs(sx-sy)>1e-6 or abs(np.dot(axes[:,0],axes[:,1]))>1e-6:return None
        line_width=round(line_width*sx/sw,6)
        for path in paths:
            pw,ph=float(path.get('w',str(w))),float(path.get('h',str(h)))
            if min(pw,ph)<=0:return None
            f='none' if path.get('fill')=='none' else fill if path.get('fill','norm')=='norm' else None
            s='none' if path.get('stroke') in ('0','false') else stroke
            if f is None or s is None:return None
            points=[];closed=False
            for command in path:
                tag=command.tag.rsplit('}',1)[-1]
                if tag=='close':
                    if closed: return None
                    closed=True;continue
                if tag not in ('moveTo','lnTo') or closed:return None
                if (tag=='moveTo') != (not points):return None
                point=command.find('a:pt',NS)
                p=matrix@np.array([float(point.get('x'))*w/pw,float(point.get('y'))*h/ph,1])
                if not np.isfinite(p).all():return None
                points.append([p[0]/sw,p[1]/sh])
            if len(points)<2:return None
            records.append({'points':_canonical(points,closed),'closed':closed,'fill':f,'stroke':s,
                            'stroke_width':0 if s=='none' else line_width})
        # Order is preserved: differently painted overlapping paths may depend on it.
        return records
    except (TypeError,ValueError,AttributeError,OverflowError):return None


def equivalent(a,b):
    if not a or not b or len(a)!=len(b):return False
    for x,y in zip(a,b):
        if any(x[k]!=y[k] for k in ('closed','fill','stroke')):return False
        if abs(x['stroke_width']-y['stroke_width'])>1e-6:return False
        if len(x['points'])!=len(y['points']):return False
        if not np.allclose(x['points'],y['points'],rtol=0,atol=1e-5):return False
    return True
