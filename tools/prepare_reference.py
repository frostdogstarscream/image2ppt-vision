"""Create private frozen page manifests and unverified OOXML annotation drafts."""
import argparse
import hashlib
import json
import os
import posixpath
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--pptx',required=True);p.add_argument('--images',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);(out/'labels').mkdir()
    images=Path(args.images).resolve();cases=[]
    with zipfile.ZipFile(args.pptx) as z:
        doc=ET.fromstring(z.read('ppt/presentation.xml'));size=doc.find('p:sldSz',NS);cx,cy=int(size.get('cx')),int(size.get('cy'))
        rels={r.get('Id'):r.get('Target') for r in ET.fromstring(z.read('ppt/_rels/presentation.xml.rels'))}
        for page,s in enumerate(doc.findall('p:sldIdLst/p:sldId',NS),1):
            path=posixpath.normpath('ppt/'+rels[s.get('{'+NS['r']+'}id')]);slide=ET.fromstring(z.read(path))
            image=images/f'page-{page:02}.png'
            with Image.open(image) as im:w,h=im.size
            objects=[]
            # Grouped transforms and hidden objects require explicit annotation.
            for i,node in enumerate(slide.find('p:cSld/p:spTree',NS)):
                xfrm=node.find('p:spPr/a:xfrm',NS)
                if xfrm is None:xfrm=node.find('p:xfrm',NS)
                if xfrm is None:continue
                off=xfrm.find('a:off',NS);ext=xfrm.find('a:ext',NS)
                if off is None or ext is None:continue
                box=[int(off.get('x'))/cx*w,int(off.get('y'))/cy*h,int(ext.get('cx'))/cx*w,int(ext.get('cy'))/cy*h]
                texts=[t.text or '' for t in node.findall('.//a:t',NS)]
                if texts:objects.append({'id':f'xml-{i}','kind':'text','bbox':box,'text':''.join(texts),'annotation_basis':'text_frame_not_glyph_extent'})
                else:
                    geom=node.find('p:spPr/a:prstGeom',NS);kind=geom.get('prst') if geom is not None else None
                    if kind in ('rect','roundRect','ellipse'):
                        objects.append({'id':f'xml-{i}','kind':'circle' if kind=='ellipse' else 'rectangle','bbox':box,'annotation_basis':'ooxml_shape'})
            label=out/'labels'/f'page-{page:02}.json'
            save(label,{'human_verified':False,'agent_reviewed':False,'page':page,'objects':objects,'masks':[],
                        'notes':['Draft only. Remove invisible objects, split text into visible OCR lines, manually trace complex masks.']})
            cases.append({'id':f'page-{page:02}','image':os.path.relpath(image,out).replace('\\','/'),'sha256':sha(image),'split':'development' if page in (3,6,13) else 'frozen_regression',
                          'operation':'analyze','options':{'segment':True},'labels':label.relative_to(out).as_posix(),'labels_sha256':sha(label)})
    save(out/'suite.json',{'version':'1','reference_pptx_sha256':sha(Path(args.pptx)),'renderer':'Artifact Tool; record installed version separately','human_render_acceptance':False,'cases':cases})
    for name,split in [('development','development'),('regression','frozen_regression')]:save(out/(name+'.json'),{'version':'1','cases':[c for c in cases if c['split']==split]})
    print(json.dumps({'pages':len(cases),'human_verified':False}))

if __name__=='__main__':main()
