"""Freeze selected real inputs and run the local-only pilot in a new directory."""
import argparse
from pathlib import Path
import shutil
from image2ppt_vision.storage import digest, write_json
from image2ppt_vision.conversion import convert


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',required=True,help='existing conversion directory with inputs/pages/build')
    p.add_argument('--reference',required=True,help='user-recognized reference PPTX, comparison only')
    p.add_argument('--compiler-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();baseline=Path(a.baseline).resolve();root=Path(a.compiler_root).resolve();out=Path(a.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    sources=[]
    for kind,ext in [('source','.png'),('analysis','.json'),('plans','.json')]:
        dest=out/'inputs'/kind;dest.mkdir(parents=True)
        for stem in ('03','05','09'):
            original=baseline/'inputs'/kind/(stem+ext);copied=dest/original.name
            shutil.copyfile(original,copied)
            sources.append({'kind':kind,'source':str(original),'copy':copied.relative_to(out).as_posix(),'sha256':digest(copied)})
    (out/'baseline').mkdir()
    for stem in ('03','05','09'):
        shutil.copytree(baseline/'pages'/stem,out/'baseline'/stem)
    decks=list((baseline/'build').glob('*/deck.pptx'))
    if len(decks)!=1: raise ValueError('baseline deck ambiguous')
    for original,name in [(decks[0],'qwen-baseline.pptx'),(Path(a.reference).resolve(),'user-reference.pptx')]:
        shutil.copyfile(original,out/'baseline'/name)
        sources.append({'kind':'comparison_deck','source':str(original),'copy':'baseline/'+name,'sha256':digest(original)})
    code=out/'implementation-snapshot';code.mkdir()
    for folder,pattern in [('services/vision/src/image2ppt_vision','*.py'),('scripts','*.mjs'),('scripts','*.py')]:
        for src in (root/folder).glob(pattern):
            dst=code/folder/src.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    write_json(out/'baseline.json',{'sources':sources,'compiler_compose_sha256':digest(root/'scripts/compose.mjs'),
               'real_decomposition':'NOT_RUN','reason':'MaaS credentials not configured',
               'human_acceptance':False,'expanded_to_14':False})
    result=convert(out/'inputs/source',out/'inputs/analysis',out/'inputs/plans',root,out/'local-fix')
    write_json(out/'result.json',result)
    print(result['pptx'])


if __name__=='__main__': main()
