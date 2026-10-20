"""Explicit preparation step only. Inference never calls this downloader."""
import argparse
import hashlib
import json
import urllib.request
import io
import zipfile
from pathlib import Path

FILES={
    "sam2.1_hiera_small.pt":"https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
}

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    records={}
    # OCR v4 models are distributed in this fixed upstream wheel. Extract weights
    # only; do not install its older OpenCV dependency alongside headless OpenCV.
    with urllib.request.urlopen('https://pypi.org/pypi/rapidocr-onnxruntime/1.4.4/json',timeout=60) as f:meta=json.load(f)
    wheel=next(x for x in meta['urls'] if x['filename'].endswith('.whl'))
    with urllib.request.urlopen(wheel['url'],timeout=60) as f:payload=f.read()
    if hashlib.sha256(payload).hexdigest()!=wheel['digests']['sha256']:raise ValueError('OCR wheel hash mismatch')
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        if z.testzip():raise ValueError('OCR wheel corrupt')
        for kind in ('det','rec','cls'):
            name=next(n for n in z.namelist() if n.endswith('.onnx') and ('_'+kind+'_' in n))
            dest=root/(kind+'.onnx');data=z.read(name)
            if dest.exists() and dest.read_bytes()!=data:raise ValueError('existing model differs')
            dest.write_bytes(data)
            records[dest.name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'source':wheel['url'],'archive_member':name}
    for name,url in FILES.items():
        dest=root/name
        if not dest.exists():
            tmp=dest.with_suffix(".download")
            with urllib.request.urlopen(url,timeout=60) as source,open(tmp,"wb") as out:
                while b:=source.read(1024*1024):out.write(b)
            tmp.rename(dest)
        if dest.stat().st_size<100_000:raise ValueError("invalid model download: "+name)
        records[name]={"sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),"bytes":dest.stat().st_size,"source":url}
        print(name,records[name]["sha256"],flush=True)
    manifest={"format_version":"1","sam_commit":"2b90b9f5ceec907a1c18123530e92e794ad901a4","ocr":"PP-OCRv4 Chinese","files":records}
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

if __name__=="__main__":main()
