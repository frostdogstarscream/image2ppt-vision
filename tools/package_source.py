"""Create a source-only archive and digest manifest; never include private inputs."""
import argparse
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
files=[]
for pattern in ('*.md','*.toml','*.yaml','Dockerfile*','.dockerignore','.gitignore','*.lock','*.json','src/**/*.py','src/**/*.json','tests/**/*.py','tools/*.py','tools/*.sh'):
    files.extend(root.glob(pattern))
files=sorted(set(files))
manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
(out/'source-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
with tarfile.open(out/'source.tar.gz','w:gz') as tar:
    for name in manifest:tar.add(root/name,arcname=name)
with zipfile.ZipFile(out/'image2ppt-vision-source.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in manifest:z.write(root/name,name)
with zipfile.ZipFile(out/'image2ppt-vision-source.zip') as z:
    assert z.testzip() is None
print(json.dumps({'files':len(manifest),'zip_crc':'PASS','output':str(out.resolve())}))
