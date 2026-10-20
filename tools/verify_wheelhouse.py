"""Verify an offline cache against the delivery manifest, without installing."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--directory',required=True)
p.add_argument('--manifest',required=True)
a=p.parse_args()
root=Path(a.directory)
expected=json.loads(Path(a.manifest).read_text(encoding='utf-8'))
for name,sha in expected.items():
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('invalid manifest path')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    if h.hexdigest()!=sha:raise ValueError('integrity mismatch: '+name)
print(json.dumps({'status':'PASS','files':len(expected)}))
