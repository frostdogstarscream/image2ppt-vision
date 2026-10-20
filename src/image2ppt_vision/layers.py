"""Source-bound layered assets. Review is evidence, never inferred from API success."""
from pathlib import Path
from typing import Annotated, Literal
import shutil

from PIL import Image
from pydantic import Field, StrictInt, StrictStr, ValidationError

from .plans import _Record, Identifier, BBox, MAX_PIXELS
from .storage import digest, read_json, write_json

Hash = Annotated[StrictStr, Field(pattern=r'^[a-f0-9]{64}$')]


class Repair(_Record):
    bbox: BBox
    kind: Literal['text_cleanup', 'shape_cleanup', 'background_completion']
    text_ids: list[Identifier] = Field(default_factory=list)
    shape_ids: list[Identifier] = Field(default_factory=list)
    reviewed: bool = False


class Layer(_Record):
    id: Identifier
    path: StrictStr
    sha256: Hash
    bbox: BBox
    z: StrictInt
    module: Identifier | None = None
    retained_text_ids: list[Identifier] = Field(default_factory=list)


class NativeText(_Record):
    text_id: Identifier
    z: StrictInt
    module: Identifier | None = None


class NativeShape(_Record):
    shape_id: Identifier
    z: StrictInt
    module: Identifier | None = None


class LayerManifest(_Record):
    schema_name: Literal['LayerManifest'] = Field(default='LayerManifest', alias='schema')
    version: Literal['1.0'] = '1.0'
    source_sha256: Hash
    image_size: Annotated[list[StrictInt], Field(min_length=2, max_length=2)]
    provider: Literal['import', 'ming']
    model_id: Annotated[StrictStr, Field(min_length=1, max_length=256)]
    reviewed: bool = False
    layers: Annotated[list[Layer], Field(min_length=1, max_length=256)]
    native_text: list[NativeText] = Field(default_factory=list)
    native_shapes: list[NativeShape] = Field(default_factory=list)
    repairs: list[Repair] = Field(default_factory=list)


class LayerError(ValueError):
    pass


def contains(outer, inner):
    x,y,w,h=outer; a,b,c,d=inner
    return x<=a and y<=b and a+c<=x+w and b+d<=y+h


def load_layers(path, *, source_sha256=None, image_size=None, texts=None, shapes=None, require_review=True):
    path=Path(path).resolve()
    if path.stat().st_size>2_000_000: raise LayerError('layer_manifest_too_large')
    try:
        manifest=LayerManifest.model_validate(read_json(path))
    except (ValidationError, ValueError, TypeError):
        raise LayerError('invalid_layer_manifest') from None
    w,h=manifest.image_size
    if min(w,h)<=0 or w*h>MAX_PIXELS: raise LayerError('invalid_layer_canvas')
    if source_sha256 is not None and manifest.source_sha256!=source_sha256: raise LayerError('layer_source_mismatch')
    if image_size is not None and list(image_size)!=manifest.image_size: raise LayerError('layer_canvas_mismatch')
    if require_review and not manifest.reviewed: raise LayerError('layer_review_required')
    ids=[l.id for l in manifest.layers]
    zs=[l.z for l in manifest.layers]+[t.z for t in manifest.native_text]+[s.z for s in manifest.native_shapes]
    if len(ids)!=len(set(ids)) or len(zs)!=len(set(zs)): raise LayerError('duplicate_layer_id_or_z')
    ordered=sorted([(l.z,l.module) for l in manifest.layers]+[(t.z,t.module) for t in manifest.native_text]+[(s.z,s.module) for s in manifest.native_shapes])
    for module in {m for _,m in ordered if m is not None}:
        indexes=[i for i,(_,m) in enumerate(ordered) if m==module]
        if indexes[-1]-indexes[0]+1!=len(indexes): raise LayerError('noncontiguous_layer_group')
    assets={}
    for layer in manifest.layers:
        if not contains([0,0,w,h],layer.bbox) or min(layer.bbox[2:])<=0: raise LayerError('layer_outside_canvas')
        if any(v!=int(v) for v in layer.bbox): raise LayerError('layer_requires_pixel_alignment')
        asset=(path.parent/layer.path).resolve()
        if Path(layer.path).is_absolute() or not asset.is_relative_to(path.parent): raise LayerError('layer_path_escape')
        if not asset.is_file() or digest(asset)!=layer.sha256: raise LayerError('layer_asset_missing_or_changed')
        with Image.open(asset) as im:
            if im.format!='PNG' or im.mode!='RGBA' or list(im.size)!=layer.bbox[2:]:
                raise LayerError('layer_requires_exact_rgba_png')
            im.load()
        assets[layer.id]=asset
    for repair in manifest.repairs:
        if not contains([0,0,w,h],repair.bbox) or min(repair.bbox[2:])<=0: raise LayerError('repair_outside_canvas')
        if require_review and not repair.reviewed: raise LayerError('repair_review_required')
    if texts is not None:
        lookup={t['id']:t for t in texts}
        native=[t.text_id for t in manifest.native_text]
        retained=[tid for l in manifest.layers for tid in l.retained_text_ids]
        all_ids=native+retained
        if len(all_ids)!=len(set(all_ids)) or set(all_ids)!=set(lookup): raise LayerError('layer_text_ownership_conflict')
        for layer in manifest.layers:
            if any(not contains(layer.bbox,lookup[tid]['bbox']) for tid in layer.retained_text_ids):
                raise LayerError('retained_text_outside_layer')
        for repair in manifest.repairs:
            if any(tid not in native or not contains(repair.bbox,lookup[tid]['bbox']) for tid in repair.text_ids):
                raise LayerError('invalid_cleanup_ownership')
        for tid in native:
            if not any(r.kind=='text_cleanup' and tid in r.text_ids and r.reviewed for r in manifest.repairs):
                raise LayerError('native_text_cleanup_unverified')
    if shapes is not None:
        lookup={s['id']:s for s in shapes}
        wanted=[s.shape_id for s in manifest.native_shapes]
        if len(wanted)!=len(set(wanted)) or not set(wanted).issubset(lookup): raise LayerError('invalid_native_shape')
        for sid in wanted:
            if not any(r.kind=='shape_cleanup' and r.reviewed and sid in r.shape_ids
                       and contains(r.bbox,lookup[sid]['bbox']) for r in manifest.repairs):
                raise LayerError('native_shape_cleanup_unverified')
        if any(sid not in wanted for r in manifest.repairs for sid in r.shape_ids): raise LayerError('invalid_shape_cleanup_ownership')
    return manifest,assets


def snapshot_layers(path, output, **checks):
    manifest,assets=load_layers(path,**checks)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    (output/'assets').mkdir()
    data=manifest.model_dump(mode='json',by_alias=True)
    for layer in data['layers']:
        destination=output/'assets'/f"{layer['id']}.png"
        shutil.copyfile(assets[layer['id']],destination)
        if digest(destination)!=layer['sha256']: raise LayerError('layer_changed_during_snapshot')
        layer['path']=destination.relative_to(output).as_posix()
    write_json(output/'layers.json',data)
    return output/'layers.json'


def preview_layers(path):
    """Raw image-layer composite, intentionally excluding native text candidates."""
    manifest,assets=load_layers(path,require_review=False)
    output=Path(path).parent/'review'
    output.mkdir(exist_ok=True)
    page=Image.new('RGBA',tuple(manifest.image_size),(0,0,0,0))
    for layer in sorted(manifest.layers,key=lambda l:l.z):
        with Image.open(assets[layer.id]) as image:
            page.alpha_composite(image,(int(layer.bbox[0]),int(layer.bbox[1])))
    page.save(output/'image-layers-only.png')
    for i,repair in enumerate(manifest.repairs):
        x,y,w,h=repair.bbox
        page.crop((int(x),int(y),int(x+w),int(y+h))).save(output/f'repair-{i:03}.png')
    write_json(output/'review.json',{'status':'REVIEW_REQUIRED','native_text_included':False,
               'note':'Inspect image layers for old text and repair artifacts before overlaying native text.',
               'repairs':[r.model_dump() for r in manifest.repairs],'human_acceptance':False})
