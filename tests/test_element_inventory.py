from copy import deepcopy
import numpy as np
import cv2
import pytest
from PIL import Image

from image2ppt_vision.element_inventory import propose_elements,build_inventory
from image2ppt_vision.storage import digest


def fixture():
    pixels=np.full((300,500,3),255,np.uint8)
    cv2.rectangle(pixels,(20,30),(220,260),(180,210,230),2)
    cv2.circle(pixels,(70,80),25,(0,90,230),-1)
    cv2.rectangle(pixels,(260,40),(470,140),(210,230,245),-1)
    cv2.rectangle(pixels,(280,160),(460,250),(0,130,210),-1)
    return pixels


def test_nested_objects_survive_and_source_pixels_never_change():
    pixels=fixture();before=pixels.copy()
    candidates,mask=propose_elements(pixels,[])
    np.testing.assert_array_equal(pixels,before)
    assert len(candidates)>=4 and any(c['geometry_hint']=='ellipse' for c in candidates)
    assert any(c['parent_candidate'] for c in candidates)
    assert not any(c['native_verified'] for c in candidates)


def test_ocr_crossing_is_evidence_not_silently_clipped():
    texts=[dict(id='t',bbox=[200,250,45,20],text='fixture')]
    candidates,mask=propose_elements(fixture(),texts)
    assert mask[255,210] and any('t' in c['ocr_crossing'] for c in candidates)


def test_inventory_is_bound_immutable_and_preserves_crops(tmp_path):
    source=tmp_path/'source.png';Image.fromarray(fixture()).save(source)
    analysis=dict(image_size=[500,300],inputs={'source':{'sha256':digest(source)}},objects=[])
    out=tmp_path/'inventory';result=build_inventory(source,analysis,out)
    assert digest(source)==digest(out/'source.png')==result['source_sha256']
    assert (out/'candidate-overlay.png').exists()
    for c in result['candidates']:assert digest(out/c['crop'])==c['crop_sha256']
    with pytest.raises(FileExistsError):build_inventory(source,analysis,out)
    bad=deepcopy(analysis);bad['inputs']['source']['sha256']='0'*64
    with pytest.raises(ValueError):build_inventory(source,bad,tmp_path/'bad')
