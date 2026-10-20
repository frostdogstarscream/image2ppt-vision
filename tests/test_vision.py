import json
from unittest.mock import patch
import numpy as np
import pytest
from PIL import Image,ImageDraw
from fastapi.testclient import TestClient
from image2ppt_vision.storage import Store,read_json,digest
from image2ppt_vision.contracts import Request
from image2ppt_vision.engine import execute
from image2ppt_vision.api import create_app
from image2ppt_vision.algorithms import geometry,compare,extract
from image2ppt_vision.benchmark import match,edit_distance


@pytest.fixture
def fixture(tmp_path):
    p=tmp_path/'中文图片.png';im=Image.new('RGB',(240,160),'white');d=ImageDraw.Draw(im)
    d.rectangle((20,20,120,100),outline='blue',width=3);d.ellipse((160,30,210,80),outline='red',width=3);im.save(p)
    store=Store(tmp_path/'data');a=store.upload(p);return store,a,p


def test_geometry_detects_fixture(fixture):
    _,_,p=fixture
    nodes=geometry(np.array(Image.open(p)))
    assert any(n['kind']=='rectangle' for n in nodes)
    assert any(n['kind']=='circle' for n in nodes)


def test_execution_manifest_and_original(fixture):
    s,a,p=fixture;before=digest(p)
    j=s.submit(Request(operation='analyze',asset_id=a['id'],ocr=False));execute(s.root,j['id'])
    assert s.status(j['id'])['status']=='SUCCEEDED';directory=s.path('jobs',j['id'])
    for name,r in read_json(directory/'manifest.json').items():assert digest(directory/name)==r['sha256']
    assert digest(p)==before
    assert s.submit(Request(operation='analyze',asset_id=a['id']))['id']!=j['id']


def test_missing_model_partial(fixture,monkeypatch,tmp_path):
    monkeypatch.setenv('VISION_MODELS',str(tmp_path/'missing'))
    s,a,_=fixture;j=s.submit(Request(operation='analyze',asset_id=a['id']));execute(s.root,j['id'])
    assert s.status(j['id'])['status']=='PARTIAL'


@pytest.mark.parametrize('id',['../x','https://example.org','a'*31,'z'*32])
def test_bad_id(id):
    with pytest.raises(ValueError):Request(operation='analyze',asset_id=id)


@pytest.mark.parametrize('box',[[-1,0,1,1],[0,0,0,1],[0,0,1,-1]])
def test_bad_box(box):
    with pytest.raises(ValueError):Request(operation='extract',asset_id='a'*32,bbox=box)


def test_bounds(fixture):
    s,a,_=fixture;j=s.submit(Request(operation='extract',asset_id=a['id'],bbox=[0,0,500,500]))
    with pytest.raises(ValueError):execute(s.root,j['id'])


def test_prompt_crop_contract_and_coordinates():
    from image2ppt_vision.engine import validate_bounds
    with pytest.raises(ValueError):Request(operation='segment',asset_id='a'*32,crop_prompt=True)
    r=Request(operation='segment',asset_id='a'*32,mode='prompt',bbox=[20,20,40,40],crop_prompt=True,points=[[0,0]],labels=[0])
    with pytest.raises(ValueError,match='outside prompt crop'):validate_bounds(r,100,100)
    r.points=[(30,30)];validate_bounds(r,100,100)


def test_queue_limit_and_recover(fixture):
    s,a,_=fixture
    for _ in range(20):s.submit(Request(operation='analyze',asset_id=a['id']))
    with pytest.raises(ValueError,match='queue full'):s.submit(Request(operation='analyze',asset_id=a['id']))
    id=s.next_job();s.update(id,'RUNNING');s.recover();assert s.status(id)['status']=='INTERRUPTED'
    assert 'request.json' in read_json(s.path('jobs',id)/'manifest.json')
    assert s.next_job()!=id


def test_disk_reserve(fixture):
    s,_,p=fixture
    with patch('image2ppt_vision.storage.shutil.disk_usage') as mock:
        mock.return_value.free=1
        with pytest.raises(ValueError,match='disk reserve'):s.upload(p)


def test_bad_image(fixture,tmp_path):
    s,_,_=fixture;p=tmp_path/'bad.png';p.write_bytes(b'bad')
    with pytest.raises(OSError):s.upload(p)


def test_multiframe_rejected(fixture,tmp_path):
    s,_,_=fixture;p=tmp_path/'animated.png';im=Image.new('RGB',(10,10));im.save(p,save_all=True,append_images=[Image.new('RGB',(10,10),'red')])
    with pytest.raises(ValueError):s.upload(p)


def test_alpha_diagnostics(tmp_path):
    a=np.full((20,20,3),100,dtype='uint8');m=np.zeros((20,20),dtype=bool);m[5:15,5:15]=True
    r=extract(a,m,None,tmp_path);assert r['mask_pixels']==100
    assert np.array(Image.open(tmp_path/'transparent.png'))[0,0,3]==0
    assert digest(tmp_path/'dark.png')!=digest(tmp_path/'white.png')


def test_compare_explicit_resize(tmp_path):
    a=np.zeros((20,20,3),dtype='uint8');b=np.zeros((10,10,3),dtype='uint8')
    with pytest.raises(ValueError):compare(a,b,tmp_path)
    assert compare(a,b,tmp_path,resize=True)['before']['mae']==0


def test_http_auth_and_artifact(fixture):
    s,a,p=fixture
    with TestClient(create_app(s.root,'test-secret',run_worker=False)) as c:
        assert c.get('/v1/capabilities').status_code==401
        headers={'Authorization':'Bearer test-secret'}
        with p.open('rb') as f:r=c.post('/v1/assets',headers=headers,files={'file':('中文.png',f,'image/png')})
        assert r.status_code==200
        j=c.post('/v1/jobs',headers=headers,json={'operation':'analyze','asset_id':r.json()['id'],'ocr':False}).json()
        execute(s.root,j['id'])
        url=f"/v1/jobs/{j['id']}/artifacts/result.json"
        assert c.get(url,headers=headers).status_code==200
        (s.path('jobs',j['id'])/'result.json').write_text('{}')
        assert c.get(url,headers=headers).status_code==404
        assert c.get(f"/v1/jobs/{j['id']}/artifacts/request-not-listed",headers=headers).status_code==404


def test_metrics_counterexamples():
    assert edit_distance('正确文字','错误文字')==2
    assert not match([{'bbox':[100,100,5,5]}],[{'bbox':[0,0,5,5]}],.5)
    assert not match([{'kind':'circle','bbox':[0,0,5,5]}],[{'kind':'rectangle','bbox':[0,0,5,5]}],.7)


def test_segmentation_metrics_do_not_hide_failures():
    from image2ppt_vision.benchmark import mask_metrics
    left=np.zeros((20,20),dtype=bool);left[:,:8]=True
    right=np.zeros((20,20),dtype=bool);right[:,12:]=True
    assert mask_metrics([right],[left],True)['status']=='FAIL'
    merged=np.logical_or(left,right)
    report=mask_metrics([merged,merged],[left,right],False)
    assert report['possible_merged_masks']==2 and report['duplicate_pairs']==1
    assert report['mode']=='automatic' and report['status']=='REVIEW_REQUIRED'


def sleep_child(root,connection):
    import time
    connection.recv();time.sleep(30)


def test_timeout_only_owned_worker_and_recreation(fixture,monkeypatch):
    from image2ppt_vision import worker as module
    s,a,_=fixture;original=module.child
    monkeypatch.setattr(module,'child',sleep_child)
    worker=module.Worker(s)
    first=s.submit(Request(operation='analyze',asset_id=a['id'],ocr=False,timeout_seconds=1))
    worker.run(first['id'])
    assert s.status(first['id'])['error_type']=='Timeout'
    assert not worker.process.is_alive()
    monkeypatch.setattr(module,'child',original)
    second=s.submit(Request(operation='analyze',asset_id=a['id'],ocr=False))
    try:
        worker.run(second['id']);assert s.status(second['id'])['status']=='SUCCEEDED'
    finally:worker.stop()


def test_oom_preserves_other_stages(fixture):
    s,a,_=fixture;j=s.submit(Request(operation='analyze',asset_id=a['id'],ocr=False,segment=True))
    class OutOfMemoryError(RuntimeError):pass
    with patch('image2ppt_vision.models.segment',side_effect=OutOfMemoryError()):execute(s.root,j['id'])
    assert s.status(j['id'])['status']=='PARTIAL'
    result=read_json(s.path('jobs',j['id'])/'result.json')
    assert result['objects'] and any(x.get('error_type')=='OutOfMemoryError' for x in result['stages'])


def test_mask_size_and_binary(fixture,tmp_path):
    s,a,_=fixture;p=tmp_path/'mask.png';Image.new('L',(10,10),255).save(p)
    mask=s.upload(p)
    j=s.submit(Request(operation='extract',asset_id=a['id'],mask_id=mask['id']));execute(s.root,j['id'])
    assert s.status(j['id'])['status']=='FAILED'
    Image.new('L',(240,160),100).save(p);mask=s.upload(p)
    j=s.submit(Request(operation='extract',asset_id=a['id'],mask_id=mask['id']));execute(s.root,j['id'])
    assert s.status(j['id'])['status']=='FAILED'


def test_input_limits(fixture,monkeypatch):
    import image2ppt_vision.storage as module
    s,a,p=fixture
    monkeypatch.setattr(module,'MAX_BYTES',1)
    with pytest.raises(ValueError,match='40 MB'):s.upload(p)
    monkeypatch.setattr(module,'MAX_BYTES',40_000_000);monkeypatch.setattr(module,'MAX_PIXELS',10)
    with pytest.raises(ValueError,match='million pixels'):s.upload(p)


def test_exif_mapping(fixture,tmp_path):
    s,_,_=fixture;p=tmp_path/'rotated.jpg';im=Image.new('RGB',(60,40));exif=im.getexif();exif[274]=6;im.save(p,exif=exif)
    result=s.upload(p)
    assert result['size']==[40,60]
    assert result['original_to_normalized_edges_matrix']==[[0,-1,40],[1,0,0],[0,0,1]]


def test_compare_records_original_resize_mapping(fixture,tmp_path):
    s,a,_=fixture;p=tmp_path/'small.png';Image.new('RGB',(120,80),'white').save(p)
    ref=s.upload(p)
    j=s.submit(Request(operation='compare',asset_id=a['id'],reference_id=ref['id'],resize_reference=True,bbox=[0,0,100,50]))
    execute(s.root,j['id'])
    comparison=read_json(s.path('jobs',j['id'])/'result.json')['comparison']
    assert comparison['original_reference_size']==[120,80]
    assert comparison['reference_to_source_scale']==[2,2]
    assert comparison['region']==[0,0,100,50]


def test_http_decompression_bomb_returns_validation_error(fixture):
    s,_,p=fixture
    with TestClient(create_app(s.root,'test-secret',run_worker=False)) as c:
        with patch('image2ppt_vision.storage.Image.open',side_effect=Image.DecompressionBombError()):
            with p.open('rb') as f:
                r=c.post('/v1/assets',headers={'Authorization':'Bearer test-secret'},files={'file':('page.png',f,'image/png')})
        assert r.status_code==422
