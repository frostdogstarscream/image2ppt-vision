from copy import deepcopy
import base64
import io
import json

import numpy as np
from PIL import Image
import pytest

from image2ppt_vision.layers import load_layers, snapshot_layers, LayerError
from image2ppt_vision.storage import digest, read_json, write_json
from image2ppt_vision import decomposition, reconstruction
from image2ppt_vision import cli
from test_reconstruction import fixture_inputs, text, fitted, retained, image_pixels


@pytest.fixture
def layered(tmp_path):
    pixels=np.full((60,100,3),255,np.uint8);pixels[20:30,20:40]=0
    case=fixture_inputs(tmp_path,pixels,[text('t1',[20,20,20,10])],
                        [{'id':'card','bbox':[0,0,100,60],'kind':'panel'}])
    clean=tmp_path/'clean.png';Image.new('RGBA',(100,60),'white').save(clean)
    data={'schema':'LayerManifest','version':'1.0','source_sha256':digest(case.source),
          'image_size':[100,60],'provider':'import','model_id':'synthetic-fixture','reviewed':True,
          'layers':[{'id':'background','path':'clean.png','sha256':digest(clean),'bbox':[0,0,100,60],
                     'z':0,'module':'card','retained_text_ids':[]}],
          'native_text':[{'text_id':'t1','z':1,'module':'card'}],
          'repairs':[{'bbox':[20,20,20,10],'kind':'text_cleanup','text_ids':['t1'],'reviewed':True}]}
    path=tmp_path/'layers.json';write_json(path,data)
    return case,path,data


@pytest.mark.parametrize('damage,code',[
    ('source','layer_source_mismatch'),('review','layer_review_required'),
    ('duplicate','layer_text_ownership_conflict'),('missing','layer_text_ownership_conflict'),
    ('cleanup','native_text_cleanup_unverified'),('size','layer_requires_exact_rgba_png'),
    ('hash','layer_asset_missing_or_changed'),('escape','layer_path_escape'),
    ('z','duplicate_layer_id_or_z'),('repair','repair_review_required')])
def test_layer_contract_rejects_damage(layered,damage,code):
    case,path,data=layered
    if damage=='source': data['source_sha256']='0'*64
    if damage=='review': data['reviewed']=False
    if damage=='duplicate': data['layers'][0]['retained_text_ids']=['t1']
    if damage=='missing': data['native_text']=[];data['repairs']=[]
    if damage=='cleanup': data['repairs']=[]
    if damage=='size': data['layers'][0]['bbox']=[0,0,99,60]
    if damage=='hash': data['layers'][0]['sha256']='0'*64
    if damage=='escape': data['layers'][0]['path']='../outside.png'
    if damage=='z': data['native_text'][0]['z']=0
    if damage=='repair': data['repairs'][0]['reviewed']=False
    write_json(path,data)
    with pytest.raises(LayerError,match=code):
        load_layers(path,source_sha256=digest(case.source),texts=case.analysis['objects'])


def test_layered_native_materialization_and_snapshot(layered,monkeypatch):
    case,path,data=layered
    monkeypatch.setattr(reconstruction,'estimate_text',lambda p,t,**kw:fitted(p,t))
    report=reconstruction.reconstruct(case.source,case.analysis,case.plan,case.output,layers=path)
    ir=read_json(case.output/'slideir.json')
    assert report['native_text']==1 and report['human_acceptance'] is False
    assert [n['type'] for n in ir['nodes']]==['image','text']
    assert len(ir['groups'])==1
    assert ir['assets'][0]['contains_evidence_ids']==['layer-image-background-pixels']
    with Image.open(case.output/ir['assets'][0]['path']) as im:
        assert im.getpixel((25,25))==(255,255,255,255)
    assert (case.output/'layers/layers.json').is_file()
    assert read_json(case.output/'authoring.json')['ir_sha256']==digest(case.output/'slideir.json')


def test_reject_unavailable_native_typography(layered,monkeypatch):
    case,path,_=layered;monkeypatch.setattr(reconstruction,'estimate_text',retained)
    with pytest.raises(LayerError,match='requested_native_typography_unavailable'):
        reconstruction.reconstruct(case.source,case.analysis,case.plan,case.output,layers=path)
    assert not (case.output/'slideir.json').exists()


def test_import_preserves_pixels_and_no_network(layered,monkeypatch,tmp_path):
    case,path,_=layered
    monkeypatch.setattr(decomposition,'_fetch',lambda *a:pytest.fail('offline import called network'))
    result=decomposition.decompose(case.source,tmp_path/'imported',provider='import',manifest=path)
    m,assets=load_layers(result['manifest'],texts=case.analysis['objects'])
    assert digest(assets['background'])==digest(path.parent/'clean.png')
    with pytest.raises(FileExistsError):
        decomposition.decompose(case.source,tmp_path/'imported',provider='import',manifest=path)


def test_ming_one_request_candidate_and_no_secrets(layered,monkeypatch,tmp_path):
    case,_,_=layered; calls=[]
    monkeypatch.setenv('VISION_LAYER_API_KEY','test-secret-never-save')
    encoded=base64.b64encode((tmp_path/'clean.png').read_bytes()).decode()
    def fetch(req,limit,timeout):
        calls.append(req)
        assert req.full_url.endswith('/images/edits')
        assert b'name="image"' in req.data and b'ming-image-0.1-design-layer' in req.data
        return json.dumps({'data':[{'b64_json':encoded}]}).encode()
    monkeypatch.setattr(decomposition,'_fetch',fetch)
    output=tmp_path/'ming'
    result=decomposition.decompose(case.source,output,provider='ming')
    assert len(calls)==1
    assert read_json(result['manifest'])['reviewed'] is False
    with pytest.raises(LayerError,match='layer_review_required'): load_layers(result['manifest'])
    assert 'test-secret' not in (output/'decomposition.json').read_text()


def test_ming_timeout_is_not_retried(layered,monkeypatch,tmp_path):
    case,_,_=layered;calls=[]
    monkeypatch.setenv('VISION_LAYER_API_KEY','test-secret')
    def fail(*args): calls.append(1);raise TimeoutError('sensitive payload')
    monkeypatch.setattr(decomposition,'_fetch',fail)
    with pytest.raises(LayerError,match='layer_provider_failed'):
        decomposition.decompose(case.source,tmp_path/'failed',provider='ming')
    report=read_json(tmp_path/'failed/decomposition.json')
    assert calls==[1] and report['attempts']==1 and report['status']=='FAILED'
    assert 'sensitive' not in str(report)


def test_cross_boundary_native_text_is_above_modules(tmp_path,monkeypatch):
    pixels=np.full((60,100,3),255,np.uint8);pixels[20:30,40:60]=0
    case=fixture_inputs(tmp_path,pixels,[text('cross',[40,20,20,10])],
                        [{'id':'card','bbox':[0,0,50,60],'kind':'panel'}])
    monkeypatch.setattr(reconstruction,'estimate_text',lambda p,t,**kw:fitted(p,t))
    report=reconstruction.reconstruct(case.source,case.analysis,case.plan,case.output)
    ir=read_json(case.output/'slideir.json')
    assert ir['nodes'][-1]['id']=='cross'
    assert report['typography'][0]['region']=='page-remainder'
    assert not any('cross' in g['member_ids'] for g in ir['groups'])


def test_cross_boundary_retained_text_has_one_complete_owner(tmp_path,monkeypatch):
    pixels=np.full((60,100,3),255,np.uint8);pixels[20:30,40:60]=0
    case=fixture_inputs(tmp_path,pixels,[text('cross',[40,20,20,10])],
                        [{'id':'card','bbox':[0,0,50,60],'kind':'panel'}])
    monkeypatch.setattr(reconstruction,'estimate_text',retained)
    reconstruction.reconstruct(case.source,case.analysis,case.plan,case.output)
    ir=read_json(case.output/'slideir.json')
    owners=[a for a in ir['assets'] if 'cross' in a['contains_evidence_ids']]
    assert len(owners)==1 and owners[0]['id']=='crossing-text-000'
    np.testing.assert_array_equal(image_pixels(ir,case.output),pixels)


def test_layered_shape_requires_cleaning_and_verified_source(layered,monkeypatch):
    case,path,data=layered
    pixels=case.pixels.copy();pixels[40:50,60:90]=[20,80,180]
    Image.fromarray(pixels).save(case.source)
    case.analysis['inputs']['source']['sha256']=digest(case.source)
    case.plan['source_sha256']=digest(case.source);data['source_sha256']=digest(case.source)
    case.plan['shapes']=[{'id':'bar','bbox':[60,40,30,10],'geometry':'rect','fill':'1450B4'}]
    data['native_shapes']=[{'shape_id':'bar','z':2,'module':'card'}]
    write_json(path,data)
    with pytest.raises(LayerError,match='native_shape_cleanup_unverified'):
        load_layers(path,shapes=case.plan['shapes'])
    data['repairs'].append({'kind':'shape_cleanup','bbox':[60,40,30,10],'shape_ids':['bar'],'reviewed':True})
    write_json(path,data)
    monkeypatch.setattr(reconstruction,'estimate_text',lambda p,t,**kw:fitted(p,t))
    report=reconstruction.reconstruct(case.source,case.analysis,case.plan,case.output,layers=path)
    ir=read_json(case.output/'slideir.json')
    assert report['native_verified_shapes']==1
    assert ir['nodes'][-1]['type']=='shape' and ir['nodes'][-1]['id']=='bar'


def test_cli_import_and_layered_reconstruct(layered,monkeypatch,capsys):
    case,path,_=layered
    analysis=path.parent/'analysis.json';plan=path.parent/'plan.json'
    write_json(analysis,case.analysis);write_json(plan,case.plan)
    imported=path.parent/'imported-cli'
    cli.main(['decompose','--provider','import','--input',str(case.source),'--manifest',str(path),'--output',str(imported)])
    assert json.loads(capsys.readouterr().out)['status']=='REVIEW_REQUIRED'
    assert (imported/'review/image-layers-only.png').is_file()
    monkeypatch.setattr(reconstruction,'estimate_text',lambda p,t,**kw:fitted(p,t))
    cli.main(['reconstruct','--input',str(case.source),'--analysis',str(analysis),'--plan',str(plan),
              '--layers',str(imported/'layers.json'),'--output',str(case.output)])
    assert json.loads(capsys.readouterr().out)['native_text']==1


def test_cli_layered_convert_forwards_directory(monkeypatch,capsys):
    from image2ppt_vision import conversion
    captured={}
    def convert(*args,**kwargs): captured.update(kwargs);return {'status':'fixture'}
    monkeypatch.setattr(conversion,'convert',convert)
    cli.main(['convert','--input-dir','source','--analysis-dir','analysis','--plan-dir','plans',
              '--compiler-root','root','--output','new','--layers-dir','layers'])
    assert captured=={'layers_dir':'layers'}
