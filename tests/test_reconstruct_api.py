"""CLI/HTTP share the real reconstruction module; OCR/font fitting is injected.

These fixtures verify contracts and artifact parity, not real-image accuracy.
"""
import copy
import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from image2ppt_vision import __version__, cli, client, engine, models, model_provider, reconstruction
from image2ppt_vision.api import create_app
from image2ppt_vision.contracts import Request
from image2ppt_vision.plans import validate_plan
from image2ppt_vision.storage import Store, digest, read_json, write_json


@pytest.fixture
def case(tmp_path, monkeypatch):
    source=tmp_path/'source.png'
    image=Image.new('RGB',(120,80),'white');draw=ImageDraw.Draw(image)
    draw.rectangle((20,28,80,42),fill='black');image.save(source)
    store=Store(tmp_path/'store');asset=store.upload(source)
    text={'id':'text-0000','kind':'text','text':'source text','bbox':[20,28,61,15],
          'polygon':[[20,28],[81,28],[81,43],[20,43]],'score':.99}
    analysis={'schema':'VisionAnalysis','version':'1.0','image_size':[120,80],
              'objects':[text],'inputs':{asset['id']:asset}}
    plan={'schema':'ReconstructionPlan','version':'1.0','source_sha256':digest(source),'image_size':[120,80],
          'author':{'tool':'fixture','model_id':'fixture'},
          'regions':[{'id':'content','bbox':[0,0,120,80],'kind':'panel'}]}
    analysis_path=tmp_path/'analysis.json';plan_path=tmp_path/'plan.json'
    write_json(analysis_path,analysis);write_json(plan_path,plan)
    monkeypatch.setattr(models,'recognize',lambda pixels:[dict(text)])
    monkeypatch.setattr(reconstruction,'estimate_text',lambda *args,**kwargs:{
        'eligible':False,'diagnostics':{'reason':'font_fixture_retained'}})
    forbidden=Mock(side_effect=AssertionError('reconstruct must never invoke the model provider'))
    monkeypatch.setattr(model_provider,'generate_plan',forbidden)
    return Namespace(source=source,store=store,asset=asset,text=text,analysis=analysis,plan=plan,
                     analysis_path=analysis_path,plan_path=plan_path,root=tmp_path,forbidden=forbidden)


def test_reconstruct_contract_requires_plan_and_rejects_irrelevant_options(case):
    with pytest.raises(ValueError):Request(operation='reconstruct',asset_id=case.asset['id'])
    with pytest.raises(ValueError):Request(operation='analyze',asset_id=case.asset['id'],plan=case.plan)
    for options in ({'segment':True},{'bbox':[0,0,50,50]},{'mode':'prompt'}):
        with pytest.raises(ValueError):Request(operation='reconstruct',asset_id=case.asset['id'],plan=case.plan,**options)
    request=Request(operation='reconstruct',asset_id=case.asset['id'],plan=case.plan)
    assert request.model_dump()['plan']['schema']=='ReconstructionPlan'
    assert 'schema_name' not in request.model_dump()['plan']
    assert Request.model_validate_json(request.model_dump_json()).plan.source_sha256==digest(case.source)


def test_saved_cli_and_http_generate_identical_slideir(case,capsys):
    local=case.root/'local-reconstructed'
    cli.main(['reconstruct','--input',str(case.source),'--analysis',str(case.analysis_path),
              '--plan',str(case.plan_path),'--output',str(local)])
    summary=json.loads(capsys.readouterr().out)
    assert summary['status']=='REVIEW_REQUIRED' and summary['human_acceptance'] is False
    assert (local/'slideir.json').is_file()
    with TestClient(create_app(case.store.root,'test-token',run_worker=False)) as api:
        result=api.post('/v1/jobs',headers={'Authorization':'Bearer test-token'},json={
            'operation':'reconstruct','asset_id':case.asset['id'],'plan':case.plan})
        assert result.status_code==202
        job=result.json()
        engine.execute(case.store.root,job['id'])
        status=api.get('/v1/jobs/'+job['id'],headers={'Authorization':'Bearer test-token'}).json()
        assert status['status']=='SUCCEEDED'
        assert 'reconstruction/slideir.json' in status['artifacts']
        response=api.get('/v1/jobs/'+job['id']+'/artifacts/reconstruction/slideir.json',
                         headers={'Authorization':'Bearer test-token'})
        assert response.status_code==200
        assert response.json()==read_json(local/'slideir.json')
        report=read_json(case.store.path('jobs',job['id'])/'result.json')
        assert report['reconstruction']['status']=='REVIEW_REQUIRED'
        assert report['reconstruction']['human_acceptance'] is False
        assert report['environment']['code_version']=='0.3.0'
    case.forbidden.assert_not_called()


@pytest.mark.parametrize('mutation',[
    lambda p:p.update(source_sha256='b'*64),
    lambda p:p.update(image_size=[121,80]),
    lambda p:p['regions'].append({'id':'overlap','bbox':[0,0,10,10],'kind':'panel'}),
    lambda p:p.update(code='run arbitrary code'),
])
def test_http_rejects_unbound_or_invalid_plan_before_queue(case,mutation):
    plan=copy.deepcopy(case.plan);mutation(plan)
    with TestClient(create_app(case.store.root,'test-token',run_worker=False)) as api:
        response=api.post('/v1/jobs',headers={'Authorization':'Bearer test-token'},json={
            'operation':'reconstruct','asset_id':case.asset['id'],'plan':plan})
        assert response.status_code==422
    with case.store.db() as db:
        assert db.execute('SELECT count(*) FROM jobs').fetchone()[0]==0


def test_unknown_override_fails_required_reconstruction_stage(case):
    plan=copy.deepcopy(case.plan)
    plan['text_overrides']=[{'target_id':'missing','text':'invented','reason':'fixture'}]
    job=case.store.submit(Request(operation='reconstruct',asset_id=case.asset['id'],plan=plan))
    engine.execute(case.store.root,job['id'])
    assert case.store.status(job['id'])['status']=='FAILED'
    result=read_json(case.store.path('jobs',job['id'])/'result.json')
    assert result['reconstruction'] is None
    assert any(stage['name']=='reconstruction' and stage['status']=='FAIL' for stage in result['stages'])
    assert not (case.store.path('jobs',job['id'])/'reconstruction'/'slideir.json').exists()


def test_cli_preserves_previous_revision(case,capsys):
    output=case.root/'already-exists';output.mkdir();marker=output/'preserve.txt';marker.write_text('old')
    with pytest.raises(FileExistsError):
        cli.main(['reconstruct','--input',str(case.source),'--analysis',str(case.analysis_path),
                  '--plan',str(case.plan_path),'--output',str(output)])
    assert marker.read_text()=='old'


def test_cli_without_saved_analysis_runs_shared_engine(case,monkeypatch,capsys):
    from image2ppt_vision import worker
    class InlineWorker:
        def __init__(self,store):self.store=store
        def run(self,job_id):engine.execute(self.store.root,job_id)
        def stop(self):pass
    monkeypatch.setattr(worker,'Worker',InlineWorker)
    output=case.root/'fresh-analysis'
    cli.main(['--data',str(case.store.root),'reconstruct','--input',str(case.source),
              '--plan',str(case.plan_path),'--output',str(output)])
    status=json.loads(capsys.readouterr().out)
    assert status['status']=='SUCCEEDED'
    assert status['quality_status']=='REVIEW_REQUIRED'
    assert (output/'slideir.json').is_file()
    assert read_json(output/'reconstruction.json')['ocr_lines']==1
    case.forbidden.assert_not_called()


def test_remote_client_transmits_plan_and_verifies_artifacts(case,monkeypatch):
    with TestClient(create_app(case.store.root,'test-token',run_worker=False)) as api:
        def transport(base,path,payload=None,content_type='application/json'):
            response=api.request('POST' if payload is not None else 'GET',path,content=payload,
                headers={'Authorization':'Bearer test-token','Content-Type':content_type})
            assert response.status_code<400
            if path=='/v1/jobs':engine.execute(case.store.root,response.json()['id'])
            return response.content
        monkeypatch.setattr(client,'request',transport)
        monkeypatch.setattr(client.time,'sleep',lambda seconds:None)
        output=case.root/'downloaded'
        args=Namespace(command='reconstruct',url='http://local-fixture',options=None,analysis=None,
                       output=str(output),plan=str(case.plan_path),input=str(case.source),reference=None,
                       mask=None,data=str(case.root/'client-data'))
        result=client.run(args)
        assert result['status']=='SUCCEEDED' and result['verified_files']>5
        assert (output/'reconstruction'/'slideir.json').is_file()
        assert read_json(output/'request.json')['plan']['schema']=='ReconstructionPlan'
        assert read_json(output/'reconstruction'/'reconstruction.json')['human_acceptance'] is False
    case.forbidden.assert_not_called()


def test_explicit_plan_cli_is_only_provider_path(case,monkeypatch,capsys):
    requested=[]
    def generate(source,analysis,*,provenance_path):
        requested.append((source,analysis))
        Path(provenance_path).write_text('{"source_sha256":"'+digest(case.source)+'"}',encoding='utf-8')
        return validate_plan(case.plan)
    monkeypatch.setattr(model_provider,'generate_plan',generate)
    output=case.root/'generated-plan.json'
    cli.main(['plan','--input',str(case.source),'--analysis',str(case.analysis_path),'--output',str(output)])
    assert len(requested)==1
    assert read_json(output)['source_sha256']==digest(case.source)
    assert Path(str(output)+'.provenance.json').is_file()
    assert json.loads(capsys.readouterr().out)['status']=='REVIEW_REQUIRED'


def test_capabilities_report_configuration_without_secret(case,monkeypatch):
    monkeypatch.setenv('VISION_LLM_BASE_URL','https://private-provider.example/v1')
    monkeypatch.setenv('VISION_LLM_MODEL','private-model-id')
    monkeypatch.setenv('VISION_LLM_API_KEY','private-api-key')
    monkeypatch.setenv('VISION_MODELS',str(case.root/'missing-models'))
    result=models.capabilities();encoded=json.dumps(result)
    assert result['version']==__version__=='0.3.0'
    assert result['reconstruction'] is True and result['pptx_compilation'] is False
    assert result['network_inference'] is False
    assert result['model_plan_provider']=={'available':True,'automatic':False,'configured':True,'invocation':'explicit_cli_plan'}
    for private in ('private-provider','private-model-id','private-api-key'):
        assert private not in encoded
