"""Strict API budget/replay tests. Transport and compiler are fixtures, not models."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest
from PIL import Image

from image2ppt_vision import api_evaluation as api
from image2ppt_vision import reconstruction
from image2ppt_vision.storage import digest,read_json,write_json


@pytest.fixture
def suite(tmp_path,monkeypatch):
    pages=[]
    for identifier in ('02','03'):
        source=tmp_path/(identifier+'.png');Image.new('RGB',(100,60),'white').save(source)
        analysis=tmp_path/(identifier+'.json')
        write_json(analysis,{'schema':'VisionAnalysis','version':'1.0','image_size':[100,60],
                   'inputs':{'source':{'sha256':digest(source)}},'objects':[]})
        pages.append({'id':identifier,'source':source.name,'source_sha256':digest(source),
                      'analysis':analysis.name,'analysis_sha256':digest(analysis)})
    config=tmp_path/'suite.json'
    write_json(config,{'schema':'IndependentAPISuite','pages':pages,'model':'fixture-model','base_url':'https://fixture.example/v1'})
    monkeypatch.setenv('VISION_LLM_API_KEY','fixture-secret')
    root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/compose.mjs').is_file())
    directory=api.prepare_round(config,tmp_path/'campaign',root,1)
    return config,directory,root


def fixture_transport(directory,page,*,invalid=False,status=None):
    target=directory/'attempts'/page;target.mkdir(parents=True)
    sha=digest(directory/'inputs'/page/'source.png')
    plan={'schema':'ReconstructionPlan','version':read_json(directory/'round.json')['protocol'],'source_sha256':sha,'image_size':[100,60],
          'author':{'tool':'vision-model-plan','model_id':'fixture-model'},
          'regions':[{'id':'body','bbox':[0,0,100,60],'kind':'panel'}]}
    content='not json' if invalid else json.dumps(plan)
    raw=json.dumps({'model':'fixture-returned','usage':{'prompt_tokens':12,'completion_tokens':25},
                    'choices':[{'finish_reason':'stop','message':{'content':content}}]}).encode()
    (target/'response.body').write_bytes(raw)
    write_json(target/'transport.json',{'status':'FAILED' if status else 'RECEIVED',
               'error_code':status,'saved_response_sha256':hashlib.sha256(raw).hexdigest()})


def fixture_compiler(directory,irs,output):
    output.mkdir(parents=True,exist_ok=True)
    (output/'deck.pptx').write_bytes(b'fixture not a production artifact')
    (output/'preview').mkdir(exist_ok=True)
    for i in range(len(irs)):Image.new('RGB',(100,60),'white').save(output/'preview'/f'page-{i+1:03}.png')
    write_json(output/'quality.json',{'fixture':True})
    return {k:str(output/v) for k,v in {'pptx':'deck.pptx','preview':'preview','quality':'quality.json'}.items()}


def test_reference_and_unknown_fields_rejected_before_generation(suite):
    config,_,_=suite;data=read_json(config);data['reference_pptx']='secret.pptx';write_json(config,data)
    with pytest.raises(api.EvaluationError,match='invalid_generation_suite'):api._suite(config)


def test_v2_request_and_frozen_lowering_are_explicit(suite,monkeypatch):
    config,_,root=suite
    data=read_json(config);data.update(version='1.1',first_round_protocol='2.0',max_rounds=None,max_requests=None)
    write_json(config,data)
    directory=api.prepare_round(config,config.parent/'v2-campaign',root,1)
    request=read_json(directory/'inputs/02/request.json')
    assert 'version 2.0' in request['messages'][0]['content']
    assert (directory/'frozen/services/vision/src/image2ppt_vision/plans_v2.py').is_file()
    for page in ('02','03'):
        target=directory/'attempts'/page;target.mkdir(parents=True)
        plan=dict(schema='ReconstructionPlan',version='2.0',source_sha256=digest(directory/'inputs'/page/'source.png'),
                  image_size=[100,60],author=dict(tool='vision-model-plan',model_id='fixture-model'))
        raw=json.dumps(dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(plan)))])).encode()
        (target/'response.body').write_bytes(raw)
        write_json(target/'transport.json',dict(status='RECEIVED',saved_response_sha256=hashlib.sha256(raw).hexdigest()))
    monkeypatch.setattr(api,'_compile',fixture_compiler)
    row=api._finish_page(directory,read_json(directory/'round.json')['pages'][0],read_json(directory/'round.json'))
    assert row['status']=='SUCCEEDED'
    assert read_json(directory/'pages/02/plan.json')['version']=='2.0'
    assert read_json(directory/'pages/02/lowered-plan.json')['version']=='1.1'
    assert read_json(directory/'pages/02/lowering.json')['default_text_action']=='attempt_native_after_pixel_verification'


def test_one_request_per_page_and_sealed_resume_never_retries(suite,monkeypatch):
    _,directory,_=suite;calls=[]
    def transport(d,p):calls.append(p);fixture_transport(d,p)
    monkeypatch.setattr(api,'_transport',transport);monkeypatch.setattr(api,'_compile',fixture_compiler)
    report=api.execute_round(directory)
    assert report['status']=='GENERATED' and report['requests_reserved']==2 and calls==['02','03']
    assert report['pages'][0]['model_returned']=='fixture-returned'
    assert report['pages'][0]['usage']['completion_tokens']==25
    assert api.execute_round(directory)==report and calls==['02','03']
    assert (directory/'generation.seal.json').is_file()
    payload=read_json(directory/'inputs/02/request.json')
    assert payload['enable_thinking'] is False and payload['max_tokens']==8192
    assert 'reference_pptx' not in str(payload)


def test_invalid_response_is_saved_and_next_page_continues(suite,monkeypatch):
    _,d,_=suite
    monkeypatch.setattr(api,'_transport',lambda d,p:fixture_transport(d,p,invalid=p=='02'))
    monkeypatch.setattr(api,'_compile',fixture_compiler)
    r=api.execute_round(d)
    assert r['status']=='PARTIAL' and r['requests_reserved']==2
    assert r['pages'][0]['stage']=='plan_validation' and r['pages'][0]['status']=='FAILED'
    assert r['deck']['source_order']==['03'] and r['deck']['complete_14_pages'] is False
    assert b'not json' in (d/'attempts/02/response.body').read_bytes()


@pytest.mark.parametrize('status',['model_http_401','model_http_403'])
def test_auth_failure_stops_further_calls(suite,monkeypatch,status):
    _,d,_=suite;calls=[]
    def transport(d,p):calls.append(p);fixture_transport(d,p,status=status)
    monkeypatch.setattr(api,'_transport',transport)
    r=api.execute_round(d)
    assert r['status']=='STOPPED_AUTH' and calls==['02']
    assert r['pages'][1]['status']=='NOT_RUN'
    assert not (d/'generation.seal.json').exists()


def test_missing_credentials_reserves_nothing_and_can_resume(suite,monkeypatch):
    _,d,_=suite;monkeypatch.delenv('VISION_LLM_API_KEY')
    monkeypatch.setattr(api,'_transport',lambda *a:pytest.fail('network called without key'))
    r=api.execute_round(d)
    assert r['status']=='NOT_RUN_CONFIGURATION' and r['requests_reserved']==0
    monkeypatch.setenv('VISION_LLM_API_KEY','fixture-key')
    monkeypatch.setattr(api,'_transport',fixture_transport);monkeypatch.setattr(api,'_compile',fixture_compiler)
    assert api.execute_round(d)['requests_reserved']==2


def test_auth_resume_attempts_only_remaining_pages(suite,monkeypatch):
    _,d,_=suite;calls=[]
    def denied(d,p):calls.append(p);fixture_transport(d,p,status='model_http_401')
    monkeypatch.setattr(api,'_transport',denied);assert api.execute_round(d)['status']=='STOPPED_AUTH'
    def allowed(d,p):calls.append(p);fixture_transport(d,p)
    monkeypatch.setattr(api,'_transport',allowed);monkeypatch.setattr(api,'_compile',fixture_compiler)
    assert api.execute_round(d)['status']=='PARTIAL' and calls==['02','03']


def test_reserved_but_interrupted_attempt_never_sent_again(suite,monkeypatch):
    _,d,_=suite;db=api.ledger(d.parent);api.reserve(db,1,'02');db.close();calls=[]
    def transport(d,p):calls.append(p);fixture_transport(d,p)
    monkeypatch.setattr(api,'_transport',transport);monkeypatch.setattr(api,'_compile',fixture_compiler)
    r=api.execute_round(d)
    assert calls==['03'] and r['requests_reserved']==2
    assert r['pages'][0]['error_code']=='reserved_attempt_interrupted'


def test_saved_response_resume_performs_no_second_call(suite,monkeypatch):
    _,d,_=suite;db=api.ledger(d.parent);api.reserve(db,1,'02');db.close();fixture_transport(d,'02')
    calls=[]
    def transport(d,p):calls.append(p);fixture_transport(d,p)
    monkeypatch.setattr(api,'_transport',transport);monkeypatch.setattr(api,'_compile',fixture_compiler)
    assert api.execute_round(d)['status']=='GENERATED'
    assert calls==['03']


def test_freeze_and_generation_seal_detect_changes(suite,monkeypatch):
    _,d,_=suite;monkeypatch.setattr(api,'_transport',fixture_transport);monkeypatch.setattr(api,'_compile',fixture_compiler)
    api.execute_round(d)
    (d/'attempts/02/response.body').write_text('tampered')
    with pytest.raises(api.EvaluationError,match='generation_changed_after_seal'):api.execute_round(d)
    (d/'inputs/02/analysis.json').write_text('{}')
    with pytest.raises(api.EvaluationError,match='frozen_round_changed'):api.verify_round(d)


def test_budget_is_atomic_and_bounded(tmp_path):
    db=api.ledger(tmp_path)
    for i in range(28):assert api.reserve(db,1,str(i))
    assert api.reserve(db,1,'0') is False
    with pytest.raises(api.EvaluationError,match='request_budget_exhausted'):api.reserve(db,2,'new')
    db.close()


def test_unlimited_budget_still_deduplicates_every_round_page(tmp_path):
    db=api.ledger(tmp_path)
    for i in range(40):assert api.reserve(db,1,str(i),maximum=None)
    assert api.reserve(db,1,'0',maximum=None) is False
    assert api.reserve(db,2,'0',maximum=None) is True
    assert db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]==41
    db.close()


def test_unlimited_campaign_requires_new_explicit_version(suite,monkeypatch):
    config,old,root=suite
    original=read_json(config);unlimited={**original,'max_rounds':None,'max_requests':None}
    write_json(config,unlimited)
    with pytest.raises(api.EvaluationError,match='invalid_generation_suite'):api._suite(config)
    unlimited.update(version='1.1',first_round_protocol='1.1');write_json(config,unlimited)
    with pytest.raises(api.EvaluationError,match='campaign_inputs_or_budget_changed'):
        api.prepare_round(config,old.parent,root,1,resume=True)
    campaign=old.parent.parent/'unlimited';calls=[]
    def transport(d,p):calls.append((read_json(d/'round.json')['round'],p));fixture_transport(d,p)
    monkeypatch.setattr(api,'_transport',transport);monkeypatch.setattr(api,'_compile',fixture_compiler)
    monkeypatch.setattr(api.time,'sleep',lambda seconds:None)
    for number in (1,2,3):
        directory=api.prepare_round(config,campaign,root,number)
        metadata=read_json(directory/'round.json')
        assert metadata['protocol']=='1.1' and metadata['budget'] is None and metadata['request_retries']==0
        result=api.execute_round(directory)
        assert result['status']=='GENERATED' and result['requests_reserved']==2
        assert api.execute_round(directory)==result
    assert len(calls)==6 and len(set(calls))==6
    with pytest.raises(api.EvaluationError,match='previous_round_not_finished'):
        api.prepare_round(config,campaign,root,5)
    for invalid in (0,-1,True):
        with pytest.raises(api.EvaluationError,match='invalid_round'):api.prepare_round(config,campaign,root,invalid)


def test_legacy_campaign_identity_and_budget_remain_unchanged(suite):
    config,directory,root=suite
    assert 'first_round_protocol' not in read_json(directory.parent/'campaign.json')['settings']
    assert api.prepare_round(config,directory.parent,root,1,resume=True)==directory
    with pytest.raises(api.EvaluationError,match='invalid_round'):api.prepare_round(config,directory.parent,root,3)


def test_round_two_cannot_start_before_first_finishes(suite):
    config,d,root=suite
    with pytest.raises(api.EvaluationError,match='round_one_not_finished'):
        api.prepare_round(config,d.parent,root,2)


def test_second_round_image_transport_preserves_source_binding_and_freezes_policy(suite,monkeypatch):
    config,d,root=suite
    monkeypatch.setattr(api,'_transport',fixture_transport);monkeypatch.setattr(api,'_compile',fixture_compiler)
    api.execute_round(d)
    second=api.prepare_round(config,d.parent,root,2)
    metadata=read_json(second/'round.json');image=read_json(second/'inputs/02/request-image.json')
    assert metadata['scheduling']['on_http_429']=='stop_unattempted_pages_without_retry'
    assert image['source_sha256']==digest(d/'inputs/02/source.png')
    assert image['source_dimensions']==[100,60] and image['mean_rgb_error']<=2
    assert image['transmitted_sha256']==digest(second/'inputs/02'/image['file'])
    calls=[]
    def transport(d,p):calls.append(p);fixture_transport(d,p,status='model_http_429')
    monkeypatch.setattr(api,'_transport',transport)
    result=api.execute_round(second)
    assert calls==['02'] and result['status']=='STOPPED_RATE_LIMIT'
    assert result['pages'][1]['request_count']==0 and result['pages'][1]['error_code']=='service_rate_limit_stopped'
    assert not (second/'generation.seal.json').exists()


def test_compiler_failure_keeps_page_failed(suite,monkeypatch):
    _,d,_=suite;monkeypatch.setattr(api,'_transport',fixture_transport)
    monkeypatch.setattr(api,'_compile',lambda *a:(_ for _ in ()).throw(api.EvaluationError('compiler_failed')))
    r=api.execute_round(d)
    assert r['status']=='FAILED' and all(p['stage']=='compile' for p in r['pages'])
    assert all((d/'pages'/p/'plan.json').exists() for p in ('02','03'))


def test_wall_timeout_kills_only_owned_child(suite,monkeypatch):
    _,d,_=suite
    class Process:
        killed=False
        def wait(self,timeout=None):
            if timeout is not None:raise subprocess.TimeoutExpired('fixture',timeout)
        def kill(self):self.killed=True
    process=Process()
    monkeypatch.setattr(api.subprocess,'Popen',lambda *a,**kw:process)
    api._transport(d,'02')
    assert process.killed
    assert read_json(d/'attempts/02/transport.json')['error_code']=='model_wall_timeout'


def test_response_redaction_precedes_save(tmp_path,monkeypatch):
    from image2ppt_vision import api_eval_worker as worker
    from io import BytesIO
    request=tmp_path/'request.json';request.write_text('{}')
    settings=tmp_path/'settings.json';write_json(settings,{'base_url':'https://fixture.example/v1','timeout_seconds':120})
    monkeypatch.setenv('VISION_LLM_API_KEY','echo-fixture-secret')
    class Opener:
        def open(self,*a,**kw):return BytesIO(b'{"error":"echo-fixture-secret"}')
    monkeypatch.setattr(worker.request,'build_opener',lambda *a:Opener())
    worker.exchange(request,settings,tmp_path)
    assert b'echo-fixture-secret' not in (tmp_path/'response.body').read_bytes()
    assert read_json(tmp_path/'transport.json')['error_code']=='credential_echo_rejected'
