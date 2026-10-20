"""Strict, resumable API generation. Reference documents never enter this module."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import base64
import io
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import Annotated, Literal

from pydantic import Field, StrictInt, StrictStr, ValidationError, model_validator

from .plans import _Record, Identifier, ReconstructionPlan, parse_plan_json
from .model_provider import _source, _compact_ocr, _payload, _endpoint, _completion_content
from .storage import digest, read_json, write_json


class EvaluationError(ValueError):
    """Safe code only, never a provider body, key, or source text."""


class InputPage(_Record):
    id: Identifier
    source: StrictStr
    source_sha256: Annotated[StrictStr,Field(pattern=r'^[a-f0-9]{64}$')]
    analysis: StrictStr
    analysis_sha256: Annotated[StrictStr,Field(pattern=r'^[a-f0-9]{64}$')]


class GenerationSuite(_Record):
    schema_name: Literal['IndependentAPISuite'] = Field(default='IndependentAPISuite',alias='schema')
    version: Literal['1.0','1.1']='1.0'
    pages: Annotated[list[InputPage],Field(min_length=1,max_length=14)]
    base_url: StrictStr
    model: Annotated[StrictStr,Field(min_length=1,max_length=256)]
    enable_thinking: Literal[False]=False
    max_tokens: Literal[8192]=8192
    timeout_seconds: Literal[120]=120
    max_rounds: Annotated[StrictInt,Field(gt=0)] | None=2
    max_requests: Annotated[StrictInt,Field(gt=0)] | None=28
    first_round_protocol: Literal['1.0','1.1','2.0']='1.0'

    @model_validator(mode='after')
    def legacy_budget(self):
        if self.version=='1.0' and (self.max_rounds!=2 or self.max_requests!=28 or self.first_round_protocol!='1.0'):
            raise ValueError('legacy campaign budget and protocol are immutable')
        return self


def utc():
    return datetime.now(timezone.utc).isoformat()


def json_hash(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def _suite(path):
    try:
        suite=GenerationSuite.model_validate(read_json(path))
    except (ValidationError,ValueError,TypeError):
        raise EvaluationError('invalid_generation_suite') from None
    _endpoint(suite.base_url)
    if len({p.id for p in suite.pages})!=len(suite.pages):raise EvaluationError('duplicate_page_id')
    if any(p.id in ('.','..') or '/' in p.id or '\\' in p.id or ':' in p.id for p in suite.pages):
        raise EvaluationError('invalid_page_id')
    if any(ord(c)<32 for c in suite.model):raise EvaluationError('invalid_model_id')
    return suite


@contextmanager
def campaign_lock(root):
    """OS lock is released on process exit, including an interrupted run."""
    path=Path(root)/'.campaign.lock'
    with path.open('a+b') as handle:
        if path.stat().st_size==0:handle.write(b'0');handle.flush()
        handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            raise EvaluationError('campaign_already_running') from None
        try:yield
        finally:
            handle.seek(0)
            if os.name=='nt':msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def ledger(root):
    db=sqlite3.connect(Path(root)/'budget.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS attempts (round INTEGER, page TEXT, reserved_at TEXT, PRIMARY KEY(round,page))')
    db.commit()
    return db


def reserve(db, round_no, page, maximum=28):
    db.execute('BEGIN IMMEDIATE')
    try:
        if db.execute('SELECT 1 FROM attempts WHERE round=? AND page=?',(round_no,page)).fetchone():
            db.rollback();return False
        if maximum is not None and db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]>=maximum:raise EvaluationError('request_budget_exhausted')
        db.execute('INSERT INTO attempts VALUES(?,?,?)',(round_no,page,utc()));db.commit();return True
    except BaseException:
        db.rollback();raise


def _copy_checked(source, target, expected):
    if digest(source)!=expected:raise EvaluationError('input_hash_mismatch')
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    if digest(target)!=expected:raise EvaluationError('input_changed_during_snapshot')


def _freeze_code(compiler_root, destination):
    root=Path(compiler_root).resolve()
    for directory in ('scripts','src/image2ppt','services/vision/src/image2ppt_vision'):
        for source in (root/directory).rglob('*'):
            if source.is_file() and source.suffix in ('.py','.mjs','.json') and '__pycache__' not in source.parts:
                target=destination/source.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,target)
    if (root/'runtime.local.json').is_file():
        config=read_json(root/'runtime.local.json')
        write_json(destination/'runtime.local.json',{k:config[k] for k in ('node_modules','python','presentations_skill') if k in config})
    if not (destination/'scripts/compose.mjs').is_file():raise EvaluationError('compiler_missing')
    _pin_runtime(root,destination)


def _pin_runtime(root,destination):
    """Pin the local runtime and font candidates without collecting environment secrets."""
    from .typography import _candidates
    profile=Path(os.environ.get('USERPROFILE',str(Path.home())))
    primary=profile/'.cache/codex-runtimes/codex-primary-runtime/dependencies'
    config=read_json(root/'runtime.local.json') if (root/'runtime.local.json').exists() else {}
    modules=next((Path(v).resolve() for v in (os.environ.get('IMAGE2PPT_NODE_MODULES'),config.get('node_modules'),primary/'node/node_modules',root/'.runtime/bundle/codex-primary-runtime/dependencies/node/node_modules')
                  if v and (Path(v)/'@oai/artifact-tool/package.json').is_file()),None)
    bundled_node=primary/'node/bin/node.exe'
    node=Path(os.environ.get('VISION_NODE') or (str(bundled_node) if bundled_node.is_file() else shutil.which('node') or '')).resolve()
    python=Path(os.environ.get('IMAGE2PPT_PYTHON') or config.get('python') or primary/'python/python.exe').resolve()
    skill=Path(config['presentations_skill']).resolve() if config.get('presentations_skill') else None
    if skill is None:
        base=profile/'.codex/plugins/cache/openai-primary-runtime/presentations'
        skill=next((p/'skills/presentations' for p in sorted(base.iterdir(),reverse=True) if (p/'skills/presentations/container_tools/artifact_tool_utils.mjs').is_file()),None) if base.is_dir() else None
    if not modules or not node.is_file() or not python.is_file() or skill is None:raise EvaluationError('runtime_not_available_for_freeze')
    write_json(destination/'runtime.local.json',{'node_modules':str(modules),'python':str(python),'presentations_skill':str(skill)})
    fonts=[{**f,'sha256':digest(f['path'])} for f in _candidates(None)]
    external={str(p):digest(p) for p in (node,python,modules/'@oai/artifact-tool/package.json')}
    external.update({str(p):digest(p) for p in (skill/'container_tools').rglob('*') if p.is_file() and p.suffix in ('.py','.mjs','.json')})
    external.update({f['path']:f['sha256'] for f in fonts})
    versions={n:importlib.metadata.version(n) for n in ('pydantic','Pillow','numpy')}
    write_json(destination/'runtime-pin.json',{'node':str(node),'python_version':sys.version,'packages':versions,'fonts':fonts,'external_files':external})


def _verify_runtime(directory):
    path=Path(directory)/'frozen/runtime-pin.json'
    if not path.exists():return  # Historical prepared rounds remain readable.
    pin=read_json(path)
    if sys.version!=pin['python_version'] or any(importlib.metadata.version(n)!=v for n,v in pin['packages'].items()):raise EvaluationError('frozen_python_runtime_changed')
    if any(not Path(p).is_file() or digest(p)!=sha for p,sha in pin['external_files'].items()):raise EvaluationError('frozen_external_runtime_changed')


def _request(page, suite, protocol='1.0'):
    sha,size,blob=_source(page/'source.png')
    analysis=read_json(page/'analysis.json');ocr=_compact_ocr(analysis,size,sha)
    payload=_payload(sha,size,blob,ocr,suite.model)
    payload.update(enable_thinking=False,max_tokens=8192)
    if protocol=='1.0':
        # Keep the legacy structure proposal for round 1, but publish the complete
        # machine schema rather than an abbreviated hand-written approximation.
        schema=ReconstructionPlan.model_json_schema(by_alias=True)
        if 'tables' in schema.get('properties',{}):
            schema['properties'].pop('tables',None)
            schema['properties']['version']={'const':'1.0','type':'string'}
            for name in ('TableProposal','TableMerge'):schema.get('$defs',{}).pop(name,None)
        payload['messages'][0]['content']+='\nComplete JSON schema (version 1.0 only): '+json.dumps(schema,ensure_ascii=False)
    else:
        from .table_reconstruction import table_planning_instruction
        from PIL import Image
        import numpy as np
        if protocol=='2.0':
            from .plans_v2 import planning_instruction_v2
            payload['messages'][0]['content']=planning_instruction_v2()
        else:payload['messages'][0]['content']=table_planning_instruction()
        # Preserve source pixels for reconstruction; reduce only the transport
        # image with one universal, same-resolution policy for round two.
        with Image.open(io.BytesIO(blob)) as image:
            image=image.convert('RGB');buffer=io.BytesIO()
            image.save(buffer,format='JPEG',quality=95,subsampling=0,optimize=True)
            encoded=buffer.getvalue()
            with Image.open(io.BytesIO(encoded)) as check:
                mae=float(np.abs(np.asarray(image,dtype=float)-np.asarray(check.convert('RGB'),dtype=float)).mean())
        use_jpeg=mae<=2.0 and len(encoded)<len(blob)
        transmitted=encoded if use_jpeg else blob
        media='jpeg' if use_jpeg else 'png';name='model-input.'+('jpg' if use_jpeg else 'png')
        (page/name).write_bytes(transmitted)
        write_json(page/'request-image.json',{'policy':'same-resolution JPEG95 4:4:4 when mean RGB error <=2 and smaller; otherwise PNG',
            'source_sha256':sha,'source_dimensions':size,'transmitted_sha256':hashlib.sha256(transmitted).hexdigest(),
            'transmitted_bytes':len(transmitted),'normalized_png_bytes':len(blob),'mean_rgb_error':mae if use_jpeg else 0.,'file':name})
        payload['messages'][1]['content'][1]['image_url']['url']='data:image/'+media+';base64,'+base64.b64encode(transmitted).decode('ascii')
    return payload


def prepare_round(config, output, compiler_root, round_no, *, resume=False):
    config=Path(config).resolve();suite=_suite(config);root=Path(output).resolve()
    if isinstance(round_no,bool) or not isinstance(round_no,int) or round_no<1 or (suite.max_rounds is not None and round_no>suite.max_rounds):
        raise EvaluationError('invalid_round')
    root.mkdir(parents=True,exist_ok=True)
    campaign=suite.model_dump(mode='json',by_alias=True)
    if suite.version=='1.0':campaign.pop('first_round_protocol',None)
    # Source paths are frozen separately; no reference path is a valid field.
    identity={**campaign,'pages':[{'id':p.id,'source_sha256':p.source_sha256,'analysis_sha256':p.analysis_sha256} for p in suite.pages]}
    campaign_file=root/'campaign.json'
    if campaign_file.exists():
        if read_json(campaign_file)['identity_sha256']!=json_hash(identity):raise EvaluationError('campaign_inputs_or_budget_changed')
    else:write_json(campaign_file,{'schema':'APIEvaluationCampaign','version':'1.0','created_at':utc(),
                                 'identity_sha256':json_hash(identity),'settings':identity})
    destination=root/f'round-{round_no}'
    if destination.exists():
        if not resume:raise EvaluationError('round_exists_use_resume')
        verify_round(destination);return destination
    if round_no>1:
        previous=root/f'round-{round_no-1}/generation.seal.json'
        if not previous.is_file():raise EvaluationError('round_one_not_finished' if round_no==2 else 'previous_round_not_finished')
        verify_generation(previous.parent)
        report=read_json(previous.parent/'generation.json')
        if report.get('requests_reserved',0)!=len(suite.pages):raise EvaluationError('round_one_incomplete' if round_no==2 else 'previous_round_incomplete')
    destination.mkdir();code=destination/'frozen';code.mkdir()
    _freeze_code(compiler_root,code)
    settings={k:campaign[k] for k in ('base_url','model','enable_thinking','max_tokens','timeout_seconds')}
    write_json(destination/'settings.json',settings)
    rows=[]
    protocol=suite.first_round_protocol if round_no==1 or suite.first_round_protocol=='2.0' else '1.1'
    for item in suite.pages:
        page=destination/'inputs'/item.id;page.mkdir(parents=True)
        _copy_checked((config.parent/item.source).resolve(),page/'source.png',item.source_sha256)
        _copy_checked((config.parent/item.analysis).resolve(),page/'analysis.json',item.analysis_sha256)
        payload=_request(page,suite,protocol)
        write_json(page/'request.json',payload)
        rows.append({'id':item.id,'source_sha256':item.source_sha256,'analysis_sha256':item.analysis_sha256,
                     'request_sha256':digest(page/'request.json')})
    write_json(destination/'round.json',{'schema':'APIEvaluationRound','version':'1.0','round':round_no,
               'created_at':utc(),'protocol':protocol,'pages':rows,
               'session_correction':False,'request_retries':0,'budget':suite.max_requests})
    if round_no>=2 or suite.version=='1.1':
        metadata=read_json(destination/'round.json')
        metadata['scheduling']={'minimum_request_start_interval_seconds':65,'on_http_429':'stop_unattempted_pages_without_retry'}
        write_json(destination/'round.json',metadata)
    bootstrap="from pathlib import Path\nimport sys\nr=Path(__file__).resolve().parent\nsys.path.insert(0,str(r/'frozen/services/vision/src'))\nfrom image2ppt_vision.api_evaluation import execute_round\nprint(__import__('json').dumps(execute_round(r)))\n"
    (destination/'run.py').write_text(bootstrap,encoding='utf-8')
    hashes={p.relative_to(destination).as_posix():digest(p) for p in destination.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    write_json(destination/'freeze.json',{'files':hashes,'created_at':utc()})
    return destination


def verify_round(directory):
    directory=Path(directory).resolve()
    for relative,expected in read_json(directory/'freeze.json')['files'].items():
        file=(directory/relative).resolve()
        if not file.is_relative_to(directory) or not file.is_file() or digest(file)!=expected:
            raise EvaluationError('frozen_round_changed')
    _verify_runtime(directory)


def _transport(directory, page):
    folder=directory/'attempts'/page
    folder.mkdir(parents=True,exist_ok=False)
    code=("import sys;sys.path.insert(0,sys.argv[1]);"
          "from image2ppt_vision.api_eval_worker import exchange;exchange(*sys.argv[2:])")
    command=[sys.executable,'-I','-c',code,str(directory/'frozen/services/vision/src'),
             str(directory/'inputs'/page/'request.json'),str(directory/'settings.json'),str(folder)]
    start=time.monotonic()
    process=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    try:
        process.wait(timeout=read_json(directory/'settings.json')['timeout_seconds'])
    except subprocess.TimeoutExpired:
        process.kill();process.wait()
        write_json(folder/'transport.json',{'status':'FAILED','error_code':'model_wall_timeout',
                   'elapsed_seconds':round(time.monotonic()-start,4),'budget_consumed':True})
    if not (folder/'transport.json').exists():
        write_json(folder/'transport.json',{'status':'FAILED','error_code':'model_worker_interrupted','budget_consumed':True})


def _compile(directory, irs, output):
    from .conversion import _compiler, _final_json
    root=directory/'frozen'
    pin=directory/'frozen/runtime-pin.json'
    if pin.exists():node=Path(read_json(pin)['node'])
    else:root,node=_compiler(root)
    # The unchanged compiler's finalizer confines publication to its own ROOT.
    # Build there, then copy verified immutable bytes into the round's page/deck
    # directory; never loosen the finalizer's workspace boundary.
    staging=root/'.api-staging'/__import__('uuid').uuid4().hex
    cmd=[str(node),str(root/'scripts/compose.mjs'),*map(str,irs),'--output',str(staging)]
    env={k:v for k,v in os.environ.items() if k not in ('IMAGE2PPT_NODE_MODULES','IMAGE2PPT_PYTHON','VISION_LLM_API_KEY','VISION_LAYER_API_KEY')}
    completed=subprocess.run(cmd,cwd=root,shell=False,capture_output=True,timeout=1800,env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/('compiler-'+__import__('uuid').uuid4().hex[:10]+'.json'),{
        'returncode':completed.returncode,'stdout_sha256':hashlib.sha256(completed.stdout).hexdigest(),
        'stderr_sha256':hashlib.sha256(completed.stderr).hexdigest(),'staging':str(staging.relative_to(directory))})
    if completed.returncode:raise EvaluationError('compiler_failed')
    result=_final_json(completed.stdout)
    for field in ('pptx','preview','quality'):
        path=Path(result[field]).resolve()
        if not path.is_relative_to(staging.resolve()) or not path.exists():raise EvaluationError('compiler_artifact_invalid')
    built=Path(result['output_dir']).resolve();published=output/built.name
    if not built.is_relative_to(staging):raise EvaluationError('compiler_artifact_invalid')
    shutil.copytree(built,published)
    for field in ('pptx','preview','quality','compare','output_dir'):
        if field in result:result[field]=str(published/Path(result[field]).resolve().relative_to(built))
    quality=read_json(result['quality'])
    if digest(result['pptx'])!=quality['pptx_sha256']:raise EvaluationError('copied_compiler_artifact_changed')
    return result


def _finish_page(directory, item, metadata):
    """Deterministic continuation can consume an already saved response, never call API."""
    from .reconstruction import reconstruct
    page=item['id'];attempt=directory/'attempts'/page;target=directory/'pages'/page
    target.mkdir(parents=True,exist_ok=True)
    row={'id':page,'status':'FAILED','stage':'transport','request_count':1}
    if not (attempt/'transport.json').is_file():
        row['error_code']='reserved_attempt_interrupted';return row
    transport=read_json(attempt/'transport.json');row['transport']=transport
    if transport['status']!='RECEIVED':row['error_code']=transport['error_code'];return row
    raw=(attempt/'response.body').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=transport['saved_response_sha256']:raise EvaluationError('saved_response_changed')
    row['stage']='plan_validation'
    try:
        envelope=json.loads(raw)
        row['model_returned']=envelope.get('model')
        usage=envelope.get('usage')
        row['usage']={k:v for k,v in (usage or {}).items() if isinstance(v,(int,float,dict))} if isinstance(usage,dict) else None
        content=_completion_content(raw)
        inputs=directory/'inputs'/page;analysis=read_json(inputs/'analysis.json')
        texts=[t for t in analysis['objects'] if t['kind']=='text']
        if metadata['protocol']=='2.0':
            from .plans_v2 import parse_plan_v2,lower_plan_v2,materialize_groups_v2
            plan=parse_plan_v2(content,source_sha256=item['source_sha256'],image_size=analysis['image_size'],texts=texts)
        else:
            plan=parse_plan_json(content,source_sha256=item['source_sha256'],image_size=analysis['image_size'],
                            ocr_ids={t['id'] for t in texts})
        settings=read_json(directory/'settings.json')
        if plan.version!=metadata['protocol']:raise EvaluationError('plan_protocol_mismatch')
        if plan.author.model_id!=settings['model'] or plan.author.tool!='vision-model-plan':raise EvaluationError('plan_author_mismatch')
        write_json(target/'plan.json',plan.to_dict())
        render_plan=plan
        if plan.version=='2.0':
            render_plan,lowering=lower_plan_v2(plan,texts)
            write_json(target/'lowered-plan.json',render_plan.to_dict())
            write_json(target/'lowering.json',lowering)
        row['stage']='reconstruction'
        revision=target/('reconstruct-'+__import__('uuid').uuid4().hex[:10])
        pin=directory/'frozen/runtime-pin.json'
        report=reconstruct(inputs/'source.png',analysis,render_plan,revision,**({'font_candidates':read_json(pin)['fonts']} if pin.exists() else {}))
        if plan.version=='2.0':
            report=materialize_groups_v2(plan,lowering,revision,report)
        row.update(ir=str((revision/'slideir.json').relative_to(directory)),
                   native_text=report['native_text'],native_tables=report.get('native_tables',0),
                   retained_raster_text_count=len(report['retained_raster_text']))
        row['stage']='compile'
        compiled=_compile(directory,[revision/'slideir.json'],target/'build')
        row.update(status='SUCCEEDED',stage='complete',artifacts={k:str(Path(compiled[k]).relative_to(directory)) for k in ('pptx','preview','quality')})
    except Exception as exc:
        row['error_code']=str(exc) if isinstance(exc,EvaluationError) else row['stage']+'_failed'
        row['error_type']=type(exc).__name__
    write_json(target/'validation.json',{'status':row['status'],'stage':row['stage'],
               'error_code':row.get('error_code'),'human_acceptance':False})
    return row


def execute_round(directory):
    directory=Path(directory).resolve();verify_round(directory)
    root=directory.parent;metadata=read_json(directory/'round.json');round_no=metadata['round']
    with campaign_lock(root):
        if (directory/'generation.seal.json').exists():
            verify_generation(directory);return read_json(directory/'generation.json')
        db=ledger(root)
        rows={r['id']:r for r in read_json(directory/'generation.json').get('pages',[])} if (directory/'generation.json').exists() else {}
        report={'schema':'APIGenerationResult','version':'1.0','round':round_no,'status':'RUNNING','pages':[],
                'source_order':[p['id'] for p in metadata['pages']],'requests_reserved':0,'human_acceptance':False}
        auth_stopped=False;rate_stopped=False;last_start=None
        try:
            key=os.environ.get('VISION_LLM_API_KEY','')
            configured=bool(key.strip()) and all(32<=ord(c)<=126 for c in key)
            for item in metadata['pages']:
                page=item['id']
                new_attempt=False
                existing=db.execute('SELECT 1 FROM attempts WHERE round=? AND page=?',(round_no,page)).fetchone()
                if page in rows and rows[page].get('status') in ('SUCCEEDED','FAILED') and existing:
                    row=rows[page]
                elif existing:
                    row=_finish_page(directory,item,metadata)
                elif not configured or auth_stopped or rate_stopped:
                    row={'id':page,'status':'NOT_RUN','stage':'configuration',
                         'error_code':'model_configuration_required' if not configured else 'service_rate_limit_stopped' if rate_stopped else 'authentication_stopped','request_count':0}
                else:
                    interval=metadata.get('scheduling',{}).get('minimum_request_start_interval_seconds',0)
                    if last_start is not None and interval:
                        remaining=interval-(time.monotonic()-last_start)
                        if remaining>0:time.sleep(remaining)
                    reserve(db,round_no,page,maximum=metadata.get('budget',28))
                    last_start=time.monotonic()
                    new_attempt=True
                    try:_transport(directory,page)
                    except Exception:
                        pass  # Reservation remains consumed. Never retry an ambiguous send.
                    row=_finish_page(directory,item,metadata)
                if new_attempt and row.get('error_code') in ('model_http_401','model_http_403'):auth_stopped=True
                if new_attempt and row.get('error_code')=='model_http_429' and metadata.get('scheduling',{}).get('on_http_429')=='stop_unattempted_pages_without_retry':rate_stopped=True
                report['pages'].append(row)
                report['requests_reserved']=db.execute('SELECT COUNT(*) FROM attempts WHERE round=?',(round_no,)).fetchone()[0]
                report['campaign_requests_reserved']=db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
                write_json(directory/'generation.json',report)
            succeeded=[r for r in report['pages'] if r['status']=='SUCCEEDED']
            if succeeded:
                try:
                    result=_compile(directory,[directory/r['ir'] for r in succeeded],directory/'deck')
                    report['deck']={**{k:str(Path(result[k]).relative_to(directory)) for k in ('pptx','preview','quality')},
                                    'source_order':[r['id'] for r in succeeded],
                                    'complete_14_pages':len(succeeded)==14}
                except Exception:
                    report['deck_error']='deck_compilation_failed'
            report.update(status='NOT_RUN_CONFIGURATION' if not configured else 'STOPPED_AUTH' if auth_stopped else 'STOPPED_RATE_LIMIT' if rate_stopped else
                          'GENERATED' if len(succeeded)==len(metadata['pages']) and 'deck' in report else 'PARTIAL' if succeeded else 'FAILED',
                          succeeded_pages=len(succeeded),expected_pages=len(metadata['pages']),completed_at=utc())
            write_json(directory/'generation.json',report)
            if configured and not any(r['status']=='NOT_RUN' for r in report['pages']):
                files={p.relative_to(directory).as_posix():digest(p) for folder in ('pages','attempts','deck') for p in (directory/folder).rglob('*') if p.is_file()}
                files['generation.json']=digest(directory/'generation.json');files['freeze.json']=digest(directory/'freeze.json')
                write_json(directory/'generation.seal.json',{'files':files,'completed_at':utc()})
            return report
        finally:db.close()


def verify_generation(directory):
    directory=Path(directory).resolve();verify_round(directory)
    for relative,sha in read_json(directory/'generation.seal.json')['files'].items():
        path=(directory/relative).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or digest(path)!=sha:raise EvaluationError('generation_changed_after_seal')


def evaluate_api(config, output, compiler_root, round_no=1, *, resume=False, prepare_only=False):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    with campaign_lock(output):
        directory=prepare_round(config,output,compiler_root,round_no,resume=resume)
    if prepare_only:return {'status':'PREPARED','round':str(directory),'network_calls':0}
    # Execute the saved implementation, not imports from the mutable checkout.
    child=subprocess.run([sys.executable,'-I',str(directory/'run.py')],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if child.returncode:raise EvaluationError('frozen_generation_failed')
    return read_json(directory/'generation.json')
