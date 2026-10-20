"""Conversion orchestration tests: real local reconstruction, mocked compiler."""
import json
import os
import subprocess
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image, ImageDraw

from image2ppt_vision import cli, conversion, reconstruction
from image2ppt_vision.storage import digest, read_json, write_json


@pytest.fixture
def deck_case(tmp_path,monkeypatch):
    sources=tmp_path/'图片目录';analyses=tmp_path/'分析目录';plans=tmp_path/'计划目录';compiler=tmp_path/'固定编译器'
    for directory in (sources,analyses,plans,compiler):directory.mkdir()
    names=['03-纪检.png','02-材料.png']
    for name in names:
        image=Image.new('RGB',(120,80),'white');ImageDraw.Draw(image).rectangle((20,20,70,40),fill='navy')
        source=sources/name;image.save(source)
        analysis={'schema':'VisionAnalysis','version':'1.0','image_size':[120,80],
                  'inputs':{'fixture':{'original_sha256':digest(source)}},
                  'objects':[{'id':'text-0000','kind':'text','text':'fixture text','bbox':[20,20,51,21]}]}
        plan={'source_sha256':digest(source),'image_size':[120,80],'author':{'tool':'fixture','model_id':'fixture'},
              'regions':[{'id':'panel','bbox':[0,0,120,80],'kind':'panel'}]}
        write_json(analyses/(source.stem+'.json'),analysis);write_json(plans/(source.stem+'.json'),plan)
    for name in conversion.REQUIRED_COMPILER_FILES:
        path=compiler/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('// fixture only',encoding='utf-8')
    node=tmp_path/'node.exe';node.write_bytes(b'fixture executable is never run')
    monkeypatch.setenv('VISION_NODE',str(node))
    monkeypatch.setattr(conversion.shutil,'which',lambda value:None)
    monkeypatch.setattr(reconstruction,'estimate_text',lambda *args,**kwargs:{'eligible':False,'diagnostics':{'reason':'fixture'}})
    runner=Mock(side_effect=AssertionError('test must explicitly mock the compiler result'))
    monkeypatch.setattr(conversion.subprocess,'run',runner)
    return Namespace(sources=sources,analyses=analyses,plans=plans,compiler=compiler,node=node,root=tmp_path,runner=runner)


def successful_compiler(command,**kwargs):
    build=Path(command[-1]);directory=build/'test-build';directory.mkdir()
    pptx=directory/'deck.pptx';pptx.write_bytes(b'mocked compiler artifact, not a valid production PPTX')
    preview=directory/'preview';preview.mkdir()
    count=len(command)-4
    for index in range(count):Image.new('RGB',(10,10),'white').save(preview/f'page-{index+1:03}.png')
    quality=directory/'quality.json'
    write_json(quality,{'slide_count':count,'pptx_sha256':digest(pptx),'preview':{'origin':'reimported_final_pptx'},
                        'human_acceptance':False})
    result={'output_dir':str(directory),'pptx':str(pptx),'preview':str(preview),'quality':str(quality),
            'slide_count':count,'status':'REVIEW_REQUIRED'}
    return subprocess.CompletedProcess(command,0,'fixture warning\n'+json.dumps(result,ensure_ascii=False,indent=2),'')


def run_case(case,output=None,**kwargs):
    return conversion.convert(case.sources,case.analyses,case.plans,case.compiler,
                              output or case.root/'new-conversion',**kwargs)


def test_conversion_sorted_unicode_inputs_fixed_command_and_snapshots(deck_case):
    case=deck_case;case.runner.side_effect=successful_compiler
    originals={str(path):digest(path) for root in (case.sources,case.analyses,case.plans) for path in root.iterdir()}
    result=run_case(case)
    manifest=read_json(result['conversion'])
    assert manifest['source_order']==['02-材料.png','03-纪检.png']
    assert result['slide_count']==2 and result['human_acceptance'] is False
    assert result['output_dir']==str(case.root/'new-conversion')
    assert manifest['status']=='REVIEW_REQUIRED' and manifest['stage']=='complete'
    assert all(page['status']=='REVIEW_REQUIRED' for page in manifest['pages'])
    command=case.runner.call_args.args[0];options=case.runner.call_args.kwargs
    assert command[:2]==[str(case.node),str(case.compiler/'scripts/compose.mjs')]
    assert [Path(path).parent.name for path in command[2:-2]]==['02-材料','03-纪检']
    assert command[-2:] == ['--output',str(case.root/'new-conversion'/'build')]
    assert options['shell'] is False and options['cwd']==str(case.compiler)
    assert options['creationflags']==(subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    assert options['timeout']==1800
    assert 'fixture warning' not in Path(result['conversion']).read_text('utf-8')
    for path,expected in originals.items():assert digest(path)==expected
    for name in manifest['source_order']:
        assert digest(case.root/'new-conversion'/'inputs'/'source'/name)==digest(case.sources/name)
        assert (case.root/'new-conversion'/'pages'/Path(name).stem/'authoring.json').is_file()


@pytest.mark.parametrize('change,expected',[
    ('missing_analysis','analysis_stems_mismatch'),('extra_analysis','analysis_stems_mismatch'),
    ('missing_plan','plan_stems_mismatch'),('extra_plan','plan_stems_mismatch'),
    ('missing_compiler','fixed_compiler_files_missing'),
])
def test_preflight_rejects_missing_extra_or_unfixed_inputs_with_manifest(deck_case,change,expected):
    case=deck_case
    if change=='missing_analysis':(case.analyses/'02-材料.json').unlink()
    elif change=='extra_analysis':write_json(case.analyses/'index.json',[])
    elif change=='missing_plan':(case.plans/'02-材料.json').unlink()
    elif change=='extra_plan':write_json(case.plans/'extra.json',{})
    else:(case.compiler/'scripts/compose.mjs').unlink()
    with pytest.raises(conversion.ConversionError,match=expected):run_case(case)
    manifest=read_json(case.root/'new-conversion'/'conversion.json')
    assert manifest['status']=='FAILED' and manifest['error']['code']==expected
    case.runner.assert_not_called()


def test_previous_conversion_is_preserved(deck_case):
    case=deck_case;output=case.root/'previous';output.mkdir();marker=output/'old.pptx';marker.write_bytes(b'original')
    with pytest.raises(FileExistsError):run_case(case,output)
    assert marker.read_bytes()==b'original'
    case.runner.assert_not_called()


def test_bad_source_binding_preserves_partial_manifest_and_skips_compiler(deck_case):
    case=deck_case;plan_path=case.plans/'03-纪检.json';plan=read_json(plan_path);plan['source_sha256']='b'*64;write_json(plan_path,plan)
    with pytest.raises(conversion.ConversionError,match='conversion_reconstruction_failed'):run_case(case)
    manifest=read_json(case.root/'new-conversion'/'conversion.json')
    assert [page['status'] for page in manifest['pages']]==['REVIEW_REQUIRED','FAILED']
    assert (case.root/'new-conversion'/'pages'/'02-材料'/'slideir.json').is_file()
    case.runner.assert_not_called()


@pytest.mark.parametrize('failure',['exit','timeout','malformed_json','outside_path','wrong_slide_count','wrong_quality_hash'])
def test_compiler_failures_are_retained_without_raw_diagnostics(deck_case,failure):
    case=deck_case
    def run(command,**kwargs):
        if failure=='exit':return subprocess.CompletedProcess(command,3,'secret prompt','secret API key')
        if failure=='timeout':raise subprocess.TimeoutExpired(command,10,output=b'secret prompt',stderr=b'secret API key')
        result=successful_compiler(command,**kwargs)
        data=conversion._final_json(result.stdout)
        if failure=='malformed_json':result.stdout='secret malformed output'
        elif failure=='outside_path':data['pptx']=str(case.sources/'02-材料.png');result.stdout=json.dumps(data)
        elif failure=='wrong_slide_count':data['slide_count']=8;result.stdout=json.dumps(data)
        else:
            quality=read_json(data['quality']);quality['pptx_sha256']='c'*64;write_json(data['quality'],quality)
        return result
    case.runner.side_effect=run
    with pytest.raises(conversion.ConversionError) as raised:run_case(case)
    path=case.root/'new-conversion'/'conversion.json';text=path.read_text('utf-8');manifest=json.loads(text)
    assert 'secret' not in str(raised.value) and 'secret' not in text
    assert manifest['status']=='FAILED' and manifest['stage']=='compile'
    assert all(page['status']=='REVIEW_REQUIRED' for page in manifest['pages'])
    assert len(manifest['compiler']['stdout_sha256'])==64
    assert case.runner.call_count==1


def test_cli_convert_routes_to_formal_orchestrator(deck_case,capsys):
    case=deck_case;case.runner.side_effect=successful_compiler
    output=case.root/'cli-output'
    cli.main(['convert','--input-dir',str(case.sources),'--analysis-dir',str(case.analyses),'--plan-dir',str(case.plans),
              '--compiler-root',str(case.compiler),'--output',str(output)])
    result=json.loads(capsys.readouterr().out)
    assert result['output_dir']==str(output) and result['status']=='REVIEW_REQUIRED'
    assert (output/'conversion.json').is_file()


@pytest.mark.parametrize('text',['{"x":1}\nextra','no JSON','[]','\n{"a":{}}\nnot final'])
def test_compiler_parser_requires_final_json_object(text):
    with pytest.raises(conversion.ConversionError):conversion._final_json(text)
