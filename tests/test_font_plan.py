from copy import deepcopy
import math
import pytest
from test_supervised_elements import fixture
from image2ppt_vision.supervised_elements import materialize_elements, validate_element_plan
from image2ppt_vision.font_contract import resolve_font
from image2ppt_vision.font_plan import apply_font_plan, prepare_font_job, verify_font_export
from image2ppt_vision.element_calibration import calibrate_elements
from image2ppt_vision.storage import read_json, write_json, digest


def style():
    return {'id':'heading','role':'heading','font_family':'Microsoft YaHei','font_latin':'Arial',
            'font_size_px':26,'bold':True,'font_confidence':.7,'font_evidence':'Consistent sans-serif stroke widths; uncertain family.'}


def setup(tmp_path):
    source, inv, plan=fixture(tmp_path)
    folder=tmp_path/'input'
    materialize_elements(source,inv,plan,folder,model_id='fixture',response_sha256='0'*64)
    irfile=folder/'slideir.json'
    p={'schema':'FontPlan/1','source_sha256':digest(source),'input_ir_sha256':digest(irfile),
       'styles':[style()],'bindings':[{'text_id':'t1','style_id':'heading'}],'limitations':[]}
    return irfile,p


def test_alias_resolution_and_explicit_missing_font():
    assert resolve_font(style(),['微软雅黑','Arial'])=={'ea':'微软雅黑','latin':'Arial'}
    with pytest.raises(ValueError,match='unavailable'):resolve_font(style(),['Arial'])


def test_v2_demands_font_metadata_while_v1_remains_compatible(tmp_path):
    _,inv,p=fixture(tmp_path);validate_element_plan(p,inv)
    p['schema']='ElementPlan/2'
    with pytest.raises(ValueError,match='font family'):validate_element_plan(p,inv)
    p['text_decisions'][0].update({k:v for k,v in style().items() if k.startswith('font_')})
    validate_element_plan(p,inv)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),True,0,401])
def test_invalid_size_is_rejected_without_output(tmp_path,value):
    f,p=setup(tmp_path);p['styles'][0]['font_size_px']=value
    with pytest.raises(ValueError,match='typography'):
        apply_font_plan(f,p,tmp_path/'output',model_id='fixture',response_sha256='0'*64,available=['微软雅黑','Arial'])
    assert not (tmp_path/'output').exists()


def test_font_only_preserves_words_geometry_and_rich_text(tmp_path):
    f,p=setup(tmp_path);ir=read_json(f);n=ir['nodes'][-1]
    n['runs']=[{'text':'Cor','color':'FF0000','bold':False},{'text':'rected','bold':True}]
    write_json(f,ir);a=read_json(f.parent/'authoring.json');a['ir_sha256']=digest(f);write_json(f.parent/'authoring.json',a)
    p['input_ir_sha256']=digest(f);oldhash=digest(f)
    r=apply_font_plan(f,p,tmp_path/'output',model_id='fixture',response_sha256='0'*64,available=['微软雅黑','Arial'])
    new=read_json(tmp_path/'output/slideir.json')['nodes'][-1]
    assert new['font']=={'ea':'微软雅黑','latin':'Arial','size_pt':62.4}
    assert new['runs']==n['runs'] and new['bbox']==n['bbox'] and new['text']==n['text']
    assert r['changed']==1 and digest(f)==oldhash


@pytest.mark.parametrize('change',['source','duplicate','unknown_style','missing_font','extra_style'])
def test_bad_plan_never_written(tmp_path,change):
    f,p=setup(tmp_path)
    if change=='source':p['source_sha256']='0'*64
    if change=='duplicate':p['bindings']*=2
    if change=='unknown_style':p['bindings'][0]['style_id']='missing'
    if change=='missing_font':p['styles'][0]['font_family']='Not Installed'
    if change=='extra_style':p['styles'].append({**style(),'id':'unused'})
    with pytest.raises(ValueError):apply_font_plan(f,p,tmp_path/'output',model_id='fixture',response_sha256='0'*64,available=['微软雅黑','Arial'])
    assert not (tmp_path/'output').exists()


def test_calibration_only_suggests_for_model_locked_fonts(tmp_path,monkeypatch):
    f,p=setup(tmp_path)
    apply_font_plan(f,p,tmp_path/'output',model_id='fixture',response_sha256='0'*64,available=['微软雅黑','Arial'])
    monkeypatch.setattr('image2ppt_vision.element_calibration.estimate_text',lambda *a:{'native_eligible':True,'font_family':'宋体','font_size_px':12,'bold':False,'color':'#333333','bbox':[0,0,1,1],'diagnostics':{}})
    calibrate_elements(tmp_path/'output/slideir.json',tmp_path/'calibrated')
    before=read_json(tmp_path/'output/slideir.json')['nodes'][-1]
    after=read_json(tmp_path/'calibrated/slideir.json')['nodes'][-1]
    assert before==after
    report=read_json(tmp_path/'calibrated/calibration.json')
    row=next(r for r in report['diagnostics'] if r['kind']=='typography')
    assert not row['applied'] and row['suggestion']['font_family']=='宋体'


def test_preparation_omits_old_fonts_and_never_calls_api(tmp_path,monkeypatch):
    f,_=setup(tmp_path)
    monkeypatch.setattr('image2ppt_vision.font_plan.font_status',lambda:{'families':['微软雅黑','Arial']})
    result=prepare_font_job(f,tmp_path/'job',model='test/model',base_url='https://example.invalid/v1')
    request=read_json(tmp_path/'job/request.json')
    assert not result['api_called'] and not (tmp_path/'job/response.body').exists()
    import json
    data=json.loads(request['messages'][1]['content'][0]['text'])
    assert set(data['fixed_texts'][0])=={'id','text','bbox'}


@pytest.mark.parametrize('tamper',[None,'family','size','weight','text'])
def test_export_guard_rejects_changed_font_or_text(tmp_path,tamper):
    from zipfile import ZipFile
    import xml.etree.ElementTree as ET
    from image2ppt_vision.text_ledger import NS
    f,p=setup(tmp_path)
    apply_font_plan(f,p,tmp_path/'output',model_id='fixture',response_sha256='0'*64,available=['微软雅黑','Arial'])
    target=tmp_path/'output/slideir.json';n=read_json(target)['nodes'][-1]
    root=ET.Element('{'+NS['p']+'}sld');sp=ET.SubElement(root,'{'+NS['p']+'}sp')
    props=ET.SubElement(sp,'{'+NS['p']+'}nvSpPr');ET.SubElement(props,'{'+NS['p']+'}cNvPr',name=n['id'])
    body=ET.SubElement(sp,'{'+NS['p']+'}txBody');para=ET.SubElement(body,'{'+NS['a']+'}p');run=ET.SubElement(para,'{'+NS['a']+'}r')
    rpr=ET.SubElement(run,'{'+NS['a']+'}rPr',sz=str(round(n['font']['size_pt']*100)),b='1')
    ET.SubElement(rpr,'{'+NS['a']+'}ea',typeface=n['font']['ea']);ET.SubElement(rpr,'{'+NS['a']+'}latin',typeface=n['font']['latin'])
    t=ET.SubElement(run,'{'+NS['a']+'}t');t.text=n['text']
    if tamper=='family':rpr.find('a:ea',NS).set('typeface','黑体')
    if tamper=='size':rpr.set('sz','100')
    if tamper=='weight':rpr.set('b','0')
    if tamper=='text':t.text='changed'
    pptx=tmp_path/'test.pptx'
    with ZipFile(pptx,'w') as z:z.writestr('ppt/slides/slide1.xml',ET.tostring(root))
    if tamper is None:assert verify_font_export(target,pptx)['status']=='PASS'
    else:
        with pytest.raises(ValueError,match='mismatch'):verify_font_export(target,pptx)
