from copy import deepcopy
import json
import pytest
from PIL import Image
from image2ppt_vision.supervised_elements import validate_element_plan,materialize_elements,parse_response,apply_element_patch,raster_text_conflicts
from image2ppt_vision.storage import digest

def fixture(tmp_path):
    source=tmp_path/'s.png';Image.new('RGB',(400,200),'white').save(source)
    inv={'source_sha256':digest(source),'image_size':[400,200],'texts':[{'id':'t1','text':'OCR','bbox':[100,50,100,25]}],'candidates':[]}
    plan={'schema':'ElementPlan/1','source_sha256':digest(source),'image_size':[400,200],'background_color':'FFFFFF','objects':[{'id':'card','kind':'native_shape','bbox':[20,20,300,130],'geometry':'roundRect','fill':'EEEEFF','z':0},{'id':'icon','kind':'raster_asset','bbox':[25,35,50,50],'mask':'ellipse','contains_text_ids':[],'z':1}], 'text_decisions':[{'id':'t1','text':'Corrected','action':'native','bbox':[100,50,100,25],'font_size_px':24,'color':'000000','bold':False,'z':2}], 'limitations':['synthetic fixture only']}
    return source,inv,plan

def test_reject_page_raster_and_duplicate_ocr(tmp_path):
    _,inv,p=fixture(tmp_path);p['objects'][1]['bbox']=[0,0,400,200]
    with pytest.raises(ValueError,match='large raster'):validate_element_plan(p,inv)
    _,inv,p=fixture(tmp_path);p['text_decisions'].append(deepcopy(p['text_decisions'][0]))
    with pytest.raises(ValueError,match='exactly once'):validate_element_plan(p,inv)

def test_reject_hidden_original_text_under_native(tmp_path):
    _,inv,p=fixture(tmp_path);p['objects'][1]['bbox']=[95,45,110,40]
    with pytest.raises(ValueError,match='duplicates raster'):validate_element_plan(p,inv)

def test_retained_text_requires_actual_owner(tmp_path):
    _,inv,p=fixture(tmp_path);p['text_decisions'][0]['action']='retained_in_asset'
    with pytest.raises(ValueError,match='retention mismatch'):validate_element_plan(p,inv)
    p['objects'][1].update(bbox=[95,45,110,40],contains_text_ids=['t1'],mask='rect')
    validate_element_plan(p,inv)

def test_materialization_preserves_source_and_masks_white_interior(tmp_path):
    source,inv,p=fixture(tmp_path);out=tmp_path/'out'
    r=materialize_elements(source,inv,p,out,model_id='fixture',response_sha256='0'*64)
    assert digest(source)==inv['source_sha256'] and r['native_shapes']==1 and not r['human_acceptance']
    with Image.open(out/'assets/icon.png') as im:
        assert im.getpixel((25,25))==(255,255,255,255) and im.getpixel((0,0))[3]==0
    ir=json.loads((out/'slideir.json').read_text(encoding='utf8'))
    assert ir['nodes'][-1]['text_override_reason'] and len(ir['evidence'])==3
    with pytest.raises(FileExistsError):materialize_elements(source,inv,p,out,model_id='fixture',response_sha256='0'*64)

def test_truncated_model_plan_never_materialized(tmp_path):
    f=tmp_path/'response.body';f.write_text(json.dumps({'choices':[{'finish_reason':'length','message':{'content':'{}'}}]}))
    with pytest.raises(ValueError,match='incomplete'):parse_response(f)

def test_patch_preserves_original_and_cannot_change_source(tmp_path):
    _,inv,p=fixture(tmp_path);before=deepcopy(p)
    patch={'schema':'ElementPatch/1','source_sha256':p['source_sha256'],'objects':[{'id':'card','fill':'FFFFFF'}]}
    result=apply_element_patch(p,patch)
    assert p==before and result['objects'][0]['fill']=='FFFFFF'
    validate_element_plan(result,inv)
    patch['source_sha256']='bad'
    with pytest.raises(ValueError,match='source mismatch'):apply_element_patch(p,patch)

def test_collects_all_false_ocr_icon_conflicts(tmp_path):
    _,inv,p=fixture(tmp_path);p['objects'][1]['bbox']=[95,45,110,40]
    conflicts=raster_text_conflicts(p,inv)
    assert len(conflicts)==1 and conflicts[0]['text_id']=='t1' and conflicts[0]['asset_id']=='icon'

def test_gradient_without_fallback_fill_is_canonicalized_without_mutating_response(tmp_path):
    _,inv,p=fixture(tmp_path);p['objects'][0].pop('fill');p['objects'][0]['fill_gradient']={'angle_deg':0,'stops':[{'offset':0,'color':'FFFFFF'},{'offset':100000,'color':'AABBCC'}]};p['objects'][0]['line']=None
    result=validate_element_plan(p,inv)
    assert result['objects'][0]['fill']=='FFFFFF' and 'line' not in result['objects'][0] and 'fill' not in p['objects'][0]


def _structure_plan(p):
    p=deepcopy(p)
    p['objects']=[
        {'id':'card','kind':'native_shape','bbox':[20,20,300,130],'geometry':'roundRect','fill':'EEEEFF','z':0},
        {'id':'table','kind':'table','bbox':[30,30,180,80],'z':1,
         'rows':[['项目','数量'],['A','1']],'font':{'latin':'Arial','ea':'微软雅黑','size_px':12},
         'header_row':True,'banded_rows':True},
        {'id':'chart','kind':'chart','bbox':[220,30,150,100],'z':2,'chart_type':'bar',
         'title':'数量','categories':['A','B'],'series':[{'name':'数量','values':[1,2]}],
         'font':{'latin':'Arial','ea':'微软雅黑','size_px':12},'has_legend':False},
    ]
    p['text_decisions'][0]['z']=3
    p['groups']=[{'id':'g1','member_ids':['card','table']}]
    return p


def test_table_chart_and_group_materialize_as_native_structure(tmp_path):
    source,inv,p=fixture(tmp_path);p=_structure_plan(p)
    validate_element_plan(p,inv)
    out=tmp_path/'out'
    report=materialize_elements(source,inv,p,out,model_id='fixture',response_sha256='0'*64)
    ir=json.loads((out/'slideir.json').read_text(encoding='utf8'))
    types={node['type'] for node in ir['nodes']}
    assert {'shape','table','chart','text'}<=types
    table=next(node for node in ir['nodes'] if node['type']=='table')
    chart=next(node for node in ir['nodes'] if node['type']=='chart')
    assert table['rows']==[['项目','数量'],['A','1']] and table['header_row'] is True
    assert chart['chart_type']=='bar' and chart['categories']==['A','B']
    assert ir['groups']==[{'id':'g1','member_ids':['card','table']}]
    assert report['native_tables']==1 and report['native_charts']==1 and report['groups']==1


def test_group_must_be_contiguous_and_table_merges_must_not_overlap(tmp_path):
    source,inv,p=fixture(tmp_path);p=_structure_plan(p)
    p['groups']=[{'id':'g1','member_ids':['card','chart']}]
    with pytest.raises(ValueError,match='not contiguous'):validate_element_plan(p,inv)
    p=_structure_plan(fixture(tmp_path)[2])
    p['objects'][1]['rows']=[['','',''],['','',''],['','','']]
    p['objects'][1]['merges']=[{'row':0,'column':0,'row_span':2,'column_span':2},
                               {'row':1,'column':1,'row_span':2,'column_span':2}]
    with pytest.raises(ValueError,match='overlapping'):validate_element_plan(p,inv)


def test_table_and_chart_require_an_available_font(tmp_path):
    _,inv,p=fixture(tmp_path);p=_structure_plan(p)
    p['objects'][1]['font']['ea']='NotInstalledFamily'
    with pytest.raises(ValueError,match='font unavailable'):validate_element_plan(p,inv)
