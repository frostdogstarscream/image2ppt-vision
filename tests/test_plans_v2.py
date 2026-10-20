import json
from copy import deepcopy

import pytest

from image2ppt_vision.plans import PlanValidationError, parse_plan_json
from image2ppt_vision.plans_v2 import parse_plan_v2,lower_plan_v2,materialize_groups_v2
from image2ppt_vision.storage import write_json,read_json


def proposal():
    return dict(schema='ReconstructionPlan',version='2.0',source_sha256='a'*64,image_size=[100,100],
                author=dict(tool='vision-model-plan',model_id='fixture'))


def texts():
    return [dict(id='ocr-001',text='甲',bbox=[10,10,10,10]),dict(id='ocr-002',text='乙',bbox=[60,10,10,10])]


def parse(data,ocr=None):
    return parse_plan_v2(json.dumps(data),source_sha256='a'*64,image_size=[100,100],texts=texts() if ocr is None else ocr)


def test_sparse_defaults_keep_native_attempt_and_original_legacy_readable():
    p=parse(proposal());legacy,report=lower_plan_v2(p,texts())
    assert legacy.retain_raster_ids==[] and legacy.text_overrides==[]
    assert legacy.regions[0].bbox==[0,0,100,100]
    assert parse_plan_json(json.dumps(legacy.to_dict())).version=='1.1'
    assert report['default_text_action']=='attempt_native_after_pixel_verification'


def test_nested_semantic_groups_derive_bounds_without_pixel_regions():
    data=proposal();data['semantic_groups']=[dict(id='parent',members=['child','ocr-002']),dict(id='child',members=['ocr-001'])]
    legacy,report=lower_plan_v2(parse(data),texts())
    assert len(legacy.regions)==1
    parent=next(g for g in report['semantic_groups'] if g['id']=='parent')
    assert parent['bbox']==[10,10,60,10]


@pytest.mark.parametrize('groups',[
    [dict(id='a',members=['b']),dict(id='b',members=['a'])],
    [dict(id='a',members=['unknown'])],
    [dict(id='a',members=['ocr-001','ocr-001'])],
])
def test_invalid_semantic_graph_is_not_repaired(groups):
    data=proposal();data['semantic_groups']=groups
    with pytest.raises(PlanValidationError):parse(data)


@pytest.mark.parametrize('decisions',[
    [dict(id='ocr-001',action='correct',text='',expected_text='甲',reason='fixture')],
    [dict(id='ocr-001',action='keep_pixels',text='甲',reason='fixture')],
    [dict(id='ocr-001',action='correct',text='丙',expected_text='wrong',reason='fixture')],
    [dict(id='unknown',action='keep_pixels',reason='fixture')],
    [dict(id='ocr-001',action='keep_pixels',reason='fixture')]*2,
])
def test_invalid_or_conflicting_decisions_fail(decisions):
    data=proposal();data['text_decisions']=decisions
    with pytest.raises(PlanValidationError):parse(data)


def table_proposal():
    data=proposal()
    data['tables']=[dict(id='table',bbox=[0,0,100,100],row_edges=[0,.5,1],column_edges=[0,.5,1],
                        cells=[dict(row=0,column=0,column_span=2,ocr_ids=['ocr-002','ocr-001'])])]
    return data


def test_table_text_and_covered_cells_are_derived_and_correction_applies():
    data=table_proposal();data['text_decisions']=[dict(id='ocr-001',action='correct',text='丙',expected_text='甲',reason='fixture')]
    original=deepcopy(data);plan=parse(data);legacy,report=lower_plan_v2(plan,texts())
    table=legacy.tables[0]
    assert table.rows==[['丙\n乙',''],['','']]
    assert table.row_heights==table.column_widths==[50,50]
    assert table.bbox==legacy.regions[0].bbox
    assert report['generated_table_cells']==4 and data==original


@pytest.mark.parametrize('change',['covered_anchor','wrong_cell','missing_ocr','unknown_ocr','edges','retained'])
def test_table_contract_rejects_real_ownership_errors(change):
    data=table_proposal();t=data['tables'][0]
    if change=='covered_anchor':t['cells'].append(dict(row=0,column=1,ocr_ids=[]))
    if change=='wrong_cell':t['cells'][0]['row']=1
    if change=='missing_ocr':t['cells'][0]['ocr_ids']=['ocr-001']
    if change=='unknown_ocr':t['cells'][0]['ocr_ids']=['unknown']
    if change=='edges':t['row_edges']=[0,.6,.5,1]
    if change=='retained':data['text_decisions']=[dict(id='ocr-001',action='keep_pixels',reason='fixture')]
    with pytest.raises(PlanValidationError):parse(data)


def test_raster_regions_do_not_inherit_semantic_overlap_permission():
    data=proposal();data['raster_regions']=[dict(id='r1',bbox=[0,30,60,60],kind='photo',reason='fixture'),
        dict(id='r2',bbox=[40,40,40,40],kind='diagram',reason='fixture')]
    with pytest.raises(PlanValidationError):parse(data)
    data['raster_regions']=[dict(id='r',bbox=[0,0,15,15],kind='photo',reason='fixture')]
    with pytest.raises(PlanValidationError):parse(data)


def test_wrong_source_unknown_fields_and_duplicate_json_fail():
    data=proposal();data['source_sha256']='b'*64
    with pytest.raises(PlanValidationError):parse(data)
    data=proposal();data['reference_pptx']='forbidden'
    with pytest.raises(PlanValidationError):parse(data)
    with pytest.raises(PlanValidationError):parse_plan_v2('{"version":"2.0","version":"2.0"}',source_sha256='a'*64,image_size=[100,100],texts=[])


def test_native_grouping_does_not_reorder_or_group_shared_raster(tmp_path):
    data=proposal();data['semantic_groups']=[dict(id='text',members=['ocr-001','ocr-002'])]
    plan=parse(data);_,lowering=lower_plan_v2(plan,texts())
    nodes=[dict(id='bg',z=0),dict(id='ocr-001',z=1),dict(id='ocr-002',z=2)]
    write_json(tmp_path/'slideir.json',dict(nodes=nodes,groups=[dict(id='old',member_ids=['bg','ocr-001','ocr-002'])]))
    write_json(tmp_path/'authoring.json',{})
    result=materialize_groups_v2(plan,lowering,tmp_path,{})
    ir=read_json(tmp_path/'slideir.json')
    assert ir['nodes']==nodes and ir['groups']==[dict(id='v2-semantic-text',member_ids=['ocr-001','ocr-002'])]
    assert result['semantic_group_diagnostics'][0]['native']
    ir['nodes'][1]['id']='shared-raster';write_json(tmp_path/'slideir.json',ir)
    result=materialize_groups_v2(plan,lowering,tmp_path,{})
    assert not result['semantic_group_diagnostics'][0]['native']
