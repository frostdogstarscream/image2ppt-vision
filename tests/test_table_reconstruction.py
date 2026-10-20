from copy import deepcopy
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from image2ppt_vision.plans import validate_plan,PlanValidationError
from image2ppt_vision.reconstruction import reconstruct
from image2ppt_vision.table_reconstruction import verify_table
from image2ppt_vision.storage import read_json
from test_reconstruction import fixture_inputs,text,retained


def fixture(tmp_path):
    case=fixture_inputs(tmp_path,np.full((120,200,3),255,np.uint8),[
        text('a',[20,20,50,12],'Header'),text('b',[20,60,30,12],'123'),text('c',[110,60,30,12],'456')],
        [{'id':'table-region','bbox':[10,10,180,80],'kind':'panel'}])
    case.plan.update(version='1.1',tables=[{
        'id':'table-one','region_id':'table-region','bbox':[10,10,180,80],
        'rows':[['Header',''],['123','456']],'column_widths':[90,90],'row_heights':[40,40],
        'ocr_ids':['a','b','c'],'merges':[{'row':0,'column':0,'row_span':1,'column_span':2}]}])
    return case


@pytest.mark.parametrize('change',[
    lambda p:p.update(version='1.0'),
    lambda p:p['tables'][0].update(column_widths=[True,179]),
    lambda p:p['tables'][0].update(row_heights=['40',40]),
    lambda p:p['tables'][0].update(row_heights=[float('nan'),40]),
    lambda p:p['tables'][0].update(row_heights=[40,41]),
    lambda p:p['tables'][0]['rows'][0].__setitem__(1,'lost'),
    lambda p:p['tables'][0]['merges'].append({'row':0,'column':0,'row_span':2,'column_span':1}),
    lambda p:p['tables'][0].update(region_id='missing'),
    lambda p:p.update(retain_raster_ids=['a']),
])
def test_table_plan_rejects_unsafe_ownership_and_tracks(tmp_path,change):
    p=fixture(tmp_path).plan;change(p)
    with pytest.raises(PlanValidationError):validate_plan(p)


def test_table_cell_text_cannot_invent_or_cross_boundaries(tmp_path):
    case=fixture(tmp_path);t=case.plan['tables'][0]
    verify_table(t,case.analysis['objects'],{})
    wrong=deepcopy(t);wrong['rows'][1][0]='999'
    with pytest.raises(PlanValidationError):verify_table(wrong,case.analysis['objects'],{})
    wrong=deepcopy(t);wrong['ocr_ids']=['a','b']
    with pytest.raises(PlanValidationError):verify_table(wrong,case.analysis['objects'],{})
    case.analysis['objects'][1]['bbox']=[90,60,30,12]
    with pytest.raises(PlanValidationError):verify_table(t,case.analysis['objects'],{})


def test_table_replaces_pixels_and_preserves_native_ownership(tmp_path,monkeypatch):
    case=fixture(tmp_path)
    monkeypatch.setattr('image2ppt_vision.typography.estimate_text',retained)
    report=reconstruct(case.source,case.analysis,case.plan,case.output)
    ir=read_json(case.output/'slideir.json')
    table=next(n for n in ir['nodes'] if n['type']=='table')
    assert table['rows']==case.plan['tables'][0]['rows']
    assert table['merges']==case.plan['tables'][0]['merges']
    assert report['native_tables']==1 and report['native_table_text_lines']==3
    assert not any(set(a['contains_evidence_ids'])&{'a','b','c'} for a in ir['assets'])
    assert {'a','b','c'} <= set(table['evidence_ids'])
    root=Path(__file__).resolve().parents[3]
    script="import {loadIR} from './scripts/ir.mjs';await loadIR(process.argv[1]);"
    result=subprocess.run(['node','--input-type=module','-e',script,str(case.output/'slideir.json')],cwd=root,capture_output=True)
    assert result.returncode==0,result.stderr.decode(errors='replace')


def test_legacy_serialization_does_not_gain_table_field(tmp_path):
    p=fixture(tmp_path).plan;p.pop('tables');p['version']='1.0'
    assert 'tables' not in validate_plan(p).to_dict()
