from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from PIL import Image
import pytest

from image2ppt_vision.storage import read_json, write_json, digest
from image2ppt_vision.ocr_calibration import prepare_calibration, prepare_reviews, finalize_calibration
from image2ppt_vision.text_ledger import (NS, decide_text, reconcile_text,
    replace_text_preserving_runs, verify_ledger, structure_hash)


def machine_line():
    return {'status': 'machine_calibrated', 'selected': '人工复核材料内容',
            'blind': {'status': 'readable', 'text': '人工复核材料内容'},
            'review': {'decision': 'visual'}}


def test_three_way_preserves_previous_correction_and_conflicts():
    c = machine_line()
    assert decide_text('人工复核材科内容', c['selected'], c) == ('already_calibrated', c['selected'], False)
    assert decide_text('人工复核材科内容', '人工复核材科内容', c)[0] == 'apply_calibration'
    assert decide_text('人工复核材科内容', '请人工核对材料', c) == ('conflict_keep_current', '请人工核对材料', True)


def test_unresolved_ocr_does_not_revert_existing_planner_text():
    c = {'status': 'unresolved', 'selected': 'Q提索港宗'}
    assert decide_text('Q提索港宗', '搜索卷宗', c) == ('conflict_keep_current', '搜索卷宗', True)


def test_whitespace_preserved_but_punctuation_conflict_not_hidden():
    c = {'status': 'agreement', 'selected': 'AI文书'}
    assert decide_text('AI文书', 'AI 文书', c)[0] == 'formatting_preserved'
    c['selected'] = '“四风”'
    assert decide_text('“四风”', '"四风"', c)[0] == 'conflict_keep_current'


def test_rich_text_style_survives_internal_replacement():
    n = {'text': '人工复核材科内容', 'runs': [
        {'text': '人工复核材科', 'color': 'FF0000', 'bold': True},
        {'text': '内容', 'color': '0000FF', 'bold': False}]}
    result = replace_text_preserving_runs(n, '人工复核材料内容')
    assert result['runs'][0] == {'text': '人工复核材料', 'color': 'FF0000', 'bold': True}
    assert result['runs'][1] == n['runs'][1]
    assert n['text'] == '人工复核材科内容'


def test_cross_style_replacement_is_not_flattened():
    n = {'text': '错字', 'runs': [{'text': '错', 'color': 'FF0000'}, {'text': '字', 'color': '0000FF'}]}
    with pytest.raises(ValueError, match='styling boundary'):
        replace_text_preserving_runs(n, '正解')


def pptx_fixture(path, texts):
    root = ET.Element('{' + NS['p'] + '}sld')
    tree = ET.SubElement(ET.SubElement(root, '{' + NS['p'] + '}cSld'), '{' + NS['p'] + '}spTree')
    for index, (identifier, text) in enumerate(texts.items(), 1):
        sp = ET.SubElement(tree, '{' + NS['p'] + '}sp')
        props = ET.SubElement(sp, '{' + NS['p'] + '}nvSpPr')
        ET.SubElement(props, '{' + NS['p'] + '}cNvPr', {'id': str(index), 'name': identifier})
        body = ET.SubElement(sp, '{' + NS['p'] + '}txBody')
        para = ET.SubElement(body, '{' + NS['a'] + '}p')
        run = ET.SubElement(para, '{' + NS['a'] + '}r')
        ET.SubElement(run, '{' + NS['a'] + '}t').text = text
    with ZipFile(path, 'w') as package:
        package.writestr('ppt/slides/slide1.xml', ET.tostring(root))


@pytest.fixture
def bundle(tmp_path):
    src = tmp_path / 'source.png'; Image.new('RGB', (300, 100), 'white').save(src)
    original = '人工复核材科内容'
    evidence = {'schema': 'CVElementInventory', 'source_sha256': digest(src), 'image_size': [300, 100],
                'texts': [{'id': 't1', 'text': original, 'bbox': [10, 10, 200, 30]}]}
    cal = tmp_path / 'cal'
    prepare_calibration(src, evidence, cal, {'base_url': 'https://example.com/v1', 'model': 'fixture'})
    import json
    def respond(job, content):
        write_json(job / 'response.body', {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(content)}}]})
        write_json(job / 'transport.json', {'status': 'RECEIVED', 'saved_response_sha256': digest(job / 'response.body')})
    respond(cal / 'blind/000', {'schema': 'OCRBlind/1', 'items': [
        {'id': 't1', 'status': 'readable', 'text': '人工复核材料内容'}]})
    jobs = prepare_reviews(cal)
    respond(jobs[0], {'schema': 'OCRReview/2', 'items': [{'id': 't1', 'decision': 'visual', 'evidence': '可见料字'}]})
    finalize_calibration(cal)
    ir = {'schema_version': '0.1', 'source_image': 'source.png', 'source_sha256': digest(src),
          'canvas': {'source_px': [300, 100], 'target_px': [900, 300]}, 'background': {'color': 'FFFFFF'},
          'evidence': [{'id': 't1', 'kind': 'text', 'text': original, 'bbox': [10, 10, 200, 30]}],
          'nodes': [{'id': 'node1', 'type': 'text', 'text': original, 'bbox': [10, 10, 200, 30],
                     'font': {'latin': 'Arial', 'ea': 'Microsoft YaHei', 'size_pt': 18},
                     'evidence_ids': ['t1'], 'z': 0, 'color': '000000'}], 'assets': [], 'groups': []}
    write_json(tmp_path / 'slideir.json', ir)
    pptx_fixture(tmp_path / 'baseline.pptx', {'node1': original})
    return tmp_path, cal, ir


def test_reconcile_applies_bound_correction_and_checks_actual_file(bundle):
    root, cal, old = bundle
    report = reconcile_text(root / 'slideir.json', cal / 'calibration.json', root / 'baseline.pptx', root / 'new')
    assert report['changed_text_objects'] == 1
    result = read_json(root / 'new/slideir.json')
    assert result['nodes'][0]['text'] == '人工复核材料内容'
    assert structure_hash(result) == structure_hash(old)
    assert read_json(root / 'slideir.json') == old
    pptx_fixture(root / 'new/result.pptx', {'node1': result['nodes'][0]['text']})
    assert verify_ledger(root / 'new', root / 'new/result.pptx')['status'] == 'PASS'
    with pytest.raises(ValueError, match='export text'):
        verify_ledger(root / 'new', root / 'baseline.pptx')


def test_current_pptx_mismatch_rejected_before_merge(bundle):
    root, cal, old = bundle
    pptx_fixture(root / 'baseline.pptx', {'node1': '外部改动'})
    with pytest.raises(ValueError, match='current PPTX text'):
        reconcile_text(root / 'slideir.json', cal / 'calibration.json', root / 'baseline.pptx', root / 'new')


def test_guard_blocks_changes_after_reconciliation(bundle):
    root, cal, old = bundle
    reconcile_text(root / 'slideir.json', cal / 'calibration.json', root / 'baseline.pptx', root / 'new')
    ir = read_json(root / 'new/slideir.json'); ir['nodes'][0]['text'] = '未经记录改写'
    write_json(root / 'new/slideir.json', ir)
    with pytest.raises(ValueError, match='changed after'):
        verify_ledger(root / 'new')


def test_false_machine_claim_without_two_pass_support_is_rejected():
    c = machine_line(); c['review']['decision'] = 'unresolved'
    with pytest.raises(ValueError, match='two-pass support'):
        decide_text('人工复核材科内容', '人工复核材科内容', c)
