from copy import deepcopy
import json
from pathlib import Path

from PIL import Image
import pytest

from image2ppt_vision.ocr_calibration import (prepare_calibration, prepare_reviews,
    finalize_calibration, validate_reply, _small_chinese_edit)
from image2ppt_vision.storage import digest, read_json, write_json


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / 'source.png'
    Image.new('RGB', (500, 200), 'white').save(source)
    evidence = {'schema': 'CVElementInventory', 'source_sha256': digest(source),
                'image_size': [500, 200], 'texts': [
                    {'id': 'text-1', 'text': '人工复核材科内容', 'bbox': [10, 10, 200, 25]},
                    {'id': 'text-2', 'text': '原本正确的内容', 'bbox': [10, 50, 200, 25]},
                    {'id': 'text-3', 'text': '金额为100元', 'bbox': [10, 90, 200, 25]}]}
    settings = {'base_url': 'https://example.com/v1', 'model': 'fixture-vl', 'batch_size': 2}
    run = tmp_path / 'run'
    prepare_calibration(source, evidence, run, settings)
    return source, evidence, run


def respond(job, stage, entries):
    body = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({
        'schema': 'OCRBlind/1' if stage == 'blind' else 'OCRReview/2', 'items': entries}, ensure_ascii=False)}}]}
    write_json(job / 'response.body', body)
    write_json(job / 'transport.json', {'status': 'RECEIVED', 'saved_response_sha256': digest(job / 'response.body')})


def blind(run, readings=None):
    readings = readings or {'text-1': '人工复核材料内容', 'text-2': '原本正确的内容', 'text-3': '金额为1000元'}
    for job in sorted((run / 'blind').iterdir()):
        ids = read_json(job / 'binding.json')['ids']
        respond(job, 'blind', [{'id': i, 'status': 'readable', 'text': readings[i]} for i in ids])


def test_blind_request_never_contains_ocr_answers_and_source_is_preserved(prepared):
    source, evidence, run = prepared
    request = read_json(run / 'blind/000/request.json')
    text = ' '.join(block['text'] for block in request['messages'][1]['content'] if block['type'] == 'text')
    assert '人工复核' not in text and '原本正确' not in text
    assert read_json(run / 'original-evidence.json') == evidence
    assert digest(source) == digest(run / 'source.image')


def test_two_pass_selection_preserves_sensitive_changes_and_coordinates(prepared):
    source, evidence, run = prepared
    blind(run)
    jobs = prepare_reviews(run)
    assert len(jobs) == 1
    request = read_json(jobs[0] / 'request.json')
    assert request['response_format'] == {'type': 'json_object'}
    assert 'json' in request['messages'][0]['content'].lower()
    assert read_json(jobs[0] / 'binding.json')['ids'] == ['text-1', 'text-3']
    respond(jobs[0], 'review', [
        {'id': 'text-1', 'decision': 'visual', 'evidence': '右侧可见斗部'},
        {'id': 'text-3', 'decision': 'visual', 'evidence': '候选含额外零'}])
    result = finalize_calibration(run)
    assert result['counts'] == {'machine_calibrated': 1, 'agreement': 1, 'review_required': 1}
    calibrated = read_json(run / 'calibrated-evidence.json')
    assert calibrated['texts'][0]['text'] == '人工复核材料内容'
    assert calibrated['texts'][2]['text'] == '金额为100元'
    assert [t['bbox'] for t in calibrated['texts']] == [t['bbox'] for t in evidence['texts']]
    assert not result['human_acceptance'] and not result['accuracy_measured']
    assert read_json(run / 'original-evidence.json') == evidence
    with pytest.raises(FileExistsError): finalize_calibration(run)


def test_unavailable_api_retains_every_original(prepared):
    _, evidence, run = prepared
    assert prepare_reviews(run) == []
    result = finalize_calibration(run)
    assert result['counts'] == {'unresolved': 3}
    assert [t['text'] for t in read_json(run / 'calibrated-evidence.json')['texts']] == [t['text'] for t in evidence['texts']]


def test_source_tamper_rejected(prepared):
    _, _, run = prepared
    (run / 'source.image').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'): prepare_reviews(run)


def test_crop_tamper_rejected(prepared):
    _, _, run = prepared
    (run / 'crops/text-1.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='crop changed'): prepare_reviews(run)


def test_changed_response_is_unresolved(prepared):
    _, _, run = prepared
    blind(run)
    with (run / 'blind/000/response.body').open('a') as file: file.write(' ')
    prepare_reviews(run)
    result = finalize_calibration(run)
    assert result['counts']['unresolved'] == 3


@pytest.mark.parametrize('items', [
    [{'id': 'a', 'status': 'readable', 'text': '□'}],
    [{'id': 'a', 'status': 'not_text', 'text': '文'}],
    [{'id': 'wrong', 'status': 'readable', 'text': '文'}],
    [{'id': 'a', 'status': 'readable', 'text': '文'}, {'id': 'a', 'status': 'readable', 'text': '文'}],
])
def test_invalid_model_claims_rejected(items):
    with pytest.raises(ValueError): validate_reply({'schema': 'OCRBlind/1', 'items': items}, ['a'], 'blind')


def test_review_cannot_invent_third_candidate():
    with pytest.raises(ValueError, match='third candidate'):
        validate_reply({'schema': 'OCRReview/2', 'items': [
            {'id': 'a', 'decision': 'visual', 'text': '编造', 'evidence': '语义更通顺'}]},
            ['a'], 'review', {'a': {'ocr': '原文', 'visual': '候选'}})


@pytest.mark.parametrize('before,after', [('100元','1000元'), ('AI辅助','Al辅助'), ('全部错误','完全改写'), ('甲','乙')])
def test_sensitive_or_large_changes_do_not_auto_apply(before, after):
    assert not _small_chinese_edit(before, after)[0]


def test_input_binding_and_no_overwrite(prepared, tmp_path):
    source, evidence, run = prepared
    settings = read_json(run / 'settings.json')
    with pytest.raises(FileExistsError): prepare_calibration(source, evidence, run, settings)
    wrong = deepcopy(evidence); wrong['source_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='source mismatch'): prepare_calibration(source, wrong, tmp_path / 'new', settings)
    settings['api_key'] = 'never-on-disk'
    with pytest.raises(ValueError, match='credentials'): prepare_calibration(source, evidence, tmp_path / 'new', settings)


def test_vision_analysis_input_preserves_nontext_and_ocr_confidence(prepared, tmp_path):
    source, inventory, old = prepared
    analysis = {'schema': 'VisionAnalysis', 'image_size': inventory['image_size'],
                'inputs': {'source': {'original_sha256': digest(source)}},
                'objects': [{'kind': 'text', 'score': .72, **t} for t in inventory['texts']]
                           + [{'id': 'shape-1', 'kind': 'shape', 'bbox': [0, 0, 500, 200]}]}
    run = tmp_path / 'analysis-run'
    prepare_calibration(source, analysis, run, read_json(old / 'settings.json'))
    prepare_reviews(run); finalize_calibration(run)
    result = read_json(run / 'calibrated-evidence.json')
    assert result['objects'][-1] == analysis['objects'][-1]
    assert result['objects'][0]['score'] == .72
    assert result['objects'][0]['text'] == analysis['objects'][0]['text']


def test_cli_prepare_does_not_invoke_api(prepared, tmp_path, monkeypatch, capsys):
    from image2ppt_vision.cli import main
    source, evidence, old = prepared
    monkeypatch.delenv('VISION_LLM_API_KEY', raising=False)
    out = tmp_path / 'cli-run'
    main(['calibrate-ocr', '--input', str(source), '--analysis', str(old / 'original-evidence.json'),
          '--config', str(old / 'settings.json'), '--output', str(out), '--prepare-only'])
    assert json.loads(capsys.readouterr().out)['status'] == 'PREPARED'
    assert not list(out.rglob('transport.json'))
