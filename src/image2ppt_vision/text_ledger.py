"""Three-way text reconciliation and guards for an actual PPTX export.

The source OCR, machine calibration and current delivered text stay separate.
Keeping a provisional choice does not establish source accuracy or acceptance.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import difflib
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from .storage import digest, read_json, write_json
from .ocr_calibration import _validated_input, _small_chinese_edit

NS = {'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
      'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}


def actual_objects(pptx, slide):
    with ZipFile(pptx) as package:
        if package.testzip() is not None:
            raise ValueError('invalid PPTX CRC')
        root = ET.fromstring(package.read(f'ppt/slides/slide{slide}.xml'))
    objects = {}
    for tag in ('sp', 'pic', 'graphicFrame', 'grpSp'):
        for element in root.findall(f'.//p:{tag}', NS):
            name = element.find('.//p:cNvPr', NS)
            if name is None or not name.get('name'):
                raise ValueError('unnamed actual object')
            identifier = name.get('name')
            if identifier in objects:
                raise ValueError('duplicate actual object name')
            paragraphs = []
            for paragraph in element.findall('p:txBody/a:p', NS):
                paragraphs.append(''.join('\n' if node.tag == '{' + NS['a'] + '}br' else node.text or ''
                                          for node in paragraph.iter()
                                          if node.tag in {'{' + NS['a'] + '}t', '{' + NS['a'] + '}br'}))
            objects[identifier] = {'kind': tag, 'text': '\n'.join(paragraphs)}
    return objects


def structure_hash(ir):
    nodes = deepcopy(ir['nodes'])
    for node in nodes:
        if node['type'] == 'text':
            for key in ('text', 'runs', 'text_override_reason'):
                node.pop(key, None)
    projection = {k: ir.get(k) for k in ('canvas', 'background', 'assets', 'groups', 'evidence', 'source_sha256', 'source_image')}
    projection['nodes'] = nodes
    return hashlib.sha256(json.dumps(projection, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def replace_text_preserving_runs(node, selected):
    result = deepcopy(node)
    if not node.get('runs'):
        result['text'] = selected
        return result
    if ''.join(run['text'] for run in node['runs']) != node['text']:
        raise ValueError('invalid original rich text')
    spans, offset = [], 0
    for index, run in enumerate(node['runs']):
        spans.append((offset, offset + len(run['text']), index))
        offset += len(run['text'])
    operations = []
    for op, i, j, k, l in difflib.SequenceMatcher(None, node['text'], selected, autojunk=False).get_opcodes():
        if op == 'equal':
            continue
        owners = [(start, index) for start, end, index in spans
                  if start <= i <= j <= end and (i != j or start < i < end or i == 0 or i == offset)]
        if len(owners) != 1:
            raise ValueError('replacement crosses run styling boundary')
        start, index = owners[0]
        operations.append((index, i - start, j - start, selected[k:l]))
    for index, start, end, text in reversed(operations):
        previous = result['runs'][index]['text']
        result['runs'][index]['text'] = previous[:start] + text + previous[end:]
    if any(not run['text'] for run in result['runs']):
        raise ValueError('replacement empties styled run')
    result['text'] = selected
    if ''.join(run['text'] for run in result['runs']) != selected:
        raise ValueError('rich-text replacement mismatch')
    return result


def decide_text(original, current, calibration):
    selected = calibration['selected']
    status = calibration['status']
    if status not in {'machine_calibrated', 'agreement', 'kept_ocr', 'review_required', 'unresolved'}:
        raise ValueError('unknown calibration status')
    if status == 'machine_calibrated':
        if not _small_chinese_edit(original, selected)[0]:
            raise ValueError('machine correction violates calibration policy')
        reading, review = calibration.get('blind'), calibration.get('review')
        if not reading or reading.get('status') != 'readable' or reading.get('text') != selected or not review or review.get('decision') != 'visual':
            raise ValueError('machine correction lacks two-pass support')
        if current == selected:
            return 'already_calibrated', current, False
        if current == original:
            return 'apply_calibration', selected, False
        return 'conflict_keep_current', current, True
    if status in {'review_required', 'unresolved'}:
        return ('pending_keep_current' if current == original else 'conflict_keep_current'), current, True
    if current == selected:
        return 'agreed_current', current, False
    if re.sub(r'\s+', '', current) == re.sub(r'\s+', '', selected):
        return 'formatting_preserved', current, False
    return 'conflict_keep_current', current, True


def _confined(folder, relative):
    value = Path(relative)
    path = (folder / value).resolve()
    if value.is_absolute() or not path.is_relative_to(folder) or path == folder:
        raise ValueError('asset path escapes bundle')
    return path


def reconcile_text(ir_file, calibration_file, current_pptx, output, *, slide=1):
    ir_file, calibration_file, current_pptx = map(lambda p: Path(p).resolve(), (ir_file, calibration_file, current_pptx))
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('new ledger revision required')
    ir, calibration = read_json(ir_file), read_json(calibration_file)
    original_authoring=read_json(ir_file.parent/'authoring.json') if (ir_file.parent/'authoring.json').exists() else {}
    if original_authoring and original_authoring.get('ir_sha256')!=digest(ir_file):raise ValueError('authoring IR binding changed')
    manifest = _validated_input(calibration_file.parent)
    if calibration.get('schema') != 'OCRCalibration/1' or calibration['source_sha256'] != ir['source_sha256'] or calibration['image_size'] != ir['canvas']['source_px']:
        raise ValueError('calibration source mismatch')
    if digest(calibration_file.parent / 'calibrated-evidence.json') != calibration['calibrated_evidence_sha256']:
        raise ValueError('calibration evidence changed')
    source = _confined(ir_file.parent, ir['source_image'])
    if digest(source) != ir['source_sha256']:
        raise ValueError('IR source changed')
    actual = actual_objects(current_pptx, slide)
    if set(actual) != {n['id'] for n in ir['nodes']} | {g['id'] for g in ir.get('groups', [])}:
        raise ValueError('current PPTX object inventory differs from IR')
    for node in ir['nodes']:
        if node['type'] == 'text' and actual[node['id']]['text'] != node['text']:
            raise ValueError('current PPTX text differs from IR: ' + node['id'])
    raw = {line['id']: line for line in manifest['lines']}
    if len(calibration['lines']) != len(raw) or {line['id'] for line in calibration['lines']} != set(raw):
        raise ValueError('calibration ID coverage mismatch')
    evidence = {item['id']: item for item in ir['evidence']}
    native_owners, raster_owners = {}, {}
    for node in ir['nodes']:
        if node['type'] == 'text':
            ids = [i for i in node['evidence_ids'] if evidence[i]['kind'] == 'text']
            if len(ids) != 1 or ids[0] in native_owners:
                raise ValueError('ledger requires one text evidence per native text object')
            native_owners[ids[0]] = node
    for asset in ir['assets']:
        owners = [n for n in ir['nodes'] if n.get('asset_id') == asset['id']]
        if len(owners) != 1:
            raise ValueError('ambiguous raster ownership')
        for identifier in asset['contains_evidence_ids']:
            if identifier in raw:
                if identifier in raster_owners or identifier in native_owners:
                    raise ValueError('duplicate text evidence ownership')
                raster_owners[identifier] = owners[0]
    if set(raw) != set(native_owners) | set(raster_owners):
        raise ValueError('unaccounted OCR or native text evidence')
    updated = deepcopy(ir)
    target_nodes = {n['id']: n for n in updated['nodes']}
    rows = []
    for line in calibration['lines']:
        identifier, original = line['id'], line['original']
        if original != raw[identifier]['original'] or original != evidence[identifier]['text'] or line['bbox'] != raw[identifier]['bbox']:
            raise ValueError('original OCR binding changed')
        node = native_owners.get(identifier)
        row = {'id': identifier, 'source_bbox': raw[identifier]['bbox'], 'original_ocr': original,
               'visual_reading': line['blind'], 'calibration_status': line['status'],
               'calibration_selected': line['selected'], 'calibration_review': line['review'],
               'crop': raw[identifier]['crop'], 'crop_sha256': raw[identifier]['crop_sha256'],
               'current_text': node['text'] if node else None, 'changed': False,
               'human_verified': False, 'selection_is_provisional': True}
        if node is None:
            row.update(owner_type='raster_image', owner_id=raster_owners[identifier]['id'],
                       decision='raster_preserved', selected_text=None, needs_review=True)
        else:
            decision, selected, pending = decide_text(original, node['text'], line)
            if selected != node['text']:
                try:
                    replacement = replace_text_preserving_runs(node, selected)
                except ValueError:
                    decision, selected, pending = 'style_conflict_keep_current', node['text'], True
                else:
                    replacement['text_override_reason'] = 'Source-bound OCR API calibration; see text-ledger.json / ' + identifier + '. Machine selected, not human verified.'
                    target_nodes[node['id']].update(replacement)
                    row['changed'] = True
            row.update(owner_type='native_text', owner_id=node['id'], decision=decision,
                       selected_text=selected, needs_review=pending)
        rows.append(row)
    if structure_hash(updated) != structure_hash(ir):
        raise ValueError('text reconciliation altered layout or assets')
    output.mkdir(parents=True)
    for relative, expected in [(ir['source_image'], ir['source_sha256'])] + [(a['path'], a['sha256']) for a in ir['assets']]:
        source_path = _confined(ir_file.parent, relative)
        if digest(source_path) != expected:
            raise ValueError('asset source changed')
        destination = _confined(output, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
    (output / 'evidence-inputs').mkdir()
    shutil.copyfile(ir_file, output / 'evidence-inputs/baseline-slideir.json')
    shutil.copyfile(calibration_file, output / 'evidence-inputs/calibration.json')
    shutil.copytree(calibration_file.parent / 'crops', output / 'crops')
    updated['review_status'] = 'needs_review'
    updated['review_notes'] = [*ir.get('review_notes', []),
        'Three-way text ledger applied; existing geometry and layout preserved. Provisional conflicts remain listed in text-ledger.json.']
    write_json(output / 'slideir.json', updated)
    ledger = {'schema': 'TextLedger/1', 'source_sha256': ir['source_sha256'],
              'baseline_ir_sha256': digest(ir_file), 'baseline_pptx_sha256': digest(current_pptx),
              'baseline_pptx_slide': slide, 'calibration_sha256': digest(calibration_file),
              'output_ir_sha256': digest(output / 'slideir.json'), 'structure_sha256': structure_hash(ir),
              'rows': rows, 'counts': dict(Counter(row['decision'] for row in rows)),
              'changed_text_objects': sum(row['changed'] for row in rows),
              'needs_review_count': sum(row['needs_review'] for row in rows),
              'policy': 'Keep current text on conflict; never revert it to raw OCR implicitly.',
              'human_acceptance': False, 'powerpoint_checked': False, 'wps_checked': False}
    write_json(output / 'text-ledger.json', ledger)
    write_json(output / 'authoring.json', {'format_version': '1', 'tool': 'deterministic-three-way-text-ledger',
               'model_id': 'none; uses retained API calibration evidence',
               'created_at': datetime.now(timezone.utc).isoformat(), 'ir_sha256': ledger['output_ir_sha256'],
               'node_decisions': {row['owner_id']: row['decision'] for row in rows},
               'review_notes': updated['review_notes'], 'unresolved_ids': [],
               'model_typography': deepcopy(original_authoring.get('model_typography',{}))})
    table = []
    for row in rows:
        values = [row['id'], row['decision'], row['original_ocr'],
                  row['visual_reading']['text'] if row['visual_reading'] else '',
                  row['calibration_selected'], row['current_text'] or '[图片内保留]',
                  row['selected_text'] or '[图片内保留]']
        table.append('<tr class="' + ('pending' if row['needs_review'] else 'clear') + '">'
                     + ''.join('<td>' + html.escape(value) + '</td>' for value in values)
                     + '<td><img src="' + html.escape(row['crop'], quote=True) + '"></td></tr>')
    (output / 'text-ledger.html').write_text('''<!doctype html><meta charset="utf-8"><title>三方文字记录</title>
<style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px;vertical-align:top}td{white-space:pre-wrap}img{max-width:600px}.pending{background:#fff5db}</style>
<h1>三方文字记录</h1><p>浅黄色为待核对项。保留当前文字不代表已确认正确；图片内文字不重复生成文本框。</p>
<table><thead><tr><th>ID</th><th>决定</th><th>原始 OCR</th><th>独立视觉转录</th><th>校准选择</th><th>当前 PPT</th><th>本次输出</th><th>原图裁剪</th></tr></thead><tbody>'''
                     + ''.join(table) + '</tbody></table>', encoding='utf-8')
    verify_ledger(output)
    return {k: ledger[k] for k in ('schema', 'counts', 'changed_text_objects', 'needs_review_count', 'human_acceptance')}


def verify_ledger(folder, pptx=None, *, slide=1):
    folder = Path(folder).resolve()
    ledger, ir = read_json(folder / 'text-ledger.json'), read_json(folder / 'slideir.json')
    if ledger.get('schema') != 'TextLedger/1' or digest(folder / 'slideir.json') != ledger['output_ir_sha256']:
        raise ValueError('ledger IR changed after reconciliation')
    if structure_hash(ir) != ledger['structure_sha256']:
        raise ValueError('protected layout changed')
    expected = {row['owner_id']: row['selected_text'] for row in ledger['rows'] if row['owner_type'] == 'native_text'}
    current = {node['id']: node['text'] for node in ir['nodes'] if node['type'] == 'text'}
    if expected != current:
        raise ValueError('selected text differs from IR')
    if pptx is not None:
        actual = actual_objects(pptx, slide)
        if set(actual) != {n['id'] for n in ir['nodes']} | {g['id'] for g in ir.get('groups', [])}:
            raise ValueError('export object inventory differs from ledger')
        found = {identifier: obj['text'] for identifier, obj in actual.items() if obj['text']}
        if found != expected:
            raise ValueError('export text differs from selected ledger text')
    return {'status': 'PASS', 'native_text_objects_checked': len(expected),
            'raster_owned_text_entries': sum(r['owner_type'] == 'raster_image' for r in ledger['rows']),
            'changed_text_objects': ledger['changed_text_objects'],
            'meaning': 'selection propagation verified, not source accuracy', 'human_acceptance': False}
