from copy import deepcopy
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image

from image2ppt_vision import api_scoring as scoring
from image2ppt_vision.api_evaluation import EvaluationError
from image2ppt_vision.storage import read_json,write_json,digest


def obj(id='r',kind='text',text='项目 123',bbox=None,geometry='rect'):
    return {'id':id,'type':kind,'text':text,'bbox':bbox or [.1,.2,.3,.1],
            'geometry':geometry,'ignored_nonvisual':False}


def pptx(path,slide_xml):
    with ZipFile(path,'x') as z:
        z.writestr('ppt/presentation.xml','<p:presentation xmlns:p="'+scoring.NS['p']+'" xmlns:r="'+scoring.NS['r']+'"><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst><p:sldSz cx="1000" cy="1000"/></p:presentation>')
        z.writestr('ppt/_rels/presentation.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="slides/slide1.xml"/></Relationships>')
        z.writestr('ppt/slides/slide1.xml',slide_xml)


def test_text_is_not_matched_by_count_alone():
    expected=[obj()];actual=[obj('a',text='不同文字')]
    result=scoring.match_objects(expected,actual,'text')
    assert result['matched']==0 and result['precision']==result['recall']==0


def test_geometry_type_and_one_percent_tolerance():
    expected=[obj(kind='shape',text='')]
    assert scoring.match_objects(expected,[obj('a',kind='shape',text='',bbox=[.109,.2,.3,.1])],'shape')['status']=='PASS'
    assert scoring.match_objects(expected,[obj('a',kind='shape',text='',bbox=[.111,.2,.3,.1])],'shape')['status']=='FAIL'
    assert scoring.match_objects(expected,[obj('a',kind='shape',text='',geometry='ellipse')],'shape')['status']=='FAIL'
    assert scoring.match_objects(expected,[obj('a',kind='image',text='')],'shape')['recall']==0


def test_duplicate_objects_lower_precision():
    expected=[obj()];actual=[obj('a'),obj('b')]
    result=scoring.match_objects(expected,actual,'text')
    assert result['precision']==.5 and result['recall']==1


def test_split_text_is_reported_separately():
    reference=[obj(text='甲乙',bbox=[0,0,.4,.2])]
    actual=[obj('a',text='甲',bbox=[0,0,.2,.2]),obj('b',text='乙',bbox=[.2,0,.2,.2])]
    assert scoring.split_merge_diagnostics(reference,actual)['possible_splits']==[{'id':'r','parts':['a','b']}]


def test_group_transform_flattens_to_slide_coordinates(tmp_path):
    path=tmp_path/'group.pptx'
    xml=f'''<p:sld xmlns:p="{scoring.NS['p']}" xmlns:a="{scoring.NS['a']}"><p:cSld><p:spTree>
    <p:grpSp><p:nvGrpSpPr><p:cNvPr id="1" name="group"/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="100" y="200"/><a:ext cx="400" cy="200"/><a:chOff x="10" y="20"/><a:chExt cx="200" cy="100"/></a:xfrm></p:grpSpPr>
    <p:sp><p:nvSpPr><p:cNvPr id="2" name="text"/></p:nvSpPr><p:spPr><a:xfrm><a:off x="20" y="30"/><a:ext cx="30" cy="40"/></a:xfrm><a:prstGeom prst="rect"/></p:spPr>
    <p:txBody><a:p><a:r><a:t>原生文本</a:t></a:r></a:p></p:txBody></p:sp></p:grpSp></p:spTree></p:cSld></p:sld>'''
    pptx(path,xml)
    record=scoring.inspect_objects(path)['slides'][0]['objects'][1]
    assert record['type']=='text' and record['text']=='原生文本'
    assert record['bbox']==pytest.approx([.12,.22,.06,.08])


def test_native_table_and_custom_geometry_are_not_images(tmp_path):
    path=tmp_path/'table.pptx'
    xml=f'''<p:sld xmlns:p="{scoring.NS['p']}" xmlns:a="{scoring.NS['a']}"><p:cSld><p:spTree>
    <p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="1" name="table"/></p:nvGraphicFramePr><p:xfrm><a:off x="0" y="0"/><a:ext cx="500" cy="200"/></p:xfrm>
    <a:graphic><a:graphicData><a:tbl><a:tr><a:tc gridSpan="2"><a:txBody><a:p><a:r><a:t>123</a:t></a:r></a:p></a:txBody></a:tc><a:tc hMerge="1"><a:txBody><a:p/></a:txBody></a:tc></a:tr></a:tbl></a:graphicData></a:graphic></p:graphicFrame>
    <p:sp><p:nvSpPr><p:cNvPr id="2" name="custom"/></p:nvSpPr><p:spPr><a:xfrm><a:off x="500" y="500"/><a:ext cx="100" cy="100"/></a:xfrm><a:custGeom/></p:spPr></p:sp>
    </p:spTree></p:cSld></p:sld>'''
    pptx(path,xml);records=scoring.inspect_objects(path)['slides'][0]['objects']
    assert records[0]['type']=='table' and records[0]['cells']==[['123','']]
    assert records[0]['merges'][0]['gridSpan']=='2'
    assert records[1]['geometry']=='custom'


def test_visual_error_uses_actual_preview_and_rejects_stretch(tmp_path):
    source=tmp_path/'source.png';preview=tmp_path/'preview.png'
    Image.new('RGB',(100,60),(255,255,255)).save(source)
    Image.new('RGB',(100,60),(235,235,235)).save(preview)
    report=scoring.image_difference(source,preview,tmp_path/'diff')
    assert report['mean_absolute_rgb']==20 and report['status']=='FAIL'
    assert len(report['tiles'])==16 and (tmp_path/'diff/difference.png').is_file()
    Image.new('RGB',(100,100),'white').save(preview)
    with pytest.raises(EvaluationError,match='preview_aspect_mismatch'):scoring.image_difference(source,preview,tmp_path/'bad')


def test_missing_ocr_cannot_become_content_pass(tmp_path,monkeypatch):
    from image2ppt_vision import models
    monkeypatch.setattr(models,'verified_models',lambda:(_ for _ in ()).throw(FileNotFoundError()))
    expected=[obj(text='金额123',bbox=[.1,.2,.3,.1])]
    analysis={'image_size':[100,100],'objects':[{'kind':'text','text':'金额123','bbox':[10,20,30,10]}]}
    result=scoring.content_metrics(expected,[obj('a',text='金额124')],analysis,tmp_path/'unused.png')
    assert result['native_content']['numeric_exact'] is False
    assert result['native_content']['status']=='FAIL'
    assert result['rendered_content']['status']=='NOT_RUN_OCR_UNAVAILABLE'


def test_reference_mismatch_is_excluded_not_called_content_error(tmp_path,monkeypatch):
    from image2ppt_vision import models
    monkeypatch.setattr(models,'verified_models',lambda:(_ for _ in ()).throw(FileNotFoundError()))
    reference=[obj(text='not in image')];analysis={'image_size':[100,100],'objects':[]}
    report=scoring.content_metrics(reference,[],analysis,tmp_path/'unused.png')
    assert report['unconfirmed_reference_ids']==['r']
    assert report['native_content']['character_accuracy'] is None


def test_comparison_keeps_both_round_statuses_and_regressions(tmp_path):
    base={'reference_sha256':'a','pages':[{'id':'02','status':'PASS','structure':{k:{'precision':1,'recall':1} for k in ('text','shape','table')},'visual':{'mean_absolute_rgb':2}}]}
    second=deepcopy(base);second['pages'][0]['status']='FAIL';second['pages'][0]['structure']['text']['recall']=.5
    a=tmp_path/'a.json';b=tmp_path/'b.json';write_json(a,base);write_json(b,second)
    scoring.compare_rounds(a,b,tmp_path/'comparison.json')
    report=read_json(tmp_path/'comparison.json')
    assert report['pages'][0]['delta']['text']['recall']==-.5
    assert report['no_page_cherry_picking'] is True


def test_extra_digits_and_native_table_cells_are_counted(tmp_path,monkeypatch):
    from image2ppt_vision import models
    monkeypatch.setattr(models,'verified_models',lambda:(_ for _ in ()).throw(FileNotFoundError()))
    cell=obj(text='金额123')
    table={**obj(kind='table'), 'cell_objects':[cell]}
    analysis={'image_size':[100,100],'objects':[{'kind':'text','text':'金额123','bbox':[10,20,30,10]}]}
    result=scoring.content_metrics([table],[{**table,'cell_objects':[obj('a',text='金额123456')]}],analysis,tmp_path/'unused.png')
    assert result['source_confirmed_reference_texts']==1
    assert result['native_content']['status']=='FAIL' and result['native_content']['numeric_exact'] is False


def test_custom_geometry_same_bbox_does_not_prove_same_shape():
    a={**obj(kind='shape',geometry='custom'), 'custom_geometry_sha256':'a'}
    b={**obj('a',kind='shape',geometry='custom'), 'custom_geometry_sha256':'b'}
    assert scoring.match_objects([a],[b],'shape')['matched']==0


def test_reference_subset_preserves_namespaces_and_original_for_multiple_slides(tmp_path,monkeypatch):
    xml = pytest.importorskip("lxml.etree", reason="reference PPT evaluation requires the optional evaluation dependency")
    source=tmp_path/'reference.pptx'
    relationships='http://schemas.openxmlformats.org/package/2006/relationships'
    with ZipFile(source,'x') as z:
        z.writestr('ppt/presentation.xml',f'''<p:presentation xmlns:p="{scoring.NS['p']}" xmlns:r="{scoring.NS['r']}" xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" xmlns:p14="http://schemas.microsoft.com/office/powerpoint/2010/main" mc:Ignorable="p14"><p:sldIdLst><p:sldId id="256" r:id="r1"/><p:sldId id="257" r:id="r2"/></p:sldIdLst></p:presentation>''')
        z.writestr('ppt/_rels/presentation.xml.rels',f'<Relationships xmlns="{relationships}"><Relationship Id="r1" Type="x/slide" Target="slides/slide1.xml"/><Relationship Id="r2" Type="x/slide" Target="slides/slide2.xml"/></Relationships>')
        z.writestr('[Content_Types].xml','<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Override PartName="/ppt/slides/slide1.xml"/><Override PartName="/ppt/slides/slide2.xml"/></Types>')
        for i in (1,2):z.writestr(f'ppt/slides/slide{i}.xml','<slide/>')
    before=digest(source);calls=[]
    def renderer(root,file,output):
        with ZipFile(file) as z:
            assert z.testzip() is None
            tree=xml.fromstring(z.read('ppt/presentation.xml'))
            assert 'p14' in tree.nsmap and len(tree.find('p:sldIdLst',scoring.NS))==1
            assert len([n for n in z.namelist() if n.startswith('ppt/slides/slide')])==1
        calls.append(file);return {'pages':[{'path':'fixture.png'}]}
    monkeypatch.setattr(scoring,'_render',renderer)
    result=scoring.render_reference_pages('.',source,tmp_path/'out',[1,2])
    assert len(calls)==2 and all(r['status']=='RENDERED' for r in result.values()) and digest(source)==before
