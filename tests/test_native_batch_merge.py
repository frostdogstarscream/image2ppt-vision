import importlib.util,json
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import pytest

spec=importlib.util.spec_from_file_location('native_batch_merge',Path(__file__).resolve().parents[3]/'scripts/merge_native_batches.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def setup(tmp_path,external=False,duplicate=False,absolute=False):
    ns=module.NS;ct=module.CT;rel=module.REL
    def package(path,name,picture=False):
        root=ET.Element('{'+ns['p']+'}sld');tree=ET.SubElement(ET.SubElement(root,'{'+ns['p']+'}cSld'),'{'+ns['p']+'}spTree')
        nv=ET.SubElement(tree,'{'+ns['p']+'}nvGrpSpPr');ET.SubElement(nv,'{'+ns['p']+'}cNvPr',{'id':'1','name':''});ET.SubElement(tree,'{'+ns['p']+'}grpSpPr')
        node=ET.SubElement(tree,'{'+ns['p']+'}pic' if picture else '{'+ns['p']+'}sp');props=ET.SubElement(node,'{'+ns['p']+'}nvPicPr' if picture else '{'+ns['p']+'}nvSpPr');ET.SubElement(props,'{'+ns['p']+'}cNvPr',{'id':'2','name':name})
        if picture:ET.SubElement(ET.SubElement(node,'{'+ns['p']+'}blipFill'),'{'+ns['a']+'}blip',{'{'+ns['r']+'}embed':'rId1'})
        else:ET.SubElement(node,'{'+ns['a']+'}t').text='中文保留'
        relationships=ET.Element('{'+rel+'}Relationships')
        if picture:ET.SubElement(relationships,'{'+rel+'}Relationship',{'Id':'rId1','Type':ns['r']+'/image','Target':'https://example.test/image.png' if external else '/ppt/media/image1.png' if absolute else '../media/image1.png',**({'TargetMode':'External'} if external else {})})
        types=ET.Element('{'+ct+'}Types');ET.SubElement(types,'{'+ct+'}Default',{'Extension':'png','ContentType':'image/png'})
        with ZipFile(path,'w') as z:
            z.writestr('ppt/slides/slide1.xml',ET.tostring(root,encoding='utf-8'));z.writestr('ppt/slides/_rels/slide1.xml.rels',ET.tostring(relationships));z.writestr('[Content_Types].xml',ET.tostring(types));z.writestr('ppt/media/image1.png',b'fixture-image')
    package(tmp_path/'base.pptx','text');package(tmp_path/'part.pptx','text' if duplicate else 'icon',True)
    ir={'background':{'color':'FFFFFF'},'nodes':[{'id':'icon','type':'image','z':0},{'id':'text','type':'text','z':1}]}
    (tmp_path/'ir.json').write_text(json.dumps(ir));manifest={'base':str(tmp_path/'base.pptx'),'batches':[{'slide':1,'path':str(tmp_path/'part.pptx')}],'irs':[str(tmp_path/'ir.json')]};(tmp_path/'manifest.json').write_text(json.dumps(manifest))
    return tmp_path/'manifest.json'

@pytest.mark.parametrize('absolute',[False,True])
def test_merges_images_and_chinese_text_in_exact_z_order(tmp_path,absolute):
    f=setup(tmp_path,absolute=absolute);report=module.merge_batches(f,tmp_path/'out.pptx');assert report['copied_images']==1
    with ZipFile(tmp_path/'out.pptx') as z:
        assert not z.testzip();root=ET.fromstring(z.read('ppt/slides/slide1.xml'));nodes=root.findall('.//p:spTree/*',module.NS)
        names=[n.find('.//p:cNvPr',module.NS).get('name') for n in nodes if n.tag.endswith('sp') or n.tag.endswith('pic')]
        assert names==['icon','text'] and '中文保留' in z.read('ppt/slides/slide1.xml').decode('utf-8')
        assert z.read('ppt/media/batch-1-image1.png')==b'fixture-image'
        assert b'<Types xmlns=' in z.read('[Content_Types].xml')
    with pytest.raises(FileExistsError):module.merge_batches(f,tmp_path/'out.pptx')

def test_rejects_external_relationships(tmp_path):
    f=setup(tmp_path,external=True)
    with pytest.raises(ValueError,match='unsupported batch relationship'):module.merge_batches(f,tmp_path/'out.pptx')

def test_rejects_duplicate_names_and_unsupported_node_types(tmp_path):
    f=setup(tmp_path,duplicate=True)
    with pytest.raises(ValueError,match='duplicate'):module.merge_batches(f,tmp_path/'out.pptx')
    ir=json.loads((tmp_path/'ir.json').read_text());ir['nodes'][0]['type']='chart';(tmp_path/'ir.json').write_text(json.dumps(ir))
    with pytest.raises(ValueError,match='unsupported batch node'):module.merge_batches(f,tmp_path/'out2.pptx')
