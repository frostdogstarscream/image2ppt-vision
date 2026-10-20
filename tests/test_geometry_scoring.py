import xml.etree.ElementTree as ET
import numpy as np

from image2ppt_vision.geometry_scoring import custom_signature,equivalent


def signature(points,*,scale=100,fill='FF0000',stroke='000000',curve=False,matrix=None):
    commands=''.join(f'<a:{"moveTo" if i==0 else "cubicBezTo" if curve else "lnTo"}><a:pt x="{x*scale}" y="{y*scale}"/></a:{"moveTo" if i==0 else "cubicBezTo" if curve else "lnTo"}>' for i,(x,y) in enumerate(points))
    xml=f'''<spPr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
    <a:xfrm><a:ext cx="100" cy="100"/></a:xfrm><a:custGeom><a:pathLst><a:path w="{scale}" h="{scale}">
    {commands}<a:close/></a:path></a:pathLst></a:custGeom>
    <a:solidFill><a:srgbClr val="{fill}"/></a:solidFill><a:ln w="1"><a:solidFill><a:srgbClr val="{stroke}"/></a:solidFill></a:ln></spPr>'''
    return custom_signature(ET.fromstring(xml),np.eye(3) if matrix is None else matrix,[1000,1000])


def test_scale_start_vertex_direction_and_collinear_points_are_equivalent():
    a=signature([(0,0),(1,0),(1,1),(0,1)])
    b=signature([(1,1),(1,0),(.5,0),(0,0),(0,1)],scale=2000)
    assert a and b and equivalent(a,b)


def test_same_box_different_contour_or_style_is_not_equivalent():
    a=signature([(0,0),(1,0),(1,1),(0,1)])
    assert not equivalent(a,signature([(0,0),(1,0),(1,1)]))
    assert not equivalent(a,signature([(0,0),(1,0),(1,1),(0,1)],fill='00FF00'))
    assert not equivalent(a,signature([(0,0),(1,0),(1,1),(0,1)],stroke='0000FF'))


def test_rotation_and_reflection_use_real_slide_coordinates():
    points=[(0,0),(1,0),(0,1)]
    matrix=np.array([[0,-1,100],[1,0,0],[0,0,1]])
    assert equivalent(signature(points,matrix=matrix),signature([(1,0),(1,1),(0,0)]))


def test_unsupported_geometry_is_unresolved_never_hash_matched():
    assert signature([(0,0),(1,0),(1,1)],curve=True) is None
    assert not equivalent(None,None)
