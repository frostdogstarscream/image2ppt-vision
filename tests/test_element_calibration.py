from copy import deepcopy
import numpy as np
from image2ppt_vision.element_calibration import _fill, _ink

def test_fill_measurement_excludes_child_objects_and_never_mutates_pixels():
    p=np.full((100,200,3),245,np.uint8);p[30:70,50:150]=[0,90,180];before=p.copy()
    parent={'id':'p','type':'shape','geometry':'roundRect','bbox':[0,0,200,100],'fill':'000000'}
    child={'id':'c','type':'shape','geometry':'rect','bbox':[50,30,100,40],'fill':'005AB4'}
    result=_fill(p,parent,[parent,child]);assert result['color']=='F5F5F5';np.testing.assert_array_equal(p,before)

def test_unsupported_geometry_and_nonuniform_fill_refused():
    p=np.full((100,200,3),255,np.uint8);node={'id':'p','type':'shape','geometry':'rightArrow','bbox':[0,0,200,100],'fill':'000000'}
    assert _fill(p,node,[node]) is None
    node['geometry']='rect';p[:,:100]=0
    assert _fill(p,node,[node]) is None

def test_gradient_bar_declared_dark_is_measured_light():
    """Page 05's panel header: a light gradient bar the plan calls dark blue."""
    p=np.empty((60,400,3),np.uint8)
    for x in range(400):
        p[:,x]=[int(235+20*x/399)]*3
    node={'id':'bar','type':'shape','geometry':'rect','bbox':[0,0,400,60],'fill':'1a5fb4',
          'fill_gradient':{'angle_deg':0,'stops':[{'offset':0,'color':'1a5fb4'},{'offset':100000,'color':'0d4a9e'}]}}
    result=_fill(p,node,[node])
    assert result is not None and int(result['color'][:2],16)>=228

def test_declared_gradient_survives_when_source_is_a_gradient():
    p=np.empty((60,400,3),np.uint8)
    for x in range(400):
        p[:,x]=[int(240-120*x/399)]*3
    node={'id':'bar','type':'shape','geometry':'rect','bbox':[0,0,400,60],'fill':'f0f0f0',
          'fill_gradient':{'angle_deg':0,'stops':[{'offset':0,'color':'f0f0f0'},{'offset':100000,'color':'787878'}]}}
    result=_fill(p,node,[node])
    assert result is not None and 'gradient' in result

def test_text_ink_separates_dark_glyphs_from_a_light_bar():
    p=np.full((30,80,3),240,np.uint8);p[8:22,10:50]=[20,74,195]
    ink=_ink(p,[0,0,80,30])
    assert ink is not None
    assert abs(int(ink['color'][:2],16)-20)<12 and ink['separation']>100

def test_text_ink_refuses_white_on_white():
    """A proposal that is invisible in the source has nothing to measure."""
    p=np.full((30,80,3),255,np.uint8)
    assert _ink(p,[0,0,80,30]) is None
