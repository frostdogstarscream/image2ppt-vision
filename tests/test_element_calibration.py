from copy import deepcopy
import numpy as np
from image2ppt_vision.element_calibration import _fill

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
