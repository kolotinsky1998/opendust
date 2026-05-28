import pytest
from jax import random
from jaxfmm.hierarchy import gen_hierarchy

@pytest.fixture(scope="module")
def unitcube(request):
    if(len(request.param)==2):
        seed = 23556
        N, kwargs = request.param
    else:
        N, seed, kwargs = request.param
    pts = random.uniform(random.key(seed),(N,3),minval=-0.5,maxval=0.5)
    chrgs = random.uniform(random.key(seed+1),N,minval=-1,maxval=1)
    hier = gen_hierarchy(pts,**kwargs)
    return chrgs, hier