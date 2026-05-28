import jax.numpy as jnp
import jaxfmm.fmm as fmm
from jaxfmm.transf import *
import jaxfmm.debug_helpers as dbg
import pytest

@pytest.mark.parametrize('unitcube', [[128, {"N_max": 16}]], indirect=True)
def test_M2M(unitcube):
    chrgs, hier = unitcube
    pad_pts = hier["pts"][hier["idcs"][0][0],:]
    pad_chrgs = chrgs[hier["idcs"][0][0]]

    mpls = fmm.get_initial_mpls(pad_pts, pad_chrgs,hier["boxcenters"][1:],hier["p"])
    mpls_merged = M2M(mpls,hier["boxcenters"][1:],hier["boxcenters"][0,None],hier["unqs"],hier["invs"])

    mpls_direct = fmm.get_initial_mpls(pad_pts.reshape((1,-1,3)), pad_chrgs.reshape((1,-1)),hier["boxcenters"][0,None],hier["p"])
    assert jnp.allclose(mpls_merged, mpls_direct, rtol=1e-6, atol=1e-6)

@pytest.mark.parametrize('unitcube', [[128, {"N_max": 16}]], indirect=True)
def test_M2L(unitcube):
    chrgs, hier = unitcube
    to_shift = jnp.array([2.0,-4.0,5.0])

    mpls = fmm.get_initial_mpls(hier["pts"][None,:], chrgs[None,:],hier["boxcenters"][0,None],hier["p"])
    locs = M2L(mpls,hier["boxcenters"][0,None],hier["boxcenters"][0,None]+to_shift,jnp.array([[0,0]]),hier["unqs"],hier["invs"])
    locs_direct = dbg.get_locs(hier["pts"][None,:],chrgs[None,:],hier["boxcenters"][0,None]+to_shift,hier["p"])
    assert jnp.allclose(locs, locs_direct, rtol=1e-4, atol=1e-4)

@pytest.mark.parametrize('unitcube', [[128, {"N_max": 16}]], indirect=True)
def test_L2L(unitcube):
    chrgs, hier = unitcube
    pad_pts = hier["pts"][hier["idcs"][0][0],:]
    pad_chrgs = chrgs[hier["idcs"][0][0]]
    to_shift = jnp.array([2.0,-4.0,5.0])
    
    locs = dbg.get_locs(pad_pts.reshape((-1,3))+to_shift,pad_chrgs.flatten(),hier["boxcenters"][0,None],hier["p"])
    locs_dist = L2L(locs,hier["boxcenters"][0,None],hier["boxcenters"][1:],hier["unqs"],hier["invs"])
    locs_direct = dbg.get_locs(pad_pts.reshape((1,-1,3))+to_shift, pad_chrgs.reshape((1,-1)),hier["boxcenters"][1:],hier["p"])
    assert jnp.allclose(locs_dist, locs_direct,rtol=1e-6,atol=1e-2)