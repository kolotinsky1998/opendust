import pytest
import jax.numpy as jnp
import jaxfmm.fmm as fmm
import jaxfmm.debug_helpers as dbg
from jaxfmm.basis import mpl_idx
from jaxfmm.hierarchy import gen_hierarchy

@pytest.mark.parametrize('unitcube', [[128, {}]], indirect=True)
def test_mpl(unitcube):
    chrgs, hier = unitcube
    mpls = fmm.get_initial_mpls(hier["pts"][None,...], chrgs[None,...],hier["boxcenters"][0:1],hier["p"])
    eval_pts = hier["pts"] + jnp.array([1.0,-2.0,1.14])

    pot_mpl = dbg.eval_mpls(mpls,eval_pts,hier["boxcenters"][0],hier["p"])
    pot_dir = fmm.eval_potential_direct(hier["pts"],chrgs,eval_pts)
    assert jnp.allclose(pot_dir,pot_mpl,rtol=1e-3,atol=1e-3)

@pytest.mark.parametrize('unitcube', [[128, {}]], indirect=True)
def test_loc(unitcube):
    chrgs, hier = unitcube
    shift = jnp.array([1.0,-2.0,1.44])
    locs = dbg.get_locs(hier["pts"],chrgs,hier["boxcenters"][0:1]+shift,hier["p"])
    eval_pts = hier["pts"] + shift

    pot_loc = fmm.eval_local(locs,eval_pts[None,...],jnp.arange(eval_pts.shape[0]),hier["boxcenters"][0:1]+shift,hier["p"])
    pot_dir = fmm.eval_potential_direct(hier["pts"],chrgs,eval_pts)
    assert jnp.allclose(pot_dir,pot_loc,rtol=1e-3,atol=1e-3)

def test_mpl_coeffs():  # TODO: also check for correct normalization
    for n in range(9):
        for m in range(-n,n+1):
            pts, chrgs = dbg.gen_multipole_dist(m,n,eps=10.0)    # special point charge distribution corresponding to multipole moments - set eps large to minimize error
            tree_info = gen_hierarchy(pts)
            coeff = fmm.get_initial_mpls(pts[None,...],chrgs[None,...],tree_info["boxcenters"][0:1],n)[0,:]
            test = jnp.where(jnp.abs(coeff)>1e-5)[0]   # only the (m,n) coefficient should be nonzero
            assert test.shape[0] == 1
            assert test[0] == mpl_idx(m,n)