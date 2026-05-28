import jax.numpy as jnp
from jaxfmm import *
from jax import random
import pytest

@pytest.mark.parametrize("periodic_axes", [(0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2)])
def test_pbc(periodic_axes):    # TODO: use unitcube fixture, improve test
    ### generate system
    N = 128
    pbc_lvls = 1
    key = random.key(768)
    pts = random.uniform(key,(N,3),minval=-1,maxval=1)
    chrgs = random.uniform(key,N,minval=-1,maxval=1)

    ### balance charges
    possum = chrgs[chrgs>0].sum()
    negsum = jnp.abs(chrgs[chrgs<0].sum())
    chrgs = chrgs.at[chrgs>0].multiply(negsum/possum)

    ### compute periodic potential
    tree_info = gen_hierarchy(pts, N_max=64, periodic_axes=periodic_axes, pbc_lvls=pbc_lvls, debug_info=True, L0_boxcen=jnp.array([0.0,0.0,0.0]),L0_boxlen=jnp.array([2.0,2.0,2.0]), pbc_no_monopole=False)
    pot_FMM = eval_potential(chrgs, **tree_info)

    ### dupe the points - NOTE: only works for cubic level zero cells
    dupes = [(tree_info["pbc_ws"][0][0].shape[0])**((pbc_lvls+2.0)/len(periodic_axes)) if i in periodic_axes else 0 for i in range(3)]
    tmp = jnp.mgrid[-(dupes[0]//2):(dupes[0]//2)+1,-(dupes[1]//2):(dupes[1]//2)+1,-(dupes[2]//2):(dupes[2]//2)+1].T.reshape((-1,3)) * tree_info["boxlens"][0][0]
    pts_nonper = (pts[None,:,:] + tmp[:,None,:]).reshape((-1,3))
    chrgs_nonper = jnp.tile(chrgs, (tmp.shape[0],))

    ### compute analytic potential
    pot_dir = eval_potential_direct(pts_nonper,chrgs_nonper,pts)
    err = jnp.linalg.norm(pot_dir - pot_FMM)/jnp.linalg.norm(pot_dir)

    assert err < 3e-2