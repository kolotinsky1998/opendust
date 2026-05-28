import pytest
import jaxfmm.fmm as fmm
import jax.numpy as jnp
from jax import random as rng

@pytest.mark.parametrize('unitcube', [[2**15, {}], [2**15, {"p": 4, "eval_pts": rng.uniform(rng.key(7),(2**12,3),minval=0.25,maxval=0.75)}], [2**12, {"p": 4, "eval_pts": rng.uniform(rng.key(8),(2**15,3),minval=0.25,maxval=0.75)}]], indirect=True)
def test_potential(unitcube):
    chrgs, hier = unitcube
    res = fmm.eval_potential(chrgs, **hier)
    res_dir = fmm.eval_potential_direct(hier["pts"], chrgs, hier["eval_pts"])
    L2err = jnp.linalg.norm(res - res_dir)/jnp.linalg.norm(res_dir)
    assert L2err < 6e-3

@pytest.mark.parametrize('unitcube', [[2**15, {}]], indirect=True)
def test_field(unitcube):
    chrgs, hier = unitcube
    res = fmm.eval_potential(chrgs,**hier, field=True)
    res_dir = fmm.eval_potential_direct(hier["pts"],chrgs, field=True)
    L2err = jnp.linalg.norm(res-res_dir)/jnp.linalg.norm(res_dir)
    assert L2err < 3e-3