import jax.numpy as jnp
import jaxfmm.hierarchy as trees

### TODO: find better tests for the hierarchy gen

def test_tree_connectivity_unitcube():
    nside, sidelen = 64, 10.0
    pts = (jnp.mgrid[:nside,:nside,:nside].T/(nside-1) * sidelen - sidelen/2).reshape((-1,3))

    max_l = trees.get_max_l(pts.shape[0], 128)
    assert max_l==4

    idcs, rev_idcs, boxcenters, boxlens = trees.balanced_tree(pts,max_l)
    assert idcs.shape[1]==64
    assert jnp.all(idcs < pts.shape[0])
    assert jnp.allclose(pts, pts[idcs].reshape((-1,3))[rev_idcs])
    for i in range(len(boxlens)):
        assert jnp.allclose(boxlens[i], pts[nside//(2**i)-1,0] - pts[0,0])

    mpl_cnct, dir_cnct = trees.gen_connectivity(boxcenters, boxlens,boxcenters,boxlens,no_cross_level=True)[4:6]

    assert mpl_cnct.shape[0] == 667584
    assert dir_cnct.shape[0] == 70336