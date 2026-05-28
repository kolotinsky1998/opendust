import jax
import jax.numpy as jnp
from jaxfmm.transf import M2M, M2L, L2L
from jaxfmm.basis import eval_regular_basis, eval_regular_basis_grad, inv_mpl_idx
from jaxfmm.hierarchy import handle_padding
from functools import partial

__all__ = ["eval_potential", "eval_potential_direct"]

@partial(jax.jit, static_argnames=['p', 'mem_limit'])
def get_initial_mpls(padded_pts, padded_chrgs, boxcenters, p, mem_limit = jnp.inf):
    r"""
    Get initial multipole expansions for each box on the highest level.
    """
    mem = padded_chrgs.size * (p+1)**2 * padded_chrgs.dtype.itemsize
    batch_size = int((mem_limit/mem) * boxcenters.shape[0]) + 1 if mem > mem_limit else boxcenters.shape[0]

    def impl_body(args):
        padded_pts_loc, padded_chrgs_loc, boxcenters_loc = args
        dist = padded_pts_loc - boxcenters_loc
        return (eval_regular_basis(dist,p) * padded_chrgs_loc).sum(axis=0)
    return jax.lax.map(impl_body, [padded_pts, padded_chrgs[...,None], boxcenters[:,None,:]], batch_size=batch_size)

@partial(jax.jit, static_argnames=['mem_limit', 'src_ofs', 'max_src_lvl'])
def go_down(coeff, boxcenters, src_ofs, max_src_lvl, unqs, invs, mem_limit = jnp.inf):   # TODO: speed up compile time
    r"""
    Using multipole-to-multipole transformation, descend the hierarchy.
    """
    mpls = jnp.zeros((src_ofs[max_src_lvl+1],coeff.shape[1]))
    mpls = mpls.at[src_ofs[max_src_lvl]:src_ofs[max_src_lvl+1]].set(coeff)
    for i in range(max_src_lvl,0,-1):
        mpls = mpls.at[src_ofs[i-1]:src_ofs[i]].set(M2M(mpls[src_ofs[i]:src_ofs[i+1]],boxcenters[src_ofs[i]:src_ofs[i+1]],boxcenters[src_ofs[i-1]:src_ofs[i]], unqs, invs, mem_limit))
    return mpls

@partial(jax.jit, static_argnames=['lvl_info', 'mem_limit', 'trg_ofs'])
def go_up(locs, eval_boxcenters, trg_ofs, unqs, invs, lvl_info, mem_limit = jnp.inf):   # TODO: speed up compile time
    r"""
    Using multipole-to-local and local-to-local transformation, ascend the hierarchy.
    """
    l_eval = lvl_info[-2][0]    # highest eval level needed
    for j in range(l_eval):
        locs = locs.at[trg_ofs[j+1]:trg_ofs[j+2]].add(L2L(locs[trg_ofs[j]:trg_ofs[j+1]], eval_boxcenters[trg_ofs[j]:trg_ofs[j+1]], eval_boxcenters[trg_ofs[j+1]:trg_ofs[j+2]], unqs, invs, mem_limit))
    return locs

@partial(jax.jit, static_argnames=['p', 'field', 'mem_limit'])
def eval_local(locs, padded_eval_pts, rev_idcs, boxcenters, p, field = False, mem_limit = jnp.inf):
    r"""
    Evaluate local expansions on the highest level.
    """
    mem = rev_idcs.size * (p+1)**2 * (1+2*field) * locs.dtype.itemsize
    batch_size = int((mem_limit/mem) * locs.shape[0]) + 1 if mem > mem_limit else locs.shape[0]
    def evloc_body(args):
        pts_loc, box_loc, loc_loc = args
        if(field):
            reg = eval_regular_basis_grad(pts_loc - box_loc, p)
            loc_loc = loc_loc[...,None]  # need additional newaxis for vector components
        else:
            reg = eval_regular_basis(pts_loc - box_loc,p)
        ms, _ = inv_mpl_idx(jnp.arange((p+1)**2))
        prefac = ((2-(ms==0))*(1-2*(ms<0)))[None,None,:]
        if(field):
            prefac = prefac[...,None]
        padded_res = (loc_loc * reg * prefac).sum(axis=2)
        return padded_res
    padded_res = jax.lax.map(evloc_body, [padded_eval_pts, boxcenters[:,None], locs[:,None]], batch_size=batch_size)
    if(field):
        return -padded_res.reshape((-1,3))[rev_idcs] / (4*jnp.pi)
    else:
        return padded_res.flatten()[rev_idcs] / (4*jnp.pi)

@partial(jax.jit, static_argnames=['field', 'mem_limit'])
def eval_direct(padded_pts, padded_chrgs, padded_eval_pts, rev_idcs, dir_cnct, img_cnct = jnp.array([[]]), field = False, mem_limit = jnp.inf):
    r"""
    Evaluate the near-field potential directly (P2P).
    """
    if(dir_cnct.size == 0):   # nothing to compute
        return 0
    mem = dir_cnct.shape[0] * padded_pts.shape[1] * padded_pts.itemsize
    batch_size = int((mem_limit/mem)*dir_cnct.shape[0]) + 1 if mem_limit<mem else dir_cnct.shape[0]
    pot_glob = jnp.zeros(padded_eval_pts.shape[:2+field])
    nbox = padded_pts.shape[0]

    def scan_body(buf, idcs):
        distsvec = padded_eval_pts[idcs[:,0],None,:,:]
        partner = idcs[:,1]
        if(img_cnct.shape[1] != 0): # PBC enabled
               distsvec -= img_cnct[partner//nbox,None,None,:] # image offset
               partner %= nbox    # local image position
        distsvec -= padded_pts[partner,:,None,:]
        distsnorm = jnp.linalg.norm(distsvec,axis=-1)
        distsnorm = 1/jnp.where(distsnorm==0,jnp.inf,distsnorm)
        chrgs = padded_chrgs.at[partner].get(mode="fill",fill_value=0.0)
        if(field):
            pot = ((chrgs[...,None]*(distsnorm**3))[...,None] * distsvec).sum(axis=(1))
        else:
            pot = (distsnorm*chrgs[...,None]).sum(axis=(1))
        return buf.at[idcs[:,0]].add(pot,indices_are_sorted=True), None

    if(batch_size < dir_cnct.shape[0]):
        num_pad = batch_size - dir_cnct.shape[0]%batch_size
        fill_value = jnp.iinfo(dir_cnct.dtype).max
        scandir = jax.lax.pad(dir_cnct, fill_value, ((0,num_pad,0), (0,0,0))).reshape((-1,batch_size,2))
        pot_glob, _ = jax.lax.scan(scan_body,pot_glob,scandir)
    else:
        pot_glob, _ = scan_body(pot_glob, dir_cnct)

    if(field):
        return pot_glob.reshape((-1,3))[rev_idcs]/(4*jnp.pi)
    else:
        return pot_glob.flatten()[rev_idcs]/(4*jnp.pi)

@partial(jax.jit, static_argnames=['field', 'lvl_info', 'mem_limit', 'src_ofs', 'trg_ofs'])
def eval_potential(chrgs, pts, eval_pts, idcs, boxcenters, eval_boxcenters, src_ofs, trg_ofs, mpl_cnct, dir_cnct, unqs, invs, lvl_info, img_cnct, PBC_op, field = False, mem_limit = jnp.inf, **kwargs):
    r"""
    Evaluate the potential/field via FMM. Requires an array of charge values and the hierarchy information generated by gen_hierarchy.
    """
    if(len(lvl_info) == 1): # only direct interactions - compute potential directly
        return eval_potential_direct(pts,chrgs,eval_pts,field)
    p = invs[-1].shape[0] - 1
    pad_arr = handle_padding(pts, chrgs, eval_pts, idcs)
    coeff = get_initial_mpls(pad_arr[0], pad_arr[1], boxcenters[src_ofs[lvl_info[-2][1]]:src_ofs[lvl_info[-2][1]+1]], p, mem_limit)
    coeff = go_down(coeff, boxcenters, src_ofs, lvl_info[-2][1], unqs, invs, mem_limit)
    loc_init = PBC_op@coeff[0]    # PBC far field
    coeff = M2L(coeff, boxcenters, eval_boxcenters, mpl_cnct, unqs, invs, img_cnct, mem_limit)
    coeff = coeff.at[0].add(loc_init)
    coeff = go_up(coeff, eval_boxcenters, trg_ofs, unqs, invs, lvl_info, mem_limit)
    return eval_local(coeff[trg_ofs[-2]:trg_ofs[-1]], pad_arr[2], idcs[1][1], eval_boxcenters[trg_ofs[-2]:trg_ofs[-1]], p, field, mem_limit) + \
           eval_direct(pad_arr[3], pad_arr[4], pad_arr[5], idcs[3][1], dir_cnct, img_cnct, field, mem_limit)

@partial(jax.jit, static_argnames=['field'])
def eval_potential_direct(pts, chrgs, eval_pts = None, field = False):
    r"""
    Evaluate the potential directly via pairwise sums.

    :param pts: Array containing point positions.
    :type padded_pts: jnp.array
    :param chrgs: Array containing point charges.
    :type chrgs: jnp.array
    :param eval_pts: Array containing points to evaluate the potential at. Defaults to pts.
    :type eval_pts: jnp.array, optional
    :param field: Optionally evaluate the field (negative gradient) instead of the potential.
    :type field: bool, optional

    :return: Electrostatic potential (or field) of the points and corresponding charges.
    :rtype: jnp.array
    """
    if(eval_pts is None):
        eval_pts = pts
    res = jnp.zeros(eval_pts.shape[:field+1])
    def eval_direct_body(i, val):
        distsvec = pts[:,:] - eval_pts[i,None,:]
        inv_dists = jnp.linalg.norm(distsvec,axis=-1)
        inv_dists = 1/jnp.where(inv_dists==0,jnp.inf,inv_dists) # take out self-interaction
        if(field):
            val = val.at[i].set(-((chrgs * inv_dists**3)[:,None] * distsvec).sum(axis=0))
        else:
            val = val.at[i].set((chrgs * inv_dists).sum())
        return val
    return jax.lax.fori_loop(0,eval_pts.shape[0],eval_direct_body,res)/(4*jnp.pi)