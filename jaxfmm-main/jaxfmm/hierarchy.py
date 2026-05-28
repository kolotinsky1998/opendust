import jax
import jax.numpy as jnp
from functools import partial
from math import log2, ceil
from jaxfmm.pbc import gen_img_connectivity, gen_pbc_op
from jaxfmm.rotation import precompute_combs
from jaxfmm.util import get_gpu_l2_cache_size

__all__ = ["gen_hierarchy"]

def get_max_l(N_tot, N_max, s = 3):
    r"""
    Compute number of levels in the hierarchy.
    """
    max_l = int(ceil(log2(N_tot/N_max)/s))
    return 0 if max_l < 0 else max_l

@partial(jax.jit, static_argnames = ["max_l", "s"])
def balanced_tree(pts, max_l, s = 3):
    r"""
    Generate a balanced 2^n-tree hierarchy.
    """
    n_chi = 2**s
    idcs = jnp.arange(pts.shape[0])[None,:]

    for l in range(max_l*s):    # carry out s splits on max_l levels in total (we cannot make this a for_i loop as the shape constantly changes)
        splitpos = idcs.shape[1]//2   # split position in the middle
        needpad = idcs.shape[1]%2     # modulo tells us if padding must be inserted

        pts_sorted = pts.at[idcs].get(mode="fill",fill_value=-jnp.nan**2)  # padded values get converted to NaNs - NOTE: due to the argpartition implementation, we need to make sure that all the NaNs have the same sign
        axis_to_split = jnp.argmax(jnp.nanmax(pts_sorted,axis=1) - jnp.nanmin(pts_sorted,axis=1),axis=1)    # nanmax and -min to ignore NaNs
        idcs = idcs[jnp.arange(idcs.shape[0])[:,None],jnp.argpartition(pts_sorted[jnp.arange(axis_to_split.shape[0]),:,axis_to_split],splitpos,axis=1)] # splitting, the NaNs introduced below get transported to the beginning

        # padding so the next array has the correct shape
        idcs = jax.lax.pad(idcs,pts.shape[0],[(0,0,0),(0,needpad,0)])   # pad at the end with out of range values
        idcs = idcs.reshape((-1,idcs.shape[1]//2))                      # now that we padded, we can safely reshape this
    
    idcs = jnp.sort(idcs,axis=1)    # sorting is good for locality but might be overkill TODO: swap only first and last positions instead of full sort?
    rev_idcs = jnp.argsort(idcs.flatten())[:pts.shape[0]]     # reverse sorting indices, to undo the sorting
    pts_sorted = pts.at[idcs].get(mode="fill",fill_value=jnp.nan)  
    boxcenters, boxlens = [jnp.zeros((n_chi**l,3)) for l in range(max_l+1)], [jnp.zeros((n_chi**l,3)) for l in range(max_l+1)]

    for l in range(max_l,-1,-1):
        minc, maxc = jnp.nanmin(pts_sorted,axis=1), jnp.nanmax(pts_sorted,axis=1)    # nanmax and nanmin to ignore NaNs
        boxlens[l] = maxc - minc    # in principle we only need to save the norm of this, but it is nice to have for visualizations
        boxcenters[l] = minc + boxlens[l]/2
        if(l > 0):
            pts_sorted = pts_sorted.reshape((-1,pts_sorted.shape[1]*n_chi,3))
    return idcs, rev_idcs, boxcenters, boxlens

def reduce_max_lvl(pts, idcs, s, old_max_l, new_max_l):
    r"""
    Reduce max level of a balanced tree after it has already been created.
    """
    ldiff = old_max_l - new_max_l
    idcs = idcs.reshape((idcs.shape[0]//(2**(s*ldiff)),-1))
    trim = idcs.shape[1] - (idcs.size - pts.shape[0])//idcs.shape[0]
    idcs = jnp.sort(idcs,axis=-1)[:,:trim] # we know exactly how much padding is accumulated!
    rev_idcs = jnp.argsort(idcs.flatten())[:pts.shape[0]] # also fix the reverse sorting
    return idcs, rev_idcs

def gen_connectivity(boxcenters, boxlens, eval_boxcenters, eval_boxlens, theta = 0.75, s = 3, no_cross_level = False, periodic_axes = ()): # TODO: is there a way to make this JIT compilable?
    r"""
    Compute connectivity information for a given hierarchy.
    """
    n_l, eval_n_l = len(boxcenters), len(eval_boxcenters)  # number of levels of the source and eval hierarchy
    n_chi = 2**s     # number of child boxes per split
    periodic = len(periodic_axes) > 0
    if(periodic):
        img_cnct = gen_img_connectivity(boxlens[0], theta, periodic_axes)[0]
    else:
        img_cnct = jnp.array([[]])

    n_img = img_cnct.shape[0]
    l_src, l_eval = 0, 0
    mpl_cnct, lvl_info = [], []
    nbox_tot = sum([boxcenter.shape[0] for boxcenter in boxcenters])  # this is the value we need to pad the periodic interactions
    non_wellseps = jnp.arange(n_img)[None,:]
    trg_ofs = tuple(jnp.cumsum(jnp.array([0]+[tbox.shape[0] for tbox in eval_boxcenters])).tolist())
    src_ofs = tuple(jnp.cumsum(jnp.array([0]+[sbox.shape[0] for sbox in boxcenters])).tolist())
    keepgoing = True
    while(keepgoing):
        evalend, srcend = (l_eval >= (eval_n_l-1)), (l_src >= (n_l-1))
        keepgoing = not (evalend and srcend)    # we keep going until we reach the final level in both hierarchies
        nbox, nbox_eval = boxcenters[l_src].shape[0], eval_boxcenters[l_eval].shape[0]

        if(periodic):
            rhs = boxcenters[l_src][non_wellseps%nbox] + img_cnct.at[non_wellseps//nbox].get(mode="fill",fill_value=jnp.nan)
        else:
            rhs = boxcenters[l_src].at[non_wellseps].get(mode="fill",fill_value=jnp.nan)
        d = jnp.linalg.norm(eval_boxcenters[l_eval][:,None,:] - rhs,axis=-1)  # distances between boxcenters

        r1 = jnp.linalg.norm(eval_boxlens[l_eval][:,None,:],axis=-1)/2  # eval box radii
        r2 = jnp.linalg.norm(boxlens[l_src][non_wellseps%nbox if periodic else non_wellseps],axis=-1)/2 # source box radii
        tmp = r1 > r2
        R = jnp.where(tmp,r1,r2)     # R = max(r1,r2)
        r = jnp.where(tmp,r2,r1)     # r = min(r1,r2)

        ws = (R + theta*r <= theta*d)     # well-separatedness criterion, NaNs always return False
        non_ws = (R + theta*r > theta*d)  # need to check again to exclude NaNs
        ws_nums = ws.sum(axis=1)          # number of well-separated boxes for each box
        non_ws_nums = non_ws.sum(axis=1)  # number of non-well-separated boxes for each box

        if(ws_nums.sum() > 0):       # do not save empty cnct information
            trgid = jnp.repeat(jnp.arange(nbox_eval),ws_nums) + trg_ofs[l_eval] # TODO: replace this with a CSR style row-index array, which only contains the start and endpoint of a given index.
            srcid = non_wellseps[ws] 
            srcid = jnp.where(srcid>=nbox,(srcid%nbox + (nbox_tot*(srcid//nbox))),srcid)
            srcid += src_ofs[l_src]
            pairs = jnp.stack((trgid,srcid),axis=1)
            mpl_cnct.append(pairs)
            lvl_info.append((l_eval, l_src))

        if(not (evalend and srcend)):
            to_pad_non_ws = jnp.max(non_ws_nums) - non_ws_nums    # how much padding per box must be inserted for non-well-separated boxes
            non_wellsep_padding = jnp.repeat(jnp.cumsum(non_ws_nums),to_pad_non_ws)   # the correct indices for the insert below
            non_wellseps = jnp.insert(non_wellseps[non_ws],non_wellsep_padding,n_img*nbox).reshape((nbox_eval,-1))    # we overwrite the old values here
            if(non_wellseps.shape[1] == 0):   # we are already done, quit out
                break

        r1mean, r2mean = jnp.mean(r1), jnp.mean(r2)
        if((not srcend) and ((r1mean < 1.05 * r2mean) or evalend or no_cross_level)): # TODO: play around with the first condition and factors therein
            l_src += 1          # src_boxes >= eval_boxes -> increase src lvl
            non_wellseps = jnp.repeat(non_wellseps,n_chi,axis=1)*n_chi + jnp.tile(jnp.arange(n_chi),non_wellseps.shape[1])   # indices change
        if((not evalend) and ((r1mean >= 1.05 * r2mean) or srcend or no_cross_level)):
            l_eval += 1             # eval boxes > src boxes -> increase eval lvl
            non_wellseps = jnp.repeat(non_wellseps,n_chi,axis=0)    # size of non-wellseps grows

    lvl_info.append((l_eval, l_src))

    ### stack everything together and cut unneeded information
    boxcenters = jnp.concatenate(boxcenters[:lvl_info[-1][1]+1])
    eval_boxcenters = jnp.concatenate(eval_boxcenters[:lvl_info[-1][0]+1])
    src_ofs = src_ofs[:lvl_info[-1][1]+2]
    trg_ofs = trg_ofs[:lvl_info[-1][0]+2]

    if(len(mpl_cnct)>0):
        mpl_cnct = jnp.concatenate(mpl_cnct,axis=0) # TODO: ensure that this data is sorted properly
        mpl_cnct = mpl_cnct[jnp.argsort(mpl_cnct[:,0])] # sort along axis 0
    else:
        mpl_cnct = jnp.array([[]],dtype=jnp.int32)

    if(non_ws.sum()>0): # if nonempty
        trgid = jnp.repeat(jnp.arange(nbox_eval),non_ws_nums)# + trg_ofs[l_eval] TODO: include the offsets for the dir_cnct or not?
        srcid = non_wellseps[non_ws]# + src_ofs[l_src]
        dir_cnct = jnp.stack((trgid,srcid),axis=1) # TODO: ensure that this data is sorted properly
        dir_cnct = dir_cnct[jnp.argsort(dir_cnct[:,0])] # sort along axis 0
    else:
        dir_cnct = jnp.array([[]],dtype=jnp.int32)

    return boxcenters, eval_boxcenters, src_ofs, trg_ofs, mpl_cnct, dir_cnct, tuple(lvl_info), img_cnct

def gen_hierarchy(pts, eval_pts = None, N_max = 128, theta = 0.77, s = 3, p = 3, periodic_axes = (), pbc_lvls = 5, pbc_no_monopole = True, L0_boxcen = None, L0_boxlen = None, no_cross_level = False, debug_info = False, mem_limit = None, **kwargs):
    r"""
    Generate the balanced tree and connectivity for the FMM.

    :param pts: Array of shape (N,3) containing the positions of N point charges.
    :type pts: jnp.array
    :param eval_pts, optional: Array of shape (N_eval,3) containing the positions of N evaluation points. Defaults to None, where eval_pts = pts.
    :type eval_pts: jnp.array
    :param N_max: Maximum allowed number of point charges per box.
    :type N_max: int, optional
    :param theta: Well-separatedness parameter, determines accuracy.
    :type theta: float, optional
    :param s: How many splits per level and box are been performed. Each box has 2^s children.
    :type s: int, optional
    :param p: Maximum expansion order.
    :type p: int, optional
    :param periodic_axes: Tuple indicating which dimensions (0, 1 and/or 2) are periodic.
    :type periodic_axes: tuple
    :param pbc_lvls: How many virtual levels are introduced in the hierarchy. If pbc_lvls < 0, only the non-well-separated boxes on level zero will be added.
    :type pbc_lvls: int
    :param pbc_no_monopole: Disable the contribution from distant monopoles for PBC. True by default, to negate possible errors introduced by almost zero total charge.
    :type pbc_no_monopole: bool
    :param L0_boxcen: Center of the source box at level zero, useful for periodic boundary conditions. The default (None) will generate the smallest possible axis-aligned box.
    :type L0_boxcen: jnp.array, optional
    :param L0_boxlen: Sidelengths of the source box at level zero, useful for periodic boundary conditions. The default (None) will generate the smallest possible axis-aligned box.
    :type L0_boxlen: jnp.array, optional
    :param no_cross_level: For pts = eval_pts, disable cross-level comparisons.
    :type no_cross_level: bool, optional
    :param debug_info: Whether to include further information in the hierarchy (boxlengths, PBC shifts).
    :type debug_info: bool, optional
    :param mem_limit: Memory limit (in Bytes) for the potential evaluation. The default (None) attempts to estimate memory limits based on the available L2-cache.
    :type mem_limit: float, optional

    :return: Dictionary containing full hierarchy information.
    :rtype: dict
    """
    if(eval_pts is None):
        eval_pts = pts
    periodic = len(periodic_axes) > 0
    sz_eps = 1.01   # safety factor for checking if coordinates are inside level zero boxes
    
    max_l = get_max_l(pts.shape[0], N_max, s)
    eval_max_l = get_max_l(eval_pts.shape[0], N_max, s)
    idcs, rev_idcs, boxcenters, boxlens = balanced_tree(pts, max_l, s)

    old_boxcen, old_boxlen = boxcenters[0].copy(), boxlens[0].copy()
    if(L0_boxcen is not None):  # overwrite with specified value
        boxcenters[0] = L0_boxcen[None,:]
    if(L0_boxlen is not None):  # overwrite with specified value
        boxlens[0] = L0_boxlen[None,:]
    if(jnp.any((old_boxcen[0] - old_boxlen[0]/2 - boxcenters[0]) < -sz_eps*boxlens[0]/2) or jnp.any((old_boxcen[0] + old_boxlen[0]/2 - boxcenters[0]) > sz_eps*boxlens[0]/2)):
        raise ValueError("Level zero box does not contain all source points.")

    if(pts is eval_pts):
        eval_idcs, eval_boxcenters, eval_boxlens = idcs, boxcenters, boxlens
    else:
        no_cross_level = False  # we must deal with cross-level contributions
        eval_idcs, _, eval_boxcenters, eval_boxlens = balanced_tree(eval_pts, eval_max_l, s)

        if(periodic):
            if(jnp.any((eval_boxcenters[0] - eval_boxlens[0]/2 - boxcenters[0]) < -sz_eps*boxlens[0]/2) or jnp.any((eval_boxcenters[0] + eval_boxlens[0]/2 - boxcenters[0]) > sz_eps*boxlens[0]/2)):
                raise ValueError("All evaluation points must be contained in the level zero source box for PBC.")
            eval_boxcenters[0], eval_boxlens[0] = boxcenters[0], boxlens[0]  # for PBC, we need identical level zero boxes

    boxcenters, eval_boxcenters, src_ofs, trg_ofs, mpl_cnct, dir_cnct, lvl_info, img_cnct = gen_connectivity(boxcenters, boxlens, eval_boxcenters, eval_boxlens, theta, s, no_cross_level, periodic_axes)

    ### we adapt the sorting idcs to the "true" max levels - the if statements prevent unnecessary work
    only_dir = (len(lvl_info) == 1)
    idcs_list = [(idcs, rev_idcs) if only_dir else reduce_max_lvl(pts,idcs,s,max_l,lvl_info[-2][1])] # src padding at max mpl level

    if(only_dir or (pts is eval_pts and lvl_info[-2][0]==lvl_info[-2][1])):   # eval padding at max mpl level
        idcs_list.append(idcs_list[0])
    else:
        idcs_list.append(reduce_max_lvl(eval_pts,eval_idcs,s,eval_max_l,lvl_info[-2][0]))

    if(only_dir or (lvl_info[-2][1]==lvl_info[-1][1])):                       # src padding at max dir level
        idcs_list.append(idcs_list[0])
    else:
        idcs_list.append(reduce_max_lvl(pts,idcs,s,max_l,lvl_info[-1][1]))

    if(only_dir or (lvl_info[-2][0]==lvl_info[-1][0])):                       # eval padding at max dir level
        idcs_list.append(idcs_list[1])
    elif(pts is eval_pts and lvl_info[-1][0]==lvl_info[-1][1]):
        idcs_list.append(idcs_list[2])
    else:
        idcs_list.append(reduce_max_lvl(eval_pts,eval_idcs,s,eval_max_l,lvl_info[-1][0]))
    # idcs_list[0][1], idcs_list[2][1] = None, None   # TODO: we do not need these rev idcs, remove them!

    if(periodic):
        PBC_op, pbc_ws = gen_pbc_op(boxlens[0], theta, periodic_axes, p, pbc_lvls, pbc_no_monopole)
        if(pbc_lvls < 0): # optionally, only add non-ws boxes on level 0
            PBC_op = jnp.zeros(((p+1)**2,(p+1)**2))
        if(len(periodic_axes)==3 and not jnp.all(jnp.isclose(boxlens[0],boxlens[0][0,0]))):
            raise ValueError("3D PBC are currently only supported for cubic level zero boxes. You can manually set the dimensions of the level zero box by specifying L0_boxcen and L0_boxlen.")
    else:   # for open boundary conditions, we get no local expansion on level 0
        PBC_op = jnp.zeros(((p+1)**2,(p+1)**2))
        pbc_ws = []

    unqs, invs = precompute_combs(p)

    if mem_limit is None:
        if (pts.device.platform == 'cpu'):  # TODO: check which device we need to use here
            mem_limit = 512 * 1024  # 512 KB TODO: get an actual L2-cache estimate
        else:
            try:
                mem_limit = get_gpu_l2_cache_size(pts.device.local_hardware_id)
            except:
                mem_limit = jnp.inf 
                print("Warning: mem_limit could not be determined automatically, setting mem_limit = jnp.inf.") # TODO: use logging instead of a print

    hierarchy = {"pts": pts,
                 "eval_pts": eval_pts,
                 "idcs": idcs_list,
                 "boxcenters": boxcenters,
                 "eval_boxcenters": eval_boxcenters,
                 "src_ofs": src_ofs,
                 "trg_ofs": trg_ofs,
                 "mpl_cnct": mpl_cnct,
                 "dir_cnct": dir_cnct,
                 "unqs": unqs, "invs": invs,
                 "lvl_info": lvl_info,
                 "img_cnct": img_cnct,
                 "s": s,
                 "p": p,
                 "theta": theta,
                 "periodic_axes": periodic_axes,
                 "pbc_lvls": pbc_lvls if periodic else 0,
                 "pbc_no_monopole": pbc_no_monopole,
                 "PBC_op": PBC_op,
                 "mem_limit": mem_limit
                }
    if(debug_info):
        hierarchy["boxlens"] = jnp.concatenate(boxlens)
        hierarchy["eval_boxlens"] = jnp.concatenate(eval_boxlens)
        hierarchy["pbc_ws"] = pbc_ws
    return hierarchy

@jax.jit
def handle_padding(pts, chrgs, eval_pts, idcs):
    r"""
    Generate the padded arrays required for computing the potential.
    """
    padded_pts = pts.at[idcs[0][0]].get(mode="fill",fill_value=0.0)   # TODO: we could buffer these arrays if desired...
    padded_chrgs = chrgs.at[idcs[0][0]].get(mode="fill",fill_value=0.0)  
    loc_padded_eval_pts = padded_pts if idcs[1] is idcs[0] else eval_pts.at[idcs[1][0]].get(mode="fill",fill_value=0.0)
    dir_padded_pts = padded_pts if idcs[2] is idcs[0] else pts.at[idcs[2][0]].get(mode="fill",fill_value=0.0)
    dir_padded_chrgs = padded_chrgs if idcs[2] is idcs[0] else chrgs.at[idcs[2][0]].get(mode="fill",fill_value=0.0)
    if(idcs[3] is idcs[2]):
        dir_padded_eval_pts = dir_padded_pts
    elif(idcs[3] is idcs[1]): 
        dir_padded_eval_pts = loc_padded_eval_pts
    else:
        dir_padded_eval_pts = eval_pts.at[idcs[3][0]].get(mode="fill",fill_value=0.0)
    return padded_pts, padded_chrgs, loc_padded_eval_pts, dir_padded_pts, dir_padded_chrgs, dir_padded_eval_pts