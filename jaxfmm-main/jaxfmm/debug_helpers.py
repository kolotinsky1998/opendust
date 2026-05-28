import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxfmm.basis import inv_mpl_idx, eval_singular_basis
from time import perf_counter
import numpy as np
import pyvista as pv
import os
from functools import partial
import matplotlib.pyplot as plt

__all__ = ["gen_stats", "gen_hierarchy_vtk", "gen_wellsep_vtk", "gen_pts_vtk", "time_function"]

def gen_stats(pts, eval_pts, idcs, boxcenters, eval_boxcenters, mpl_cnct, dir_cnct, img_cnct, lvl_info, s, periodic_axes, p, theta, print_stats = False, **kwargs):
    """Generate hierarchy stats. Use the hierarchy information from gen_hierarchy as input."""
    stats = {"p": p,
             "theta": theta,
             "max_l_src": lvl_info[-1][1],
             "max_l_trg": lvl_info[-1][0],
             "num_pts": pts.shape[0],
             "num_eval_pts": eval_pts.shape[0],
             "num_child": 2**s,
             "pts_per_box": idcs[2][0].shape[1],
             "eval_pts_per_box": idcs[3][0].shape[1]}
    periodic = len(periodic_axes)>0
    mem_hierarch = idcs[0][0].nbytes + idcs[0][1].nbytes
    if(idcs[1] is not idcs[0]):
        mem_hierarch += idcs[1][0].nbytes + idcs[1][1].nbytes
    if(idcs[2] is not idcs[0]):
        mem_hierarch += idcs[2][0].nbytes + idcs[2][1].nbytes
    if((idcs[3] is not idcs[1]) and (idcs[3] is not idcs[2])):
        mem_hierarch += idcs[3][0].nbytes + idcs[3][1].nbytes
    mem_hierarch = sum([elem.nbytes for elem in (boxcenters, mpl_cnct, dir_cnct, img_cnct)])
    if(boxcenters is not eval_boxcenters):
        mem_hierarch += eval_boxcenters.nbytes
    
    stats["M2L_num"] = mpl_cnct.size//2
    stats["lvl_info"] = lvl_info
    stats["direct_num"] = dir_cnct.size//2
    stats["compression_ratio"] = dir_cnct.shape[0] * idcs[3][0].shape[1] * idcs[2][0].shape[1] / (pts.shape[0]*eval_pts.shape[0])
    stats["mem_hierarch"] = mem_hierarch
    stats["mem_coeffs"] = 4*2*((p+1)**2)*((2**s)**(stats["max_l_src"]+1)-1)/7 # 4 bytes, 2 arrays, (p+1)**2 coeffs and nboxes
    stats["mem_pts_chrgs"] = (pts.nbytes*4)/3
    if(eval_pts is not pts):
        stats["mem_pts_chrgs"] += eval_pts.nbytes
    if(print_stats):
        print("---------------------FMM Hierarchy Stats---------------------")
        print("p = %i, theta = %.2f, %i children per box"%(p, theta, stats["num_child"]))
        print("%i sources, %i targets"%(stats["num_pts"],stats["num_eval_pts"]))
        print("%i sources per leaf, %i targets per leaf"%(stats["pts_per_box"], stats["eval_pts_per_box"]))
        print("max src lvl: %i, max trg lvl: %i"%(stats["max_l_src"], stats["max_l_trg"]))
        if(periodic): 
            print("Periodic boundary on axes: ", periodic_axes)
        print("")
        print("Total interactions: %11i"%(stats["M2L_num"]+stats["direct_num"]))
        print("      of which M2L: %11i"%(stats["M2L_num"]))
        print("      of which P2P: %11i"%(stats["direct_num"]))
        print("Compression (vs dir.): %.2e"%(stats["compression_ratio"]))
        print("")
        print("Memory used by the hierarchy:            %.2e Bytes"%stats["mem_hierarch"])
        print("Memory used by mpl + local coeffs (p=%i): %.2e Bytes"%(p, stats["mem_coeffs"])) 
        print("Memory used by the points + charges:     %.2e Bytes"%stats["mem_pts_chrgs"])
        print("-------------------------------------------------------------")
    return stats

def make_pyvista_mesh(boxcenters, boxlens):
    r"""
    Generate pyvista mesh of boxes with centers at boxcenters and diagonals boxlens.
    """
    pv_shifts = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
    pts = np.zeros((boxcenters.shape[0],8,3))
    for i in range(pv_shifts.shape[0]):
        pts[:,i,:] = boxcenters + boxlens/2 * pv_shifts[i,None,:]
    pts = pts.reshape((-1,3))
    cells = np.arange(pts.shape[0],dtype=np.int32).reshape((-1,8))
    return pv.UnstructuredGrid({pv.CellType.HEXAHEDRON: cells}, pts)

def gen_hierarchy_vtk(boxcenters, boxlens, src_ofs, eval_boxcenters=None, eval_boxlens=None, trg_ofs=None, dir="hierarchy", **kwargs):
    r"""
    Output a series of vtk files showing the FMM hierarchy on every level. Does not show virtual PBC images.
    """
    if(eval_boxcenters is None or eval_boxlens is None or trg_ofs is None):
        eval_boxcenters = boxcenters
        eval_boxlens = boxlens
        trg_ofs = src_ofs

    eval_max_l = len(trg_ofs)-1
    max_l = len(src_ofs)-1
    if not os.path.exists(dir):
        os.makedirs(dir)
    for l in range(max_l):
        mesh = make_pyvista_mesh(boxcenters[src_ofs[l]:src_ofs[l+1]],boxlens[src_ofs[l]:src_ofs[l+1]])
        mesh.cell_data["src_level"] = np.ones(mesh.points.shape[0]//8)*l
        mesh.save("%s/src_level_%i.vtk"%(dir,l))
    
    if(eval_boxcenters is not boxcenters or eval_boxlens is not boxlens):
        for l in range(eval_max_l):
            mesh = make_pyvista_mesh(eval_boxcenters[trg_ofs[l]:trg_ofs[l+1]],eval_boxlens[trg_ofs[l]:trg_ofs[l+1]])
            mesh.cell_data["eval_level"] = np.ones(mesh.points.shape[0]//8)*l
            mesh.save("%s/eval_level_%i.vtk"%(dir,l))

def gen_pts_vtk(pts, eval_pts, dir="hierarchy", **kwargs):
    r"""
    Output the hierarchy points as vtk.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
    point_cloud = pv.PolyData(np.array(pts))
    point_cloud.save("%s/src_pts.vtk"%(dir))
    point_cloud = pv.PolyData(np.array(eval_pts))
    point_cloud.save("%s/eval_pts.vtk"%(dir))

def gen_wellsep_vtk(id, boxcenters, boxlens, eval_boxcenters, eval_boxlens, mpl_cnct, dir_cnct, src_ofs, trg_ofs, img_cnct, s, pbc_ws, dir="hierarchy", **kwargs):
    r"""
    Output a vtk file showing all the boxes that are considered for potential calculation, given an id of a box on the highest level. Includes virtual PBC images.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
    n_chi = 2**s
    n_l_src = len(src_ofs) - 1
    n_l_trg = len(trg_ofs) - 1
    nbox_src = boxcenters.shape[0]
    periodic = len(pbc_ws)>0
    meshlist = []

    ### periodic far-field images
    L = boxlens[0]
    for l, img_cnct_ws in enumerate(pbc_ws):
        mesh = make_pyvista_mesh(boxcenters[0] - img_cnct_ws[1],jnp.tile(L,(img_cnct_ws[1].shape[0],1)))
        mesh.cell_data["eval_level"] = jnp.ones(mesh.points.shape[0]//8)*(-l)
        mesh.cell_data["src_level"] = jnp.ones(mesh.points.shape[0]//8)*(-l)
        meshlist.append(mesh)
        L += jnp.ptp(img_cnct_ws[0],axis=0)

    ### M2L
    if(mpl_cnct.size > 0):
        for i in range(n_l_trg):
            loc_id = id//(n_chi**(n_l_src-i-1)) + src_ofs[i]
            extract = mpl_cnct[(mpl_cnct[:,0] == loc_id),1]
            if(extract.size>0):
                if(periodic):
                    mesh = make_pyvista_mesh(boxcenters[extract%nbox_src] + img_cnct[extract//nbox_src], boxlens[extract%nbox_src])
                else:
                    mesh = make_pyvista_mesh(boxcenters[extract], boxlens[extract])
                mesh.cell_data["eval_level"] = jnp.ones(mesh.points.shape[0]//8) * i
                src_level = (extract[:,None]%nbox_src>=jnp.array(src_ofs)[None,:]).sum(axis=1) - 1
                mesh.cell_data["src_level"] = jnp.ones(mesh.points.shape[0]//8) * src_level
                meshlist.append(mesh)

    ### direct
    nbox_src_dir = src_ofs[-1] - src_ofs[-2]
    if(dir_cnct.size > 0):
        extract = dir_cnct[(dir_cnct[:,0] == id),1] # NOTE: the dir_cnct has no offset currently, so we add it manually
        boxcenters_loc = boxcenters[src_ofs[-2]:src_ofs[-1]]
        boxlens_loc = boxlens[src_ofs[-2]:src_ofs[-1]]
        if(extract.size>0):
            if(periodic):
                mesh = make_pyvista_mesh(boxcenters_loc[extract%nbox_src_dir] + img_cnct[extract//nbox_src_dir], boxlens_loc[extract%nbox_src_dir])
            else:
                mesh = make_pyvista_mesh(boxcenters_loc[extract], boxlens_loc[extract])
            mesh.cell_data["eval_level"] = jnp.ones(mesh.points.shape[0]//8) * n_l_trg   # NOTE: we do not subtract 1 to mark the direct interactions
            mesh.cell_data["src_level"] = jnp.ones(mesh.points.shape[0]//8) * n_l_src
            meshlist.append(mesh)

    ### the id box itself
    mesh = make_pyvista_mesh(eval_boxcenters[id+trg_ofs[-2]][None,:],eval_boxlens[id+trg_ofs[-2]][None,:]*1.001)   # the multiplication fixes pyvista getting confused for eval_pts = pts
    mesh.cell_data["eval_level"] = jnp.ones(1) * (n_l_trg + 1)
    mesh.cell_data["src_level"] = jnp.ones(1) * (n_l_src + 1)
    meshlist.append(mesh)

    mergedmesh = pv.merge(meshlist)
    mergedmesh.save("%s/wellsep_%i.vtk"%(dir,id))

@partial(jax.jit, static_argnames=['p'])
def eval_mpls(mpls, eval_pts, boxcenters, p):
    r"""
    Evaluate multipole expansions.
    """
    sing = eval_singular_basis(eval_pts - boxcenters[None,:],p)
    ms, _ = inv_mpl_idx(jnp.arange((p+1)**2))
    prefac = ((2-(ms==0))*(1-2*(ms<0)))[None,None,:]
    res = (mpls * sing * prefac).sum(axis=2)
    return res / (4*jnp.pi)

@partial(jax.jit, static_argnames=['p'])
def get_locs(padded_pts, padded_chrgs, boxcenters, p):
    r"""
    Obtain local expansions directly.
    """
    dist = padded_pts - boxcenters[:,None]
    sing = eval_singular_basis(dist,p)
    return (sing * padded_chrgs[...,None]).sum(axis=1)

def binom(x, y):
  return jnp.exp(jsp.special.gammaln(x + 1) - jsp.special.gammaln(y + 1) - jsp.special.gammaln(x - y + 1))

def gen_multipole_dist(m, n, eps = 0.5):
    r"""
    Generate a point charge distribution corresponding to a specific multipole moment (Majic, Matt. (2022). 
    Point charge representations of multipoles. European Journal of Physics. 43. 10.1088/1361-6404/ac578b.)
    """
    if(m == 0):   # axial
        k = jnp.arange(-n, n+1, 2)
        chrgs = (-1)**((n-k)/2) * binom(n, (n-k)/2.0) / (jsp.special.factorial(n) * (2*eps)**n)
        pts = jnp.zeros((k.shape[0],3))
        pts = pts.at[:,2].set(k*eps)
    else:         # (stacked) bracelet
        rotate = m < 0
        m = abs(m)      # we work with the real basis and rotate later
        knum = n-m+1
        jnum = 2*m
        j = jnp.tile(jnp.arange(jnum),knum)
        k = jnp.repeat(jnp.arange(-n+m,n-m+1,2),jnum)
        phi = (j-0.5) * jnp.pi/m if rotate else j * jnp.pi/m
        pts = jnp.array([eps*jnp.cos(phi), eps*jnp.sin(phi), k*eps]).T
        chrgs = 4**(m-1) * jsp.special.factorial(m-1) / ((2*eps)**n * jsp.special.factorial(n-m)) * (-1)**((n-m-k)/2 + j) * binom(n-m,(n-m-k)/2)
    return pts, chrgs

def time_function(func, nruns = 10, print_times = True, **kwargs):
    r"""
    Time function func, obtaining the best of nruns runs and optionally printing the results. Function args are supplied as additional kwargs.
    """
    best = jnp.inf
    for i in range(nruns):
        T0 = perf_counter()
        res = jax.block_until_ready(func(**kwargs))
        T1 = perf_counter()
        best = min(T1-T0,best)
        if(i==0):
            comp = best
    if(print_times):
        print("%s compilation: %.2e s"%(func.__name__,comp))
        print("%s runtime: %.2e s"%(func.__name__,best))
    return res, comp, best

def plot_sparsity(mpl_cnct, dir_cnct, trg_ofs, src_ofs, show_plot = True, **kwargs):
    r"""
    Plot the sparsity pattern of the interactions in a given hierarchy. This gives an indication of how well the FMM compresses the number of interactions compared to a direct evaluation.
    """
    s = 0.5
    fig, axs = plt.subplots(1,2)
    if(mpl_cnct.size < 1):
        mpl_cnct = np.array([[[np.nan,np.nan]]])
    if(dir_cnct.size < 1):
        dir_cnct = np.array([[[np.nan,np.nan]]])
    data = [(dir_cnct[...,0]+trg_ofs[-2],dir_cnct[...,1]+src_ofs[-2]), (mpl_cnct[...,0],mpl_cnct[...,1])]

    titles = ["Near Field Connectivity", "Far Field Connectivity"]
    fig.set_size_inches(14,6)
    for arr, title, ax in zip(data, titles, axs):
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.set_title(title, y=-0.1)
        ax.vlines(jnp.array(trg_ofs)[1:-1]-0.5,0,trg_ofs[-1],'lightgray')
        ax.hlines(jnp.array(src_ofs)[1:-1]-0.5,0,src_ofs[-1],'lightgray')
        ax.scatter(arr[0],arr[1],s=s,zorder=500)
        ax.set_xlabel("Target ID")
        ax.set_ylabel("Source ID")
        ax.set_xlim(trg_ofs[0]-1,trg_ofs[-1])
        ax.set_ylim(src_ofs[0]-1,src_ofs[-1])
        ax.yaxis.set_inverted(True)

    if(show_plot):
        plt.show()
    return fig, axs