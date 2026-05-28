import jax.numpy as jnp
from jaxfmm.basis import mpl_idx, inv_mpl_idx
import jax
from jaxfmm.basis import eval_regular_basis, eval_singular_basis
from functools import partial

def gen_img_connectivity(L0_boxlen, theta, periodic_axes):
    r"""
    Similar to gen_connectivity, but computes connectivity information for periodic images.
    The generated patterns are rectangular cuboids, which helps mitigate numerical inaccuracies for the resulting PBC operator.
    """
    R = jnp.linalg.norm(L0_boxlen/2)                           # r = R in this case, all boxes are the same
    num_images = jnp.int32((1+theta)/(theta*L0_boxlen) * R)[0] # lower bound of non-ws images in axis directions
    non_per = jnp.array([False if i in periodic_axes else True for i in range(3)])
    num_images = num_images.at[non_per].set(0)
    mul_facs = 2*num_images + 1                                # total number of boxes in axis directions
    ids = [slice(-num_images[i],num_images[i]+1) if i in periodic_axes else slice(0,1) for i in range(3)]
    img_ids = jnp.reshape(jnp.mgrid[ids].T,(-1,3))  # spawn all the images

    L0_boxlen_up = L0_boxlen * mul_facs   # boxlength on the parent level
    R_up = jnp.linalg.norm(L0_boxlen_up/2)
    num_images_up = jnp.int32((1+theta)/(theta*L0_boxlen_up) * R_up)[0] # upper bound of ws images in axis directions
    ids_up = [slice(-num_images_up[i],num_images_up[i]+1) if i in periodic_axes else slice(0,1) for i in range(3)]
    img_ids_up = jnp.reshape(jnp.mgrid[ids_up].T,(-1,3))                                # spawn parent images
    img_ids_up = img_ids_up[jnp.any(img_ids_up!=0,axis=-1)]                             # we take out the non-ws area
    img_ids_up = ((mul_facs*img_ids_up)[:,None,:] + img_ids[None,:,:]).reshape((-1,3))  # spawn children for each remaining parent

    return img_ids*L0_boxlen, img_ids_up*L0_boxlen, mul_facs

@partial(jax.jit, static_argnames = ["p"])
def gen_M2Mop(p, dists, reg_zeros = []):    # TODO: generate via rotations
    r"""
    Generate M2M operator matrix for PBC.
    """
    reg = eval_regular_basis(dists,p).sum(axis=0)   # sign of dists does not matter here (symmetry)
    reg = reg.at[reg_zeros].set(0)  # manually set terms to zero that should be zero (symmetry)
    M2Mop = jnp.zeros(((p+1)**2,(p+1)**2))

    for j in range(p+1):
        for k in range(0,j+1):  # Real coeffs
            for n in range(j+1):
                for m in range(max(k+n-j,-n),min(k+j-n,n)+1):
                    M2Mop = M2Mop.at[mpl_idx(k,j),mpl_idx(abs(k-m),j-n)].add((-1)**((abs(k)-abs(m)-abs(k-m))//2) * reg[...,mpl_idx(abs(m),n)])
                    M2Mop = M2Mop.at[mpl_idx(k,j),mpl_idx(-abs(k-m),j-n)].add(-(-1)**((abs(k)-abs(m)-abs(k-m))//2) * jnp.sign(m)*jnp.sign(k-m)*reg[mpl_idx(-abs(m),n)])
        for k in range(-j,0):   # Imag coeffs
            for n in range(j+1):
                for m in range(max(k+n-j,-n),min(k+j-n,n)+1):
                    M2Mop = M2Mop.at[mpl_idx(k,j),mpl_idx(-abs(k-m),j-n)].add(-(-1)**((abs(k)-abs(m)-abs(k-m))//2) * jnp.sign(k-m)*reg[...,mpl_idx(abs(m),n)])
                    M2Mop = M2Mop.at[mpl_idx(k,j),mpl_idx(abs(k-m),j-n)].add(-(-1)**((abs(k)-abs(m)-abs(k-m))//2) * jnp.sign(m)*reg[...,mpl_idx(-abs(m),n)])

    return M2Mop

@partial(jax.jit, static_argnames = ["p"])
def gen_M2Lop(p, dists, sing_zeros = []): # TODO: generate via rotations
    r"""
    Generate M2L operator matrix for PBC.
    """
    sing = eval_singular_basis(dists,2*p).sum(axis=0)   # sign of dists does not matter here (symmetry)
    sing = sing.at[sing_zeros].set(0) # manually set terms to zero that should be zero (symmetry)
    M2Lop = jnp.zeros(((p+1)**2,(p+1)**2))

    for j in range(p+1):
        for k in range(1,j+1):  # -Imag coeffs!
            for n in range(p+1):
                for m in range(max(k-j-n,-n),min(j+n+k,n)+1):
                    M2Lop = M2Lop.at[mpl_idx(k,j),mpl_idx(abs(m),n)].add((-1)**((abs(k-m)-abs(k)-abs(m))//2) * (-jnp.sign(m-k)*sing[mpl_idx(-abs(m-k),j+n)]))
                    M2Lop = M2Lop.at[mpl_idx(k,j),mpl_idx(-abs(m),n)].add((-1)**((abs(k-m)-abs(k)-abs(m))//2) * jnp.sign(m)*sing[mpl_idx(abs(m-k),j+n)])
        for k in range(-j,1):   # Real coeffs!
            for n in range(p+1):
                for m in range(max(k-j-n,-n),min(j+n+k,n)+1):
                    M2Lop = M2Lop.at[mpl_idx(k,j),mpl_idx(abs(m),n)].add((-1)**((abs(k-m)-abs(k)-abs(m))//2) * sing[mpl_idx(abs(m-k),j+n)])
                    M2Lop = M2Lop.at[mpl_idx(k,j),mpl_idx(-abs(m),n)].add((-1)**((abs(k-m)-abs(k)-abs(m))//2) * jnp.sign(m)* jnp.sign(m-k) * sing[mpl_idx(-abs(m-k),j+n)])
    return M2Lop

def gen_pbc_op(L, theta, periodic_axes, p, pbc_lvls, pbc_no_monopole):
    r"""
    Generate full PBC operator matrix.
    """
    par_zero = [jnp.zeros(((i+1)*p+1)**2,dtype=jnp.uint8) for i in range(2)]
    for i, func in enumerate((eval_regular_basis,eval_singular_basis)):
        par = func(jnp.array([1,2,3]),(i+1)*p)
        uneven = jnp.any(jnp.stack((func(jnp.array([-1,2,3]),(i+1)*p)/par < 0,
                                    func(jnp.array([1,-2,3]),(i+1)*p)/par < 0,
                                    func(jnp.array([1,2,-3]),(i+1)*p)/par < 0),axis=-1),axis=-1)  # as long as the pattern is symmetric, these cancel
        #cubic = jnp.abs(func(jnp.array([1,1,1]),p))<1e-6   # these components only cancel for patterns that are both symmetric and cubic
        par_zero[i] = jnp.where(par_zero[i].at[uneven].set(1))[0]
    reg_zeros, sing_zeros = par_zero

    img_non_ws, img_ws, mul_facs = gen_img_connectivity(L,theta,periodic_axes)
    M2Mop = jnp.eye((p+1)**2)
    res = gen_M2Lop(p,img_ws,sing_zeros)
    img_cnct_list = [[img_non_ws, img_ws]]
    for i in range(1,pbc_lvls+1):
        L *= mul_facs
        img_non_ws, img_ws, mul_facs = gen_img_connectivity(L,theta,periodic_axes)
        M2Mop = gen_M2Mop(p, img_non_ws,reg_zeros)@M2Mop
        M2Lop = gen_M2Lop(p, img_ws,sing_zeros)
        res += M2Lop@M2Mop
        img_cnct_list.append([img_non_ws, img_ws])
    if(pbc_no_monopole):    # set monopole contribution to zero
        res = res.at[0,0].set(0)

    ### edit the pbc op such that it works with the new basis TODO: deal with this in M2Lop, M2Mop
    ms, ns = inv_mpl_idx(jnp.arange((p+1)**2))
    flipidx = mpl_idx(-ms,ns)
    res *= (-1+2*(ms<0))**ns[:,None]
    res = res[flipidx,:]

    return res, img_cnct_list