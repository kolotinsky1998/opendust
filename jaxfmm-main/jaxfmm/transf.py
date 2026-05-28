import jax
import jax.numpy as jnp
from jaxfmm.rotation import rot_azimuth, rot_polar, cart_to_sph
from jax.scipy.special import factorial as fac
from jaxfmm.basis import mpl_idx, inv_mpl_idx
from functools import partial

@partial(jax.jit, static_argnames=["mem_limit"])
def M2M(coeffs, oldboxdims, newboxdims, unqs, invs, mem_limit = jnp.inf):
    p = invs[-1].shape[0]-1
    n_chi = oldboxdims.shape[0]//newboxdims.shape[0]
    coeffs = coeffs.reshape((coeffs.shape[0]//n_chi,n_chi,coeffs.shape[1]))
    oldboxdims = oldboxdims.reshape((-1,n_chi,3))
    mem = invs[-1].size * oldboxdims.shape[0] * n_chi * coeffs.dtype.itemsize
    batch_size = int((mem_limit/mem) * coeffs.shape[0]) + 1 if mem > mem_limit else coeffs.shape[0]

    def M2M_batched(args):
        coeffs, oldboxdims, newboxdims = args
        r, theta, phi = cart_to_sph(oldboxdims - newboxdims)
        ### rotate
        coeffs = rot_azimuth(coeffs, phi)
        coeffs = rot_polar(unqs,invs,theta,coeffs)
        ### shift in the rotated system
        new_coeffs = coeffs.copy()
        for j in range(1,p+1):
            fact = (r**j) / fac(j)
            ms, ns = inv_mpl_idx(jnp.arange((p-j+1)**2))
            idxs = ms + ns + (ns+j)**2 + j
            new_coeffs = new_coeffs.at[...,idxs].add(fact[...,None] * coeffs[...,:mpl_idx(p-j,p-j)+1])
        ### rotate back
        coeffs = rot_polar(unqs,invs,theta,new_coeffs,True)
        coeffs = rot_azimuth(coeffs, phi,True)
        return coeffs.sum(axis=(-2))
    return jax.lax.map(M2M_batched,[coeffs,oldboxdims,newboxdims[:,None,:]],batch_size = batch_size)

@partial(jax.jit, static_argnames=["mem_limit"])
def L2L(coeffs_glob, oldboxdims, newboxdims, unqs, invs, mem_limit = jnp.inf):
    p = invs[-1].shape[0]-1
    n_chi = newboxdims.shape[0]//oldboxdims.shape[0]
    coeffs_glob = coeffs_glob[:,None,:]
    newboxdims = newboxdims.reshape((-1,n_chi,3))
    mem = invs[-1].size * newboxdims.shape[0] * n_chi * coeffs_glob.dtype.itemsize
    batch_size = int((mem_limit/mem) * coeffs_glob.shape[0]) + 1 if mem > mem_limit else coeffs_glob.shape[0]

    def L2L_batched(args):
        coeffs, oldboxdims, newboxdims = args
        r ,theta, phi = cart_to_sph(newboxdims - oldboxdims)
        ### rotate
        coeffs = rot_azimuth(coeffs,phi,False,True)
        coeffs = rot_polar(unqs,invs,theta,coeffs,False,True)
        ### shift in the rotated system
        new_coeffs = coeffs.copy()
        for j in range(1,p+1):
            fact = (r**j) / fac(j)
            ms, ns = inv_mpl_idx(jnp.arange((p-j+1)**2))
            idxs = ms + ns + (ns+j)**2 + j
            new_coeffs = new_coeffs.at[...,:mpl_idx(p-j,p-j)+1].add(fact[...,None] * coeffs[...,idxs])
        ### rotate back
        coeffs = rot_polar(unqs,invs,theta,new_coeffs,True,True)
        coeffs = rot_azimuth(coeffs,phi,True,True)
        return coeffs
    coeffs = jax.lax.map(L2L_batched,[coeffs_glob,oldboxdims[:,None,:],newboxdims],batch_size = batch_size)
    return coeffs.reshape((-1,coeffs.shape[-1]))

@partial(jax.jit, static_argnames=['mem_limit'])
def M2L(mpls, src_centers, trg_centers, mpl_cnct, unqs, invs, img_cnct = jnp.array([[]]), mem_limit = jnp.inf):
    """
    This function carries out ALL M2L interactions all at once!
    """
    p = invs[-1].shape[0]-1

    locs = jnp.zeros((trg_centers.shape[0],(p+1)**2))
    mem = 2 * mpl_cnct.shape[0] * (p+1)**2 * mpls.itemsize
    batch_size = int((mem_limit/mem)*mpl_cnct.shape[0]) + 1 if mem_limit<mem else mpl_cnct.shape[0]
    n_boxs = src_centers.shape[0]

    def scan_body(buf, idcs):
        if(img_cnct.shape[1] > 0):  # both pbc images and real boxes, shift appropriately
            trg_ids = idcs[:,1]%n_boxs
            r, theta, phi = cart_to_sph(trg_centers[idcs[:,0]] - src_centers[trg_ids] - img_cnct[idcs[:,1]//n_boxs])
        else:   # open boundary conditions
            trg_ids = idcs[:,1]
            r, theta, phi = cart_to_sph(trg_centers[idcs[:,0]] - src_centers[trg_ids])
        coeffs_loc = mpls[trg_ids]

        coeffs_loc = rot_azimuth(coeffs_loc, phi)
        coeffs_loc = rot_polar(unqs,invs,theta,coeffs_loc)
        ### carry out M2L transformation in the rotated system
        for j in range(p+1):
            ns = jnp.arange(j,p+1)
            pws = ns[:,None] + ns[None,:]
            fact = (-1)**(ns[:,None]+j) * fac(pws)[None,...] / ((r[...,None,None])**((pws + 1)[None,...])) # NOTE: the factor (-1) in front probably originates from the sign of the shift vector...
            ### real update
            idxs = mpl_idx(j,ns)

            coeffs_loc = coeffs_loc.at[...,idxs].set((fact * coeffs_loc[...,None,idxs]).sum(axis=-1))
            if(j>0):    # imag update
                idxs = mpl_idx(-j,ns)
                coeffs_loc = coeffs_loc.at[...,idxs].set(-(fact * coeffs_loc[...,None,idxs]).sum(axis=-1))  # imag update - note the inverse sign!
        ### rotate back
        coeffs_loc = rot_polar(unqs,invs,theta,coeffs_loc,True,True)
        coeffs_loc = rot_azimuth(coeffs_loc, phi,True,True)
        
        return buf.at[idcs[:,0]].add(coeffs_loc, indices_are_sorted=True), None

    if(batch_size<mpl_cnct.shape[0]):
        num_pad = batch_size - mpl_cnct.shape[0]%batch_size
        fill_value = jnp.iinfo(mpl_cnct.dtype).max
        scanmpl = jax.lax.pad(mpl_cnct, fill_value, ((0,num_pad,0), (0,0,0))).reshape((-1,batch_size,2))
        locs, _ = jax.lax.scan(scan_body,locs,scanmpl)
    else:
        locs, _ = scan_body(locs,mpl_cnct)

    return locs