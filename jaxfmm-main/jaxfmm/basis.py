import jax
import jax.numpy as jnp
from functools import partial

def mpl_idx(m, n):
    r"""
    Compute flattened array position of multipole coefficient C^m_n with order m and degree n.
    """
    return n**2 + (m+n)

def inv_mpl_idx(idx):
    r"""
    Compute order m and degree n of flattened array position idx.
    """
    n = jnp.int32(jnp.floor(jnp.sqrt(idx)+1e-4))
    m = idx - n*(n+1)
    return m, n

@partial(jax.jit, static_argnames=['p'])
def eval_regular_basis(rvec, p):    # TODO: swap signs again and check if this removes the (-1)^m in the rotation matrix?
    r"""
    Evaluate real regular basis functions (Laplace kernel) with a recursion relation, based on 
    [Gumerov, N. A. et al. Fast multipole methods on graphics processors. J. Comp. Phys., B 227, 8290 (2008)].
    """
    x, y, z = rvec[...,0], rvec[...,1], rvec[...,2]
    r2 = (rvec**2).sum(axis=-1)
    coeff = jnp.ones((*rvec.shape[:-1], (p+1)**2))  # coeff[...,mpl_idx(0,0)] = 1

    if(p>0):
        coeff = coeff.at[...,mpl_idx(-1,1)].set(-0.5*y)
        coeff = coeff.at[...,mpl_idx(0,1)].set(z)
        coeff = coeff.at[...,mpl_idx(1,1)].set(0.5*x)
    for n in range(2,p+1):    # first/second recursion: extreme values and their neighbors
        coeff = coeff.at[...,mpl_idx(-n,n)].set((x*coeff[...,mpl_idx(1-n,n-1)] - y*coeff[...,mpl_idx(n-1,n-1)])/(2*n))
        coeff = coeff.at[...,mpl_idx(1-n,n)].set(z*coeff[...,mpl_idx(1-n,n-1)])
        for m in range(2-n,n-1):   # third recursion: all values inbetween
            coeff = coeff.at[...,mpl_idx(m,n)].set(((2*n-1)*z*coeff[...,mpl_idx(m,n-1)] - 
                                                    r2*coeff[...,mpl_idx(m,n-2)])/((n-abs(m))*(n+abs(m))))
        coeff = coeff.at[...,mpl_idx(n-1,n)].set(z*coeff[...,mpl_idx(n-1,n-1)])
        coeff = coeff.at[...,mpl_idx(n,n)].set((x*coeff[...,mpl_idx(n-1,n-1)] + y*coeff[...,mpl_idx(1-n,n-1)])/(2*n))
    return coeff

@partial(jax.jit, static_argnames=['p'])
def eval_regular_basis_grad(rvec, p):
    r"""
    Evaluate the gradient of real regular basis functions (Laplace kernel) with a recursion relation.
    """
    scal_coeff = eval_regular_basis(rvec, p)
    x, y, z = rvec[...,0], rvec[...,1], rvec[...,2]
    coeff = jnp.zeros((*rvec.shape[:-1], (p+1)**2, 3)) # coeff[...,mpl_idx(0,0),:] = 0

    if(p>0):
        coeff = coeff.at[...,mpl_idx(1,1),0].set(0.5)
        coeff = coeff.at[...,mpl_idx(0,1),2].set(1.0)
        coeff = coeff.at[...,mpl_idx(-1,1),1].set(-0.5)
    for n in range(2,p+1):    
        # first recursion: extreme values
        coeff = coeff.at[...,mpl_idx(n,n),:].set(x[...,None]*coeff[...,mpl_idx(n-1,n-1),:] + 
                                                 y[...,None]*coeff[...,mpl_idx(1-n,n-1),:])
        coeff = coeff.at[...,mpl_idx(n,n),0].add(scal_coeff[...,mpl_idx(n-1,n-1)])
        coeff = coeff.at[...,mpl_idx(n,n),1].add(scal_coeff[...,mpl_idx(1-n,n-1)])
        coeff = coeff.at[...,mpl_idx(n,n),:].divide(2*n)

        coeff = coeff.at[...,mpl_idx(-n,n),:].set(x[...,None]*coeff[...,mpl_idx(1-n,n-1),:] - 
                                                  y[...,None]*coeff[...,mpl_idx(n-1,n-1),:])
        coeff = coeff.at[...,mpl_idx(-n,n),0].add(scal_coeff[...,mpl_idx(1-n,n-1)])
        coeff = coeff.at[...,mpl_idx(-n,n),1].subtract(scal_coeff[...,mpl_idx(n-1,n-1)])
        coeff = coeff.at[...,mpl_idx(-n,n),:].divide(2*n)

        # second recursion: neighbors of extreme values
        coeff = coeff.at[...,mpl_idx(n-1,n),:].set(z[...,None]*coeff[...,mpl_idx(n-1,n-1),:])
        coeff = coeff.at[...,mpl_idx(n-1,n),2].add(scal_coeff[...,mpl_idx(n-1,n-1)])
        
        coeff = coeff.at[...,mpl_idx(1-n,n),:].set(z[...,None]*coeff[...,mpl_idx(1-n,n-1),:])
        coeff = coeff.at[...,mpl_idx(1-n,n),2].add(scal_coeff[...,mpl_idx(1-n,n-1)])

        # third recursion: all values inbetween
        for m in range(-n+2,n-1):
            coeff = coeff.at[...,mpl_idx(m,n),:].set((2*n-1)*z[...,None]*coeff[...,mpl_idx(m,n-1),:] - 
                                                     (rvec**2).sum(axis=-1)[...,None]*coeff[...,mpl_idx(m,n-2),:] - 
                                                     2*scal_coeff[...,mpl_idx(m,n-2),None]*rvec)
            coeff = coeff.at[...,mpl_idx(m,n),2].add((2*n-1)*scal_coeff[...,mpl_idx(m,n-1)])
            coeff = coeff.at[...,mpl_idx(m,n),:].divide((n-abs(m)) * (n+abs(m)))
    return coeff

@partial(jax.jit, static_argnames=['p'])
def eval_singular_basis(rvec, p):
    r"""
    Evaluate real singular basis functions (Laplace kernel) with a recursion relation.
    """
    x, y, z = rvec[...,0], rvec[...,1], rvec[...,2]
    r2 = (rvec**2).sum(axis=-1)
    coeff = jnp.zeros((*rvec.shape[:-1], (p+1)**2))
    coeff = coeff.at[...,mpl_idx(0,0)].set(1/jnp.sqrt(r2))

    if(p>0):
        coeff = coeff.at[...,mpl_idx(1,1)].set(coeff[...,mpl_idx(0,0)]*x/r2)
        coeff = coeff.at[...,mpl_idx(0,1)].set(coeff[...,mpl_idx(0,0)]*z/r2)
        coeff = coeff.at[...,mpl_idx(-1,1)].set(coeff[...,mpl_idx(0,0)]*y/r2)
    for n in range(2,p+1): # first/second recursion: extreme values and their neighbors
        coeff = coeff.at[...,mpl_idx(n,n)].set((2*n-1)*(x*coeff[...,mpl_idx(n-1,n-1)] - 
                                                        y*coeff[...,mpl_idx(1-n,n-1)])/r2)
        coeff = coeff.at[...,mpl_idx(n-1,n)].set((2*n-1)*z*coeff[...,mpl_idx(n-1,n-1)]/r2)
        for m in range(2-n,n-1): # third recursion: all values inbetween
            coeff = coeff.at[...,mpl_idx(m,n)].set(((2*n-1)*z*coeff[...,mpl_idx(m,n-1)] - 
                                                    (n-1-m)*(n-1+m)*coeff[...,mpl_idx(m,n-2)])/r2)
        coeff = coeff.at[...,mpl_idx(1-n,n)].set((2*n-1)*z*coeff[...,mpl_idx(1-n,n-1)]/r2)
        coeff = coeff.at[...,mpl_idx(-n,n)].set((2*n-1)*(x*coeff[...,mpl_idx(1-n,n-1)] + 
                                                         y*coeff[...,mpl_idx(n-1,n-1)])/r2)
    return coeff